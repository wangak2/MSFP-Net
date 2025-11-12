import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm3d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W, D)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_threshold: float = 0.5):
        super().__init__()
        self.gn = GroupBatchnorm3d(oup_channels, group_num=group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = F.softmax(self.gn.gamma, dim=0)
        reweights = self.sigmoid(gn_x * w_gamma)
        info_mask = w_gamma > self.gate_threshold
        noninfo_mask = w_gamma <= self.gate_threshold
        x_1 = info_mask * reweights * x
        x_2 = noninfo_mask * reweights * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class FF(nn.Module):
    def __init__(self, dim, layer_scale_init_value=0.5):
        super().__init__()
        self.conv_1 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv_4 = nn.Conv3d(dim * 3, dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm3d(dim)
        self.nonlin = nn.Sigmoid()
        self.act = nn.LeakyReLU()

    def forward(self, x, y, z):
        x_w = self.conv_1(x)
        y_w = self.conv_2(y)
        z_w = self.conv_3(z)
        w = x_w + y_w + z_w
        w = self.norm(w)
        w = self.nonlin(w)
        output = torch.cat((x, y, z), 1)
        output = self.conv_4(output)
        max_map = self.max_pool(output)
        output = max_map + output
        output = w * output
        return output


class MultiFrequencyChannelAttention3D(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_w, dct_h, dct_d,
                 frequency_branches=8,
                 frequency_selection='low',
                 reduction=16,
                 layer_scale_init_value=0.5):
        super(MultiFrequencyChannelAttention3D, self).__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_w = dct_w
        self.dct_h = dct_h
        self.dct_d = dct_d

        mapper_x, mapper_y, mapper_z = get_freq_indices_3d(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_w // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_h // 8) for temp_y in mapper_y]
        mapper_z = [temp_z * (dct_d // 6) for temp_z in mapper_z]

        assert len(mapper_x) == len(mapper_y) == len(mapper_z)

        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx),
                                 self.get_dct_filter(dct_w, dct_h, dct_d, mapper_x[freq_idx],
                                                     mapper_y[freq_idx], mapper_z[freq_idx], in_channels))

        self.conv = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm3d(in_channels // 2)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm3d(in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        batch_size, C, W, H, D = x.shape
        x_pooled = x
        if W != self.dct_w or H != self.dct_h or D != self.dct_d:
            x_pooled = F.adaptive_avg_pool3d(x, (self.dct_w, self.dct_h, self.dct_d))

        sum_w = 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                sum_w += x_pooled * params
        sum_w = sum_w / self.num_freq

        sum_w = self.conv(sum_w)
        sum_w = self.norm2(sum_w)
        sum_w = self.act(sum_w)
        sum_w = self.conv2(sum_w)
        w = self.norm(sum_w)
        w = torch.sigmoid(w)
        x1 = x * w
        return x1

    def get_dct_filter(self, tile_size_x, tile_size_y, tile_size_z, mapper_x, mapper_y, mapper_z, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y, tile_size_z)
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                for t_z in range(tile_size_z):
                    dct_filter[:, t_x, t_y, t_z] = (self.build_filter(t_x, mapper_x, tile_size_x) *
                                                    self.build_filter(t_y, mapper_y, tile_size_y) *
                                                    self.build_filter(t_z, mapper_z, tile_size_z))
        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return result if freq == 0 else result * math.sqrt(2)


def get_freq_indices_3d(method):
    assert method in ['low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    all_low_indices_x = [0, 1, 0, 0, 1, 1, 0, 1, 3, 5, 5, 0, 0, 4, 3, 2, 2]
    all_low_indices_y = [0, 0, 1, 0, 1, 1, 1, 0, 5, 2, 3, 4, 0, 5, 2, 5, 1]
    all_low_indices_z = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 0, 3, 2, 3, 1, 0]
    mapper_x = all_low_indices_x[:num_freq]
    mapper_y = all_low_indices_y[:num_freq]
    mapper_z = all_low_indices_z[:num_freq]
    return mapper_x, mapper_y, mapper_z


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, stride=(1, 1, 1), dilation=1, layer_scale_init_value=1e-6):
        super(ResBlock, self).__init__()
        self.conv_p1 = nn.Conv3d(planes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv3d(planes // 2, planes // 2, kernel_size=kernel_size,
                                padding=padding, stride=stride, groups=planes // 2)
        self.conv1 = nn.Conv3d(planes // 2, planes, kernel_size=1, stride=1, padding=0, groups=planes // 2)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=1, stride=1, padding=0, groups=planes // 2)
        self.norm2_I = nn.InstanceNorm3d(planes // 2, affine=True)
        self.norm3_I = nn.InstanceNorm3d(planes, affine=True)
        self.nonlin = nn.LeakyReLU()

    def channel_shuffle(self, x, groups):
        B, C, H, W, D = x.size()
        channels_per_group = C // groups
        x = x.view(B, groups, channels_per_group, H, W, D)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(B, -1, H, W, D)

    def forward(self, x):
        x1 = self.conv_p1(x)
        out1 = self.dwconv(x1)
        out1 = self.norm2_I(out1)
        out1 = self.nonlin(out1)
        out2 = self.conv1(out1)
        out2 = self.conv2(out2)
        out2 = self.norm3_I(out2)
        out2 = self.nonlin(out2)
        return self.channel_shuffle(out2, 4)


class MFMS(nn.Module):
    def __init__(self, in_features, filters, layer_scale_init_value=1e-6):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.dim = in_features
        self.res1 = ResBlock(filters, filters, kernel_size=7, padding=(3, 3, 3), stride=(1, 1, 1), dilation=4)
        self.res2 = ResBlock(filters, filters, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1), dilation=2)
        self.res3 = ResBlock(filters, filters, kernel_size=3, padding=(1, 1, 1), stride=(1, 1, 1), dilation=1)
        self.norm2 = nn.BatchNorm3d(filters)
        self.conv_down = nn.Conv3d(in_features, filters, kernel_size=3, stride=2, padding=1)
        self.ff = FF(dim=filters)
        self.sru = SRU(oup_channels=filters, group_num=2, gate_threshold=0.5)

        c2w = {64: 48, 128: 24, 256: 12, 512: 6}
        c2hd = {64: 64, 128: 32, 256: 16, 512: 8}
        self.mfca = MultiFrequencyChannelAttention3D(filters, c2hd[filters], c2hd[filters], c2w[filters])

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((filters)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_down(x)
        res = x
        x = self.sru(x)
        x1 = self.res1(x)
        x2 = self.res2(x)
        x3 = self.res3(x)
        out1 = self.ff(x1, x2, x3)
        out1 = self.mfca(out1)

        out1 = out1.permute(0, 2, 3, 4, 1)
        res = res.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            out1 = self.gamma * out1
            res = (1 - self.gamma) * res
        out1 = out1.permute(0, 4, 1, 2, 3)
        res = res.permute(0, 4, 1, 2, 3)

        out = out1 + res
        out = self.norm2(out)
        out = self.act(out)
        return out