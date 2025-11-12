import torch
import torch.nn as nn
import torch.nn.functional as F

from unetr_block import UnetrBasicBlock, UnetrUpBlock
from PIEBlock import PIE
from MFMSBlock import MFMS
from timm.models.layers import DropPath

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format != "channels_first":
            raise NotImplementedError("Only channels_first is supported for 3D.")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # x: (B, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class encoder_block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1, 1) * x
        x = input + self.drop_path(x)
        return x


class msfp_conv(nn.Module):
    def __init__(self, in_chans=1, dims=[64, 128, 256, 512], out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.out_indices = out_indices

        # Stem and downsampling layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, dims[0], kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(dims[0]),
            nn.ReLU(),
        )
        self.downsample_layers = nn.ModuleList([stem])
        self.downsample_layers.append(MFMS(in_features=dims[0], filters=dims[1]))
        self.downsample_layers.append(MFMS(in_features=dims[1], filters=dims[2]))
        self.downsample_layers.append(MFMS(in_features=dims[2], filters=dims[3]))

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.resconv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.resconv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.pconv = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = self.pconv(x)
        out = self.resconv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.resconv2(out)
        out = self.norm(out)
        out = self.act(out)
        return out + residual


class MSFP(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=2,
        feat_size=[64, 128, 256, 512],
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = feat_size

        self.msfp_3d = msfp_conv(
            in_chans=self.in_chans,
            dims=self.feat_size,
            out_indices=[0, 1, 2, 3]
        )

        # Cross-resolution fusion convs
        self.conv2_to3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_to2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.conv4_to5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5_to4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, bias=False)

        # Decoder upsample layers
        self.transposeconv_stage1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage4 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, bias=False)

        # SPM blocks
        self.res1 = PIE(64, scale=8)
        self.res2 = PIE(128, scale=4)
        self.res3 = PIE(256, scale=2)
        self.res4 = PIE(512, scale=1)

        # Norm & activation
        self.norm1 = nn.BatchNorm3d(64)
        self.norm2 = nn.BatchNorm3d(128)
        self.norm3 = nn.BatchNorm3d(256)
        self.norm4 = nn.BatchNorm3d(512)
        self.act = nn.ReLU()

        # Decoder refinement blocks
        self.stage0_de = ResBlock(512 * 2, 512)
        self.stage1_de = ResBlock(256 * 2, 256)
        self.stage2_de = ResBlock(128 * 2, 128)
        self.stage3_de = ResBlock(64 * 2, 64)

        # Final classification head
        self.cls_conv = nn.Conv3d(64, out_chans, kernel_size=1)

    def unify_shape(self, x4, x5):
        x4_5 = self.act(self.norm4(self.conv4_to5(x4)))
        x5_fuse = x4_5 + x5

        x5_4 = self.act(self.norm3(self.conv5_to4(x5)))
        x4_fused = x5_4 + x4
        return x4_fused, x5_fuse

    def forward(self, x_in):
        outs = self.msfp_3d(x_in)
        x2, x3, x4, x5 = outs[0], outs[1], outs[2], outs[3]

        # High-level fusion (x4, x5)
        x4_fused, x5_fused = self.unify_shape(x4, x5)
        x5_fused, _ = self.res4(x5_fused)
        x4_fused, _ = self.res3(x4_fused)

        # Low-level fusion (x2, x3)
        x3_up = self.act(self.norm1(self.conv3_to2(x3)))
        x2_down = self.act(self.norm2(self.conv2_to3(x2)))
        enc2 = x2 + x3_up
        enc3 = x3 + x2_down

        enc3, _ = self.res2(enc3)
        enc2, _ = self.res1(enc2)

        # Refine low-level again
        enc3_up = self.act(self.norm1(self.conv3_to2(enc3)))
        enc2_down = self.act(self.norm2(self.conv2_to3(enc2)))
        enc2_fuse = enc2 + enc3_up
        enc3_fuse = enc3 + enc2_down

        enc3_fuse, _ = self.res2(enc3_fuse)
        enc2_fuse, _ = self.res1(enc2_fuse)

        # Decoder
        en_out = torch.cat((x5_fused, x5), dim=1)
        en_out = self.stage0_de(en_out)

        dec4 = self.transposeconv_stage1(en_out)
        dec3 = self.stage1_de(torch.cat((dec4, x4_fused), dim=1))

        dec3 = self.transposeconv_stage2(dec3)
        dec2 = self.stage2_de(torch.cat((dec3, enc3_fuse), dim=1))

        dec2 = self.transposeconv_stage3(dec2)
        dec1 = self.stage3_de(torch.cat((dec2, enc2_fuse), dim=1))

        out = self.transposeconv_stage4(dec1)
        out = self.cls_conv(out)

        return out