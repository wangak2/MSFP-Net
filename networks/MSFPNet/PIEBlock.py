import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm
from effTrans_block_layers import effTrans_layers


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
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
        out = out + residual

        return out


class PIE(nn.Module):
    def __init__(self, in_channel, scale=2, h=8, w=8, l=6, layer_scale_init_value=1e-6):
        super(PIE, self).__init__()
        self.scale = scale
        self.resblock_fea = ResBlock(in_channel, in_channel)
        self.resblock_fea2 = ResBlock(in_channel, 2)
        self.max_pool = nn.MaxPool3d(kernel_size=scale * 2 + 1, stride=scale, padding=scale)
        self.resblock_ref = ResBlock(in_channel, in_channel)
        self.conv_channel1 = ResBlock(2, in_channel)
        self.conv_channel2 = ResBlock(in_channel, 2)

        self.h = h
        self.w = w
        self.l = l
        self.dim = in_channel
        self.sigmoid = nn.Sigmoid()
        self.Delight_Trans = effTrans_layers(in_channel)
        self.norm = nn.BatchNorm3d(in_channel)
        self.norm2 = nn.BatchNorm3d(2)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((2)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, feature, refined_shape_prior):
        b, c, _ = refined_shape_prior.size()

        feature = self.resblock_fea(feature)
        feature_attn = self.resblock_fea2(feature)
        feature_attn = self.max_pool(feature_attn)
        feature_attn = self.norm2(feature_attn)
        feature_attn = self.sigmoid(feature_attn)

        refined_shape_prior = refined_shape_prior.contiguous().view(b, c, self.h, self.w, self.l)
        feature_attn = feature_attn.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            feature_attn = self.gamma * feature_attn
        feature_attn = feature_attn.permute(0, 4, 1, 2, 3)

        refined_shape_prior = feature_attn * refined_shape_prior + refined_shape_prior
        previous_class_center = refined_shape_prior.flatten(2)

        refined_shape_prior = self.conv_channel1(refined_shape_prior)
        refined_shape_prior = refined_shape_prior.flatten(2).permute(2, 0, 1)
        mask = None
        refined_shape_prior = self.Delight_Trans(refined_shape_prior, mask)
        refined_shape_prior = refined_shape_prior.permute(1, 2, 0).view(b, self.dim, self.h, self.w, self.l)

        refined_shape_prior = F.interpolate(refined_shape_prior, scale_factor=self.scale, mode="trilinear")
        refined_shape_prior = self.resblock_ref(refined_shape_prior)
        refined_shape_prior = self.norm(refined_shape_prior)
        refined_shape_prior = self.sigmoid(refined_shape_prior)

        class_feature = feature * refined_shape_prior

        refined_shape_prior = (
            F.interpolate(self.conv_channel2(class_feature), scale_factor=(1.0 / self.scale), mode="trilinear").flatten(2)
            + previous_class_center
        )

        return class_feature, refined_shape_prior