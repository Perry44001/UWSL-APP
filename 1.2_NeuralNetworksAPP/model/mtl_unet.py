# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:37:44 2022

@author: admin
"""

import torch
import torch.nn as nn


def make_model(length_freq):
    return MultiTaskLossWrapper(1, 2)


class ChannelAttention(nn.Module):
    # CBAM
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.active = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.active(x)
        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, flow):
        super(Conv, self).__init__()
        self.dp = 0.3
        self.flow = flow
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels)
        self.p_conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.flow == 'M':
            self.p_relu = nn.PReLU(in_channels)
        else:
            self.p_relu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout2d(p=self.dp)

    def forward(self, x):
        if self.flow == 'M':
            out = self.bn(self.conv(self.dropout(self.p_relu(x))))
        elif self.flow == 'EX':
            out = self.dropout(self.p_relu(self.bn(self.p_conv_1(self.conv1(x)))))
        else:
            out = self.dropout(self.p_relu(self.bn(self.conv(x))))
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            Conv(in_channels, mid_channels, flow='E'),
            Conv(mid_channels, out_channels, flow='E')
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class UNet_Encoder(nn.Module):
    def __init__(self, input_channels, in_channels, bi_linear=True):
        super(UNet_Encoder, self).__init__()
        self.n_channels = input_channels
        self.bi_linear = bi_linear
        self.inc = DoubleConv(input_channels, in_channels * 2)
        self.down1 = Down(in_channels * 2, in_channels * 4)
        self.down2 = Down(in_channels * 4, in_channels * 8)
        self.down3 = Down(in_channels * 8, in_channels * 16)
        factor = 2 if bi_linear else 1
        self.down4 = Down(in_channels * 16, in_channels * 32 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5


class Local(nn.Module):
    def __init__(self, input_channel, in_channels):
        super(Local, self).__init__()
        self.UE = UNet_Encoder(input_channel, in_channels)
        self.conv_EX1r = Conv(in_channels * 16, in_channels * 24, flow='E')
        self.conv_EX2r = Conv(in_channels * 24, in_channels * 32, flow='E')
        self.ca1 = ChannelAttention(in_channels * 32, ratio=4)
        self.sa1 = SpatialAttention()
        self.avg_r = nn.AdaptiveAvgPool2d((1, 1))
        self.fcr = nn.Linear(in_channels * 32, 1)

        self.conv_EX1d = Conv(in_channels * 16, in_channels * 24, flow='E')
        self.conv_EX2d = Conv(in_channels * 24, in_channels * 32, flow='E')
        self.ca2 = ChannelAttention(in_channels * 32, ratio=4)
        self.sa2 = SpatialAttention()
        self.avg_d = nn.AdaptiveAvgPool2d((1, 1))
        self.fcd = nn.Linear(in_channels * 32, 1)

    def forward(self, x):
        out = self.UE(x)
        out_r = self.avg_r(self.conv_EX2r(self.conv_EX1r(out)))
        out_r = self.ca1(out_r) * out_r
        out_r = self.sa1(out_r) * out_r
        r = self.fcr(out_r.view(out_r.size(0), out_r.size(1))).view(-1)
        out_d = self.avg_d(self.conv_EX2d(self.conv_EX1d(out)))
        out_d = self.ca2(out_d) * out_d
        out_d = self.sa2(out_d) * out_d
        d = self.fcd(out_d.view(out_d.size(0), out_d.size(1))).view(-1)
        return [r, d]


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, in_channel, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = Local(in_channel, in_channels=16)
        self.task_num = task_num
        self.log_vars2 = nn.Parameter(torch.zeros(task_num))

    def forward(self, input_feature, target):
        outputs = self.model(input_feature)
        precision1 = torch.exp(-self.log_vars2[0])
        precision2 = torch.exp(-self.log_vars2[1])
        mtl_loss = torch.sum(0.5 * precision1 * (target[0] - outputs[0]) ** 2., -1) + \
                   torch.sum(0.5 * precision2 * (target[1] - outputs[1]) ** 2., -1) + \
                   0.5 * self.log_vars2[0] + 0.5 * self.log_vars2[1]

        return mtl_loss, self.log_vars2.data.tolist(), outputs


if __name__ == '__main__':
    length_freq = 151
    mtl = MultiTaskLossWrapper(1, 2)

    inputs = torch.randn(5, 1, length_freq * 2, 18 * 18)
    r = torch.rand(5)
    z = torch.rand(5)
    targets = [r, z]
    loss, log_vars2, output = mtl(inputs, [r, z])
    print(loss, log_vars2, output)
