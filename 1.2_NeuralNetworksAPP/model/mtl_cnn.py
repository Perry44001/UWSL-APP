# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:26:16 2022

@author: admin
"""

import torch
import torch.nn as nn


def make_model(length_freq):
    return MultiTaskLossWrapper(2, length_freq*2)

# 深度可分离卷积块
# in_channels 输入的通道数    out_channels 输出的通道数    flow 指定使用PReLU层的输入是深度可分离卷积层的输入还是输出
class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, flow):
        super(DepthConv, self).__init__()
        self.dp = 0.3
        self.flow = flow
        self.channel_wise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                           kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels)
        self.point_wise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.flow == 'M':
            self.p_relu = nn.PReLU(in_channels)
        else:
            self.p_relu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout2d(p=self.dp)

    def forward(self, x):
        if self.flow == 'M':
            out = self.bn(self.point_wise_conv(self.channel_wise_conv(self.dropout(self.p_relu(x)))))
        else:
            out = self.dropout(self.p_relu(self.bn(self.point_wise_conv(self.channel_wise_conv(x)))))
        return out

# 常规卷积块
class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalConv, self).__init__()

        self.dp = 0.3
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.p_relu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout2d(p=self.dp)

    def forward(self, x):
        out = self.dropout(self.p_relu(self.bn(self.conv(x))))
        return out

# 中间块
class middle_block(nn.Module):
    def __init__(self):
        super(middle_block, self).__init__()
        self.conv = DepthConv
        self.conv1 = self.conv(512, 512, flow='M')
        self.conv2 = self.conv(512, 512, flow='M')
        self.conv3 = self.conv(512, 512, flow='M')

    def forward(self, x):
        residual = x
        out = residual + self.conv3(self.conv2(self.conv1(x)))
        return out


class MTL_CNN(nn.Module):
    def __init__(self, input_channels):
        super(MTL_CNN, self).__init__()
        # Entry flow
        self.d_conv = DepthConv   # 深度卷积块（残差）
        self.n_conv = NormalConv  # 常规卷积块
        self.block = middle_block()  # 中间块

        self.n_conv_E1 = self.n_conv(input_channels, 256)
        self.n_conv_E11 = self.n_conv(256, 256)
        self.n_conv_E2 = self.n_conv(256, 384)
        self.n_conv_E22 = self.n_conv(384, 384)
        self.d_conv_E3 = self.d_conv(384, 512, flow='E')
        self.d_conv_E33 = self.d_conv(512, 512, flow='E')
        self.n_conv_E3 = self.n_conv(384, 512)

        # Middle flow
        self.stage1 = self.make_layer(self.block)
        self.conv_M2 = self.d_conv(512, 640, flow='M')
        self.conv_M22 = self.d_conv(640, 640, flow='M')

        # Exit flow
        self.conv_EX1r = self.d_conv(640, 768, flow='EX')
        self.conv_EX2r = self.d_conv(768, 1024, flow='EX')
        self.avg_r = nn.AdaptiveAvgPool2d(1)
        self.fcr = nn.Linear(1024, 1)

        self.conv_EX1d = self.d_conv(640, 768, flow='EX')
        self.conv_EX2d = self.d_conv(768, 1024, flow='EX')
        self.avg_d = nn.AdaptiveAvgPool2d(1)
        self.fcd = nn.Linear(1024, 1)

    def forward(self, x):
        # Entry flow
        out = self.n_conv_E22(self.n_conv_E2(self.n_conv_E11(self.n_conv_E1(x))))
        residual = self.n_conv_E3(out)
        out = self.d_conv_E33(self.d_conv_E3(out))
        out = out + residual

        # Middle flow
        out = self.conv_M22(self.conv_M2(self.stage1(out)))

        # Exit flow
        out_r = self.avg_r(self.conv_EX2r(self.conv_EX1r(out)))
        r = self.fcr(out_r.view(out_r.size(0), out_r.size(1))).view(-1)
        out_d = self.avg_d(self.conv_EX2d(self.conv_EX1d(out)))
        d = self.fcd(out_d.view(out_d.size(0), out_d.size(1))).view(-1)
        return [r, d]

    def make_layer(self, block):
        block_list = []
        for i in range(0, 7):
            block_list.append(block)

        return nn.Sequential(*block_list)


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, input_channels):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = MTL_CNN(input_channels=input_channels)
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

    mtl = MultiTaskLossWrapper(2, 151*2)
    inputs = torch.randn(5, 151 * 2, 18, 18)
    r = torch.rand(5)
    z = torch.rand(5)
    targets = [r, z]
    loss, log_vars2, output = mtl(inputs, [r, z])
    print(loss, log_vars2, output)
