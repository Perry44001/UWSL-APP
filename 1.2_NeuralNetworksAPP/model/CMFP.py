# -*- coding: utf-8 -*-
"""
Created on 2023年2月13日10:01:14

@author: admin
"""
import numpy as np
import torch
import torch.nn as nn


def make_model(length_freq, M_array):
    return MultiTaskLossWrapper(length_freq * 2, M_array)


class CMFP(nn.Module):
    # def __init__(self, input_channels, M_array, r_target, d_target):
    def __init__(self, input_channels, M_array):
        super(CMFP, self).__init__()
        self.n_conv_E1 = nn.Conv2d(in_channels=input_channels, out_channels=1,
                                   kernel_size=(M_array, 1), stride=1, padding=0, bias=False)
        # self.r_target = r_target
        # self.d_target = d_target

    def forward(self, x):
        # Entry flow
        out = torch.squeeze(self.n_conv_E1(x))
        # print('out', out)
        # index = torch.argmax(out, dim=1)
        # r = torch.ones_like(index)
        # d = torch.ones_like(index)
        # for i in range(len(index)):
        #     r_index = int(index[i] / len(self.r_target))
        #     d_index = index[i] - int(index[i] / len(self.r_target)) * len(self.d_target)
        #
        #     # Exit flow
        #     r[i] = self.r_target[r_index]
        #     d[i] = self.d_target[d_index]
        # return [r, d]
        return out


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, input_channels, M_array):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = CMFP(input_channels=input_channels, M_array=M_array)

    def forward(self, input_feature, r_target, d_target):
        outputs = self.model(input_feature)
        index = torch.argmax(outputs, dim=1)
        r = torch.ones_like(index)
        d = torch.ones_like(index)
        for i in range(len(index)):
            r_index = int(index[i] / len(r_target))
            d_index = index[i] - int(index[i] / len(r_target)) * len(d_target)

            # Exit flow
            r[i] = r_target[r_index]
            d[i] = d_target[d_index]
        output = [r, d]

        return outputs, d, output


if __name__ == '__main__':
    # mtl = CMFP(151 * 2, 13)
    mtl = MultiTaskLossWrapper(151 * 2, 13)
    inputs = torch.randn(5, 151 * 2, 13, 9)
    # r = torch.rand(5)
    # z = torch.rand(5)
    # targets = [r, z]
    r_target = np.arange(1, 3.1, 1)
    d_target = np.arange(1, 3.1, 1)
    outputs, d, output = mtl(inputs, r_target, d_target)
    print('outputs', outputs)
    print('d', d)
    print('output', output)
