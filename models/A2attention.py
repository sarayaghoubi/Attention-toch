""""
the article A2-Nets: Double Attention Networks
it performs simple attention o three convolutional layer
number of parameters about : 360,000

"""

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class AttentionModule(nn.Module):

    def __init__(self, in_channels, c_m, c_n, mid, oup, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = torch.nn.Conv2d(in_channels, c_m, kernel_size=(3, 3), stride=2, padding=(2, 2))
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=(3, 3), stride=2, padding=(2, 2))
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=(3, 3), stride=2, padding=(2, 2))
        self.oup = oup
        self.hidden_c = mid
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(mid, oup, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1))
        attention_vectors = F.softmax(V.view(b, self.c_n, -1))
        # step 1: feature gating
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
        # step 2: feature distribution
        tmp_z = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        # tmp_z = tmp_z.view(b, self.c_m, -1)  # b,c_m,h,w
        if self.reconstruct:
            tmp_z = self.conv_reconstruct(tmp_z)
        x,y,z = tmp_z.shape
        return torch.unsqueeze(tmp_z.view(self.hidden_c,self.in_channels,-1),3)


class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, mid_c, oup, reconstruct=True):
        super().__init__()
        self.hidden = mid_c
        self.Attention_one = AttentionModule(in_channels, c_m, c_n, mid_c, oup, reconstruct)
        self.Attention_two = AttentionModule(in_channels, c_m, c_n, mid_c, oup, reconstruct)
        self.maxPooling = nn.AdaptiveAvgPool2d(mid_c*c_m)
        self.fc = nn.Sequential(
            nn.Linear(in_channels*c_m*c_n*mid_c*mid_c, oup),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.Attention_one(x)
        x = self.Attention_two(x)
        x1 = self.maxPooling(x)
        x1 = x1.view(self.batch,-1)
        x2 = self.fc(x1)
        return x2


def count_parameters(models):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)

#   checked \/
