"""
This is an implementation (also had some modifications to be compatible with the audio-classifier model)
of the article: https://arxiv.org/abs/2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes
the original model consists of (in order) 2 Residual network+ 2 transformer modules
                                            has about minimum 17 million parameters


In this script:
    CoAtNetV1 :  580,000
    CoAtNetV2 :  530,000

"""

import torch
import torch.nn as nn
from einops import rearrange

# To Do List:
"""
1- check for replacing relu with gelu
2- also check for using pre-norm blocks
3- check for adding the second linear layer with sigmoid activating function( exactly as was mentioned in the article)
"""


####


class SE(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, oup, bias=False),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y


class MBConv(nn.Module):
    def __init__(self, inp, oup_list):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup_list[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup_list[0]),
            nn.ReLU(inplace=False),
            # dw
            nn.Conv2d(oup_list[0], oup_list[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup_list[1]),
            nn.ReLU(inplace=False),

            nn.Conv2d(oup_list[1], oup_list[2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup_list[2]),
            nn.ReLU(inplace=False),
            # dw
        )

    def forward(self, x):
        return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, audio_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = audio_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.iw, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * 2, 172),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        # q, k, v = map(lambda t: rearrange(
        #     t, 'b n (h d) -> b h n d', ), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        # relative_bias = self.relative_bias_table.gather(
        #     0, self.relative_index.repeat(1, self.heads))
        # relative_bias = rearrange(
        #     relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        # dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, inp, audio_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        self.attn = Attention(inp, audio_size, heads, dim_head, dropout)

    def forward(self, x):
        x = x + self.attn(x)
        return x


class CoAtNetV1(nn.Module):

    # structure : conv+conv+att+att
    def __init__(self, audio_size, channels, oup):
        super().__init__()
        self.s1 = MBConv(channels[0], channels[1:4])
        self.s2 = MBConv(channels[3], channels[4:7])

        self.s3 = AttentionBlock(channels[6], audio_size)
        self.s4 = AttentionBlock(channels[7], audio_size)

        self.lin_layer = SE(channels[8], oup)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.lin_layer(x)
        return x


class CoAtNetV2(nn.Module):
    # conv+att conv+att
    def __init__(self, audio_size, channels, n_classes):
        super().__init__()
        self.s1 = MBConv(channels[0], channels[1:4])
        self.s2 = AttentionBlock(channels[3], channels[4], audio_size)

        self.s3 = MBConv(channels[2], channels[3:6])
        self.s4 = AttentionBlock(channels[5], channels[5], audio_size)

        self.lin_layer = SE(channels[7], n_classes)

    def forward(self, x):
        x1 = self.s1(x)
        x1 = self.s2(x1)
        x2 = self.s3(x)
        x2 = self.s4(x2)
        x = self.lin_layer(x1 + x2)
        return x


def count_parameters(models):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)

# print(f' the number of parameters : {count_parameters(model)}')
