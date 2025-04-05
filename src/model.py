import math
import torch
from torch import nn
import torch.nn.functional as F

def round_filters(filters, width_mult):
    filters *= width_mult
    new_filters = max(8, int(filters + 4) // 8 * 8)  # arrotonda a multipli di 8
    return int(new_filters)

def round_repeats(repeats, depth_mult):
    return int(math.ceil(repeats * depth_mult))

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.se(x)
        return x * scale

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super().__init__()
        mid_channels = in_channels * expansion_factor
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        )

        self.se = SqueezeExcite(mid_channels)

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res_connect:
            out += identity
        return out

class EfficientNetScaled(nn.Module):
    def __init__(self, phi=0, num_classes=3):
        super().__init__()

        # Parametri compound scaling
        width_mult = 1.0 * (1.2 ** phi)
        depth_mult = 1.0 * (1.1 ** phi)

        base_channels = 32
        stem_out = round_filters(base_channels, width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU6(inplace=True)
        )

        # Specifica base: (expand, out_c, repeats, stride)
        settings = [
            # expansion, out_channels, num_blocks, stride
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        ]

        blocks = []
        input_c = stem_out

        for expansion, out_c, repeats, stride in settings:
            out_c = round_filters(out_c, width_mult)
            repeat = round_repeats(repeats, depth_mult)
            for i in range(repeat):
                s = stride if i == 0 else 1
                blocks.append(MBConvBlock(input_c, out_c, expansion_factor=expansion, stride=s))
                input_c = out_c

        self.blocks = nn.Sequential(*blocks)

        head_c = round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(input_c, head_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_c),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(head_c, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
