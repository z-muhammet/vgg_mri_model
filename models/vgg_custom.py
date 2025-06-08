import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Define number of groups for GroupNorm
num_groups = 32

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y

class TrainOnlyNoise(nn.Module):
    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1,
                 use_se: bool = False, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.core = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.GELU(),
            SEBlock(out_ch) if use_se else nn.Identity(),
        )
        self.shortcut = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_ch),
            )
        )
    def forward(self, x):
        return self.core(x) + self.shortcut(x)

class VGGCustom(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.block_out_channels = [64, 128, 256]  # Output channels for each block (258 -> 256)
        self.blocks = nn.ModuleList([
            # Block 1: 4 Conv, last one with stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(in_channels, 32),   # 3→32
                ConvBlock(32, 64,stride=2),            # 32→64
                nn.Dropout2d(0.1),
            ),
            # Block 2: 3 Conv, last one with stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(64, 128),            # 64→128
                ConvBlock(128, 256,stride=1),  # 128→256
                ConvBlock(256, 512, use_se=True, dilation=2,stride=2), # 256→512
                nn.Dropout2d(0.15),
            ),
            # Block 3: 3 Conv, last one with stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(128, 128),              # 128→128
                ConvBlock(128, 128),              # 128→128
                ConvBlock(128, 256, stride=2, use_se=True), # 128→256 (258 -> 256)
                nn.Dropout2d(0.3),
            ),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Match adapters with block indices
        self.adaptors = nn.ModuleList([
            nn.Linear(64, 512),    # 0: Block-1 output 64 → 512
            nn.Identity(),         # 2: Block-2 output already 512
            nn.Linear(256, 512),   # 1: Block-3 output 256 → 512 (258 -> 256)
        ])

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, num_classes),
        )
        self._init_weights()
        self.freeze_blocks_until(0)  # Yalnızca ilk blok açık
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_blocks_until(self, last_open_idx: int):
        for i, blk in enumerate(self.blocks):
            req_grad = i <= last_open_idx
            for p in blk.parameters():
                p.requires_grad = req_grad
        self.opened_blocks = last_open_idx + 1

    def forward(self, x):
        last_idx = -1
        for i, blk in enumerate(self.blocks):
            if i < self.opened_blocks:
                x = blk(x)
                last_idx = i
        x = self.avgpool(x).view(x.size(0), -1)
        # Dynamic adaptor based on last opened block
        x = self.adaptors[last_idx](x)
        return self.classifier(x)