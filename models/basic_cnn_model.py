import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Define number of groups for GroupNorm
num_groups = 32
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
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        kernel_size = 3
        pad = dilation * (kernel_size // 2)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=pad,
                              dilation=dilation, bias=False)
        
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch) 
        self.act = nn.ReLU()
        self.drop = nn.Dropout2d(dropout) 
        self.noise = TrainOnlyNoise(std=0.01)
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.noise(out)
        return out


class BasicCNNModel(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.block_out_channels = [64, 128, 256]  # Output channels for each block (258 -> 256)
        self.blocks = nn.ModuleList([
            # Block 1: 4 Conv, last one with stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(in_channels, 32),
                ConvBlock(32, 64, stride=2, dilation=1,dropout=0.1),
            ),
            # Block 2: 3 Conv, last one with stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(64, 128, dilation=4),
                ConvBlock(128, 256, stride=1, dilation=2),
                ConvBlock(256, 512, dilation=2, stride=2,dropout= 0.2),
            ),
            # Block 3: 3 Conv, last one with stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(512, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 256, stride=2),
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
            nn.BatchNorm1d(512, momentum=0.01),
            nn.Mish(),
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