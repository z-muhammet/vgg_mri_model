import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, use_se: bool = False):
        super().__init__()
        self.core = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            SEBlock(out_ch) if use_se else nn.Identity(),
        )
        self.shortcut = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
    def forward(self, x):
        return self.core(x) + self.shortcut(x)

class VGGCustom(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.block_out_channels = [64, 128, 258]  # Her bloğun çıkış kanal sayısı
        self.blocks = nn.ModuleList([
            # Block 1: 4 Conv, sonuncusu stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(in_channels, 32),   # 3→32
                ConvBlock(32, 64),            # 32→64
                nn.MaxPool2d(kernel_size=2),  # spatial ↓
                nn.Dropout2d(0.05),
            ),
            # Block 2: 3 Conv, sonuncusu stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(64, 64),            # 64→64
                ConvBlock(64, 128, use_se=True),  # 64→128
                ConvBlock(128, 128),              # 128→128
                nn.Dropout2d(0.15),
            ),
            # Block 3: 3 Conv, sonuncusu stride=2 (downsampling)
            nn.Sequential(
                ConvBlock(128, 128),              # 128→128
                ConvBlock(128, 128),              # 128→128
                ConvBlock(128, 258, stride=2, use_se=True), # 128→258
                nn.Dropout2d(0.3),
            ),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Adaptörler: her blok çıkışından 258'e
        self.adaptors = nn.ModuleList([
            nn.Identity(),  # 258 ise gerek yok
            nn.Linear(64, 258),   # 64→258
            nn.Linear(128, 258),  # 128→258
        ])
        self.classifier = nn.Sequential(
            nn.Linear(258, 38),     # Son blok sonunda 258 kanal
            nn.BatchNorm1d(38),
            nn.SiLU(),
            nn.Linear(38, num_classes)
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
        last_block_idx = -1
        for i, blk in enumerate(self.blocks):
            if i < self.opened_blocks:
                x = blk(x)
                last_block_idx = i
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # Adaptör ile 258 kanala çek
        if last_block_idx == 2:
            x = self.adaptors[0](x)  # 258→258 (Identity)
        elif last_block_idx == 1:
            x = self.adaptors[2](x)  # 128→258
        elif last_block_idx == 0:
            x = self.adaptors[1](x)  # 64→258
        return self.classifier(x)
