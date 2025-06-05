import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.residual = nn.Identity()
        
    def forward(self, x):
        return self.se(self.conv(x)) + self.residual(x)

class VGGCustom(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.current_block = 0

        self.block1 = nn.Sequential(
            ConvBlock(3, 40),      # 35 * 1.15
            ConvBlock(40, 77),     # 67 * 1.15
            ConvBlock(77, 154, stride=2),    # 134 * 1.15
            nn.Dropout2d(0.11),
        )

        self.block2 = nn.Sequential(
            ConvBlock(154, 77),    # 67 * 1.15
            ConvBlock(77, 38),     # 33 * 1.15
            ConvBlock(38, 38, stride=2),     # 33 * 1.15
            nn.Dropout2d(0.15),
        )
        
        self.block3 = nn.Sequential(
            ConvBlock(38, 76),     # 66 * 1.15
            ConvBlock(76, 152),    # 132 * 1.15
            ConvBlock(152, 304, stride=2),   # 264 * 1.15
            nn.Dropout2d(0.05),
        )

        self.blocks = nn.ModuleList([self.block1, self.block2, self.block3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(304 * 4 * 4, num_classes)  # 304 son ConvBlock'un çıkış kanalı sayısı
        )

        self._initialize_weights()
        self.freeze_blocks_until(0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
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

    def open_next_block(self):
        if self.current_block == 0:
            # İlk bloğu aç
            for param in self.block1.parameters():
                param.requires_grad = True
            self.current_block = 1
            return True
        elif self.current_block == 1:
            # İkinci bloğu aç
            for param in self.block2.parameters():
                param.requires_grad = True
            self.current_block = 2
            return True
        elif self.current_block == 2:
            # Üçüncü bloğu aç
            for param in self.block3.parameters():
                param.requires_grad = True
            self.current_block = 3
            return True
        return False

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.avgpool(x)
        return self.classifier(x)
