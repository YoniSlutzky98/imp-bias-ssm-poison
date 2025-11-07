import torch
import torch.nn as nn

from .mamba2 import Mamba2, Mamba2Config

class Mamba2Block(nn.Module):
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        cfg = Mamba2Config(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba = Mamba2(cfg)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Mamba2 returns (output, cache), we only need the output
        mamba_out, _ = self.mamba(x)
        return self.drop(self.norm(mamba_out + x))

class Mamba2Model(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=128, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(d_input, d_model//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model//2), nn.GELU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU()
        )
        self.blocks = nn.Sequential(*[Mamba2Block(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_output))

    def forward(self, x):
        B = x.size(0)
        x = self.stem(x).flatten(2).transpose(1, 2)   # (B, seq_len, d_model)
        x = self.blocks(x)                           # Process through Mamba blocks
        x = x.mean(1)                                # Global average pooling
        return self.head(x)