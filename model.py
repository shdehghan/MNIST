import torch
import torch.nn as nn

# Building block

class BottleneckBlock(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        # 1×1 — channel reduction (bottleneck entry)
        self.conv1 = nn.Conv2d(in_ch,  mid_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)

        # 3×3 — spatial feature extraction on cheap channel count
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)

        # 1×1 — channel expansion (bottleneck exit)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut: projection when shape changes, identity otherwise
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))          # BN before residual add

        return self.relu(out + identity)

# Full model

class SudokuMNISTModel(nn.Module):

    def __init__(self):
        super().__init__()

        #  Stem 
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 84 → 42
        )

        
        # Block 1
        self.block1 = BottleneckBlock(in_ch=16, mid_ch=8,  out_ch=16, stride=1)

        # Block 2
        self.block2 = BottleneckBlock(in_ch=16, mid_ch=8,  out_ch=32, stride=2)

        # Block 3
        self.block3 = BottleneckBlock(in_ch=32, mid_ch=16, out_ch=32, stride=2)

        # Block 4
        self.block4 = BottleneckBlock(in_ch=32, mid_ch=16, out_ch=64, stride=2)

        FEAT = 64  # backbone output channels

        # head 1 
        self.gap                 = nn.AdaptiveAvgPool2d(1)
        self.missing_digit_head  = nn.Linear(FEAT, 10)

        self.row_pool = nn.AdaptiveAvgPool2d((3, 1))
        self.col_pool = nn.AdaptiveAvgPool2d((1, 3))

        # head 2 
        self.ordering_head = nn.Sequential(
            nn.Linear(FEAT, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
        )
        # head 1
        self.sum_head = nn.Sequential(
            nn.Linear(FEAT, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # Initialisation 
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    # Spatial feature extraction
    def _spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        
        # (B, C, 6, 6) → pool cols → (B, C, 3, 1) → squeeze → permute → (B, 3, C)
        row_feats = self.row_pool(x).squeeze(-1).permute(0, 2, 1)

        # (B, C, 6, 6) → pool rows → (B, C, 1, 3) → squeeze → permute → (B, 3, C)
        col_feats = self.col_pool(x).squeeze(-2).permute(0, 2, 1)

        return torch.cat([row_feats, col_feats], dim=1)   # (B, 6, C)

    # Forward pass 
    def forward(self, x: torch.Tensor):
        
        # Shared backbone
        x = self.stem(x)     # (B, 16, 42, 42)
        x = self.block1(x)   # (B, 16, 42, 42)
        x = self.block2(x)   # (B, 32, 21, 21)
        x = self.block3(x)   # (B, 32, 11, 11)
        x = self.block4(x)   # (B, 64,  6,  6)

        # ── Head 1: missing digit 
        global_feat    = self.gap(x).flatten(1)              # (B, 64)
        missing_logits = self.missing_digit_head(global_feat)  # (B, 10)

        # Heads 2 & 3: per-row / per-col 
        spatial         = self._spatial_features(x)          # (B, 6, 64)
        ordering_logits = self.ordering_head(spatial)        # (B, 6, 3)
        sum_preds       = self.sum_head(spatial).squeeze(-1) # (B, 6)

        return missing_logits, ordering_logits, sum_preds

# Utilities

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: nn.Module) -> None:
    """Print per-module parameter counts."""
    print(f"{'Module':<35} {'Params':>8}")
    print("─" * 45)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<33} {n:>8,}")
    print("─" * 45)
    print(f"  {'TOTAL':<33} {count_parameters(model):>8,}")

# Quick sanity check

if __name__ == "__main__":
    model = SudokuMNISTModel()

    x = torch.randn(4, 1, 84, 84)
    missing, ordering, sums = model(x)

    print("Output shapes:")
    print(f"  missing_logits   : {tuple(missing.shape)}")   # (4, 10)
    print(f"  ordering_logits  : {tuple(ordering.shape)}")  # (4, 6, 3)
    print(f"  sum_preds        : {tuple(sums.shape)}")       # (4, 6)
    print()
    parameter_breakdown(model)
    print()
    assert count_parameters(model) <= 20_000, "Parameter budget exceeded!"
    print("✓ Parameter budget OK")