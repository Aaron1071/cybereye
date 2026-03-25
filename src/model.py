"""
model.py – All model variants for the CyberEye ablation study.

Variants:
  CNNBaseline      – ResNet-18 fine-tuned (Week 2 baseline)
  HybridCNNViT     – ResNet-18 + shallow Transformer (Week 3, 2 layers)
  DeeperHybrid     – ResNet-18 + deeper Transformer (3-4 layers, larger embed)
  FullViTHybrid    – ResNet-18 + full ViT-style (CLS token, deeper, larger dim)

Positional encoding: all hybrid models use nn.Parameter, which IS learned
(gradient flows through it during backprop—this is the standard lightweight
approach used in DeiT and similar models).
"""

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────
# 1.  CNN Baseline  (unchanged from Week 2)
# ─────────────────────────────────────────────────────────────
class CNNBaseline(nn.Module):
    """ResNet-18 with replaced classification head. Week-2 baseline."""

    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
# 2.  Hybrid CNN–ViT  (Week 3, 2-layer Transformer)
# ─────────────────────────────────────────────────────────────
class HybridCNNViT(nn.Module):
    """
    ResNet-18 feature extractor  →  1×1 projection  →  Transformer encoder
    →  global average pool  →  linear classifier.

    Positional encoding is a learned nn.Parameter (same as DeiT / PVT).
    Initialised with small random noise; gradient updates it during training.
    """

    def __init__(
        self,
        num_classes: int = 26,
        embed_dim: int = 256,   # projection dimension (token size)
        num_heads: int = 4,
        num_layers: int = 2,    # Transformer depth  ← Week 3 default
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── CNN backbone (remove avgpool + fc) ──────────────────────────
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # ── Project 512 CNN channels → embed_dim ────────────────────────
        self.conv_proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # ResNet-18 outputs 7×7 feature map for 224×224 input → 49 tokens
        num_tokens = 7 * 7

        # ── Learned positional encoding ──────────────────────────────────
        # nn.Parameter means it IS learned (backprop updates these values).
        # Initialised close to zero so it starts as a small correction.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )

        # ── Transformer encoder ─────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Classification head ─────────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)           # (B, 512, 7, 7)
        x = self.conv_proj(x)                   # (B, embed_dim, 7, 7)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, 49, embed_dim)
        x = x + self.pos_embedding              # add learned positions
        x = self.transformer(x)                 # (B, 49, embed_dim)
        x = x.mean(dim=1)                       # global average pool over tokens
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# 3.  Deeper Hybrid  (Week 4 – larger embed, more layers)
# ─────────────────────────────────────────────────────────────
class DeeperHybrid(nn.Module):
    """
    Same topology as HybridCNNViT but with:
      • embed_dim = 384  (50 % larger than 256)
      • num_layers = 4   (double the depth)
      • num_heads  = 6   (divisible into 384)
      • dropout    = 0.1

    Ablation note: compare directly against HybridCNNViT to measure the
    contribution of depth + dimension independently.
    """

    def __init__(
        self,
        num_classes: int = 26,
        embed_dim: int = 384,   # larger embedding dimension
        num_heads: int = 6,     # 384 / 6 = 64 head dim
        num_layers: int = 4,    # deeper Transformer
        dropout: float = 0.1,
    ):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.conv_proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        num_tokens = 7 * 7
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.conv_proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# 4.  Full ViT-Style Hybrid  (Week 4 – CLS token + LayerNorm)
# ─────────────────────────────────────────────────────────────
class FullViTHybrid(nn.Module):
    """
    Full ViT-style hybrid with:
      • Prepended [CLS] token (classification is read from CLS, not avg-pool)
      • Learned positional encoding over (1 + num_tokens) positions
      • Pre-norm Transformer layers (LayerNorm before attention)
      • MLP classification head: Linear → GELU → Dropout → Linear
      • embed_dim = 384, num_layers = 4, num_heads = 6

    The CLS token aggregates global context across all patch tokens,
    which is the standard ViT classification approach (Dosovitskiy 2021).
    Using it instead of average pooling often improves recall on rare families.
    """

    def __init__(
        self,
        num_classes: int = 26,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── CNN backbone ────────────────────────────────────────────────
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.conv_proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        num_patch_tokens = 7 * 7   # 49 patch tokens

        # ── [CLS] token ─────────────────────────────────────────────────
        # A single learned vector prepended to the patch sequence.
        # After Transformer layers, only this token's output is classified.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional encoding for (1 cls + 49 patch) tokens ───────────
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1 + num_patch_tokens, embed_dim) * 0.02
        )

        # ── Transformer encoder (pre-norm via norm_first=True) ───────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),   # final norm after last layer
        )

        # ── MLP classification head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for projection, truncated normal for cls token."""
        nn.init.kaiming_normal_(self.conv_proj.weight, mode="fan_out")
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN patch tokens
        x = self.feature_extractor(x)               # (B, 512, 7, 7)
        x = self.conv_proj(x)                        # (B, embed_dim, 7, 7)
        x = x.flatten(2).transpose(1, 2)             # (B, 49, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)               # (B, 50, embed_dim)

        # Add positional encoding
        x = x + self.pos_embedding                   # (B, 50, embed_dim)

        # Transformer
        x = self.transformer(x)                      # (B, 50, embed_dim)

        # Read CLS token output only
        cls_out = x[:, 0]                            # (B, embed_dim)

        return self.classifier(cls_out)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)

    for ModelClass, name in [
        (CNNBaseline,  "CNNBaseline"),
        (HybridCNNViT, "HybridCNNViT (2-layer, 256d)"),
        (DeeperHybrid, "DeeperHybrid (4-layer, 384d)"),
        (FullViTHybrid,"FullViTHybrid (4-layer, 384d, CLS)"),
    ]:
        m = ModelClass(num_classes=26)
        out = m(dummy)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:45s}  output: {out.shape}  params: {params:,}")
