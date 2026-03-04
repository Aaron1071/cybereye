import torch
import torch.nn as nn
from torchvision import models

class CNNBaseline(nn.Module):  # nn.Module = base class (teach: Inheritance = extend parent)
    def __init__(self, num_classes=26):  # __init__ = constructor (setup layers)
        super().__init__()  # Call parent init
        self.model = models.resnet18(weights="IMAGENET1K_V1")  # Pre-trained (transfer learning—why? Starts with general features, fine-tune for malware; connect: Like using pre-trained sigs in antivirus, adapt to new IoT threats)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Linear = fully connected layer (replace for 26 classes; in_features = input size)

    def forward(self, x):  # Forward = data flow method
        return self.model(x)  # Pass through

#The light version 
class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=26, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()

        # ----- CNN Backbone -----
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # remove avgpool & fc

        # Project CNN channels -> embed_dim
        self.conv_proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # Assume feature map is 7x7 after ResNet (for 224x224 input)
        num_tokens = 7 * 7

        # ----- Positional Encoding -----
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # ----- Transformer Encoder -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ----- Classification Head -----
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.feature_extractor(x)            # (B, 512, 7, 7)
        x = self.conv_proj(x)                    # (B, embed_dim, 7, 7)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, 49, embed_dim)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Global average pooling over tokens
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x