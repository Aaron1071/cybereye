import torch.nn as nn
from torchvision import models

class CNNBaseline(nn.Module):  # nn.Module = base class (teach: Inheritance = extend parent)
    def __init__(self, num_classes=26):  # __init__ = constructor (setup layers)
        super().__init__()  # Call parent init
        self.model = models.resnet18(weights="IMAGENET1K_V1")  # Pre-trained (transfer learning—why? Starts with general features, fine-tune for malware; connect: Like using pre-trained sigs in antivirus, adapt to new IoT threats)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Linear = fully connected layer (replace for 26 classes; in_features = input size)

    def forward(self, x):  # Forward = data flow method
        return self.model(x)  # Pass through

# Add hybrid later (obj 1: CNN-ViT)