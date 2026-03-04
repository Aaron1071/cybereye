import os
import collections
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Paths (full Windows—r'' raw string ignores backslashes; term: Raw = no escape)
TRAIN_PATH = r"C:/Users/aaron/Downloads/CyberEye/data/archive/malevis_train_val_300x300/train"
VAL_PATH = r"C:/Users/aaron/Downloads/CyberEye/data/archive/malevis_train_val_300x300/val"

if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
    raise FileNotFoundError("Paths missing—copy MaleVis folders")

print(f"Classes: {len(os.listdir(TRAIN_PATH))}")

# Transforms (aug for train—teach: Aug = data augmentation, boosts variety like simulating obfuscated binaries in forensics)
train_transform = transforms.Compose([
    transforms.Resize(256),  # Resize = scale image (256 intermediate for crop; why? Standard for pre-trained models)
    transforms.CenterCrop(224),  # Crop = cut center (224 ViT/ResNet input; term: Crop = extract region)
    transforms.RandomHorizontalFlip(0.5),  # Flip = mirror horiz (p=prob 0.5; connect: Mirrors packed ELF patterns)
    transforms.RandomRotation(15),  # Rotation = turn ±15 deg (handles alignment vars in binary viz)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Jitter = vary color (RGB noise sim—teach: Brightness/contrast = pixel intensity tweaks)
    transforms.ToTensor(),  # ToTensor = convert to tensor ([C,H,W] format; term: Tensor = multi-dim array for GPU)
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # Normalize = scale to mean/std (stability for pre-train; connect: Like normalizing log data in forensics for anomaly detection)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_loaders(batch_size=16):
    train_ds = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)  # ImageFolder = loads folder data (classes from subdirs; teach: Supervised = labeled data)
    val_ds = datasets.ImageFolder(VAL_PATH, transform=val_transform)

    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}")

    train_counts = collections.Counter([y for _, y in train_ds])  # Counter = dict for counts (imbalance check; connect: Count artifacts in forensics disk scan)
    val_counts = collections.Counter([y for _, y in val_ds])
    print("Train counts:", train_counts)
    print("Val counts:", val_counts)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)  # DataLoader = batches data (shuffle randomizes—avoids order bias; num_workers=0 Windows safe)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_counts, train_ds.classes  # Return for train.py

# Denorm viz (fix clipping—teach: Denorm = reverse normalize for display)
def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  # View = reshape (3 channels to broadcast)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img * std + mean
    img = img.clamp(0, 1)  # Clamp = bound (0-1 RGB valid)
    return img

if __name__ == "__main__":  # Guard (run if script, not import—modular)
    train_loader, _, _, classes = get_loaders()
    images, labels = next(iter(train_loader))  # Next/iter = get batch
    plt.imshow(denorm(images[0]).permute(1,2,0).numpy())  # Permute = reorder dims (HWC for plt; numpy = to array)
    plt.title(classes[labels[0]])
    plt.show()  # Show window (connect: Viz binaries like BinVis in forensics for pattern spotting)
    print("Dataset ready.")