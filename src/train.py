import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.dataset import get_loaders
from src.model import CNNBaseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, train_counts, classes = get_loaders()
num_classes = len(classes)

weights = torch.tensor([
    len(train_loader.dataset) / (num_classes * train_counts[i])
    for i in range(num_classes)
]).to(device)

from src.model import HybridCNNViT
model = HybridCNNViT(num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
os.makedirs("checkpoints", exist_ok=True)

epochs = 10
patience = 3
min_delta = 0.001

train_losses = []
val_losses = []
val_accs = []

best_val_loss = float("inf")
counter = 0
best_model_path = "best_model.pth"

for epoch in range(epochs):
    # -------- TRAIN --------
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # -------- VALIDATE --------
    model.eval()
    val_running_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_acc = accuracy_score(trues, preds) * 100

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}: "
        f"Train Loss {train_loss:.4f}, "
        f"Val Loss {val_loss:.4f}, "
        f"Val Acc {val_acc:.2f}%"
    )

    # -------- SAVE CHECKPOINT --------
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc
    }, f"checkpoints/epoch_{epoch+1}.pth")

    # -------- EARLY STOP + BEST MODEL --------
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1}")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

print("Training complete.")

# -------- PLOTS --------
epochs_range = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_vs_epochs.png")
plt.show()

plt.figure()
plt.plot(epochs_range, val_accs, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("acc_vs_epochs.png")
plt.show()