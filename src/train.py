import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score  # For acc (teach: Sklearn = ML utils lib)
from src.dataset import get_loaders, denorm  # Import (why? Modular—reuse loaders)
from src.model import CNNBaseline
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device = hardware (GPU if avail—why? Parallel for convolutions; connect: GPU in forensics for password cracking)

train_loader, val_loader, train_counts, classes = get_loaders()
num_classes = len(classes)

weights = torch.tensor([  # Weights (inverse count—high for rares; connect: Balance like weighting rare logs in SIEM for threat hunting)
    len(train_loader.dataset) / (num_classes * train_counts[i])
    for i in range(num_classes)
]).to(device)

model = CNNBaseline(num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)  # Loss = error measure (CrossEntropy for multi-class; term: Criterion = loss func)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer = updater (Adam adaptive; parameters = weights; lr = learning rate—step size)

os.makedirs("checkpoints", exist_ok=True)  # Makedirs = create dirs (exist_ok no error if exists)

epochs = 10
train_losses, val_losses, val_accs = [], [], []
best_val_loss = float('inf')  # Inf = infinity (start high)
patience, counter, min_delta = 3, 0, 0.001  # Early stop params (patience = wait epochs; min_delta = improvement threshold—teach: Hyperparam = tunable setting)

for epoch in range(epochs):
    model.train()  # Train mode (enables updates)
    running_loss = 0.0
    for images, labels in train_loader:  # Loop batches
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero grads (reset)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backprop (compute grads)
        optimizer.step()  # Step (update)
        running_loss += loss.item()  # Item = scalar
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()  # Eval mode
    val_running_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():  # No grad (save mem)
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            pred = outputs.argmax(dim=1).cpu().numpy()  # Argmax = top class
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())
    val_loss = val_running_loss / len(val_loader)
    val_acc = accuracy_score(trues, preds) * 100  # *100 for %
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")

    # Checkpoint (save dict—why? Resume; connect: Like saving forensic tool states)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, f"checkpoints/epoch_{epoch+1}.pth")

    # Early stop (new: If no improve, counter++)
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break  # Break = exit loop

torch.save(model.state_dict(), "cnn_final_model.pth")  # Final save

# Graphs (liked—separate figs; save for report; connect: Plot like entropy graphs in malware forensics)
epochs_range = list(range(1, len(train_losses) + 1))  # List = sequence (1 to epochs)
plt.figure()  # Figure = canvas
plt.plot(epochs_range, train_losses, label="Train Loss")  # Plot = line (label for legend)
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epochs")  # Xlabel = axis label
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()  # Legend = key
plt.grid(True)  # Grid = lines for readability
plt.savefig("loss_vs_epochs.png")  # Savefig = export png
plt.show()  # Show window

plt.figure()
plt.plot(epochs_range, val_accs, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("acc_vs_epochs.png")
plt.show()

if __name__ == "__main__":
    print("Train complete.")