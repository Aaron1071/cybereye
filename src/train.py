"""
train.py – Training script for CyberEye malware classification.

New in Week 4:
  • FocalLoss class  (addresses class imbalance better than weighted CE)
  • --loss flag: choose 'weighted_ce' | 'focal'
  • --model flag: choose 'cnn' | 'hybrid' | 'deeper' | 'full_vit'
  • Gradient clipping kept from Week 3
  • Best-model checkpointing + early stopping kept from Week 3

Usage examples:
  python train.py                              # default: hybrid + focal
  python train.py --model cnn   --loss weighted_ce
  python train.py --model deeper --loss focal
  python train.py --model full_vit --loss focal
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.dataset import get_loaders
from src.model import CNNBaseline, HybridCNNViT, DeeperHybrid, FullViTHybrid


# ─────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Why it helps here:
      - The MaleVis validation set has class 18 ("Other") with 1 482 samples
        while other classes have ~150.  Standard CE (even weighted) still lets
        easy majority examples dominate the gradient.
      - Focal loss down-weights well-classified examples (high p_t) so the
        model focuses on hard, mis-classified ones (Neshta, VBKrypt, Sality).

    Args:
        alpha  : per-class weight tensor (same as CE weight= argument).
                 Pass class-frequency-inverse weights for best results.
        gamma  : focusing parameter.  0 = standard CE.  2 is the default
                 from the original paper and works well here.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha          # (num_classes,) or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE per sample, shape (B,)
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")

        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce)

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CyberEye training script")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["cnn", "hybrid", "deeper", "full_vit"],
        help="Model variant to train",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="focal",
        choices=["weighted_ce", "focal"],
        help="Loss function",
    )
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--patience",  type=int,   default=3)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--gamma",     type=float, default=2.0,
                        help="Focal loss gamma (ignored for weighted_ce)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Loss   : {args.loss}  (gamma={args.gamma})")

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, train_counts, classes = get_loaders(
        batch_size=args.batch
    )
    num_classes = len(classes)

    # ── Class weights (inverse-frequency) ────────────────────
    weights = torch.tensor([
        len(train_loader.dataset) / (num_classes * train_counts[i])
        for i in range(num_classes)
    ]).to(device)

    # ── Model ─────────────────────────────────────────────────
    model_map = {
        "cnn":      CNNBaseline(num_classes),
        "hybrid":   HybridCNNViT(num_classes),     # 2-layer, 256d
        "deeper":   DeeperHybrid(num_classes),     # 4-layer, 384d
        "full_vit": FullViTHybrid(num_classes),    # 4-layer, 384d, CLS token
    }
    model = model_map[args.model].to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    # ── Loss ─────────────────────────────────────────────────
    if args.loss == "focal":
        criterion = FocalLoss(alpha=weights, gamma=args.gamma)
        print(f"Using FocalLoss(gamma={args.gamma}) with class weights")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using weighted CrossEntropyLoss")

    # ── Optimiser ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Output dirs ───────────────────────────────────────────
    run_name = f"{args.model}_{args.loss}"
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = f"best_model_{run_name}.pth"

    # ── Training loop ─────────────────────────────────────────
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float("inf")
    min_delta = 0.001
    counter = 0

    for epoch in range(args.epochs):

        # ── Train ──────────────────────────────────────────────
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

        # ── Validate ────────────────────────────────────────────
        model.eval()
        val_running_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_acc  = accuracy_score(trues, preds) * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs}  "
            f"Train Loss {train_loss:.4f}  "
            f"Val Loss {val_loss:.4f}  "
            f"Val Acc {val_acc:.2f}%"
        )

        # ── Save per-epoch checkpoint ───────────────────────────
        torch.save({
            "epoch":              epoch + 1,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":         train_loss,
            "val_loss":           val_loss,
            "val_acc":            val_acc,
        }, f"{ckpt_dir}/epoch_{epoch+1}.pth")

        # ── Best-model saving + early stopping ─────────────────
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved  (val_loss={val_loss:.4f})")
        else:
            counter += 1
            if counter >= args.patience:
                print(f"  Early stop at epoch {epoch+1}  (patience={args.patience})")
                break

    print("Training complete.")

    # ── Plots ─────────────────────────────────────────────────
    epochs_range = range(1, len(train_losses) + 1)
    prefix = f"plots/{run_name}"
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss – {run_name}")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{prefix}_loss.png"); plt.close()

    plt.figure()
    plt.plot(epochs_range, val_accs, label="Val Accuracy", color="green")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy – {run_name}")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{prefix}_acc.png"); plt.close()

    print(f"Plots saved to plots/{run_name}_*.png")


if __name__ == "__main__":
    main()
