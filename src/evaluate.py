"""
evaluate.py – Evaluate any trained model variant.

Usage:
  python -m src.evaluate                             # default: hybrid + focal
  python -m src.evaluate --model cnn  --run cnn_weighted_ce
  python -m src.evaluate --model deeper --run deeper_focal
  python -m src.evaluate --model full_vit --run full_vit_focal

The --run argument identifies which best_model_<run>.pth to load.
If omitted, defaults to best_model_hybrid_focal.pth (or best_model.pth
as Week-3 fallback).
"""

import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.dataset import get_loaders
from src.model import CNNBaseline, HybridCNNViT, DeeperHybrid, FullViTHybrid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["cnn", "hybrid", "deeper", "full_vit"],
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Checkpoint run name, e.g. 'hybrid_focal'. "
            "Loads best_model_<run>.pth. "
            "Defaults to <model>_focal (or cnn_weighted_ce for cnn)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ────────────────────────────────────────────────────
    _, val_loader, _, classes = get_loaders()
    num_classes = len(classes)

    # ── Model ───────────────────────────────────────────────────
    model_map = {
        "cnn":      CNNBaseline(num_classes),
        "hybrid":   HybridCNNViT(num_classes),
        "deeper":   DeeperHybrid(num_classes),
        "full_vit": FullViTHybrid(num_classes),
    }
    model = model_map[args.model].to(device)

    # ── Checkpoint ──────────────────────────────────────────────
    if args.run:
        ckpt_path = f"best_model_{args.run}.pth"
    else:
        # Sensible defaults
        default_run = "cnn_weighted_ce" if args.model == "cnn" else f"{args.model}_focal"
        ckpt_path = f"best_model_{default_run}.pth"

    # Week-3 backward compat: fall back to best_model.pth for hybrid
    import os
    if not os.path.exists(ckpt_path) and args.model == "hybrid":
        fallback = "best_model.pth"
        if os.path.exists(fallback):
            print(f"[info] Falling back to legacy checkpoint: {fallback}")
            ckpt_path = fallback

    print(f"Loading: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ── Inference ───────────────────────────────────────────────
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            trues.extend(labels.numpy())

    # ── Report ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Model: {args.model}  |  Checkpoint: {ckpt_path}")
    print(f"{'─'*60}")
    print(classification_report(trues, preds, target_names=classes, zero_division=0))

    # ── Confusion matrix ────────────────────────────────────────
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.3,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    run_label = args.run or f"{args.model}_focal"
    plt.title(f"Confusion Matrix – {run_label}")
    plt.tight_layout()
    out_path = f"confusion_matrix_{run_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved: {out_path}")


if __name__ == "__main__":
    main()
