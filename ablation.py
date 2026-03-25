"""
ablation.py – Ablation study: CNN vs Hybrid vs Deeper Hybrid vs FullViT

Trains (or loads) all four variants and prints a side-by-side comparison
table of accuracy, macro-F1, and per-class F1 for the challenging families.

Usage:
  python ablation.py               # train all + compare
  python ablation.py --skip_train  # load existing best_model_*.pth files + compare only

Structure of results dict saved to ablation_results.json for later analysis.
"""

import os
import json
import argparse
import subprocess
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.dataset import get_loaders
from src.model import CNNBaseline, HybridCNNViT, DeeperHybrid, FullViTHybrid
from src.train import FocalLoss


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name":       "CNN Baseline",
        "key":        "cnn_weighted_ce",
        "model_cls":  CNNBaseline,
        "loss":       "weighted_ce",
        "model_arg":  "cnn",
    },
    {
        "name":       "Hybrid (2L-256d)",
        "key":        "hybrid_focal",
        "model_cls":  HybridCNNViT,
        "loss":       "focal",
        "model_arg":  "hybrid",
    },
    {
        "name":       "Deeper Hybrid (4L-384d)",
        "key":        "deeper_focal",
        "model_cls":  DeeperHybrid,
        "loss":       "focal",
        "model_arg":  "deeper",
    },
    {
        "name":       "Full ViT Hybrid (CLS)",
        "key":        "full_vit_focal",
        "model_cls":  FullViTHybrid,
        "loss":       "focal",
        "model_arg":  "full_vit",
    },
]

# Families most relevant for the research hypothesis
FOCUS_FAMILIES = ["Neshta", "VBKrypt", "Sality", "Injector", "Expiro", "Other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Evaluate one model
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, val_loader):
    """Return (preds, trues) arrays from the validation set."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            trues.extend(labels.numpy())
    return preds, trues


def load_best_model(variant, num_classes):
    """Instantiate model and load best checkpoint weights."""
    model = variant["model_cls"](num_classes=num_classes).to(device)
    ckpt_path = f"best_model_{variant['key']}.pth"

    if not os.path.exists(ckpt_path):
        # Fallback: try legacy single best_model.pth (Week-3 compatible)
        fallback = "best_model.pth"
        if variant["key"].startswith("hybrid") and os.path.exists(fallback):
            ckpt_path = fallback
            print(f"  [warn] Using fallback checkpoint: {fallback}")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run training first:  python train.py --model {variant['model_arg']} "
                f"--loss {variant['loss']}"
            )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


# ─────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────
def print_comparison_table(results, classes):
    """Print a formatted comparison table."""
    header = f"{'Variant':<30}  {'Accuracy':>9}  {'Macro F1':>9}  {'Wtd F1':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r['name']:<30}  "
            f"{r['accuracy']:>8.2f}%  "
            f"{r['macro_f1']:>9.4f}  "
            f"{r['weighted_f1']:>9.4f}"
        )
    print("=" * len(header))

    # Focus family breakdown
    print(f"\n{'Per-class F1 for challenging families':}")
    print(f"{'Family':<18}", end="")
    for r in results:
        print(f"  {r['name'][:16]:>16}", end="")
    print()
    print("-" * (18 + 18 * len(results)))
    for family in FOCUS_FAMILIES:
        if family not in classes:
            continue
        idx = classes.index(family)
        print(f"{family:<18}", end="")
        for r in results:
            f1_val = r["per_class_f1"][idx]
            print(f"  {f1_val:>16.4f}", end="")
        print()


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
def plot_comparison(results, classes):
    os.makedirs("plots", exist_ok=True)

    # ── Overall metrics bar chart ──────────────────────────────
    names      = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    macro_f1s  = [r["macro_f1"] * 100 for r in results]
    weighted_f1s = [r["weighted_f1"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, accuracies,   width, label="Accuracy (%)",   color="#4C72B0")
    ax.bar(x,         macro_f1s,    width, label="Macro F1 (%)",   color="#DD8452")
    ax.bar(x + width, weighted_f1s, width, label="Weighted F1 (%)",color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(70, 100)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation Study – Overall Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("plots/ablation_overall.png", dpi=150)
    plt.close()
    print("Saved: plots/ablation_overall.png")

    # ── Focus-family F1 comparison ─────────────────────────────
    focus = [f for f in FOCUS_FAMILIES if f in classes]
    focus_idxs = [classes.index(f) for f in focus]

    fig, ax = plt.subplots(figsize=(12, 6))
    x2 = np.arange(len(focus))
    bar_width = 0.8 / len(results)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, r in enumerate(results):
        vals = [r["per_class_f1"][idx] for idx in focus_idxs]
        offset = (i - len(results) / 2 + 0.5) * bar_width
        ax.bar(x2 + offset, vals, bar_width, label=r["name"], color=colors[i])

    ax.set_xticks(x2)
    ax.set_xticklabels(focus)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation Study – F1 on Challenging Families")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("plots/ablation_focus_families.png", dpi=150)
    plt.close()
    print("Saved: plots/ablation_focus_families.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and load existing best_model_*.pth files",
    )
    parser.add_argument("--epochs",   type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────
    _, val_loader, _, classes = get_loaders()
    num_classes = len(classes)

    # ── Train each variant (unless --skip_train) ─────────────
    if not args.skip_train:
        for v in VARIANTS:
            print(f"\n{'='*60}")
            print(f"TRAINING: {v['name']}")
            print(f"{'='*60}")
            cmd = [
                sys.executable, "train.py",
                "--model",   v["model_arg"],
                "--loss",    v["loss"],
                "--epochs",  str(args.epochs),
                "--patience",str(args.patience),
            ]
            subprocess.run(cmd, check=True)

    # ── Evaluate each variant ─────────────────────────────────
    results = []
    for v in VARIANTS:
        print(f"\nEvaluating: {v['name']} ...")
        try:
            model = load_best_model(v, num_classes)
        except FileNotFoundError as e:
            print(f"  SKIP – {e}")
            continue

        preds, trues = evaluate_model(model, val_loader)

        report = classification_report(
            trues, preds,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        per_class_f1 = [
            report[cls]["f1-score"] for cls in classes
        ]

        results.append({
            "name":         v["name"],
            "accuracy":     accuracy_score(trues, preds) * 100,
            "macro_f1":     f1_score(trues, preds, average="macro",    zero_division=0),
            "weighted_f1":  f1_score(trues, preds, average="weighted", zero_division=0),
            "per_class_f1": per_class_f1,
        })

    if not results:
        print("No results to compare. Run without --skip_train first.")
        return

    # ── Print table ────────────────────────────────────────────
    print_comparison_table(results, classes)

    # ── Plots ──────────────────────────────────────────────────
    plot_comparison(results, classes)

    # ── Save JSON ──────────────────────────────────────────────
    json_safe = []
    for r in results:
        d = dict(r)
        d["per_class_f1"] = [float(v) for v in r["per_class_f1"]]
        json_safe.append(d)

    with open("ablation_results.json", "w") as f:
        json.dump({"classes": classes, "results": json_safe}, f, indent=2)

    print("\nFull results saved to: ablation_results.json")
    print("Plots saved to:        plots/ablation_*.png")


if __name__ == "__main__":
    main()
