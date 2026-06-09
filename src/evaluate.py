"""
Danh gia chi tiet model nhan dien mon an Viet Nam.

Script nay load model tu checkpoint (.pth) va chay tren tap Test
de xuat ra cac chi so danh gia: Confusion Matrix, Classification Report,
Per-class Accuracy, va bang so sanh nhieu model.

Su dung:
    # Danh gia 1 model:
    py -m src.evaluate --checkpoint checkpoints/best_model.pth

    # Danh gia va so sanh 2 model:
    py -m src.evaluate --checkpoint checkpoints/best_model.pth --checkpoint2 checkpoints/best_model_resnet18.pth

Ket qua luu vao:
    - figures/confusion_matrix_<model_name>.png
    - figures/per_class_accuracy_<model_name>.png
    - figures/classification_report_<model_name>.csv
    - figures/model_comparison.png  (neu co 2 model)
"""

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model import create_model
from src.preprocess import get_dataloaders
from src.food_metadata import get_display_name

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load model tu checkpoint .pth file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint khong ton tai: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Lay config
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", None)
    num_classes = config.get("num_classes", 29)

    # Neu khong co config (vi du resnet18 checkpoint), thu doan tu ten file
    if model_name is None:
        model_name = checkpoint.get("model_name", "efficientnet_b0")

    # Tao model
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_acc = checkpoint.get("best_val_acc", "N/A")
    epoch = checkpoint.get("epoch", "N/A")
    print(f"[OK] Loaded: {model_name}, {num_classes} classes, epoch {epoch}, val_acc {best_acc}")

    return model, model_name, num_classes


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    class_names: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Chay model tren toan bo tap Test.

    Returns:
        all_labels: Nhan that (numpy array)
        all_preds: Du doan cua model (numpy array)
        accuracy: Accuracy tong (%)
    """
    model.eval()
    all_labels = []
    all_preds = []

    print(f"\n  Dang danh gia tren {len(test_loader.dataset)} anh test...")

    with torch.no_grad():
        for inputs, labels, _ in tqdm(test_loader, desc="Evaluating", ncols=80):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = (all_labels == all_preds).sum() / len(all_labels) * 100

    print(f"  => Test Accuracy: {accuracy:.2f}%")
    return all_labels, all_preds, accuracy


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: Path,
):
    """Ve confusion matrix 29x29 va luu file PNG."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)

    # Tinh percentage
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    # Ten hien thi tieng Viet
    display_names = [get_display_name(name) for name in class_names]

    # Ve
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "% du doan dung"},
        ax=ax,
        annot_kws={"size": 6},
    )
    ax.set_xlabel("Du doan (Predicted)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Thuc te (Actual)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix - {model_name}\n"
        f"(gia tri = % du doan dung trong tung class)",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Confusion matrix: {save_path.name}")


# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_csv: Path,
    save_png: Path,
):
    """
    Tao classification report (Precision, Recall, F1-score)
    va luu thanh CSV + hinh anh.
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)

    display_names = [get_display_name(name) for name in class_names]

    # === Luu CSV ===
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
        for i, name in enumerate(display_names):
            writer.writerow([
                name,
                f"{precision[i]:.4f}",
                f"{recall[i]:.4f}",
                f"{f1[i]:.4f}",
                int(support[i]),
            ])
        # Weighted average
        writer.writerow([])
        writer.writerow([
            "Weighted Avg",
            f"{np.average(precision, weights=support):.4f}",
            f"{np.average(recall, weights=support):.4f}",
            f"{np.average(f1, weights=support):.4f}",
            int(support.sum()),
        ])
        writer.writerow(["Overall Accuracy", f"{accuracy:.4f}", "", "", ""])

    print(f"  [OK] Classification report CSV: {save_csv.name}")

    # === Ve bieu do ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    x = np.arange(len(class_names))
    bar_width = 0.7

    metrics = [
        ("Precision", precision, "#3498db"),
        ("Recall", recall, "#e74c3c"),
        ("F1-score", f1, "#2ecc71"),
    ]

    for ax, (metric_name, values, color) in zip(axes, metrics):
        bars = ax.barh(x, values, height=bar_width, color=color, alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(display_names, fontsize=7)
        ax.set_xlabel(metric_name, fontsize=10)
        ax.set_title(f"{metric_name} per Class", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.axvline(x=np.mean(values), color="black", linestyle="--", alpha=0.5,
                   label=f"Mean: {np.mean(values):.3f}")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Hien thi gia tri
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=6)

    plt.suptitle(
        f"Classification Report - {model_name} (Accuracy: {accuracy:.1%})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    save_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Classification report PNG: {save_png.name}")

    # In ra console
    print(f"\n  {'='*65}")
    print(f"  CLASSIFICATION REPORT — {model_name}")
    print(f"  {'='*65}")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*60}")
    for i, name in enumerate(display_names):
        print(f"  {name:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
    print(f"  {'-'*60}")
    print(f"  {'Weighted Avg':<20} {np.average(precision, weights=support):>10.4f} "
          f"{np.average(recall, weights=support):>10.4f} "
          f"{np.average(f1, weights=support):>10.4f} {support.sum():>10}")
    print(f"  {'Overall Accuracy':<20} {accuracy:>10.4f}")
    print(f"  {'='*65}")

    return {
        "accuracy": accuracy,
        "precision_weighted": np.average(precision, weights=support),
        "recall_weighted": np.average(recall, weights=support),
        "f1_weighted": np.average(f1, weights=support),
    }


# ============================================================================
# PER-CLASS ACCURACY
# ============================================================================

def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: Path,
):
    """Ve bar chart accuracy tung class."""
    per_class_acc = []
    display_names = [get_display_name(name) for name in class_names]

    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum() * 100
        else:
            acc = 0
        per_class_acc.append(acc)

    # Sap xep theo accuracy giam dan
    sorted_indices = np.argsort(per_class_acc)[::-1]
    sorted_names = [display_names[i] for i in sorted_indices]
    sorted_acc = [per_class_acc[i] for i in sorted_indices]

    # Ve
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.RdYlGn(np.array(sorted_acc) / 100)
    bars = ax.barh(range(len(sorted_names)), sorted_acc, color=colors, edgecolor="white")

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Per-Class Accuracy - {model_name}\n"
        f"(Mean: {np.mean(per_class_acc):.1f}%)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, 105)
    ax.axvline(x=np.mean(per_class_acc), color="blue", linestyle="--",
               alpha=0.7, label=f"Mean: {np.mean(per_class_acc):.1f}%")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Hien thi %
    for bar, acc in zip(bars, sorted_acc):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Per-class accuracy: {save_path.name}")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def plot_model_comparison(
    results: Dict[str, Dict],
    save_path: Path,
):
    """Ve bang so sanh 2+ model."""
    model_names = list(results.keys())
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    metric_labels = ["Accuracy", "Precision\n(Weighted)", "Recall\n(Weighted)", "F1-score\n(Weighted)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] * 100 for m in metrics]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        # Hien thi gia tri
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("So sanh hieu suat cac mo hinh", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Model comparison: {save_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def evaluate_checkpoint(checkpoint_path: str, test_loader, class_names, device):
    """Danh gia 1 checkpoint va luu ket qua."""
    model, model_name, num_classes = load_model(checkpoint_path, device)

    # Chay danh gia
    y_true, y_pred, test_acc = evaluate_model(model, test_loader, class_names, device)

    # Luu ket qua
    suffix = model_name.replace("-", "_")

    # 1. Confusion Matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names, model_name,
        save_path=FIGURES_DIR / f"confusion_matrix_{suffix}.png",
    )

    # 2. Classification Report
    metrics = generate_classification_report(
        y_true, y_pred, class_names, model_name,
        save_csv=FIGURES_DIR / f"classification_report_{suffix}.csv",
        save_png=FIGURES_DIR / f"classification_report_{suffix}.png",
    )

    # 3. Per-class Accuracy
    plot_per_class_accuracy(
        y_true, y_pred, class_names, model_name,
        save_path=FIGURES_DIR / f"per_class_accuracy_{suffix}.png",
    )

    return model_name, metrics


def main():
    parser = argparse.ArgumentParser(description="Danh gia model nhan dien mon an VN")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Duong dan toi checkpoint model chinh (.pth)",
    )
    parser.add_argument(
        "--checkpoint2", type=str, default=None,
        help="Duong dan toi checkpoint model so sanh (.pth) [optional]",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size cho test loader (default: 32)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" DANH GIA MO HINH NHAN DIEN MON AN VIET NAM")
    print(f"{'='*60}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load test data
    print(f"\n[STEP 1] Loading test data...")
    import platform
    num_workers = 0 if platform.system() == "Windows" else 4
    dataloaders, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=num_workers,
        use_weighted_sampler=False,
    )
    test_loader = dataloaders["test"]

    # Lay class names tu dataset
    data_dir = PROJECT_ROOT / "data" / "raw" / "Images" / "Train"
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Classes: {len(class_names)}")

    # Evaluate model 1
    print(f"\n[STEP 2] Danh gia model chinh: {args.checkpoint}")
    name1, metrics1 = evaluate_checkpoint(args.checkpoint, test_loader, class_names, device)

    all_results = {name1: metrics1}

    # Evaluate model 2 (neu co)
    if args.checkpoint2:
        print(f"\n[STEP 3] Danh gia model so sanh: {args.checkpoint2}")
        name2, metrics2 = evaluate_checkpoint(args.checkpoint2, test_loader, class_names, device)
        all_results[name2] = metrics2

        # So sanh
        print(f"\n[STEP 4] So sanh 2 model...")
        plot_model_comparison(all_results, save_path=FIGURES_DIR / "model_comparison.png")

        # In bang so sanh
        print(f"\n  {'='*65}")
        print(f"  BANG SO SANH MO HINH")
        print(f"  {'='*65}")
        print(f"  {'Chi so':<25} {name1:>18} {name2:>18}")
        print(f"  {'-'*61}")
        print(f"  {'Accuracy':<25} {metrics1['accuracy']*100:>17.2f}% {metrics2['accuracy']*100:>17.2f}%")
        print(f"  {'Precision (weighted)':<25} {metrics1['precision_weighted']*100:>17.2f}% {metrics2['precision_weighted']*100:>17.2f}%")
        print(f"  {'Recall (weighted)':<25} {metrics1['recall_weighted']*100:>17.2f}% {metrics2['recall_weighted']*100:>17.2f}%")
        print(f"  {'F1-score (weighted)':<25} {metrics1['f1_weighted']*100:>17.2f}% {metrics2['f1_weighted']*100:>17.2f}%")
        print(f"  {'='*65}")

    # Summary
    print(f"\n{'='*60}")
    print(f" HOAN THANH! Ket qua luu tai: figures/")
    print(f"{'='*60}")
    print(f"  Files da tao:")
    for f in sorted(FIGURES_DIR.glob("*")):
        if f.is_file() and f.stem.startswith(("confusion", "classification", "per_class", "model_comp")):
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
