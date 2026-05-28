"""
Script huan luyen model nhan dien mon an Viet Nam.

Module nay cung cap:
- TrainingConfig: Dataclass cau hinh hyperparameters
- Trainer: Class training loop day du
- 2-phase training: Phase 1 train head, Phase 2 fine-tune backbone
- Mixed precision (torch.cuda.amp) cho Colab GPU
- Early stopping, checkpoint saving, training curves

Su dung:
    # Colab / CLI:
    python src/train.py --model_name efficientnet_b0 --epochs 30 --batch_size 32

    # Dry run (test pipeline):
    python src/train.py --dry_run --epochs 1 --batch_size 4
"""

import argparse
import json
import os
import platform
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho server/Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

# Import tu cac module khac trong project
from src.preprocess import get_dataloaders, DATA_DIR
from src.model import create_model, unfreeze_backbone, get_model_info


# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent
"""Thu muc goc cua project."""

CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
"""Thu muc luu model checkpoints."""

FIGURES_DIR: Path = PROJECT_ROOT / "figures"
"""Thu muc luu bieu do training."""


# ============================================================================
# TRAINING CONFIG
# ============================================================================


@dataclass
class TrainingConfig:
    """
    Cau hinh hyperparameters cho training.

    Attributes:
        model_name: Ten model ("efficientnet_b0" | "mobilenet_v3_small").
        num_classes: So luong classes (30 mon an VN).
        batch_size: Kich thuoc batch. 32 cho Colab T4 GPU (15GB VRAM).
        num_epochs: Tong so epochs (Phase 1 + Phase 2).
        learning_rate: Learning rate cho classifier head (Phase 1).
        fine_tune_lr: Learning rate cho backbone khi fine-tune (Phase 2).
        weight_decay: L2 regularization.
        patience: So epochs cho phep val_loss khong giam truoc khi early stop.
        freeze_epochs: So epochs Phase 1 (train head, freeze backbone).
        optimizer: Loai optimizer ("adam" | "sgd").
        scheduler: Loai LR scheduler ("cosine" | "step" | "plateau").
        num_workers: So worker processes cho DataLoader. 0 tren Windows.
        use_amp: Su dung mixed precision (True cho GPU).
        use_weighted_sampler: Dung WeightedRandomSampler xu ly imbalance.
        checkpoint_dir: Thu muc luu checkpoints.
        seed: Random seed de reproducible.
    """
    model_name: str = "efficientnet_b0"
    num_classes: int = 29
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 7
    freeze_epochs: int = 5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    num_workers: int = 4
    use_amp: bool = True
    use_weighted_sampler: bool = True
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    seed: int = 42

    def __post_init__(self):
        # Windows khong ho tro num_workers > 0 tot
        if platform.system() == "Windows":
            self.num_workers = 0


# ============================================================================
# TRAINER CLASS
# ============================================================================


class Trainer:
    """
    Training loop day du voi 2-phase training strategy.

    Phase 1 (freeze_epochs epochs):
        - Backbone FROZEN, chi train classifier head
        - Learning rate cao (1e-3) de hoc nhanh
        - Khong ap dung early stopping

    Phase 2 (remaining epochs):
        - Backbone UNFROZEN, fine-tune toan bo model
        - Learning rate thap (1e-5) de khong pha pretrained features
        - CosineAnnealingLR giam dan LR
        - Early stopping neu val_loss khong giam

    Attributes:
        model: Model can train.
        dataloaders: Dict chua train/val/test DataLoaders.
        criterion: Loss function.
        device: torch.device (cuda/cpu).
        config: TrainingConfig.
        history: Dict luu lich su training (loss, accuracy, lr).
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        criterion: nn.Module,
        device: torch.device,
        config: TrainingConfig,
    ) -> None:
        """
        Khoi tao Trainer.

        Args:
            model: Model da tao tu create_model().
            dataloaders: Dict voi keys "train", "val", "test".
            criterion: Loss function (CrossEntropyLoss).
            device: Device de train (cuda/cpu).
            config: TrainingConfig instance.
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion.to(device)
        self.device = device
        self.config = config

        # Optimizer - chi train params co requires_grad
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }

        # Best model tracking
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.epochs_no_improve: int = 0

        # Tao thu muc checkpoint
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Tao optimizer chi cho trainable params."""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                trainable_params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer '{self.config.optimizer}' khong ho tro. Chon: adam, sgd")

    def _create_scheduler(self):
        """Tao learning rate scheduler."""
        sched_name = self.config.scheduler.lower()
        remaining_epochs = self.config.num_epochs - self.config.freeze_epochs

        if sched_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=max(remaining_epochs, 1),
                eta_min=1e-7,
            )
        elif sched_name == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif sched_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
            )
        else:
            raise ValueError(f"Scheduler '{sched_name}' khong ho tro.")

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train 1 epoch.

        Args:
            epoch: So thu tu epoch (0-indexed).

        Returns:
            Tuple[train_loss, train_accuracy].
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loader = self.dataloaders["train"]
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision forward
            with torch.amp.autocast("cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward voi scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.1f}%",
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate model tren validation set.

        Args:
            epoch: So thu tu epoch.

        Returns:
            Tuple[val_loss, val_accuracy].
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        loader = self.dataloaders["val"]
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)

        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def fit(self) -> Dict[str, List[float]]:
        """
        Main training loop voi 2-phase strategy.

        Phase 1: Train classifier head (backbone frozen).
        Phase 2: Fine-tune toan bo model (backbone unfrozen).

        Returns:
            Dict training history.
        """
        config = self.config
        total_epochs = config.num_epochs
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"🚀 BAT DAU TRAINING")
        print(f"{'='*70}")
        print(f"  Model:          {config.model_name}")
        print(f"  Device:         {self.device}")
        print(f"  Total epochs:   {total_epochs}")
        print(f"  Phase 1:        {config.freeze_epochs} epochs (freeze backbone, lr={config.learning_rate})")
        print(f"  Phase 2:        {total_epochs - config.freeze_epochs} epochs (fine-tune, lr={config.fine_tune_lr})")
        print(f"  Batch size:     {config.batch_size}")
        print(f"  Optimizer:      {config.optimizer}")
        print(f"  Scheduler:      {config.scheduler}")
        print(f"  Early stopping: patience={config.patience}")
        print(f"  Mixed precision:{config.use_amp}")
        print(f"{'='*70}\n")

        for epoch in range(total_epochs):
            epoch_start = time.time()

            # === CHUYEN PHASE ===
            if epoch == config.freeze_epochs:
                print(f"\n{'='*70}")
                print(f"🔓 PHASE 2: UNFREEZE BACKBONE (epoch {epoch+1})")
                print(f"{'='*70}")

                # Unfreeze backbone
                unfreeze_backbone(self.model, config.model_name, num_layers=None)
                get_model_info(self.model)

                # Tao optimizer moi voi fine-tune LR
                # Dung different LR cho backbone va classifier
                backbone_params = []
                classifier_params = []

                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if "classifier" in name or "fc" in name:
                            classifier_params.append(param)
                        else:
                            backbone_params.append(param)

                self.optimizer = torch.optim.Adam([
                    {"params": backbone_params, "lr": config.fine_tune_lr},
                    {"params": classifier_params, "lr": config.fine_tune_lr * 10},
                ], weight_decay=config.weight_decay)

                # Tao scheduler moi
                self.scheduler = self._create_scheduler()

                # Reset early stopping
                self.epochs_no_improve = 0

            # === TRAIN + VALIDATE ===
            phase = "Phase 1" if epoch < config.freeze_epochs else "Phase 2"
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            epoch_time = time.time() - epoch_start

            # Luu history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # === CHECKPOINT ===
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
                best_marker = " ⭐ BEST"
            else:
                self.epochs_no_improve += 1
                best_marker = ""

            # Save last checkpoint
            self.save_checkpoint(epoch, is_best=False)

            # === LOG ===
            print(
                f"[{phase}] Epoch {epoch+1:02d}/{total_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.0f}s{best_marker}"
            )

            # === LR SCHEDULER ===
            if epoch >= config.freeze_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # === EARLY STOPPING (chi Phase 2) ===
            if epoch >= config.freeze_epochs and self.epochs_no_improve >= config.patience:
                print(f"\n⚠ Early stopping! Val acc khong cai thien sau {config.patience} epochs.")
                print(f"  Best val acc: {self.best_val_acc:.1f}% (epoch {self.best_epoch})")
                break

        # === KET THUC ===
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ TRAINING HOAN TAT!")
        print(f"{'='*70}")
        print(f"  Tong thoi gian:  {total_time/60:.1f} phut")
        print(f"  Best val acc:    {self.best_val_acc:.1f}% (epoch {self.best_epoch})")
        print(f"  Last val acc:    {val_acc:.1f}%")
        print(f"  Checkpoint dir:  {config.checkpoint_dir}")
        print(f"{'='*70}\n")

        # Plot training curves
        self.plot_training_curves()

        # Save history
        self._save_history()

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Luu model checkpoint.

        Args:
            epoch: Epoch hien tai.
            is_best: Neu True, luu nhu best_model.pth.
        """
        checkpoint = {
            "epoch": epoch + 1,
            "model_name": self.config.model_name,
            "num_classes": self.config.num_classes,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": asdict(self.config),
        }

        ckpt_dir = Path(self.config.checkpoint_dir)

        if is_best:
            path = ckpt_dir / "best_model.pth"
            torch.save(checkpoint, path)

        # Luon luu last
        path = ckpt_dir / "last_model.pth"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """
        Load checkpoint de resume training.

        Args:
            path: Duong dan checkpoint file.

        Returns:
            Epoch tiep theo de train.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint khong ton tai: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)

        start_epoch = checkpoint["epoch"]
        print(f"✓ Loaded checkpoint: epoch {start_epoch}, best_val_acc={self.best_val_acc:.1f}%")

        return start_epoch

    def plot_training_curves(self) -> None:
        """
        Ve bieu do training loss va accuracy.
        Luu vao figures/training_curves.png.
        """
        if not self.history["train_loss"]:
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], "b-o", markersize=3, label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"], "r-o", markersize=3, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], "b-o", markersize=3, label="Train Acc")
        axes[1].plot(epochs, self.history["val_acc"], "r-o", markersize=3, label="Val Acc")
        axes[1].axhline(y=self.best_val_acc, color="g", linestyle="--", alpha=0.5,
                        label=f"Best: {self.best_val_acc:.1f}%")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        axes[2].plot(epochs, self.history["lr"], "g-o", markersize=3)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        # Phase separator
        freeze_epochs = self.config.freeze_epochs
        if freeze_epochs < len(self.history["train_loss"]):
            for ax in axes:
                ax.axvline(x=freeze_epochs + 0.5, color="gray", linestyle=":", alpha=0.7)
            axes[0].text(freeze_epochs + 0.7, axes[0].get_ylim()[1] * 0.9,
                        "← Phase 1 | Phase 2 →", fontsize=8, color="gray")

        plt.suptitle(
            f"Training Curves - {self.config.model_name} "
            f"(Best Val Acc: {self.best_val_acc:.1f}% @ epoch {self.best_epoch})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        save_path = FIGURES_DIR / f"training_curves_{self.config.model_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Training curves saved: {save_path}")

    def _save_history(self) -> None:
        """Luu training history ra file JSON."""
        history_path = Path(self.config.checkpoint_dir) / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved: {history_path}")


# ============================================================================
# SEED & DEVICE SETUP
# ============================================================================


def setup_seed(seed: int = 42) -> None:
    """Dat random seed de reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed: {seed}")


def get_device() -> torch.device:
    """Lay device tot nhat (cuda > cpu)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"✓ Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print(f"⚠ Device: {device} (khong co GPU, training se cham)")
    return device


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main function - chay training tu command line."""
    parser = argparse.ArgumentParser(
        description="Train Vietnamese Food Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "mobilenet_v3_small"],
                        help="Ten model")
    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Tong so epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Phase 1")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate Phase 2")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="So epochs Phase 1")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "plateau"])

    # System
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Tat mixed precision")

    # Resume / debug
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path de resume")
    parser.add_argument("--dry_run", action="store_true", help="Chay 1 epoch voi batch_size=4")

    args = parser.parse_args()

    # === CONFIG ===
    config = TrainingConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        seed=args.seed,
    )

    # Dry run override
    if args.dry_run:
        config.num_epochs = 1
        config.batch_size = 4
        config.freeze_epochs = 0
        print("⚠ DRY RUN MODE: 1 epoch, batch_size=4")

    # === SETUP ===
    setup_seed(config.seed)
    device = get_device()

    # === DATA ===
    print(f"\n⏳ Loading data...")
    dataloaders, class_weights = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_weighted_sampler=config.use_weighted_sampler,
        check_stratified=True,
    )
    print(f"✓ Data loaded:")
    for split, loader in dataloaders.items():
        print(f"  {split:6}: {len(loader.dataset):,} samples, {len(loader)} batches")

    # === MODEL ===
    print(f"\n⏳ Creating model...")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=True,
        freeze_backbone=(config.freeze_epochs > 0),
    )
    get_model_info(model)

    # === LOSS FUNCTION ===
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"✓ Loss: CrossEntropyLoss voi class weights (xu ly imbalance)")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"✓ Loss: CrossEntropyLoss (khong co class weights)")

    # === TRAINER ===
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        config=config,
    )

    # Resume tu checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # === TRAIN ===
    history = trainer.fit()

    print(f"\n✓ Done! Best model: {config.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()
