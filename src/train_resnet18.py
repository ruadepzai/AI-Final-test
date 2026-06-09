"""
Script huan luyen ResNet-18 de so sanh voi EfficientNetB0.

File nay doc lap voi train.py (EfficientNetB0).
Chi can chay 1 lenh duy nhat:

    python -m src.train_resnet18

Ket qua se luu vao:
    - checkpoints/best_model_resnet18.pth   (model weights)
    - figures/training_curves_resnet18.png   (bieu do training)
    - checkpoints/training_history_resnet18.json (log chi tiet)
"""

import json
import os
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.preprocess import get_dataloaders, DATA_DIR
from src.model import create_model, unfreeze_backbone, get_model_info

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG — Chinh sua o day neu can
# ============================================================================

MODEL_NAME = "resnet18"
NUM_CLASSES = 29
BATCH_SIZE = 32
NUM_EPOCHS = 30          # Tong so epochs (Phase 1 + Phase 2)
FREEZE_EPOCHS = 5        # Phase 1: freeze backbone, train head
LEARNING_RATE = 1e-3     # LR Phase 1
FINE_TUNE_LR = 1e-5      # LR Phase 2
WEIGHT_DECAY = 1e-4
PATIENCE = 7             # Early stopping
SEED = 42

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train():
    """Huan luyen ResNet-18 voi 2-phase strategy."""
    
    # Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" TRAIN RESNET-18 (SO SANH VOI EFFICIENTNET-B0)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # DataLoaders
    print(f"\n[1/5] Loading data...")
    num_workers = 0 if platform.system() == "Windows" else 4
    dataloaders, class_weights = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        use_weighted_sampler=True,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Model
    print(f"\n[2/5] Creating {MODEL_NAME}...")
    model = create_model(MODEL_NAME, num_classes=NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)
    get_model_info(model)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Dirs
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": [], "epoch_time": [],
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    total_start = time.time()
    
    # ========== TRAINING LOOP ==========
    print(f"\n[3/5] Training {NUM_EPOCHS} epochs...")
    print(f"  Phase 1: Epoch 1-{FREEZE_EPOCHS} (freeze backbone, LR={LEARNING_RATE})")
    print(f"  Phase 2: Epoch {FREEZE_EPOCHS+1}-{NUM_EPOCHS} (fine-tune, LR={FINE_TUNE_LR})")
    
    # Phase 1 optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = None
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # === Phase 2: Unfreeze backbone ===
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n{'='*40}")
            print(f" PHASE 2: Unfreeze backbone")
            print(f"{'='*40}")
            unfreeze_backbone(model, MODEL_NAME)
            get_model_info(model)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS - FREEZE_EPOCHS,
                eta_min=FINE_TUNE_LR / 100,
            )
        
        # === Train ===
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
                     leave=False, ncols=100)
        
        for inputs, labels, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total * 100
        
        # === Validate ===
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss_sum += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total * 100
        
        # LR
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        
        # Print
        phase = "P1" if epoch <= FREEZE_EPOCHS else "P2"
        print(f"  [{phase}] Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.0f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            
            save_path = CHECKPOINT_DIR / "best_model_resnet18.pth"
            torch.save(checkpoint, save_path)
            print(f"  >>> NEW BEST: {best_val_acc:.2f}% (saved to {save_path.name})")
        else:
            patience_counter += 1
            if epoch > FREEZE_EPOCHS and patience_counter >= PATIENCE:
                print(f"\n  [EARLY STOP] Val acc khong tang sau {PATIENCE} epochs.")
                break
    
    total_time = time.time() - total_start
    
    # ========== SAVE RESULTS ==========
    print(f"\n[4/5] Saving results...")
    
    # Save history JSON
    history_path = CHECKPOINT_DIR / "training_history_resnet18.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {history_path.name}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs_range, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].axvline(x=FREEZE_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Phase 1 | Phase 2")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs_range, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs_range, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1].axhline(y=best_val_acc, color="green", linestyle="--", alpha=0.5,
                     label=f"Best: {best_val_acc:.1f}%")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs_range, history["lr"], "g-o", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Training Curves - ResNet-18 (Best Val Acc: {best_val_acc:.1f}%)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "training_curves_resnet18.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves: {fig_path.name}")
    
    # ========== SUMMARY ==========
    print(f"\n[5/5] DONE!")
    print(f"{'='*60}")
    print(f"  Model:          ResNet-18")
    print(f"  Best Val Acc:   {best_val_acc:.2f}%")
    print(f"  Best Epoch:     {history['val_acc'].index(best_val_acc) + 1}")
    print(f"  Total Time:     {total_time/3600:.2f} hours")
    print(f"  Checkpoint:     checkpoints/best_model_resnet18.pth")
    print(f"  Curves:         figures/training_curves_resnet18.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
