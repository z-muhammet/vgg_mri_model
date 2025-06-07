import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter
import os, time, torch, matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as T
from dataset.custom_dataset import CustomTumorDataset
from models.vgg_custom import VGGCustom
import copy
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import gc  # Garbage collector için eklendi
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import platform

# ANSI renk kodları
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Hyperparameters
INITIAL_LR = 5e-3  # Learning rate düşürüldü (5e-2'den 5e-3'e)
BATCH_SIZE = 8    # Batch size 8 olarak ayarlandı
NUM_EPOCHS = 60
WARMUP_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class weight parameters
lambda_penalty = 0.1  # Penalty artırıldı
min_weight = 0.7

# Overfitting/Underfitting prevention parameters
# Overfitting durumunu tespit etmek için kullanılan eşik değeri
# Eğer validation loss ile training loss arasındaki fark bu değerden büyükse overfitting olarak kabul edilir
OVERFIT_THRESHOLD = 0.22  # Threshold düşürüldü (önceki 0.15'ti)

# Underfitting durumunu tespit etmek için kullanılan eşik değeri  
# Eğer validation accuracy bu değerin altındaysa underfitting olarak kabul edilir
UNDERFIT_THRESHOLD = 0.50  # Threshold düşürüldü (önceki 0.65'ti)

# Overfitting durumunda izin verilen maksimum epoch sayısı
# Bu sayıya ulaşılırsa eğitim durdurulur
MAX_OVERFIT_EPOCHS = 5    # Maximum epochs allowed for overfitting

# Underfitting durumunda izin verilen maksimum epoch sayısı
# Bu sayıya ulaşılırsa eğitim durdurulur
MAX_UNDERFIT_EPOCHS = 10   # Maximum epochs allowed for underfitting

# Regularization için başlangıç faktörü
# Dropout ve weight decay gibi regularization tekniklerinin şiddetini belirler
REGULARIZATION_FACTOR = 0.1  # Regularization azaltıldı (önceki 0.1'di)

# Regularization için izin verilen maksimum değer
# Regularization faktörü bu değeri geçemez
MAX_REGULARIZATION = 0.3   # Maximum regularization azaltıldı (önceki 0.5'ti)

# Rollback parameters
ROLLBACK_ACC_THRESHOLD = 0.015  # 1.5% accuracy drop threshold
ROLLBACK_LOSS_THRESHOLD = 0.04  # 4% loss increase threshold
ROLLBACK_COOLDOWN = 3  # Minimum epochs between rollbacks
MAX_TOTAL_ROLLBACKS = 10  # Maximum total rollbacks during training
ROLLBACK_WINDOW = 20  # Window to check for rollback activity

class DynamicRegularization(nn.Module):
    def __init__(self, model, initial_factor=0.1, max_factor=0.5):
        super().__init__()
        self.model = model
        self.factor = initial_factor
        self.max_factor = max_factor
        self.original_dropout_rates = {}
        self.store_original_rates()

    def store_original_rates(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout2d):
                self.original_dropout_rates[name] = module.p

    def increase_regularization(self):
        self.factor = min(self.factor * 1.5, self.max_factor)
        self.apply_regularization()

    def decrease_regularization(self):
        self.factor = max(self.factor / 1.5, REGULARIZATION_FACTOR)
        self.apply_regularization()

    def apply_regularization(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout2d):
                original_rate = self.original_dropout_rates[name]
                module.p = min(original_rate * (1 + self.factor), 0.9)

# ---------- Trainer State dataclass -------------------------------------------------------------
@dataclass
class TrainerState:
    best_acc: float = 0.0
    best_loss: float = math.inf
    best_weights: Optional[dict] = None
    epoch: int = 0
    last_rb_epoch: int = 0
    total_rbs: int = 0
    cooldown: int = 0

# ---------- Trainer -----------------------------------------------------------------------------
class Trainer(nn.Module): # Inherit from nn.Module for DynamicRegularization
    def __init__(
        model: nn.Module,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        max_epochs: int = 60,
        overfit_thr: float = 0.25,
        underfit_thr: float = 0.55,
        plot_every: int = 5,
        checkpoint_dir: str = "models",
        class_weights=None
    ) -> None:
        super().__init__() # Call super constructor
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.overfit_thr = overfit_thr
        self.underfit_thr = underfit_thr
        self.plot_every = plot_every
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt_path = os.path.join(checkpoint_dir, "best_vgg_custom.pt")
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.amp_enabled = self.device.type == "cuda" and (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=2)
        # DynamicRegularization should be a part of the Trainer module to be saved/loaded correctly
        self.regulariser = DynamicRegularization(self.model, initial_factor=REGULARIZATION_FACTOR)
        self.state = TrainerState()
        self.hist: List[dict] = []  # for plotting
        self.last_lr = lr

    def _run_epoch(self, train: bool = True) -> Tuple[float, float]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = self.train_loader if train else self.val_loader
        self.model.train(mode=train)
        tot_loss = correct = total = 0
        pbar = loader if not train else tqdm(loader, desc="Train", leave=False)
        for i, (X, y) in enumerate(pbar):
            y = y.long()  # Etiket tipi zorunlu long
            if i == 0:
                print(f"[BatchLog] {'Train' if train else 'Val'} batch0 labels: {y.tolist()} | tensor min: {X.min().item():.3f}, max: {X.max().item():.3f} | label min: {y.min().item()}, max: {y.max().item()}, NaN: {torch.isnan(X).any().item() or torch.isnan(y).any().item()}")
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                out = self.model(X)
                loss = self.criterion(out, y)
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
            tot_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            del X, y, out, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return tot_loss / total, correct / total # Corrected indentation and placement

    def _maybe_rollback(self, val_acc: float, val_loss: float):
        cfg_acc_drop = 0.015
        cfg_loss_rise = 0.04
        cooldown_epochs = 3

        # Need at least 1 previous epoch in history to compare against.
        # _log_epoch adds the *current* epoch's state after this is called.
        # So if len(self.hist) == 0, there's no previous epoch to compare to.
        if len(self.hist) < 1:
            return

        # Get the state and metrics of the previous epoch (the last one logged).
        prev_state = self.hist[-1]

        # Compare current epoch's performance (val_acc, val_loss) with the previous epoch's metrics (prev_state)
        acc_drop = (prev_state["val_acc"] - val_acc) / (prev_state["val_acc"] + 1e-8)
        loss_rise = (val_loss - prev_state["val_loss"]) / (prev_state["val_loss"] + 1e-8)

        # Check if rollback conditions are met and if the previous state has weights (should always if logging successful)
        if (
            acc_drop >= cfg_acc_drop or loss_rise >= cfg_loss_rise
        ) and self.state.cooldown == 0 and self.state.total_rbs < MAX_TOTAL_ROLLBACKS and "weights" in prev_state: # Check prev_state for 'weights'
            print(f"\n{YELLOW}[Rollback] Triggered! Δacc={acc_drop:.3f}, Δloss={loss_rise:.3f} → restoring previous weights (epoch {prev_state['epoch']}){RESET}") # Use epoch from prev_state
            try:
                # Load the state dicts from the previous epoch
                if "weights" in prev_state and "optim_state" in prev_state:
                    self.model.load_state_dict(prev_state["weights"])
                    self.optim.load_state_dict(prev_state["optim_state"])
                    self.state.total_rbs += 1
                    # Update last_rb_epoch to the epoch we rolled back to
                    self.state.last_rb_epoch = prev_state['epoch']
                    self.state.cooldown = cooldown_epochs
                    print(f"{YELLOW}[Rollback] Total rollbacks: {self.state.total_rbs}, Cooldown: {self.state.cooldown} epochs\n{RESET}")
                else:
                    print(f"{RED}[Rollback Error] Previous epoch state in history does not contain 'weights' or 'optim_state' keys. Cannot rollback for epoch {prev_state['epoch']}.{RESET}")
            except Exception as e:
                 print(f"{RED}[Rollback Error] Could not load state dicts for epoch {prev_state['epoch']}: {e}{RESET}")
                 # If rollback fails, perhaps we should stop or handle differently.
                 # For now, just log the error and continue.


    def _maybe_open_block(self, val_acc: float, val_loss: float):
        opened = self.model.opened_blocks
        # Can only open blocks up to the total number of blocks - 1 (since opened_blocks is 0-indexed for next block)
        if opened >= len(self.model.blocks):
            return
        # Stagnation check: if validation loss has not improved significantly in the last few epochs
        # Use a window size that makes sense, e.g., half the rollback window or at least 2 epochs.
        stagnation_window = max(2, min(len(self.hist), ROLLBACK_WINDOW // 2))
        # We need enough history entries to check for stagnation over the window
        if len(self.hist) < stagnation_window:
            return

        # Check if val loss in the last 'stagnation_window' epochs is mostly >= current val_loss
        # Access history from the end: hist[-1] is most recent, hist[-stagnation_window] is the oldest in the window
        stagnant = all(h["val_loss"] >= val_loss for h in self.hist[-stagnation_window:])

        # Check if block opening conditions are met
        # Open the next block if training is stagnant or validation accuracy is high and we haven't opened all blocks
        # opened_blocks is the count of already opened blocks, which is the index of the NEXT block to open
        if (stagnant or val_acc > 0.85) and opened < len(self.model.blocks):
             # Open the block at index 'opened'
             self.model.freeze_blocks_until(opened) # freeze_blocks_until expects the index of the last block to keep open (inclusive), so passing 'opened' opens block 'opened'
             # The freeze_blocks_until method should handle the internal opened_blocks count update
             # Print the index of the block that was just opened (which is 'opened' + 1 if 1-indexed, or just 'self.model.opened_blocks' after the call)
             print(f"{CYAN}[Block] Block-{self.model.opened_blocks} opened at epoch {self.state.epoch} (val_acc={val_acc:.3f}, val_loss={val_loss:.3f}){RESET}")


    def _log_epoch(self, tr_acc, tr_loss, val_acc, val_loss):
        # Create current epoch's history entry
        cur = {
            "epoch": self.state.epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "lr": self.optim.param_groups[0]["lr"],
        }

        # Save model and optimizer state for rollback *at the end* of the current epoch.
        # These states correspond to the model's parameters after completing the current epoch's training/validation.
        # They will be used to revert *to* this state if a future epoch performs poorly.
        try:
            cur["weights"] = copy.deepcopy(self.model.state_dict())
            cur["optim_state"] = copy.deepcopy(self.optim.state_dict())
        except Exception as e:
            print(f"{RED}[Log Epoch Error] Could not deepcopy state dicts for epoch {self.state.epoch}: {e}{RESET}")
            # If deepcopy fails, we cannot save the state for this epoch for rollback.
            # This epoch cannot be a target for rollback.

        self.hist.append(cur)

        # Ensure history does not grow indefinitely and keeps recent states for rollback
        # Keep history size limited. We need enough history for rollback comparisons (ROLLBACK_WINDOW) + 1 for the current epoch state.
        # Let's keep a small buffer, e.g., ROLLBACK_WINDOW + 2.
        while len(self.hist) > ROLLBACK_WINDOW + 2:
             self.hist.pop(0)


    def train(self):
        print("\n=== Training Started ===")
        print(f"Device: {self.device}, Max Epochs: {self.max_epochs}, LR: {self.optim.param_groups[0]['lr']:.2e}")
        print(f"Batch Size: {self.train_loader.batch_size}, Scheduler: CosineAnnealingWarmRestarts\n")

        for epoch in range(1, self.max_epochs + 1):
            self.state.epoch = epoch
            t0 = time.time()

            # 1. Run training and validation epochs
            tr_loss, tr_acc = self._run_epoch(train=True)
            val_loss, val_acc = self._run_epoch(train=False)

            # 2. Step the scheduler
            self.scheduler.step()

            # 3. Log LR changes
            current_lr = self.optim.param_groups[0]['lr']
            if abs(current_lr - self.last_lr) > 1e-8:
                print(f"{YELLOW}[LR Scheduler] LR changed: {self.last_lr:.2e} → {current_lr:.2e} (epoch {epoch}){RESET}")
                self.last_lr = current_lr

            # 4. Decrement rollback cooldown
            if self.state.cooldown > 0:
                self.state.cooldown -= 1

            # 5. Decide on rollback *before* logging the current epoch's state
            # This way, _maybe_rollback compares current metrics (epoch N) to the previously logged metrics (epoch N-1)
            # and if needed, rolls back to the state of epoch N-1.
            self._maybe_rollback(val_acc, val_loss)

            # 6. Decide on block opening *after* potential rollback
            # _maybe_open_block uses the metrics of the current epoch (which might be after rollback)
            self._maybe_open_block(val_acc, val_loss)

            # 7. Adjust regularization based on overfitting/underfitting *after* potential rollback/block opening
            gap = tr_acc - val_acc
            if gap > self.overfit_thr:
                self.regulariser.increase_regularization()
                print(f"{YELLOW}[Regulariser] Overfit detected (gap={gap:.3f}), regularisation increased to {self.regulariser.factor:.2f}{RESET}")
            elif val_acc < self.underfit_thr:
                self.regulariser.decrease_regularization()
                print(f"{YELLOW}[Regulariser] Underfit detected (val_acc={val_acc:.3f}), regularisation decreased to {self.regulariser.factor:.2f}{RESET}")

            # 8. Log the final state and metrics of the *current* epoch (after all adjustments)
            self._log_epoch(tr_acc, tr_loss, val_acc, val_loss)

            # 9. Save checkpoint based on the final metrics of the current epoch
            if val_acc > self.state.best_acc:
                self.state.best_acc = val_acc
                self.state.best_loss = val_loss
                # Save the state dict that was just logged (represents the best state found so far)
                # Access the last saved state from history, which includes weights/optim_state
                best_state_for_save = self.hist[-1] if len(self.hist) > 0 and "weights" in self.hist[-1] else None

                if best_state_for_save:
                    try:
                        # Save the weights from the history entry
                        torch.save(best_state_for_save["weights"], self.ckpt_path)
                        print(f"{GREEN}[Checkpoint] New best model saved at epoch {epoch} (val_acc={val_acc:.3f}){RESET}")
                    except Exception as e:
                        print(f"{RED}[Checkpoint Error] Could not save checkpoint for epoch {epoch}: {e}{RESET}")
                else:
                    print(f"{RED}[Checkpoint Warning] Best accuracy achieved at epoch {epoch} ({val_acc:.3f}), but state dict not found in history to save.{RESET}")

            # 10. Print epoch summary log
            dt = time.time() - t0

            # Simplied block info for log format
            block_info = f"{self.model.opened_blocks}/{len(self.model.blocks)}"

            # Use ANSI colors for improved readability - Updated format
            print(
                f"{GREEN}Epoch {epoch:02d}{RESET} | "
                f"TrainAcc {BLUE}{tr_acc:.3f}{RESET} | "
                f"ValAcc {BLUE}{val_acc:.3f}{RESET} | "
                f"TrainLoss {RED}{tr_loss:.3f}{RESET} | "
                f"ValLoss {RED}{val_loss:.3f}{RESET} | "
                f"LR {YELLOW}{self.optim.param_groups[0]['lr']:.2e}{RESET} | "
                f"Blocks opened: {YELLOW}{block_info}{RESET} | "
                f"⏱️ {CYAN}{dt:.1f}s{RESET}"
            )

            # TrainerState logu - Updated format to match example
            print(f"{CYAN}[TrainerState]{RESET} epoch={self.state.epoch}, best_acc={self.state.best_acc:.3f}, total_rollbacks={self.state.total_rbs}, cooldown={self.state.cooldown}") # Removed best_loss as it wasn't in your example

            # 11. Plot history if needed
            if epoch % self.plot_every == 0 or epoch == self.max_epochs:
                self._plot_history()
                print(f"{MAGENTA}[Plotting] Saved training curves at epoch {epoch}.{RESET}")

            # 12. Early stopping check
            if val_acc >= 0.958:
                print(f"{GREEN}Target accuracy reached – stopping early.{RESET}")
                break # Corrected placement inside the loop

            # 13. Clear cache after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nTraining finished. Best val-acc = {self.state.best_acc:.3f}")

    def _plot_history(self):
        xs = [h["epoch"] for h in self.hist]
        tr_acc = [h["train_acc"] for h in self.hist]
        val_acc = [h["val_acc"] for h in self.hist]
        tr_loss = [h["train_loss"] for h in self.hist]
        val_loss = [h["val_loss"] for h in self.hist]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(xs, tr_acc, label="train")
        plt.plot(xs, val_acc, label="val")
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.grid(); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(xs, tr_loss, label="train")
        plt.plot(xs, val_loss, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(); plt.legend()
        plt.tight_layout()
        os.makedirs("logs", exist_ok=True)
        plt.savefig(f"logs/curve_epoch_{self.state.epoch}.png")
        plt.close()

def build_dataloaders(batch_size=16, num_workers=2, to_rgb=False, pin_memory=True, persistent_workers=True):
    import platform
    if platform.system() == "Windows":
        num_workers = 0
        persistent_workers = False
    aug = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
    ])
    train_ds = CustomTumorDataset("preprocessed_data/train", to_rgb=to_rgb, transform=aug)
    val_ds = CustomTumorDataset("preprocessed_data/val", to_rgb=to_rgb)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    # Class weights hesapla
    labels = [label for _, label in train_ds.samples]
    label_counts = Counter(labels)
    num_classes = len(label_counts)
    class_sample_count = torch.tensor([label_counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    class_weights = 1. / (class_sample_count + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return train_loader, val_loader, class_weights

def freeze_support_for_win():
    mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    freeze_support_for_win()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_loader, val_loader, class_weights = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
        to_rgb=True
    )
    model = VGGCustom(num_classes=3, in_channels=3)
    trainer = Trainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-4,
        max_epochs=60,
        plot_every=5,
        class_weights=class_weights
    )
    trainer.train()
