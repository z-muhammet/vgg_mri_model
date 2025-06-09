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
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, SequentialLR
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import platform
import optuna # Added for HPO

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Constants for Mixup Alpha Decay
MIXUP_DECAY_START_EPOCH = 10
MIXUP_DECAY_END_EPOCH = 30

# Hyperparameters
INITIAL_LR = 1e-1
BATCH_SIZE = 8
NUM_EPOCHS = 60
WARMUP_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_ACCUMULATION_STEPS = 2

# Overfitting/Underfitting prevention parameters
OVERFIT_THRESHOLD = 0.22
MAX_OVERFIT_EPOCHS = 5
MAX_UNDERFIT_EPOCHS = 10
REGULARIZATION_FACTOR = 0.01
MAX_REGULARIZATION = 0.3
ROLLBACK_GAP_THRESH = 0.3
MAX_WEIGHT_DECAY  = 1e-2
PENALTY_FACTOR    = 1.5
ROLLBACK_ACC_THRESHOLD = 0.02
ROLLBACK_LOSS_THRESHOLD = 0.04
ROLLBACK_COOLDOWN = 3
MAX_TOTAL_ROLLBACKS = 10
ROLLBACK_WINDOW = 20
PENALTY_GAP_LOW   = 0.1
PENALTY_GAP_HIGH  = 0.29

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

@dataclass
class TrainerState:
    best_acc: float = 0.0
    best_loss: float = math.inf
    best_weights: Optional[dict] = None
    epoch: int = 0
    last_rb_epoch: int = 0
    total_rbs: int = 0
    cooldown: int = 0
    no_imp_epochs: int = 0

class Trainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = INITIAL_LR,
        max_epochs: int = 60,
        overfit_thr: float = 0.35,
        underfit_thr: float = 0.55,
        plot_every: int = 5,
        checkpoint_dir: str = "models",
        class_weights: torch.Tensor = None,
        target_lr: float = 5e-3,
        scheduler_type: str = "reduce_on_plateau"
    ) -> None:
        super().__init__()
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
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=target_lr, weight_decay=1e-5)

        # Define schedulers for SequentialLR
        warmup_steps = WARMUP_EPOCHS * len(train_loader) # Use train_loader directly
        warmup_scheduler = LinearLR(
            self.optim,
            start_factor=1e-8, # Must be > 0
            end_factor=1.0,
            total_iters=warmup_steps
        )

        total_training_steps = max_epochs * len(train_loader)
        remaining_steps = total_training_steps - warmup_steps

        onecycle_scheduler = OneCycleLR(
            self.optim,
            max_lr=target_lr, # Use target_lr as max_lr
            total_steps=remaining_steps,
            pct_start=0.2,
            div_factor=10,
            final_div_factor=10,
            anneal_strategy='cos'
        )

        self.scheduler = SequentialLR(
            self.optim,
            schedulers=[warmup_scheduler, onecycle_scheduler],
            milestones=[warmup_steps]
        )

        self.class_weights_cpu = class_weights.clone().detach().cpu()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_cpu.to(self.device))
        self.amp_enabled = self.device.type == "cuda" and (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        self.scaler = GradScaler(self.device.type, enabled=self.amp_enabled)
        self.regulariser = DynamicRegularization(self.model, initial_factor=REGULARIZATION_FACTOR)
        self.state = TrainerState()
        self.hist: List[dict] = []
        self.rollback_epochs: List[int] = []
        self.target_lr = target_lr
        self.mixup_alpha = 0.2
        self._mixup_initial_alpha = self.mixup_alpha
        self._last_logged_mixup_alpha = -1.0
        self.scheduler_type = scheduler_type

    def _run_epoch(self, train: bool = True) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """
        Runs a single training or validation epoch.

        Returns:
        avg_loss (float): Average loss for the epoch.
        avg_acc (float): Average accuracy for the epoch.
        per_class_correct (torch.Tensor): Number of correct predictions per class.
        per_class_total (torch.Tensor): Total samples per class.
        """
        num_classes = len(self.class_weights_cpu)
        per_class_correct = torch.zeros(num_classes, device=self.device)
        per_class_total   = torch.zeros(num_classes, device=self.device)

        loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()

        tot_loss = 0.0
        correct  = 0
        total    = 0

        pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for i, (X, y) in enumerate(pbar):
                y = y.long().to(self.device, non_blocking=True)
                X = X.to(self.device, non_blocking=True)

                if train:
                    X, y_a, y_b, lam = mixup_data(X, y, alpha=self.mixup_alpha)
                    with autocast(self.device.type, enabled=self.amp_enabled):
                        out  = self.model(X)
                        loss = lam * self.criterion(out, y_a) + (1 - lam) * self.criterion(out, y_b)
                    
                    # Gradient accumulation
                    self.scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
                    
                    # Perform optimizer step
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad(set_to_none=True)
                        self.scheduler.step() # Call scheduler step here, after optimizer update
                else:
                    with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                        out  = self.model(X)
                        loss = self.criterion(out, y)

                # Batch metrics
                batch_size = y.size(0)
                tot_loss  += loss.item() * batch_size
                preds      = out.argmax(dim=1)
                correct   += (preds == y).sum().item()
                total     += batch_size

                # Per-class counters
                for cls in range(num_classes):
                    mask = (y == cls)
                    per_class_correct[cls] += (preds[mask] == cls).sum().item()
                    per_class_total[cls]   += mask.sum().item()

                # Cleanup
                del X, y, out, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = tot_loss / total
        avg_acc  = correct / total

        return avg_loss, avg_acc, per_class_correct, per_class_total

    def _maybe_rollback(self, val_acc: float, val_loss: float):
        cfg_acc_drop = 0.015
        cfg_loss_rise = 0.04
        cooldown_epochs = 3

        if len(self.hist) < 1:
            return

        prev_state = self.hist[-1]

        # Compare current epoch's performance with previous epoch's metrics
        acc_drop = (prev_state["val_acc"] - val_acc) / (prev_state["val_acc"] + 1e-8)
        loss_rise = (val_loss - prev_state["val_loss"]) / (prev_state["val_loss"] + 1e-8)

        # Check if rollback conditions are met
        if (
            acc_drop >= cfg_acc_drop or loss_rise >= cfg_loss_rise
        ) and self.state.cooldown == 0 and self.state.total_rbs < MAX_TOTAL_ROLLBACKS and "weights" in prev_state:
            print(f"\n{YELLOW}[Rollback] Triggered! Δacc={acc_drop:.3f}, Δloss={loss_rise:.3f} → restoring previous weights (epoch {prev_state['epoch']}){RESET}")
            try:
                # Load the state dicts from the previous epoch
                if "weights" in prev_state and "optim_state" in prev_state:
                    self.model.load_state_dict(prev_state["weights"])
                    self.optim.load_state_dict(prev_state["optim_state"])
                    self.state.total_rbs += 1
                    self.state.last_rb_epoch = prev_state['epoch']
                    self.state.cooldown = cooldown_epochs
                    print(f"{YELLOW}[Rollback] Total rollbacks: {self.state.total_rbs}, Cooldown: {self.state.cooldown} epochs\n{RESET}")
                else:
                    print(f"{RED}[Rollback Error] Previous epoch state in history does not contain 'weights' or 'optim_state' keys. Cannot rollback for epoch {prev_state['epoch']}.{RESET}")
            except Exception as e:
                 print(f"{RED}[Rollback Error] Could not load state dicts for epoch {prev_state['epoch']}: {e}{RESET}")
            self.rollback_epochs.append(self.state.epoch)

    def _maybe_open_block(self, val_acc: float, val_loss: float):
        opened = self.model.opened_blocks
        if opened >= len(self.model.blocks):
            return

        # Prevent opening Block 2 if rollbacks are insufficient or no improvements
        if opened == 1 and self.state.no_imp_epochs <= 5:
            print(f"{YELLOW}[Block] Skipping opening Block-2 at epoch {self.state.epoch} - insufficient rollbacks ({self.state.total_rbs}) or improvements ({self.state.no_imp_epochs}){RESET}")
            return

        # Prevent opening Block 3 if total rollbacks haven't reached the maximum
        if opened == 2 and self.state.total_rbs < MAX_TOTAL_ROLLBACKS:
             return

        # Check for stagnation: if validation loss has not improved significantly in recent epochs
        stagnation_window = max(2, min(len(self.hist), ROLLBACK_WINDOW // 2))
        if len(self.hist) < stagnation_window:
            return

        stagnant = all(h["val_loss"] >= val_loss for h in self.hist[-stagnation_window:])

        # Open the next block if training is stagnant or validation accuracy is high
        if (stagnant or val_acc > 0.75) and opened < len(self.model.blocks):
             self.model.freeze_blocks_until(opened)
             print(f"{CYAN}[Block] Block-{self.model.opened_blocks} opened at epoch {self.state.epoch} (val_acc={val_acc:.3f}, val_loss={val_loss:.3f}){RESET}")

    def _log_epoch(self, tr_acc, tr_loss, val_acc, val_loss):
        cur = {
            "epoch": self.state.epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "lr": self.optim.param_groups[0]["lr"],
        }

        # Save model and optimizer state for rollback
        try:
            # Move model parameters to CPU and copy
            with torch.no_grad():
                cpu_weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            cur["weights"] = cpu_weights
            # Move optimizer state tensors to CPU
            optim_cpu = {}
            for k, v in self.optim.state_dict().items():
                if isinstance(v, dict):
                    optim_cpu[k] = {
                        sk: sv.detach().cpu() if torch.is_tensor(sv) else sv
                        for sk, sv in v.items()
                    }
                else:
                    optim_cpu[k] = v
            cur["optim_state"] = optim_cpu
        except Exception as e:
            print(f"{RED}[Log Epoch Error] Could not deepcopy state dicts for epoch {self.state.epoch}: {e}{RESET}")

        self.hist.append(cur)

        # Ensure history does not grow indefinitely
        while len(self.hist) > ROLLBACK_WINDOW + 2:
            self.hist.pop(0)
            gc.collect()

    def _normalize_class_weights(self):
        # Ağırlıklar toplamını sınıf sayısına eşitle
        cw = self.class_weights_cpu
        cw = cw / cw.sum() * len(cw)
        self.class_weights_cpu = cw
        # Loss fonksiyonuna da kopyalayın
        with torch.no_grad():
            self.criterion.weight.data.copy_(cw.to(self.device))
        print(f"{GREEN}[Normalize] Class weights normalized: {cw.cpu().numpy()}{RESET}")

    def train(self, trial: Optional[optuna.Trial] = None):
        print("\n=== Training Started ===")
        print(f"Device: {self.device}, Max Epochs: {self.max_epochs}, LR: {self.optim.param_groups[0]['lr']:.2e}")
        print(f"Batch Size: {self.train_loader.batch_size}, Scheduler: SequentialLR (Warmup + OneCycleLR)\n")

        def adjust_learning_rate(optimizer, new_lr):
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        num_classes = len(self.class_weights_cpu)

        for epoch in range(1, self.max_epochs + 1):
            self.state.epoch = epoch
            t0 = time.time()

            # Run training and validation epochs
            tr_loss, tr_acc, tr_corr_tr, tr_tot_tr = self._run_epoch(train=True)
            val_loss, val_acc, val_corr_val, val_tot_val = self._run_epoch(train=False)

            # Optuna Pruning
            if trial is not None:
                trial.report(val_acc, epoch)
                # Handle the case where the checkpoint path for the best trial needs to be stored
                if val_acc > self.state.best_acc: # If this is a new best accuracy for this trial
                    trial.set_user_attr("ckpt_path", self.ckpt_path)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Mixup Alpha adjustment (Linear decay from initial_alpha to 0 by epoch 45)
            mixup_end_epoch = 30

            if self.state.epoch <= mixup_end_epoch:
                # Calculate decay progress. At epoch 1, progress is 0. At epoch 45, progress is 1.
                decay_progress = (self.state.epoch - 1) / (mixup_end_epoch - 1 + 1e-8) # Add small epsilon to avoid division by zero if end_epoch is 1
                new_mixup_alpha = self._mixup_initial_alpha * (1 - decay_progress)
                self.mixup_alpha = max(0.0, new_mixup_alpha) # Ensure it doesn't go below 0
                if self.mixup_alpha != self._last_logged_mixup_alpha:
                    print(f"{YELLOW}[Mixup] Mixup alpha linearly decayed to {self.mixup_alpha:.2f} at epoch {self.state.epoch}{RESET}")
                    self._last_logged_mixup_alpha = self.mixup_alpha
            elif self.state.epoch > mixup_end_epoch:
                if self.mixup_alpha > 0.0:
                    self.mixup_alpha = 0.0
                    print(f"{YELLOW}[Mixup] Mixup alpha set to {self.mixup_alpha:.2f} at epoch {self.state.epoch}{RESET}")
                    self._last_logged_mixup_alpha = self.mixup_alpha

            # Decrease cooldown counter
            if self.state.cooldown > 0:
                self.state.cooldown -= 1

            # Calculate gap
            gap     = tr_acc - val_acc
            gap_abs = abs(gap)

            if gap_abs >= ROLLBACK_GAP_THRESH:
                self._maybe_rollback(val_acc, val_loss)

            else:
                # Class-based weight update (ClassPenalty) for moderate gap
                if PENALTY_GAP_LOW <= gap_abs < PENALTY_GAP_HIGH:
                    train_acc_cls = tr_corr_tr / (tr_tot_tr + 1e-8)
                    val_acc_cls   = val_corr_val / (val_tot_val + 1e-8)
                    per_class_gap = train_acc_cls - val_acc_cls
                    for cls, g in enumerate(per_class_gap.cpu()):
                        if g > PENALTY_GAP_LOW:
                            self.class_weights_cpu[cls] *= 1.03  # increase weight for overfitting class
                        elif g < -PENALTY_GAP_LOW:
                            self.class_weights_cpu[cls] *= 0.97  # decrease weight for underperforming/unusual class
                    self.class_weights_cpu = self.class_weights_cpu / self.class_weights_cpu.sum() * len(self.class_weights_cpu)
                    with torch.no_grad():
                        self.criterion.weight.data.copy_(self.class_weights_cpu.to(self.device))
                    print(f"{YELLOW}[ClassPenalty] per_class_gap: {per_class_gap.cpu().numpy()}")
                    print(f"{YELLOW}[ClassPenalty] new weights: {self.class_weights_cpu.cpu().numpy()}{RESET}")

            # Normalize class weights every 10 epochs
            if self.state.epoch % 10 == 0:
                self._normalize_class_weights()

            # Attempt to open block
            self._maybe_open_block(val_acc, val_loss)

            # Regularization for over/underfitting
            if gap > self.overfit_thr:
                self.regulariser.increase_regularization()
                print(f"{YELLOW}[Regulariser] Overfit detected (gap={gap:.3f}), regularisation increased to {self.regulariser.factor:.2f}{RESET}")
            elif val_acc < self.underfit_thr:
                self.regulariser.decrease_regularization()
                print(f"{YELLOW}[Regulariser] Underfit detected (val_acc={val_acc:.3f}), regularisation decreased to {self.regulariser.factor:.2f}{RESET}")

            # Epoch log and checkpoint
            self._log_epoch(tr_acc, tr_loss, val_acc, val_loss)
            if val_acc > self.state.best_acc:
                self.state.best_acc = val_acc
                self.state.best_loss = val_loss
                self.state.best_weights = self.hist[-1]["weights"]
                self.state.no_imp_epochs = 0
                best_state = self.hist[-1]
                torch.save(best_state["weights"], self.ckpt_path)
                print(f"{GREEN}[Checkpoint] New best model saved at epoch {epoch} (val_acc={val_acc:.3f}){RESET}")
            else:
                self.state.no_imp_epochs += 1

            # Summarize epoch results
            dt = time.time() - t0
            block_info = f"{self.model.opened_blocks}/{len(self.model.blocks)}"
            print(
                f"{GREEN}Epoch {epoch:02d}{RESET} | "
                f"TrainAcc {BLUE}{tr_acc:.3f}{RESET} | "
                f"ValAcc {BLUE}{val_acc:.3f}{RESET} | "
                f"TrainLoss {RED}{tr_loss:.3f}{RESET} | "
                f"ValLoss {RED}{val_loss:.3f}{RESET} | "
                f"LR {YELLOW}{self.optim.param_groups[0]['lr']:.2e}{RESET} | "
                f"Blocks {YELLOW}{block_info}{RESET} | "
                f"NoImp {MAGENTA}{self.state.no_imp_epochs}{RESET} | "
                f"⏱️ {CYAN}{dt:.1f}s{RESET}"
            )
            print(f"{CYAN}[TrainerState]{RESET} epoch={epoch}, best_acc={self.state.best_acc:.3f}, total_rbs={self.state.total_rbs}, cooldown={self.state.cooldown}, no_imp_epochs={self.state.no_imp_epochs}")

            # Clear cache and report memory
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nTraining finished. Best val-acc = {self.state.best_acc:.3f}")
        self._plot_history()

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
        # Add vertical lines for rollback epochs on the accuracy plot
        for rb_epoch in self.rollback_epochs:
            plt.axvline(x=rb_epoch, color='gray', linestyle='--', linewidth=1, label='Rollback')
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.grid(); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(xs, tr_loss, label="train")
        plt.plot(xs, val_loss, label="val")
        # Add vertical lines for rollback epochs on the loss plot
        for rb_epoch in self.rollback_epochs:
            plt.axvline(x=rb_epoch, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(); plt.legend()
        plt.tight_layout()
        os.makedirs("logs", exist_ok=True)
        plt.savefig(f"logs/curve_epoch_{self.state.epoch}.png")
        plt.close()

def get_class_balanced_weights(labels, beta=0.99):
    counts = torch.bincount(labels)
    effective_num = 1.0 - torch.pow(beta, counts.float())
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(counts)
    return weights

def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def build_dataloaders(batch_size=8, num_workers=2, to_rgb=False, pin_memory=True, persistent_workers=True):
    import platform
    if platform.system() == "Windows":
        num_workers = 0
        persistent_workers = False
    aug = None
    train_ds = CustomTumorDataset("preprocessed_data/train", to_rgb=to_rgb, transform=aug)
    val_ds = CustomTumorDataset("preprocessed_data/val", to_rgb=to_rgb)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    cb_weights = torch.tensor([1.1035788, 0.93700093, 0.9594203], dtype=torch.float32)
    cb_weights = cb_weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return train_loader, val_loader, cb_weights

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
        lr=1e-1,
        max_epochs=60,
        plot_every=10,
        class_weights=class_weights
    )
    trainer.train()
