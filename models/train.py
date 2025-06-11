import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter
import os, time, torch, matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.custom_dataset import CustomTumorDataset
from models.vgg_custom import VGGCustom
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
import numpy as np
import gc
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, SequentialLR, CosineAnnealingLR, ReduceLROnPlateau
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import WeightedRandomSampler

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Constants for Mixup Alpha Decay
# MIXUP_DECAY_START_EPOCH = 20
# MIXUP_DECAY_END_EPOCH = 30

# New global constant for Cosine Annealing start
COSINE_START_EPOCH = 20

# New global constant for SWA start

# Hyperparameters
INITIAL_LR = 1e-2
BATCH_SIZE = 12
NUM_EPOCHS = 60
SWA_START_EPOCH = NUM_EPOCHS + 1
WARMUP_EPOCHS = 0
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_ACCUMULATION_STEPS = 4

# SWA specific LR
SWA_LR = 5e-4

# Overfitting/Underfitting prevention parameters
OVERFIT_THRESHOLD = 0.22
MAX_OVERFIT_EPOCHS = 5
MAX_UNDERFIT_EPOCHS = 10
REGULARIZATION_FACTOR = 0.001
MAX_REGULARIZATION = 0.01
ROLLBACK_GAP_THRESH = 0.3
MAX_WEIGHT_DECAY  = 1e-2
PENALTY_FACTOR    = 1.01
ROLLBACK_ACC_THRESHOLD = 0.03
ROLLBACK_LOSS_THRESHOLD = 0.08
ROLLBACK_COOLDOWN = 5
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

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.to(logits.device)

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
        max_epochs: int = 60,
        overfit_thr: float = 0.35,
        underfit_thr: float = 0.55,
        plot_every: int = 5,
        checkpoint_dir: str = "models",
        class_weights: torch.Tensor = None,
        target_lr: float = 1e-2,
        scheduler_type: str = "reduce_on_plateau",
        pct_start: float = 0.2,
        div_factor: float = 10,
        final_div_factor: float = 10,
        mixup_alpha: float = 0.2,
        weight_decay: float = 5e-4
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

        # Initialize SWA model
        self.swa_model = AveragedModel(model)

        # Initialize Temperature Scaler
        self.temperature_scaler = TemperatureScaler()

        # Combine model and temperature scaler parameters for optimizer
        all_params = list(self.model.parameters())
        self.optim = torch.optim.AdamW(all_params, lr=target_lr, weight_decay=weight_decay)

        # Log the initial learning rate from the optimizer
        print(f"{CYAN}[Trainer Init] Initial optimizer LR: {self.optim.param_groups[0]['lr']:.2e}{RESET}")

        total_optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)

        if scheduler_type == "one_cycle":
            # Phase 1: OneCycleLR (Epoch 0 to COSINE_START_EPOCH-1)
            onecycle_total_iters = COSINE_START_EPOCH * total_optimizer_steps_per_epoch
            onecycle_scheduler = OneCycleLR(
                self.optim,
                max_lr=target_lr, # Use target_lr as max_lr
                total_steps=onecycle_total_iters,
                pct_start=pct_start, # Standard for OneCycleLR warm-up
                div_factor=div_factor, # from HPO
                final_div_factor=final_div_factor, # from HPO
                anneal_strategy='cos'
            )

            # Phase 2: CosineAnnealingLR (Epoch COSINE_START_EPOCH to SWA_START_EPOCH-1)
            cosine_epochs_duration = (SWA_START_EPOCH - 1) - COSINE_START_EPOCH + 1
            if cosine_epochs_duration < 0: # Ensure non-negative duration
                cosine_epochs_duration = 0
            cosine_total_iters = cosine_epochs_duration * total_optimizer_steps_per_epoch
            cosine_scheduler = CosineAnnealingLR(
                self.optim,
                T_max=cosine_total_iters,
                eta_min=1e-5
            )

            # Phase 3: ConstantLR for SWA (Epoch SWA_START_EPOCH to NUM_EPOCHS)
            swa_epochs_duration = self.max_epochs - SWA_START_EPOCH + 1
            if swa_epochs_duration < 0:
                swa_epochs_duration = 0
            swa_total_iters = swa_epochs_duration * total_optimizer_steps_per_epoch
            swa_scheduler = LinearLR(
                self.optim,
                start_factor=1.0, # Keep LR constant at the start of this phase
                end_factor=1.0, # Keep LR constant
                total_iters=max(1, swa_total_iters) # Ensure at least 1 iter if duration > 0
            )

            schedulers_list = []
            milestones_list = []

            # 1) OneCycle always
            schedulers_list.append(onecycle_scheduler)

            # 2) Add Cosine phase and its transition milestone
            if cosine_epochs_duration > 0:
                schedulers_list.append(cosine_scheduler)
                milestones_list.append(COSINE_START_EPOCH * total_optimizer_steps_per_epoch)
            
            # 3) Add SWA phase and its transition milestone only if it's actually added
            if swa_epochs_duration > 0:
                schedulers_list.append(swa_scheduler)
                milestones_list.append(SWA_START_EPOCH * total_optimizer_steps_per_epoch)

            self.scheduler = SequentialLR(
                self.optim,
                schedulers=schedulers_list,
                milestones=milestones_list
            )
            self.scheduler_is_per_batch = True
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optim,
                mode='min',
                factor=0.5,
                patience=EARLY_STOPPING_PATIENCE // 2, # Half of early stopping patience
                verbose=True,
                min_lr=1e-7
            )
            self.scheduler_is_per_batch = False
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.class_weights_cpu = class_weights.clone().detach().cpu()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_cpu.to(self.device))
        print(f"{CYAN}[Trainer Init] Class weights: {self.class_weights_cpu.numpy()}{RESET}")
        self.amp_enabled = self.device.type == "cuda" and (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        self.scaler = GradScaler(self.device.type, enabled=self.amp_enabled) # Updated GradScaler init
        self.regulariser = DynamicRegularization(self.model, initial_factor=REGULARIZATION_FACTOR)
        self.state = TrainerState()
        self.hist: List[dict] = []
        self.rollback_epochs: List[int] = []
        self.target_lr = target_lr
        self.mixup_alpha = mixup_alpha
        self._mixup_initial_alpha = self.mixup_alpha
        self._last_logged_mixup_alpha = -1.0
        self.scheduler_type = scheduler_type

    def _get_current_augmenter(self, epoch: int) -> Optional[A.Compose]:
        # Early training phase (0-10 epochs): Slightly higher augmentation
        if epoch < 10:
            return A.Compose([
                A.Rotate(limit=10, p=0.4), # Reduced limit and p
                A.RandomRotate90(p=0.2), # Reduced p
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.4), # Reduced p
                A.Affine(
                    translate_percent={'x': 0.03, 'y': 0.03}, # Reduced
                    scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)}, # Reduced
                    rotate=0,
                    shear={'x': 0.0, 'y': 0.0},
                    p=0.4 # Reduced p
                ),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4), # Reduced limits and p
            ])
        # Middle training phase (10-40 epochs): Moderate augmentation (softened)
        elif epoch < 40:
            return A.Compose([
                A.Rotate(limit=5, p=0.2), # Reduced limit and p
                A.RandomRotate90(p=0.1), # Reduced p
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.2), # Reduced p
                A.Affine(
                    translate_percent={'x': 0.01, 'y': 0.01}, # Reduced
                    scale={'x': (0.98, 1.02), 'y': (0.98, 1.02)}, # Reduced
                    rotate=0,
                    shear={'x': 0.0, 'y': 0.0},
                    p=0.2 # Reduced p
                ),
                A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.03, p=0.2), # Reduced limits and p
            ])
        # Final training phase (40+ epochs): Minimal augmentation
        else:
            return A.Compose([
                A.HorizontalFlip(p=1.0),
            ])

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
                    preds = out.argmax(dim=1) # Predictions from raw output during training

                    # Gradient accumulation
                    self.scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
                    
                    # Perform optimizer step
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad(set_to_none=True)
                        if self.scheduler_is_per_batch: # Only step per-batch schedulers here
                            self.scheduler.step()
                        if self.state.epoch >= SWA_START_EPOCH:
                            self.swa_model.update_parameters(self.model)
                else:
                    with autocast(self.device.type, enabled=self.amp_enabled):
                        out  = self.model(X)
                        # Apply temperature scaling for validation
                        out_calibrated = self.temperature_scaler(out)
                        loss = self.criterion(out_calibrated, y)
                    preds = out_calibrated.argmax(dim=1) # Predictions from calibrated output during validation

                # Batch metrics
                batch_size = y.size(0)
                tot_loss  += loss.item() * batch_size
                correct   += (preds == y).sum().item()
                total     += batch_size

                # Per-class counters (only for validation, as Mixup makes it misleading during training)
                if not train:
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
        cfg_acc_drop = 0.03
        cfg_loss_rise = 0.08
        cooldown_epochs = 5

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

        # New condition for opening Block 2: open if total rollbacks indicate some instability or if stagnation is detected.
        # This makes sure the model had a chance to stabilize or if it's not improving, we try opening a new block.
        if opened == 1: # Block 2 is not allowed to open for now.
            print(f"{YELLOW}[Block] Skipping opening Block-2 at epoch {self.state.epoch} - Block 2 is not allowed to open for now.{RESET}")
            return

        # Prevent opening Block 3 if total rollbacks haven't reached a certain threshold (implies instability)
        if opened == 2: # 3rd block is not allowed to open.
             print(f"{YELLOW}[Block] Skipping opening Block-3 at epoch {self.state.epoch} - 3rd block is not allowed to open.{RESET}")
             return

        # Check for stagnation: if validation loss has not improved significantly in recent epochs
        stagnation_window = max(2, min(len(self.hist), ROLLBACK_WINDOW // 2))
        if len(self.hist) < stagnation_window:
            return

        stagnant = all(h["val_loss"] >= val_loss for h in self.hist[-stagnation_window:])

        # Open the next block if training is stagnant or validation accuracy is high
        # Added a condition for total_rbs for opening block 2 to ensure we've had some instability before trying to open it.
        if (stagnant or val_acc > 0.55) and opened < len(self.model.blocks):
            # Specific condition for opening Block 2 to ensure it's not too early or unstable
            if opened == 1 and self.state.no_imp_epochs < 3: # If Block 2, and not enough stagnation, defer opening
                print(f"{YELLOW}[Block] Deferring Block-2 opening at epoch {self.state.epoch} - aiming for more stability first before opening Block 2.{RESET}")
                return

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
            "mixup_alpha": self.mixup_alpha,
            "class_weights": self.class_weights_cpu.cpu().numpy().copy()
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

    def train(self):
        print("\n=== Training Started ===")
        print(f"Device: {self.device}, Max Epochs: {self.max_epochs}, LR: {self.optim.param_groups[0]['lr']:.2e}")
        print(f"Batch Size: {self.train_loader.batch_size}, Scheduler: {self.scheduler_type}\n")

        def adjust_learning_rate(optimizer, new_lr):
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        num_classes = len(self.class_weights_cpu)

        print(f"{GREEN}Starting training for {self.max_epochs} epochs...{RESET}")
        for epoch in range(self.state.epoch, self.max_epochs):
            self.state.epoch = epoch

            # Dynamically set augmentation based on epoch
            current_augmenter = self._get_current_augmenter(epoch)
            self.train_loader.dataset.transform = current_augmenter

            # Epoch training
            t0 = time.time()
            tr_loss, tr_acc, tr_tot_tr, tr_corr_tr = self._run_epoch(train=True)

            # Apply SWA LR at start of SWA phase
            if self.state.epoch == SWA_START_EPOCH:
                for pg in self.optim.param_groups:
                    pg['lr'] = SWA_LR
                print(f"{CYAN}[SWA] SWA LR set to {SWA_LR:.2e} at epoch {self.state.epoch}{RESET}")

            # Run training and validation epochs
            val_loss, val_acc, val_corr_val, val_tot_val = self._run_epoch(train=False)

            # Step scheduler if it's not a per-batch scheduler
            if not self.scheduler_is_per_batch:
                self.scheduler.step(val_loss)

            # Mixup Alpha adjustment based on new schedule
            if self.state.epoch < 10: # Only apply mixup for epochs 0-9
                self.mixup_alpha = self._mixup_initial_alpha
                if self.mixup_alpha != self._last_logged_mixup_alpha:
                    print(f"{YELLOW}[Mixup] Mixup alpha set to {self.mixup_alpha:.2f} at epoch {self.state.epoch}{RESET}")
                    self._last_logged_mixup_alpha = self.mixup_alpha
            elif self.state.epoch >= 10: # From epoch 10 onwards, mixup alpha is 0
                if self.mixup_alpha > 0.0: # Only log when it changes to 0
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
                if PENALTY_GAP_LOW <= gap_abs < PENALTY_GAP_HIGH and self.state.epoch >= 30:
                    train_acc_cls = tr_corr_tr / (tr_tot_tr + 1e-8)
                    val_acc_cls   = val_corr_val / (val_tot_val + 1e-8)
                    per_class_gap = train_acc_cls - val_acc_cls
                    for cls, g in enumerate(per_class_gap.cpu()):
                        if g > PENALTY_GAP_LOW:
                            self.class_weights_cpu[cls] *= 1.03  # increase weight for overfitting class
                        elif g < -PENALTY_GAP_LOW:
                            self.class_weights_cpu[cls] *= 0.97  # decrease weight for underperforming/unusual class
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
        
        # Save the fully trained model
        full_ckpt_path = os.path.join(os.path.dirname(self.ckpt_path), "full_vgg_custom.pt")
        torch.save(self.model.state_dict(), full_ckpt_path)
        print(f"{GREEN}[Checkpoint] Full trained model saved to {full_ckpt_path}{RESET}")

        # Update BatchNorm for SWA model
        if self.max_epochs >= SWA_START_EPOCH:
            print(f"{CYAN}[SWA] Updating BatchNorm for SWA model...{RESET}")
            update_bn(self.train_loader, self.swa_model, device=self.device)
            # Save the SWA model
            swa_ckpt_path = os.path.join(os.path.dirname(self.ckpt_path), "best_vgg_custom_swa.pt")
            torch.save(self.swa_model.state_dict(), swa_ckpt_path)
            print(f"{GREEN}[SWA] SWA model saved to {swa_ckpt_path}{RESET}")

            # Calibrate TemperatureScaler on the SWA model
            print(f"{CYAN}[Calibration] Starting temperature calibration on SWA model...{RESET}")
            self.temperature_scaler.to(self.device) # Ensure scaler is on the correct device
            self.temperature_scaler.eval()
            optimizer = torch.optim.LBFGS([self.temperature_scaler.temperature], lr=0.01, max_iter=50)
            
            # Collect all logits and labels from the validation set using the SWA model
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for X, y in tqdm(self.val_loader, desc="Collecting SWA Logits", leave=False):
                    X = X.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    logits = self.swa_model(X) # Use SWA model for calibration
                    all_logits.append(logits)
                    all_labels.append(y)
            
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)

            def closure():
                optimizer.zero_grad()
                loss = self.criterion(self.temperature_scaler(all_logits), all_labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            print(f"{GREEN}[Calibration] Temperature calibrated on SWA model to: {self.temperature_scaler.temperature.item():.4f}{RESET}")
        
        self._plot_history()

    def _plot_history(self):
        xs = [h["epoch"] for h in self.hist]
        tr_acc = [h["train_acc"] for h in self.hist]
        val_acc = [h["val_acc"] for h in self.hist]
        tr_loss = [h["train_loss"] for h in self.hist]
        val_loss = [h["val_loss"] for h in self.hist]
        lrs = [h["lr"] for h in self.hist] # Extract learning rates
        mixup_alphas = [h["mixup_alpha"] for h in self.hist] # Extract mixup alphas
        class_weights_history = [h["class_weights"] for h in self.hist] # Extract class weights

        plt.figure(figsize=(15, 10)) # Increased figure size

        # Plot Accuracy
        plt.subplot(2, 2, 1) # Changed to 2x2 grid
        plt.plot(xs, tr_acc, label="train")
        plt.plot(xs, val_acc, label="val")
        # Add vertical lines for rollback epochs on the accuracy plot
        for rb_epoch in self.rollback_epochs:
            plt.axvline(x=rb_epoch, color='gray', linestyle='--', linewidth=1, label='Rollback')
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.grid(); plt.legend()

        # Plot Loss
        plt.subplot(2, 2, 2) # New subplot for Loss
        plt.plot(xs, tr_loss, label="train")
        plt.plot(xs, val_loss, label="val")
        # Add vertical lines for rollback epochs on the loss plot
        for rb_epoch in self.rollback_epochs:
            plt.axvline(x=rb_epoch, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(); plt.legend()

        # Plot Learning Rate
        plt.subplot(2, 2, 3) # New subplot for LR
        plt.plot(xs, lrs, label="LR")
        plt.xlabel("epoch"); plt.ylabel("Learning Rate"); plt.grid(); plt.legend()
        plt.yscale('log') # Use log scale for LR for better visualization

        # Plot Mixup Alpha and Class Weights
        plt.subplot(2, 2, 4) # New subplot for Mixup Alpha and Class Weights
        plt.plot(xs, mixup_alphas, label="Mixup Alpha", linestyle='--')
        # Plot each class weight
        num_classes = len(class_weights_history[0]) if class_weights_history else 0
        for i in range(num_classes):
            plt.plot(xs, [cw[i] for cw in class_weights_history], label=f"Class {i} Weight")

        plt.xlabel("epoch"); plt.ylabel("Value"); plt.grid(); plt.legend()

        plt.tight_layout()
        os.makedirs("logs", exist_ok=True)
        plt.savefig(f"logs/curve_epoch_{self.state.epoch}.png")
        plt.close()

def get_class_balanced_weights(labels: List[int], beta: float = 0.99) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels))
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

def build_dataloaders(batch_size=12, num_workers=0, to_rgb=False, pin_memory=True, persistent_workers=True):
    import platform
    if platform.system() == "Windows":
        num_workers = 0
        persistent_workers = False

    # 1) Dataset'leri oluştur
    train_ds = CustomTumorDataset("preprocessed_data/train", to_rgb=to_rgb, transform=None)
    val_ds   = CustomTumorDataset("preprocessed_data/val",   to_rgb=to_rgb, transform=None)
    test_ds  = CustomTumorDataset("preprocessed_data/test", to_rgb=to_rgb, transform=None)

    # 2) Train etiketlerini toplayın
    train_labels = [label for _, label in train_ds.samples]

    # 3) Class-balanced weights ve sampler
    cb_weights = get_class_balanced_weights(train_labels, beta=0.99)
    sample_weights = [cb_weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # 4) DataLoader'ları oluşturun (shuffle=False, sampler ile)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,       # shuffle yerine sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True # Add drop_last=True to handle single-sample batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # 5) Loss için aynı ağırlıkları döndürün
    return train_loader, val_loader, test_loader, cb_weights.to(DEVICE)

def freeze_support_for_win():
    mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    freeze_support_for_win()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        to_rgb=True
    )
    print(f"[DEBUG] Train dataset size: {len(train_loader.dataset)}")
    print(f"[DEBUG] Test dataset size: {len(test_loader.dataset)}")

    model = VGGCustom(num_classes=3, in_channels=3)
    trainer = Trainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        target_lr=INITIAL_LR,
        max_epochs=NUM_EPOCHS,
        plot_every=10,
        class_weights=class_weights,
        scheduler_type="one_cycle",
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4,
        mixup_alpha=0.2,
        weight_decay=5e-4
    )
    trainer.train()

    # Test the best model on the test dataset
    print(f"{GREEN}\n=== Testing Best Model ==={RESET}")
    best_model_path = os.path.join("models", "best_vgg_custom.pt")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
        trainer.model.eval() # Set model to evaluation mode
        test_loss, test_acc, _, _ = trainer._run_epoch(train=False)
        print(f"{GREEN}Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.3f}{RESET}")
    else:
        print(f"{RED}Best model checkpoint not found at {best_model_path}. Please train the model first.{RESET}")
