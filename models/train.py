import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter
import os, time, torch, matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.custom_dataset import CustomTumorDataset
from models.basic_cnn_model import BasicCNNModel
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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

COSINE_START_EPOCH = 20

INITIAL_LR = 1e-2
BATCH_SIZE = 12
NUM_EPOCHS = 60
SWA_START_EPOCH = NUM_EPOCHS + 1 
WARMUP_EPOCHS = 0
EARLY_STOPPING_PATIENCE = 18
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_ACCUMULATION_STEPS = 4

SWA_LR = 5e-4

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
ROLLBACK_WINDOW = NUM_EPOCHS + 1 
PENALTY_GAP_LOW   = 0.1
PENALTY_GAP_HIGH  = 0.29

class DynamicRegularization(nn.Module):
    # Initializes the dynamic regularization module.
    def __init__(self, model, initial_factor=0.1, max_factor=0.5):
        super().__init__()
        self.model = model
        self.factor = initial_factor
        self.max_factor = max_factor
        self.original_dropout_rates = {}
        self.store_original_rates()

    # Stores the original dropout rates of the model.
    def store_original_rates(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout2d):
                self.original_dropout_rates[name] = module.p

    # Increases the regularization factor and applies it to dropout rates.
    def increase_regularization(self):
        self.factor = min(self.factor * 1.5, self.max_factor)
        self.apply_regularization()

    # Decreases the regularization factor and applies it to dropout rates.
    def decrease_regularization(self):
        self.factor = max(self.factor / 1.5, REGULARIZATION_FACTOR)
        self.apply_regularization()

    # Applies the current regularization factor to dropout layers.
    def apply_regularization(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout2d):
                original_rate = self.original_dropout_rates[name]
                module.p = min(original_rate * (1 + self.factor), 0.9)

class TemperatureScaler(nn.Module):
    # Initializes the temperature scaler.
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    # Scales the input logits by the learnable temperature parameter.
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
        pct_start: float = 0.4,
        div_factor: float = 10.0,
        final_div_factor: float = 1e3,
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

        self.swa_model = AveragedModel(model)

        self.temperature_scaler = TemperatureScaler()

        all_params = list(self.model.parameters())
        self.optim = torch.optim.AdamW(all_params, lr=target_lr, weight_decay=weight_decay)

        print(f"{CYAN}[Trainer Init] Initial optimizer LR: {self.optim.param_groups[0]['lr']:.2e}{RESET}")

        total_optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)

        if scheduler_type == "one_cycle":

            onecycle_total_iters = COSINE_START_EPOCH * total_optimizer_steps_per_epoch
            onecycle_scheduler = OneCycleLR(
                self.optim,
                max_lr=target_lr, 
                total_steps=onecycle_total_iters,
                pct_start=0.4, 
                div_factor=10.0, 
                final_div_factor=1e3, 
                anneal_strategy='cos'
            )

            cosine_epochs_duration = (SWA_START_EPOCH - 1) - COSINE_START_EPOCH + 1
            if cosine_epochs_duration < 0: 
                cosine_epochs_duration = 0
            cosine_total_iters = cosine_epochs_duration * total_optimizer_steps_per_epoch
            cosine_scheduler = CosineAnnealingLR(
                self.optim,
                T_max=cosine_total_iters,
                eta_min=1e-5
            )

            swa_epochs_duration = self.max_epochs - SWA_START_EPOCH + 1
            if swa_epochs_duration < 0:
                swa_epochs_duration = 0
            swa_total_iters = swa_epochs_duration * total_optimizer_steps_per_epoch
            swa_scheduler = LinearLR(
                self.optim,
                start_factor=1.0, 
                end_factor=1.0, 
                total_iters=max(1, swa_total_iters) 
            )

            schedulers_list = []
            milestones_list = []

            schedulers_list.append(onecycle_scheduler)

            if cosine_epochs_duration > 0:
                schedulers_list.append(cosine_scheduler)
                milestones_list.append(COSINE_START_EPOCH * total_optimizer_steps_per_epoch)

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
                factor=0.2,
                patience=3,
                min_lr=1e-7
            )
            self.scheduler_is_per_batch = False
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.class_weights_cpu = class_weights.clone().detach().cpu()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_cpu.to(self.device))
        print(f"{CYAN}[Trainer Init] Class weights: {self.class_weights_cpu.numpy()}{RESET}")
        self.amp_enabled = self.device.type == "cuda" and (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        self.scaler = GradScaler(self.device.type, enabled=self.amp_enabled) 
        self.regulariser = DynamicRegularization(self.model, initial_factor=REGULARIZATION_FACTOR)
        self.state = TrainerState()
        self.hist: List[dict] = []
        self.rollback_epochs: List[int] = []
        self.target_lr = target_lr
        self.mixup_alpha = mixup_alpha
        self._mixup_initial_alpha = self.mixup_alpha
        self._last_logged_mixup_alpha = -1.0
        self.scheduler_type = scheduler_type

    # Dynamically determines and returns the augmentation pipeline for the current epoch.
    def _get_current_augmenter(self, epoch: int) -> Optional[A.Compose]:

        if epoch < 10:
            return A.Compose([
                A.Rotate(limit=5, p=0.3),
                A.RandomRotate90(p=0.1),
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.3),
                A.Affine(
                    translate_percent={'x': 0.02, 'y': 0.02},
                    scale={'x': (0.96, 1.04), 'y': (0.96, 1.04)},
                    rotate=0,
                    shear={'x': 0.0, 'y': 0.0},
                    p=0.3
                ),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
                A.GaussNoise(std_range=(0.005, 0.015), mean_range=(0.0, 0.0), p=0.1),
            ])

        elif epoch < 40:
            return A.Compose([
                A.Rotate(limit=3, p=0.1),
                A.RandomRotate90(p=0.05),
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.1),
                A.Affine(
                    translate_percent={'x': 0.005, 'y': 0.005},
                    scale={'x': (0.99, 1.01), 'y': (0.99, 1.01)},
                    rotate=0,
                    shear={'x': 0.0, 'y': 0.0},
                    p=0.1
                ),
                A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.1),
                A.GaussNoise(std_range=(0.003, 0.008), mean_range=(0.0, 0.0), p=0.05),
            ])

        else:
            return A.Compose([
                A.HorizontalFlip(p=1.0),
            ])

    # Runs a single training or validation epoch, returning metrics and collected labels/predictions.
    def _run_epoch(self, train: bool = True) -> Tuple[float, float, torch.Tensor, torch.Tensor, Optional[List[int]], Optional[List[int]]]:
        """
        Runs a single training or validation epoch.

        Returns:
        avg_loss (float): Average loss for the epoch.
        avg_acc (float): Average accuracy for the epoch.
        per_class_correct (torch.Tensor): Number of correct predictions per class.
        per_class_total (torch.Tensor): Total samples per class.
        all_labels (Optional[List[int]]): List of all true labels for the epoch (only for validation/test).
        all_predictions (Optional[List[int]]): List of all predicted labels for the epoch (only for validation/test).
        """
        num_classes = len(self.class_weights_cpu)
        per_class_correct = torch.zeros(num_classes, device=self.device)
        per_class_total   = torch.zeros(num_classes, device=self.device)

        loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()

        tot_loss = 0.0
        correct  = 0
        total    = 0

        all_labels_epoch = []
        all_predictions_epoch = []

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
                    preds = out.argmax(dim=1) 

                    self.scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()

                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad(set_to_none=True)
                        if self.scheduler_is_per_batch: 
                            self.scheduler.step()
                        if self.state.epoch >= SWA_START_EPOCH:
                            self.swa_model.update_parameters(self.model)
                else:
                    with autocast(self.device.type, enabled=self.amp_enabled):
                        out  = self.model(X)

                        out_calibrated = self.temperature_scaler(out)
                        loss = self.criterion(out_calibrated, y)
                    preds = out_calibrated.argmax(dim=1) 

                    all_labels_epoch.extend(y.cpu().numpy())
                    all_predictions_epoch.extend(preds.cpu().numpy())

                batch_size = y.size(0)
                tot_loss  += loss.item() * batch_size
                correct   += (preds == y).sum().item()
                total     += batch_size

                if not train:
                    for cls in range(num_classes):
                        mask = (y == cls)
                        per_class_correct[cls] += (preds[mask] == cls).sum().item()
                        per_class_total[cls]   += mask.sum().item()

                del X, y, out, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = tot_loss / total
        avg_acc  = correct / total

        if train:
            return avg_loss, avg_acc, per_class_correct, per_class_total, None, None
        else:
            return avg_loss, avg_acc, per_class_correct, per_class_total, all_labels_epoch, all_predictions_epoch

    # Optionally rolls back model weights if performance drops significantly.
    def _maybe_rollback(self, val_acc: float, val_loss: float):
        cfg_acc_drop = 0.03
        cfg_loss_rise = 0.08
        cooldown_epochs = 5

        if len(self.hist) < 1:
            return

        prev_state = self.hist[-1]

        acc_drop = (prev_state["val_acc"] - val_acc) / (prev_state["val_acc"] + 1e-8)
        loss_rise = (val_loss - prev_state["val_loss"]) / (prev_state["val_loss"] + 1e-8)

        if (
            acc_drop >= cfg_acc_drop or loss_rise >= cfg_loss_rise
        ) and self.state.cooldown == 0 and self.state.total_rbs < MAX_TOTAL_ROLLBACKS and "weights" in prev_state:
            print(f"\n{YELLOW}[Rollback] Triggered! Δacc={acc_drop:.3f}, Δloss={loss_rise:.3f} → restoring previous weights (epoch {prev_state['epoch']}){RESET}")
            try:

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

    # Conditionally opens a new block in the model based on performance and epoch.
    def _maybe_open_block(self, val_acc: float, val_loss: float):
        opened = self.model.opened_blocks
        if opened >= len(self.model.blocks):
            return

        if opened == 2:
            print(f"{YELLOW}[Block] Skipping opening Block-3 at epoch {self.state.epoch} - 3rd block is not allowed to open.{RESET}")
            return

        stagnation_window = max(2, min(len(self.hist), ROLLBACK_WINDOW // 2))
        if len(self.hist) < stagnation_window:
            return

        stagnant = all(h["val_loss"] >= val_loss for h in self.hist[-stagnation_window:])

        if (stagnant or val_acc > 0.70) and opened < 3:

            if opened == 1 and self.state.epoch < 15: 
                print(f"{YELLOW}[Block] Deferring Block-2 opening at epoch {self.state.epoch} - Waiting for epoch 15 to open Block 2.{RESET}")
                return
            self.model.freeze_blocks_until(opened)
            print(f"{CYAN}[Block] Block-{self.model.opened_blocks} opened at epoch {self.state.epoch} (val_acc={val_acc:.3f}, val_loss={val_loss:.3f}){RESET}")

    # Logs epoch-wise metrics and saves model/optimizer state for potential rollback.
    def _log_epoch(self, tr_acc, tr_loss, val_acc, val_loss, val_per_class_correct: torch.Tensor, val_per_class_total: torch.Tensor):
        cur = {
            "epoch": self.state.epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "lr": self.optim.param_groups[0]["lr"],
            "mixup_alpha": self.mixup_alpha,
            "class_weights": self.class_weights_cpu.cpu().numpy().copy(),
            "val_per_class_correct": val_per_class_correct.cpu().numpy().copy(),
            "val_per_class_total": val_per_class_total.cpu().numpy().copy()
        }

        try:

            with torch.no_grad():
                cpu_weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            cur["weights"] = cpu_weights
            cur["opened_blocks"] = self.model.opened_blocks

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

        while len(self.hist) > ROLLBACK_WINDOW + 2:
            self.hist.pop(0)
            gc.collect()

    # Normalizes class weights to maintain a consistent sum across classes.
    def _normalize_class_weights(self):

        cw = self.class_weights_cpu
        cw = cw / cw.sum() * len(cw)
        self.class_weights_cpu = cw

        with torch.no_grad():
            self.criterion.weight.data.copy_(cw.to(self.device))
        print(f"{GREEN}[Normalize] Class weights normalized: {cw.cpu().numpy()}{RESET}")

    # Main training loop for the model.
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

            current_augmenter = self._get_current_augmenter(epoch)
            self.train_loader.dataset.transform = current_augmenter

            t0 = time.time()
            tr_loss, tr_acc, tr_tot_tr, tr_corr_tr, _, _ = self._run_epoch(train=True)

            if self.state.epoch == SWA_START_EPOCH:
                for pg in self.optim.param_groups:
                    pg['lr'] = SWA_LR
                print(f"{CYAN}[SWA] SWA LR set to {SWA_LR:.2e} at epoch {self.state.epoch}{RESET}")

            val_loss, val_acc, val_corr_val, val_tot_val, _, _ = self._run_epoch(train=False)

            if not self.scheduler_is_per_batch:
                self.scheduler.step(val_loss)

            # Enforce minimum LR between epoch 40 and 50
            # if 40 <= self.state.epoch < 50:
            #     current_lr = self.optim.param_groups[0]['lr']
            #     if current_lr < 1e-3:
            #         for param_group in self.optim.param_groups:
            #             param_group['lr'] = 1e-3
            #         print(f"{YELLOW}[LR Adjustment] LR forced to 1e-3 at epoch {self.state.epoch}{RESET}")

            MIXUP_DECAY_END_EPOCH = 15 
            new_mixup_alpha = 0.0
            if self.state.epoch <= MIXUP_DECAY_END_EPOCH:
                new_mixup_alpha = max(0.0, self._mixup_initial_alpha - (self._mixup_initial_alpha / MIXUP_DECAY_END_EPOCH) * self.state.epoch)
            else:
                new_mixup_alpha = 0.0 

            if new_mixup_alpha != self.mixup_alpha:
                self.mixup_alpha = new_mixup_alpha
                print(f"{YELLOW}[Mixup] Mixup alpha set to {self.mixup_alpha:.2f} at epoch {self.state.epoch}{RESET}")

            if self.state.cooldown > 0:
                self.state.cooldown -= 1

            gap     = tr_acc - val_acc
            gap_abs = abs(gap)

            if gap_abs >= ROLLBACK_GAP_THRESH:
                self._maybe_rollback(val_acc, val_loss)

            else:

                if PENALTY_GAP_LOW <= gap_abs < PENALTY_GAP_HIGH and self.state.epoch >= 30:
                    train_acc_cls = tr_corr_tr / (tr_tot_tr + 1e-8)
                    val_acc_cls   = val_corr_val / (val_tot_val + 1e-8)
                    per_class_gap = train_acc_cls - val_acc_cls
                    for cls, g in enumerate(per_class_gap.cpu()):
                        if g > PENALTY_GAP_LOW:
                            self.class_weights_cpu[cls] *= 1.03  
                        elif g < -PENALTY_GAP_LOW:
                            self.class_weights_cpu[cls] *= 0.97  
                    with torch.no_grad():
                        self.criterion.weight.data.copy_(self.class_weights_cpu.to(self.device))
                    print(f"{YELLOW}[ClassPenalty] per_class_gap: {per_class_gap.cpu().numpy()}")
                    print(f"{YELLOW}[ClassPenalty] new weights: {self.class_weights_cpu.cpu().numpy()}{RESET}")

            if self.state.epoch % 10 == 0:
                self._normalize_class_weights()

            self._maybe_open_block(val_acc, val_loss)

            if gap > self.overfit_thr:
                self.regulariser.increase_regularization()
                print(f"{YELLOW}[Regulariser] Overfit detected (gap={gap:.3f}), regularisation increased to {self.regulariser.factor:.2f}{RESET}")
            elif val_acc < self.underfit_thr:
                self.regulariser.decrease_regularization()
                print(f"{YELLOW}[Regulariser] Underfit detected (val_acc={val_acc:.3f}), regularisation decreased to {self.regulariser.factor:.2f}{RESET}")

            self._log_epoch(tr_acc, tr_loss, val_acc, val_loss, val_corr_val, val_tot_val)
            if val_acc > self.state.best_acc:
                self.state.best_acc = val_acc
                self.state.best_loss = val_loss
                self.state.best_weights = self.hist[-1]["weights"]
                self.state.no_imp_epochs = 0
                best_state = self.hist[-1]
                torch.save({"opened_blocks": best_state["opened_blocks"], "model_state": best_state["weights"]}, self.ckpt_path)
                print(f"{GREEN}[Checkpoint] New best model saved at epoch {epoch} (val_acc={val_acc:.3f}){RESET}")
            else:
                self.state.no_imp_epochs += 1

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

            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nTraining finished. Best val-acc = {self.state.best_acc:.3f}")

        full_ckpt_path = os.path.join(os.path.dirname(self.ckpt_path), "full_vgg_custom.pt")
        torch.save({
            "opened_blocks": self.model.opened_blocks,
            "model_state": self.model.state_dict()
        }, full_ckpt_path)
        print(f"{GREEN}[Checkpoint] Full trained model saved to {full_ckpt_path}{RESET}")

        if self.max_epochs >= SWA_START_EPOCH:
            print(f"{CYAN}[SWA] Updating BatchNorm for SWA model...{RESET}")
            update_bn(self.train_loader, self.swa_model, device=self.device)

            swa_ckpt_path = os.path.join(os.path.dirname(self.ckpt_path), "best_vgg_custom_swa.pt")
            torch.save(self.swa_model.state_dict(), swa_ckpt_path)
            print(f"{GREEN}[SWA] SWA model saved to {swa_ckpt_path}{RESET}")

            print(f"{CYAN}[Calibration] Starting temperature calibration on SWA model...{RESET}")
            self.temperature_scaler.to(self.device) 
            self.temperature_scaler.eval()
            optimizer = torch.optim.LBFGS([self.temperature_scaler.temperature], lr=0.01, max_iter=50)

            all_logits = []
            all_labels = []
            with torch.no_grad():
                for X, y in tqdm(self.val_loader, desc="Collecting SWA Logits", leave=False):
                    X = X.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    logits = self.swa_model(X) 
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

    # Plots and saves the training history including accuracy, loss, LR, Mixup alpha, and class weights.
    def _plot_history(self):
        xs = [h["epoch"] for h in self.hist]
        tr_acc = [h["train_acc"] for h in self.hist]
        val_acc = [h["val_acc"] for h in self.hist]
        tr_loss = [h["train_loss"] for h in self.hist]
        val_loss = [h["val_loss"] for h in self.hist]
        lrs = [h["lr"] for h in self.hist] 
        mixup_alphas = [h["mixup_alpha"] for h in self.hist] 
        class_weights_history = [h["class_weights"] for h in self.hist] 
        val_per_class_correct_history = [h["val_per_class_correct"] for h in self.hist]
        val_per_class_total_history = [h["val_per_class_total"] for h in self.hist]

        plt.figure(figsize=(18, 12)) 

        plt.subplot(2, 3, 1) 
        plt.plot(xs, tr_acc, label="train")
        plt.plot(xs, val_acc, label="val")

        for rb_epoch in self.rollback_epochs:
            plt.axvline(x=rb_epoch, color='gray', linestyle='--', linewidth=1, label='Rollback')
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.grid(); plt.legend()

        plt.subplot(2, 3, 2) 
        plt.plot(xs, tr_loss, label="train")
        plt.plot(xs, val_loss, label="val")

        for rb_epoch in self.rollback_epochs:
            plt.axvline(x=rb_epoch, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(); plt.legend()

        plt.subplot(2, 3, 3) 
        plt.plot(xs, lrs, label="LR")
        plt.xlabel("epoch"); plt.ylabel("Learning Rate"); plt.grid(); plt.legend()
        plt.yscale('log') 

        plt.subplot(2, 3, 4) 
        plt.plot(xs, mixup_alphas, label="Mixup Alpha", linestyle='--')

        num_classes = len(class_weights_history[0]) if class_weights_history else 0
        for i in range(num_classes):
            plt.plot(xs, [cw[i] for cw in class_weights_history], label=f"Class {i} Weight")

        plt.xlabel("epoch"); plt.ylabel("Value"); plt.grid(); plt.legend()

        plt.subplot(2, 3, 5) 
        if val_per_class_correct_history and val_per_class_total_history:
            num_classes_plot = len(val_per_class_correct_history[0])
            for i in range(num_classes_plot):
                per_class_acc = [
                    (val_per_class_correct_history[j][i] / (val_per_class_total_history[j][i] + 1e-8))
                    for j in range(len(xs))
                ]
                plt.plot(xs, per_class_acc, label=f"Class {i} Acc")
            plt.xlabel("epoch"); plt.ylabel("Per-Class Accuracy"); plt.grid(); plt.legend()
        else:
            plt.text(0.5, 0.5, "No per-class data available", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        os.makedirs("logs", exist_ok=True)
        plt.savefig(f"logs/curve_epoch_{self.state.epoch}.png")
        plt.close()

def get_class_balanced_weights(labels: List[int], beta: float = 0.99) -> torch.Tensor:
    # Computes class-balanced weights based on label frequencies and a beta factor.
    counts = torch.bincount(torch.tensor(labels))
    effective_num = 1.0 - torch.pow(beta, counts.float())
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(counts)
    return weights

def mixup_data(x, y, alpha=0.2):
    # Applies Mixup augmentation to input data and labels.
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
    # Builds and returns training, validation, and test data loaders with class-balanced sampling.
    import platform
    if platform.system() == "Windows":
        num_workers = 0
        persistent_workers = False

    train_ds = CustomTumorDataset("preprocessed_data/train", to_rgb=to_rgb, transform=None)
    val_ds   = CustomTumorDataset("preprocessed_data/val",   to_rgb=to_rgb, transform=None)
    test_ds  = CustomTumorDataset("preprocessed_data/test", to_rgb=to_rgb, transform=None)

    train_labels = [label for _, label in train_ds.samples]

    cb_weights = get_class_balanced_weights(train_labels, beta=0.99)
    sample_weights = [cb_weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,       
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True 
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

    return train_loader, val_loader, test_loader, cb_weights.to(DEVICE)

def freeze_support_for_win():
    # Configures multiprocessing for Windows compatibility.
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

    model = BasicCNNModel(num_classes=3, in_channels=3)
    trainer = Trainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        target_lr=INITIAL_LR,
        max_epochs=NUM_EPOCHS,
        plot_every=10,
        class_weights=class_weights,
        scheduler_type="reduce_on_plateau",
        pct_start=0.4,
        div_factor=10.0,
        final_div_factor=1e3,
        mixup_alpha=0.2,
        weight_decay=5e-4
    )
    trainer.train()

    print(f"{GREEN}\n=== Testing Best Model ==={RESET}")
    best_model_path = os.path.join("models", "best_vgg_custom.pt")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.opened_blocks = ckpt["opened_blocks"]
        trainer.model.eval()
        test_loss, test_acc, test_corr_val, test_tot_val, all_labels, all_predictions = trainer._run_epoch(train=False) 
        print(f"{GREEN}Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.3f}{RESET}")

        print("📈 Karışıklık Matrisi ve Sınıflandırma Raporu çiziliyor...")
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.class_names, yticklabels=test_loader.dataset.class_names)
        plt.xlabel('Tahmin Edilen Etiket')
        plt.ylabel('Gerçek Etiket')
        plt.title('Karışıklık Matrisi')
        plt.tight_layout()
        plt.savefig(os.path.join("logs", "confusion_matrix_test.png")) 
        plt.close() 

        print("\nSınıflandırma Raporu:")
        print(classification_report(all_labels, all_predictions, target_names=test_loader.dataset.class_names))

    else:
        print(f"{RED}Best model checkpoint not found at {best_model_path}. Please train the model first.{RESET}")