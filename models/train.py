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
BATCH_SIZE = 16    # Batch size artırıldı
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

class DynamicRegularization:
    def __init__(self, model, initial_factor=0.1, max_factor=0.5):
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

def build_dataloaders():
    # Tıbbi görüntüler için özel augmentation
    medical_aug = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Küçük kaydırmalar
        T.RandomRotation(degrees=10),  # Küçük açılı döndürme
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Hafif bulanıklaştırma
    ])
    
    # Validation için sadece temel dönüşümler
    val_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    train_ds = CustomTumorDataset("preprocessed_data/train", transform=medical_aug)
    val_ds   = CustomTumorDataset("preprocessed_data/val",   transform=val_transform)
    
    print(f"{CYAN}[LOG] Eğitim veri kümesinin sinif listesi:{RESET}", train_ds.classes)
    print(f"{CYAN}[LOG] valid veri kümesinin sinif listesi:{RESET}", val_ds.classes)
    train_labels = [label for _, label in train_ds.samples]
    val_labels   = [label for _, label in val_ds.samples]
    print(f"{CYAN}[LOG] train label dağılımı:{RESET}", Counter(train_labels))
    print(f"{CYAN}[LOG] valid label dağılımı:{RESET}", Counter(val_labels))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=2,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                             num_workers=2,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=2)
    return train_loader, val_loader

def run_epoch(model, loader, criterion, optim=None, gpu_aug=None, scaler=None):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss = correct = total = 0

    if train:
        iterator = tqdm(loader, leave=True, desc="Train")
    else:
        iterator = loader

    for i, (X, y) in enumerate(iterator):
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        # GPU augmentation kaldırıldı
        
        with torch.set_grad_enabled(train):
            with autocast('cuda', enabled=True):
                out = model(X)
                loss = criterion(out, y)
                if train:
                    loss = loss / 2  # Gradient accumulation steps
            
            if train:
                scaler.scale(loss).backward()
                if (i + 1) % 2 == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)

        tot_loss += loss.item() * y.size(0) * 2
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

        # Her batch sonrası bellek temizliği
        del X, y, out, loss
        torch.cuda.empty_cache()
        if i % 10 == 0:  # Her 10 batch'te bir garbage collection
            gc.collect()

    return tot_loss / total, correct / total

def plot_metrics(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, lr_hist):
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(2, 1, 1)
    plt.plot(train_acc_hist, label='Train Accuracy', color='blue')
    plt.plot(val_acc_hist, label='Validation Accuracy', color='red')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(2, 1, 2)
    plt.plot(train_loss_hist, label='Train Loss', color='blue')
    plt.plot(val_loss_hist, label='Validation Loss', color='red')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_logs.png')
    plt.close()

# Block opening strategy
def should_open_block(epoch, val_acc, train_acc, val_loss, train_loss):
    # İlk 3 epoch'ta blok açma
    if epoch <= 3:
        return True
        
    # Validation accuracy 0.85'i geçtiyse ve train/val farkı 0.1'den küçükse
    if val_acc > 0.85 and abs(train_acc - val_acc) < 0.1:
        return True
        
    # Validation loss 0.3'ün altındaysa
    if val_loss < 0.3:
        return True
        
    return False

def handle_rollback(model, cache, current_metrics, optim, regularizer, scheduler):
    """
    Handles rollback logic based on different conditions
    Returns: (should_rollback, rollback_epochs, new_lr)
    """
    if len(cache) < 2:  # En az 2 epoch'luk cache gerekli
        return False, 0, None
        
    current_val_acc = current_metrics['val_acc']
    current_val_loss = current_metrics['val_loss']
    
    # Son 3 epoch'un metriklerini al
    recent_metrics = cache[-3:]
    
    # Validation accuracy'deki değişim oranı
    acc_change = abs(current_val_acc - recent_metrics[0]['val_acc']) / (recent_metrics[0]['val_acc'] + 1e-8)
    
    # Validation loss'daki değişim oranı
    loss_change = abs(current_val_loss - recent_metrics[0]['val_loss']) / (recent_metrics[0]['val_loss'] + 1e-8)
    
    # Yüksek accuracy durumunda farklı strateji
    if current_val_acc > 0.85:
        if acc_change < 0.01:  # Eşik artırıldı (0.005'ten 0.01'e)
            return True, 2, optim.param_groups[0]['lr'] * 0.7  # Daha az agresif düşüş (0.5'ten 0.7'ye)
        return False, 0, None  # Yüksek accuracy'de rollback yapma
    
    # Learning rate değişikliği sonrası performans düşüşü
    if len(cache) >= 2 and 'lr_changed' in cache[-1]:
        prev_acc = cache[-2]['val_acc']
        if current_val_acc < prev_acc * 0.85:  # Eşik artırıldı (0.70'ten 0.85'e)
            return True, 1, optim.param_groups[0]['lr'] * 0.8  # Daha az agresif düşüş (0.7'den 0.8'e)
        return False, 0, None
    
    # Overfitting durumu (accuracy gap > 0.20)
    if acc_change > 0.20:  # Eşik düşürüldü (0.25'ten 0.20'ye)
        if acc_change > 0.40:  # Eşik düşürüldü (0.5'ten 0.40'a)
            return True, 2, optim.param_groups[0]['lr'] * 0.8  # Daha az agresif düşüş (0.7'den 0.8'e)
        else:  # Hafif overfitting
            return True, 1, optim.param_groups[0]['lr'] * 0.8
    
    # Underfitting durumu (accuracy gap < 0.20)
    if acc_change < 0.20 and current_val_acc < 0.55:
        if acc_change < 0.10 and len(cache) >= 3:  # Ciddi underfitting
            return True, 2, optim.param_groups[0]['lr'] * 1.5
        else:  # Hafif underfitting
            return True, 1, optim.param_groups[0]['lr'] * 1.3
    
    # Validation loss'da ani artış
    if loss_change > 0.5:
        return True, 1, optim.param_groups[0]['lr'] * 0.8  # Daha az agresif düşüş (0.7'den 0.8'e)
    
    return False, 0, None

def train():
    # CUDA optimizasyonları
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train_loader, val_loader = build_dataloaders()

    model = VGGCustom(num_classes=len(train_loader.dataset.classes)).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    # GPU augmentation kaldırıldı
    gpu_aug = None

    # Cache boyutunu sınırla
    max_cache_size = 5
    cache = []
    stuck_count = 0
    window = 6
    val_loss_hist = []
    val_acc_hist = []
    train_loss_hist = []
    train_acc_hist = []
    best_val_acc = 0
    best_val_loss = float('inf')
    patience = EARLY_STOPPING_PATIENCE
    no_improve_epochs = 0
    block_no_improve_epochs = 0
    lr_hist = []

    # Eğitim fonksiyonunda, class_weights başlangıçta eşit olarak başlatılır
    class_weights = torch.ones(3, dtype=torch.float32, device=DEVICE)

    # Initialize regularization and schedulers
    regularizer = DynamicRegularization(model, REGULARIZATION_FACTOR, MAX_REGULARIZATION)
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    reduce_lr = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5)
    
    # Learning rate değişikliği takibi için
    last_lr = INITIAL_LR
    
    # Overfitting/Underfitting tracking
    overfit_epochs = 0
    underfit_epochs = 0
    best_state = None
    best_epoch = 0

    consecutive_rollbacks = 0
    skip_epochs = 0
    last_metrics = None
    last_rollback_acc = None

    for ep in range(1, NUM_EPOCHS + 1):
        # Eğer skip_epochs varsa, o kadar epoch'u atla
        if skip_epochs > 0:
            print(f"{YELLOW}[!] Skipping epoch {ep} ({skip_epochs} epochs remaining){RESET}")
            skip_epochs -= 1
            continue

        t0 = time.time()
        
        tr_l, tr_a = run_epoch(model, train_loader, criterion,
                               optim=optim, gpu_aug=gpu_aug, scaler=scaler)
        vl_l, vl_a = run_epoch(model, val_loader, criterion)

        current_metrics = {
            'train_loss': tr_l,
            'val_loss': vl_l,
            'train_acc': tr_a,
            'val_acc': vl_a
        }

        # Learning rate değişikliği kontrolü
        current_lr = optim.param_groups[0]['lr']
        lr_changed = abs(current_lr - last_lr) > 1e-6
        last_lr = current_lr

        # Cache güncelle
        cache.append({
            "model": copy.deepcopy(model.state_dict()),
            "optim": copy.deepcopy(optim.state_dict()),
            "train_loss": tr_l,
            "val_loss": vl_l,
            "train_acc": tr_a,
            "val_acc": vl_a,
            "lr_changed": lr_changed
        })
        if len(cache) > max_cache_size:
            cache.pop(0)

        # Geri alma kontrolü
        should_rollback, rollback_epochs, new_lr = handle_rollback(
            model, cache, current_metrics, optim, regularizer, scheduler
        )

        if should_rollback and len(cache) >= rollback_epochs + 1:
            # Ardışık rollback kontrolü
            if last_rollback_acc is not None:
                if abs(current_metrics['val_acc'] - last_rollback_acc) < 0.01:  # Eşik artırıldı (0.005'ten 0.01'e)
                    consecutive_rollbacks += 1
                else:
                    consecutive_rollbacks = 1
            else:
                consecutive_rollbacks = 1

            if consecutive_rollbacks >= 3:
                print(f"{YELLOW}[!] 3 consecutive rollbacks detected! Skipping next 2 epochs{RESET}")
                skip_epochs = 2
                consecutive_rollbacks = 0
                last_rollback_acc = None
                continue

            print(f"{YELLOW}[!] Rollback triggered! Going back {rollback_epochs} epochs{RESET}")
            
            # Model ve optimizer durumunu geri al
            rollback_state = cache[-(rollback_epochs + 1)]
            model.load_state_dict(rollback_state['model'])
            optim.load_state_dict(rollback_state['optim'])
            
            # Learning rate güncelle
            if new_lr is not None:
                for param_group in optim.param_groups:
                    param_group['lr'] = new_lr
                print(f"{YELLOW}[!] Learning rate adjusted to: {new_lr:.2e}{RESET}")
                # Scheduler'ları sıfırla
                scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
                reduce_lr = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5)
            
            # Cache'i güncelle
            cache = cache[:-(rollback_epochs)]
            last_rollback_acc = current_metrics['val_acc']
            
            # Epoch'u tekrar çalıştır
            continue

        last_metrics = current_metrics

        # Validation sonrası dinamik class weight güncellemesi
        if ep % 3 == 0:
            all_val_labels = []
            all_val_preds = []
            model.eval()
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    out = model(X)
                    preds = out.argmax(1)
                    all_val_labels.extend(y.cpu().numpy())
                    all_val_preds.extend(preds.cpu().numpy())
                    del X, y, out, preds
                    torch.cuda.empty_cache()
            cm = confusion_matrix(all_val_labels, all_val_preds, labels=[0,1,2])
            print(f"{YELLOW}[DYN-WEIGHT][Ep{ep}] Confusion Matrix:{RESET}\n{cm}")
            fp = np.zeros(3)
            for c in range(3):
                fp[c] = cm[:,c].sum() - cm[c,c]
            total = np.sum(cm)
            penalty_ratio = fp / (total + 1e-8)
            print(f"{YELLOW}[DYN-WEIGHT][Ep{ep}] False Positive: {fp}, Penalty Ratio: {penalty_ratio}{RESET}")
            old_weights = class_weights.cpu().numpy().copy()
            for c in range(3):
                class_weights[c] = max(min_weight, class_weights[c] * (1 - lambda_penalty * penalty_ratio[c]))
            class_weights = class_weights / class_weights.sum() * 3
            print(f"{YELLOW}[DYN-WEIGHT][Ep{ep}] Old Weights: {old_weights}, New Weights: {class_weights.cpu().numpy()}{RESET}")
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Agresif bellek temizliği
        torch.cuda.empty_cache()
        gc.collect()

        train_loss_hist.append(tr_l)
        val_loss_hist.append(vl_l)
        train_acc_hist.append(tr_a)
        val_acc_hist.append(vl_a)
        if len(val_loss_hist) > window:
            val_loss_hist.pop(0)
            train_loss_hist.pop(0)
            val_acc_hist.pop(0)
            train_acc_hist.pop(0)

        # Overfitting detection and prevention
        accuracy_gap = tr_a - vl_a
        if accuracy_gap > OVERFIT_THRESHOLD:
            overfit_epochs += 1
            if overfit_epochs >= MAX_OVERFIT_EPOCHS:
                print(f"{RED}[!] Overfitting detected! Increasing regularization...{RESET}")
                regularizer.increase_regularization()
                # Recover best state
                if best_state is not None:
                    model.load_state_dict(best_state)
                    print(f"{GREEN}[✓] Recovered best model state from epoch {best_epoch}{RESET}")
                overfit_epochs = 0
        else:
            overfit_epochs = 0
            
        # Underfitting detection and prevention
        if vl_a < UNDERFIT_THRESHOLD:
            underfit_epochs += 1
            if underfit_epochs >= MAX_UNDERFIT_EPOCHS:
                print(f"{RED}[!] Underfitting detected! Decreasing regularization...{RESET}")
                regularizer.decrease_regularization()
                underfit_epochs = 0
        else:
            underfit_epochs = 0
            
        # Save best state
        if vl_a > best_val_acc:
            best_val_acc = vl_a
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            torch.save(model.state_dict(), "models/best_vgg_custom.pt")
            print(f"{GREEN}[✓] New best model saved! Accuracy: {vl_a:.4f}{RESET}")
            
        # Learning rate scheduling
        scheduler.step()
        reduce_lr.step(vl_a)

        # Validation metrics iyileşme kontrolü
        improved = False
        if vl_a > best_val_acc:
            best_val_acc = vl_a
            improved = True
        if vl_l < best_val_loss:
            best_val_loss = vl_l
            improved = True
            
        if improved:
            no_improve_epochs = 0
            torch.save(model.state_dict(), "models/best_vgg_custom.pt")
            print(f"{GREEN}[✓] Yeni en iyi model kaydedildi! Doğruluk: {vl_a:.4f}{RESET}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"{RED}[!] Early stopping: {patience} epoch boyunca iyileşme yok.{RESET}")
            break

        # Progressive block açma - validation loss'a göre
        opened = sum([any(p.requires_grad for p in blk.parameters()) for blk in model.blocks])
        if opened == 2:
            block_open_threshold = 8
        else:
            block_open_threshold = 3
            
        # Validation loss iyileşme kontrolü - son 3 epoch'a bak
        if ep > 3 and vl_l > min(val_loss_hist[-3:]):  # Son 3 epoch'un en kötü loss'u
            block_no_improve_epochs += 1
        else:
            block_no_improve_epochs = 0
            
        if block_no_improve_epochs >= block_open_threshold and hasattr(model, 'freeze_blocks_until'):
            if opened < len(model.blocks):
                model.freeze_blocks_until(opened)
                print(f"{BLUE}[+] {ep}. epoch → Block-{opened+1} açıldı{RESET}")
                block_no_improve_epochs = 0

        current_lr = optim.param_groups[0]['lr']
        print(
            f"{BOLD}[Ep {ep:02d}]{RESET} "
            f"TrainAcc: {tr_a:.3f} | ValAcc: {vl_a:.3f} | "
            f"TrainLoss: {tr_l:.3f} | ValLoss: {vl_l:.3f} | "
            f"LR: {current_lr:.2e} | NoImp: {no_improve_epochs} | "
            f"Block: {opened} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        if vl_a >= 0.958:
            print(f"[✓] %{vl_a:.4f} doğruluk – eğitim bitti.")
            break

        # Her epoch sonunda metrikleri kaydet
        train_acc_hist.append(tr_a)
        val_acc_hist.append(vl_a)
        train_loss_hist.append(tr_l)
        val_loss_hist.append(vl_l)
        lr_hist.append(current_lr)
        
        # Her epoch sonunda grafik çiz
        plot_metrics(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, lr_hist)

        # Her epoch sonunda bellek temizliği
        torch.cuda.empty_cache()
        gc.collect()

    # Eğitim sonunda grafikleri oluştur
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_hist, label='Train Acc')
    plt.plot(val_acc_hist, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.tight_layout()
    plt.savefig('logs/training_logs.png')
    plt.close()

    # Grafik oluşturma sonrası bellek temizliği
    plt.clf()
    plt.close('all')
    torch.cuda.empty_cache()
    gc.collect()

def freeze_support_for_win():
    mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    freeze_support_for_win()
    train()
