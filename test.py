import os, random
import torch
from torch.utils.data import DataLoader, Subset
from dataset.custom_dataset import CustomTumorDataset
from models.basic_cnn_model import BasicCNNModel
from models.train import build_dataloaders # Only import build_dataloaders as Trainer is not used here
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report # Add classification_report
import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model ve veri yükleme
print("🔁 Veri yükleniyor...")
# CustomTumorDataset, içinde zaten resize, normalize ve ToTensorV2 dönüşümlerini uygular.
# train.py'deki build_dataloaders fonksiyonu da to_rgb=True ile çağırır.
test_root = os.path.join("preprocessed_data", "test") # Dizininize göre güncelleyin
test_dataset = CustomTumorDataset(test_root, to_rgb=True) # Eğitimdeki to_rgb ayarıyla aynı olmalı

# Keras kodundaki gibi 1 000 rastgele örnek seç
num_samples = 912
if num_samples > len(test_dataset):
    print(f"Test seti {num_samples}'den küçük! Tüm test setini kullanılıyor: {len(test_dataset)} örnek.")
    num_samples = len(test_dataset)

indices = random.sample(range(len(test_dataset)), num_samples)
subset = Subset(test_dataset, indices)
test_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False) # Windows için num_workers=0

# Sınıf isimlerini al (datasetten doğrudan)
class_names = test_dataset.class_names

# 3 ⟶ Eğitilmiş modeli yükleme
model = BasicCNNModel(num_classes=len(class_names), in_channels=3).to(DEVICE)
checkpoint_path = os.path.join("models", "full_vgg_custom.pt") # Yüklenecek modelin yolu

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Eğitimdeki konfigürasyonu geri getir:
    model.freeze_blocks_until(1)   # 0 ve 1 açık  → opened_blocks = 2

    model.eval()          # ⚠️ kritik: dropout/noise katmanlarını kapatır
    print(f"🤖 Eğitilmiş model {checkpoint_path} yüklendi.")
else:
    print(f"❌ Model bulunamadı: {checkpoint_path}. Lütfen önce modeli eğitin.")
    exit()

# 4 ⟶ Rastgele örneklerde doğruluk hesabı
correct = 0
total = 0
all_preds, all_labels = [], []

print("🔍 Tahminler yapılıyor...")
with torch.no_grad():           # gradyan gerekmez
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.append(preds.item()) # batch_size=1 olduğu için item() kullanıyoruz
        all_labels.append(labels.item()) # batch_size=1 olduğu için item() kullanıyoruz

# Doğruluk oranı hesapla ve düzgün print et
accuracy = 100.0 * correct / total # total, döngü içinde hesaplanıyor
print(f"\n🎯 {total} test örneğinden {correct} tanesi doğru tahmin edildi.")
print(f"✅ Doğruluk Oranı: {accuracy:.2f}%")

# 📈 Karışıklık Matrisi Çizimi
print("📈 Karışıklık Matrisi çiziliyor...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Karışıklık Matrisi')
plt.show()

print("\nSınıflandırma Raporu:")
print(classification_report(all_labels, all_preds, target_names=class_names))