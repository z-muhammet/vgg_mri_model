import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from dataset.custom_dataset import CustomTumorDataset
from models.vgg_custom import VGGCustom

# ========= Yardımcı =========
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_to_file(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ========= Klasör & Log Hazırlığı =========
ensure_dir("logs")
ensure_dir("models")

EVAL_LOG = "logs/evaluate.log"
if not os.path.exists(EVAL_LOG):
    with open(EVAL_LOG, "w", encoding="utf-8") as f:
        f.write("=== Model Test Sonuçları ===\n\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Veri Yükle =========
test_dataset = CustomTumorDataset("preprocessed_data/test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========= Modeli Yükle =========
model = VGGCustom(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("models/vgg_custom_trained.pt", map_location=DEVICE))
model.eval()

# ========= Tahmin Et =========
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ========= Metrikler =========
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["glioma", "meningioma", "unknown"], digits=4)

log_to_file(EVAL_LOG, "[✓] Confusion Matrix:\n" + str(cm))
log_to_file(EVAL_LOG, "\n[✓] Classification Report:\n" + report)

print("[✓] Değerlendirme tamamlandı. Sonuçlar logs/evaluate.log dosyasına yazıldı.")
