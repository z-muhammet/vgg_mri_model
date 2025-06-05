import torch
import numpy as np
import random
from models.vgg_custom import VGGCustom
from dataset.custom_dataset import CustomTumorDataset
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

MODEL_PATH = "models/best_vgg_custom2.pt"
TEST_PATH = "preprocessed_data/test"
NUM_SAMPLES = 1000

# --- ANSI renk kodları (opsiyonel) ---
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# --- Modeli oluştur ve yükle ---
model = VGGCustom(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- Test verisini yükle ---
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])
dataset = CustomTumorDataset(TEST_PATH, transform=transform)
class_names = dataset.classes
test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# --- Tahmin ---
correct = 0
total = 0

with torch.no_grad():
    for img, label in test_loader:
        if total >= NUM_SAMPLES:
            break

        img = img.float()
        output = model(img)
        predicted_label = torch.argmax(output, dim=1).item()
        true_label = label.item()

        is_correct = predicted_label == true_label
        status = f"{GREEN}[CORRECT]{RESET}" if is_correct else f"{RED}[WRONG]{RESET}"

        print(f"{status}  Tahmin: {class_names[predicted_label]:<12}  "
              f"Gerçek: {class_names[true_label]:<12}  (sample {total+1})")

        if is_correct:
            correct += 1
        total += 1

# --- Sonuç ---
accuracy = (correct / total) * 100
print(f"\n🎯 {total} test örneğinden {correct} tanesi doğru tahmin edildi.")
print(f"✅ Doğruluk Oranı: {accuracy:.2f}%")
