import os
import torch
from torchvision import transforms
from PIL import Image
from models.vgg_custom import VGGCustom  # kendi modelin
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Model ve test verisi yolları
MODEL_PATH = "models/best_vgg_custom.pt"
TEST_DIR = "data/test"

# Etiket çıkarıcı
def get_label_from_filename(name: str):
    if "glioma" in name:
        return 0
    elif "menin" in name:
        return 1
    elif "tumor" in name:
        return 2
    else:
        return -1

# Görüntü dönüştürücü
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Model yükle
model = VGGCustom(num_classes=3)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Test verisi tahmin
y_true, y_pred = [], []
with torch.no_grad():
    for filename in os.listdir(TEST_DIR):
        if not filename.endswith(".jpg"):
            continue
        label = get_label_from_filename(filename)
        if label == -1:
            continue

        img_path = os.path.join(TEST_DIR, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

        output = model(input_tensor)
        pred = output.argmax(1).item()

        y_true.append(label)
        y_pred.append(pred)

# Doğruluk hesapla
acc = accuracy_score(y_true, y_pred)
print(f"[TEST ACCURACY] {acc:.4f}  ({len(y_true)} örnek üzerinde)")
