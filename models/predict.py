#python predict.py --image_path ./samples/test001.jpg
import argparse
import torch
import numpy as np
from PIL import Image
from models.vgg_custom import VGGCustom

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glioma", "meningioma", "unknown"]

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    image = np.asarray(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
    return image.to(DEVICE)

def predict(image_path):
    # Modeli hazırla
    model = VGGCustom(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("models/vgg_custom_trained.pt", map_location=DEVICE))
    model.eval()

    # Görseli yükle
    image_tensor = load_and_preprocess_image(image_path)

    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)

    predicted_class = CLASS_NAMES[pred.item()]
    print(f"[✓] Tahmin: {predicted_class} ({image_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Test edilecek .jpg görsel yolu")
    args = parser.parse_args()

    predict(args.image_path)
