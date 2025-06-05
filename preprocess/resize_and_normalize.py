import os
import numpy as np
from PIL import Image
from datetime import datetime

SOURCE_ROOT = "dataset"
DEST_ROOT = "preprocessed_data"
TARGET_SIZE = (256, 256)
LOG_PATH = "logs/resize_and_normalize.log"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

def process_and_save_image(src_path, dest_path):
    try:
        image = Image.open(src_path).convert("RGB")
        image = image.resize(TARGET_SIZE)
        image_array = np.asarray(image).astype(np.float32) / 255.0
        np.save(dest_path, image_array)
        log(f"[OK] {src_path} → {dest_path}")
    except Exception as e:
        log(f"[ERROR] {src_path} → {e}")

def process_folder(split):
    source_dir = os.path.join(SOURCE_ROOT, split)
    dest_dir = os.path.join(DEST_ROOT, split)

    for class_name in os.listdir(source_dir):
        src_class_dir = os.path.join(source_dir, class_name)
        dst_class_dir = os.path.join(dest_dir, class_name)
        ensure_dir(dst_class_dir)

        for filename in os.listdir(src_class_dir):
            if filename.lower().endswith(".jpg"):
                src_file = os.path.join(src_class_dir, filename)
                dst_file = os.path.join(dst_class_dir, filename.replace(".jpg", ".npy"))
                process_and_save_image(src_file, dst_file)

if __name__ == "__main__":
    ensure_dir("logs")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("=== Resize & Normalize Log Başladı ===\n")
    for split in ["train", "val", "test"]:
        process_folder(split)
    log("=== Tüm işlemler tamamlandı ===")
