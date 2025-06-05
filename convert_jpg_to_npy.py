import os
import numpy as np
from PIL import Image

test_root = "dataset/test"
class_names = ["0_glioma", "1_menin", "2_tumor"]

for cname in class_names:
    class_dir = os.path.join(test_root, cname)
    if not os.path.isdir(class_dir):
        print(f"UYARI: {class_dir} yok, atlanıyor.")
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(".jpg"):
            jpg_path = os.path.join(class_dir, fname)
            npy_path = os.path.splitext(jpg_path)[0] + ".npy"
            img = Image.open(jpg_path).convert("RGB")
            arr = np.asarray(img).astype(np.float32) / 255.0
            np.save(npy_path, arr)
            print(f"{jpg_path} -> {npy_path}") 