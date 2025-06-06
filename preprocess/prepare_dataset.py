import os
import shutil
import random

# Etiketler
LABEL_MAP = {
    "brain_glioma": "0_glioma",
    "brain_menin": "1_menin",
    "brain_tumor": "2_tumor"
}
print(os.listdir("data"))


# Klasör yolları
SOURCE_DIR = "data"

DEST_DIR = "dataset"

# Sabit seed için
random.seed(42)

def split_and_copy(class_name, train_ratio=0.7, val_ratio=0.15):
    src_folder = os.path.join(SOURCE_DIR, class_name)
    label_folder = LABEL_MAP[class_name]

    # Dosyaları al
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(".jpg")]
    random.shuffle(files)

    total = len(files)
    train_split = int(train_ratio * total)
    val_split = int(val_ratio * total)

    train_files = files[:train_split]
    val_files = files[train_split:train_split + val_split]
    test_files = files[train_split + val_split:]

    # Hedefe kopyala
    for fname in train_files:
        shutil.copy(
            os.path.join(src_folder, fname),
            os.path.join(DEST_DIR, "train", label_folder, fname)
        )

    for fname in val_files:
        shutil.copy(
            os.path.join(src_folder, fname),
            os.path.join(DEST_DIR, "val", label_folder, fname)
        )

    for fname in test_files:
        shutil.copy(
            os.path.join(src_folder, fname),
            os.path.join(DEST_DIR, "test", label_folder, fname)
        )

    print(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


if __name__ == "__main__":
    for cls in LABEL_MAP:
        split_and_copy(cls)
