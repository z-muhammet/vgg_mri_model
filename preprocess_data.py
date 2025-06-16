import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

def preprocess_images_to_npy(input_root_dir="data", output_root_dir="preprocessed_data", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    class_name_mapping = {
        "brain_glioma": "0_glioma",
        "brain_menin": "1_menin",
        "brain_tumor": "2_tumor" 
    }

    os.makedirs(output_root_dir, exist_ok=True)
    
    all_class_images = {}

    # Tüm orijinal sınıf klasörlerini tara
    for original_class_name in os.listdir(input_root_dir):
        input_class_dir = os.path.join(input_root_dir, original_class_name)
        
        if not os.path.isdir(input_class_dir):
            continue

        if original_class_name not in class_name_mapping:
            print(f"Uyarı: Tanımlanmayan sınıf klasörü '{original_class_name}' bulundu. Atlanıyor.")
            continue

        target_class_name = class_name_mapping[original_class_name]
        all_class_images[target_class_name] = []

        image_files = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            all_class_images[target_class_name].append(os.path.join(input_class_dir, image_file))

    # Her sınıf için verileri train, val, test olarak böl
    for target_class_name, image_paths in all_class_images.items():
        random.shuffle(image_paths)
        total_images = len(image_paths)

        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        # Test count will be the rest to ensure all images are used
        test_count = total_images - train_count - val_count

        train_images = image_paths[:train_count]
        val_images = image_paths[train_count:train_count + val_count]
        test_images = image_paths[train_count + val_count:]

        print(f"Sınıf {target_class_name} için dağıtım: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        # Her bölme için çıktı dizinlerini oluştur ve görüntüleri işle
        for split_name, images_list in {"train": train_images, "val": val_images, "test": test_images}.items():
            output_split_class_dir = os.path.join(output_root_dir, split_name, target_class_name)
            os.makedirs(output_split_class_dir, exist_ok=True)
            
            for image_path in tqdm(images_list, desc=f"Processing {split_name}/{target_class_name}"):
                output_npy_path = os.path.join(output_split_class_dir, os.path.splitext(os.path.basename(image_path))[0] + ".npy")
                
                try:
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_np = np.array(img, dtype=np.float32)
                        np.save(output_npy_path, img_np)
                except Exception as e:
                    print(f"Hata oluştu {image_path}: {e}")

if __name__ == "__main__":
    print("Veri ön işleme başlatılıyor...")
    preprocess_images_to_npy()
    print("Veri ön işleme tamamlandı.")