import torch
from torch.utils.data import DataLoader
from dataset.custom_dataset import CustomTumorDataset
from models.vgg_custom import VGGCustom
import torchvision.transforms.v2 as T
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGCustom(num_classes=3).to(device)
    checkpoint = torch.load('models/best_vgg_custom.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    test_dir = 'dataset/test'
    class_names = ['0_glioma', '1_menin', '2_tumor']

    # Klasör kontrolü
    for cname in class_names:
        if not os.path.isdir(os.path.join(test_dir, cname)):
            print(f"UYARI: '{cname}' klasörü test dizininde yok!")
    test_dataset = CustomTumorDataset(root_dir=test_dir, transform=test_transform, is_training=False)
    print(f"Test örnek sayısı: {len(test_dataset)}")
    if len(test_dataset) == 0:
        print("Test veri seti boş! Lütfen dataset/test/ klasörünü ve alt klasörleri kontrol et.")
        return

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    print(f"Toplam batch: {len(test_loader)}")

    all_preds = []
    all_labels = []

    for i, (images, labels) in enumerate(test_loader):
        print(f"[Batch {i+1}/{len(test_loader)}] İşleniyor...")
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        print(f"  - Bu batchteki label dağılımı: {Counter(labels.numpy())}")
        print(f"  - Bu batchteki tahmin dağılımı: {Counter(preds.cpu().numpy())}")
        if i == 0:
            print(f"  - İlk batch gerçek label'lar: {labels.numpy()}")
            print(f"  - İlk batch tahminler: {preds.cpu().numpy()}")
        torch.cuda.empty_cache()

    print(f"\nToplam örnek: {len(all_labels)}")
    print(f"Toplam batch: {len(test_loader)}")
    print("Tüm label dağılımı:", Counter(all_labels))
    print("Tüm tahmin dağılımı:", Counter(all_preds))

    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds, 
        labels=[0,1,2], 
        target_names=['glioma', 'meningioma', 'tumor']
    ))

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['glioma', 'meningioma', 'Tumor'],
                yticklabels=['glioma', 'meningioma', 'tumor'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_model() 