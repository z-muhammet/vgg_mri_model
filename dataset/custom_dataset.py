import os
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class CustomTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.samples = []

        self.class_names = ['0_glioma', '1_menin', '2_tumor']
        self.class_to_idx = {name: int(name.split('_')[0]) for name in self.class_names}
        self.classes = self.class_names

        # Temel dönüşümler
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = self.class_to_idx[class_name]

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.npy')):
                    file_path = os.path.join(class_path, file_name)
                    self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        if file_path.lower().endswith('.npy'):
            image = np.load(file_path)
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(file_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label