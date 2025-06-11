import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional, Dict
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomTumorDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        *,
        to_rgb: bool = False,
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        cache_size: int = 128,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.to_rgb = to_rgb
        self.transform = transform
        self.target_size = target_size
        self.cache_size = cache_size
        self.cache: Dict[str, np.ndarray] = {}

        self.class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_names.sort(key=lambda n: int(n.split('_')[0]))
        self.class_to_idx = {name: int(name.split('_')[0]) for name in self.class_names}

        self.samples: List[Tuple[str, int]] = []
        for cls in self.class_names:
            cdir = os.path.join(root_dir, cls)
            for f in os.listdir(cdir):
                if f.lower().endswith('.npy'):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[cls]))

        # Örnekleri karıştır (validasyon setinde bile batchlerin homojen olmaması için)
        random.shuffle(self.samples)

        # Define the final post-augmentation transform once
        self.final_post_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1], interpolation=1), # Albumentations default: cv2.INTER_LINEAR
            A.Normalize(mean=(0.5, 0.5, 0.5) if self.to_rgb else (0.5,), std=(0.5, 0.5, 0.5) if self.to_rgb else (0.5,)), # Normalize to [-1, 1] (NumPy output)
            ToTensorV2() # Convert to tensor and [0,1] float as the very last step
        ])

    def _load_array(self, path: str) -> np.ndarray:
        if path in self.cache:
            return self.cache[path]
        arr = np.load(path).astype(np.float32)
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = arr_uint8
        return arr_uint8

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img_np = self._load_array(path) # Renamed arr_uint8 to img_np for clarity

        # Apply Albumentations transform (for augmentations) if provided
        if self.transform:
            augmented = self.transform(image=img_np)
            img_np = augmented['image']

        # Ensure img_np is in correct channel format for Normalize/ToTensorV2
        if self.to_rgb and img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1) # Convert 2D to 3D RGB
        elif not self.to_rgb and img_np.ndim == 3: # If expecting grayscale but got 3D
            img_np = img_np[..., 0] # Take one channel

        # Apply the final post-augmentation transform
        transformed_data = self.final_post_transform(image=img_np)
        tensor = transformed_data['image']

        return tensor, label

    def __len__(self):
        return len(self.samples)

    def clear_cache(self):
        self.cache.clear()