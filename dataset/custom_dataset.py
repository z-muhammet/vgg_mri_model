import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict
import random

class CustomTumorDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        *,
        to_rgb: bool = False,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        cache_size: int = 128,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.to_rgb = to_rgb
        self.transform = transform
        self.target_size = target_size
        self.cache_size = cache_size
        self.cache: Dict[str, torch.Tensor] = {}

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

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.target_size, antialias=True),
        ])
        self.normalize = transforms.Normalize(mean=[0.5] * (3 if to_rgb else 1), std=[0.5] * (3 if to_rgb else 1))

    def _load_slice(self, path: str) -> torch.Tensor:
        if path in self.cache:
            return self.cache[path]
        arr = np.load(path).astype(np.float32)

        # Log shape, dtype, and path of the numpy array right before Image.fromarray
        # print(f"[DatasetLog] Before Image.fromarray ({self.root_dir}): path={path}, shape={arr.shape}, dtype={arr.dtype}")

        # Ensure data is in 0-255 range and convert to uint8 for PIL
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if self.to_rgb:
            if arr_uint8.ndim == 2:
                 # If 2D, convert to RGB PIL Image
                 img = Image.fromarray(arr_uint8).convert('RGB')
            elif arr_uint8.ndim == 3:
                 # If 3D, assume it's already RGB-like and create PIL Image
                 # PIL expects shape (height, width, channels) for 3D uint8 arrays
                 img = Image.fromarray(arr_uint8)
            else:
                 raise ValueError(f".npy dosyasında beklenmeyen boyut (uint8 sonrası): {arr_uint8.ndim}")
        else:
            # Grayscale processing
            if arr_uint8.ndim == 3:
                # If 3D, take the first channel for grayscale
                arr_uint8 = arr_uint8[..., 0]
            elif arr_uint8.ndim != 2:
                 raise ValueError(f".npy dosyasında beklenmeyen boyut (uint8 sonrası): {arr_uint8.ndim}")
            # Convert to L mode (grayscale) PIL Image
            img = Image.fromarray(arr_uint8).convert('L')

        tensor = self.base_transform(img)
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = tensor
        return tensor

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        tensor = self._load_slice(path)

        if self.transform:
            tensor = self.transform(tensor)
        tensor = self.normalize(tensor)

        return tensor, label

    def __len__(self):
        return len(self.samples)

    def clear_cache(self):
        self.cache.clear()