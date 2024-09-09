import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T
from albumentations import Compose
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        partition: Literal["train", "val", "test"],
        labels_path: str | None = None,
        dataset_mean: list[float] = [0.485, 0.456, 0.406],
        dataset_std: list[float] = [0.229, 0.224, 0.225],
        image_size: tuple[int, int] | None = None,
        device: str = "cuda",
        do_augmentation: bool = True,
        augmentations: Compose | None = None,
    ):
        self.image_root = Path(image_root)
        self.images_paths = list(Path(image_root).glob("*"))
        self.labels_path = Path(labels_path)
        with open(self.labels_path, "r") as f:
            self.im_names2labels = json.load(f)

        self.im_names, self.labels = zip(*self.im_names2labels.items())
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.image_size = image_size
        self.device = device
        self.do_augmentation = do_augmentation
        self.partition = partition

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=dataset_mean, std=dataset_std),
            ]
        )
        self.augmentations = augmentations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor]:
        im_name, label = self.im_names[idx], self.labels[idx]
        image = PIL.Image.open(self.image_root / im_name)
        if self.image_size:
            image = image.resize(self.image_size)

        image_array = np.array(image).astype("uint8")
        if self.do_augmentation and self.augmentations:
            image_array = self.augmentations(image=image_array)["image"]
        image_tensor = self.transforms(image_array)
        return image_tensor, label
