from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_NUM_CLASSES: Dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "gtsrb": 43,
}


def default_data_root() -> Path:
    return Path.cwd() / "data"


def image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image_dataset(
    dataset_name: str,
    split: str,
    data_root: Optional[Path] = None,
    image_size: int = 224,
) -> Tuple[Dataset, int]:
    name = dataset_name.lower()
    root = Path(data_root) if data_root is not None else default_data_root()
    transform = image_transform(image_size=image_size)

    if name == "cifar10":
        dataset = datasets.CIFAR10(
            root=root,
            train=split == "train",
            download=True,
            transform=transform,
        )
    elif name == "cifar100":
        dataset = datasets.CIFAR100(
            root=root,
            train=split == "train",
            download=True,
            transform=transform,
        )
    elif name == "svhn":
        dataset = datasets.SVHN(
            root=root,
            split=split,
            download=True,
            transform=transform,
        )
    elif name == "gtsrb":
        dataset = datasets.GTSRB(
            root=root,
            split=split,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. "
            "Expected one of: cifar10, cifar100, svhn, gtsrb."
        )

    return dataset, DATASET_NUM_CLASSES[name]


def dataset_targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        samples = getattr(dataset, "samples", None) or getattr(dataset, "_samples", None)
        if samples is not None:
            targets = [int(sample[1]) for sample in samples]
    if targets is None:
        raise AttributeError("Unable to infer dataset targets for stratified folds.")
    return np.asarray(targets, dtype=np.int64)


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    data_root: Optional[Path] = None,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    train_dataset, num_classes = load_image_dataset(dataset_name, "train", data_root)
    test_dataset, _ = load_image_dataset(dataset_name, "test", data_root)
    train_loader = make_data_loader(train_dataset, batch_size, True, num_workers, seed)
    test_loader = make_data_loader(test_dataset, batch_size, False, num_workers, seed)
    return train_loader, test_loader, num_classes


def get_fold_dataloaders(
    dataset_name: str,
    fold_index: int,
    k_folds: int,
    batch_size: int = 128,
    data_root: Optional[Path] = None,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset, num_classes = load_image_dataset(dataset_name, "train", data_root)
    targets = dataset_targets(dataset)
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(targets)), targets))
    train_indices, val_indices = splits[fold_index]

    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    train_loader = make_data_loader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed + fold_index,
    )
    val_loader = make_data_loader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + fold_index,
    )
    return train_loader, val_loader, num_classes
