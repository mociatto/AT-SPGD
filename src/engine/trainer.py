from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import get_dataloaders
from src.models.split_models import EMB_DIM, ImageClient, VFLServer
from src.utils.metrics import evaluate_system


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_data_parallel(module: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(module)
    return module


def unwrap_parallel(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def module_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return unwrap_parallel(module).state_dict()


def trainable_parameters(*modules: nn.Module) -> Iterable[nn.Parameter]:
    for module in modules:
        for parameter in unwrap_parallel(module).parameters():
            if parameter.requires_grad:
                yield parameter


def train_one_epoch(
    image_client: nn.Module,
    vfl_server: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    image_client.train()
    vfl_server.train()

    total_loss = 0.0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = vfl_server(image_client(images))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_vfl_system(
    image_client: nn.Module,
    vfl_server: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module, List[float]]:
    train_device = device or default_device()
    image_client = maybe_data_parallel(image_client.to(train_device))
    vfl_server = maybe_data_parallel(vfl_server.to(train_device))

    optimizer = torch.optim.Adam(
        trainable_parameters(image_client, vfl_server),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    losses = [
        train_one_epoch(
            image_client,
            vfl_server,
            train_loader,
            criterion,
            optimizer,
            train_device,
        )
        for _ in range(epochs)
    ]
    return image_client, vfl_server, losses


def evaluate_vfl_system(
    image_client: nn.Module,
    vfl_server: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    return evaluate_system(image_client, vfl_server, data_loader, num_classes, device)


def save_checkpoint(
    checkpoint_path: Path,
    image_client: nn.Module,
    vfl_server: nn.Module,
    dataset_name: str,
    model_name: str,
    fold_index: int,
    num_classes: int,
    metrics: Dict[str, float],
    attack_test_data: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_client": module_state_dict(image_client),
            "vfl_server": module_state_dict(vfl_server),
            "dataset": dataset_name,
            "model": model_name,
            "fold": fold_index,
            "num_classes": num_classes,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    if attack_test_data is not None:
        transfer_path = checkpoint_path.parent / f"01_transfer_{dataset_name}_{model_name}.pt"
        torch.save(
            {
                "test_batch": attack_test_data["images"].detach().cpu(),
                "labels": attack_test_data["labels"].detach().cpu(),
                "clean_metrics": metrics,
            },
            transfer_path,
        )


def collect_attack_test_data(data_loader: DataLoader, max_samples: int = 512) -> Dict[str, torch.Tensor]:
    image_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []
    total_samples = 0

    for images, labels in data_loader:
        remaining = max_samples - total_samples
        if remaining <= 0:
            break
        image_batches.append(images[:remaining].detach().cpu())
        label_batches.append(labels[:remaining].detach().cpu())
        total_samples += min(int(images.size(0)), remaining)

    return {
        "images": torch.cat(image_batches, dim=0),
        "labels": torch.cat(label_batches, dim=0),
    }


def cleanup_fold() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_standard_training(
    dataset_name: str,
    model_name: str,
    data_root: Path,
    checkpoint_dir: Path,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    emb_dim: int = EMB_DIM,
    num_workers: int = 4,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    run_device = device or default_device()
    best_checkpoint = checkpoint_dir / f"01_baseline_{dataset_name}_{model_name}.pth"
    transfer_path = checkpoint_dir / f"01_transfer_{dataset_name}_{model_name}.pt"

    train_loader, val_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed,
    )

    image_client = ImageClient(model_name=model_name, dim=emb_dim)
    vfl_server = VFLServer(emb_dim=emb_dim, num_classes=num_classes)

    image_client, vfl_server, _ = train_vfl_system(
        image_client=image_client,
        vfl_server=vfl_server,
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        device=run_device,
    )

    metrics = evaluate_vfl_system(
        image_client=image_client,
        vfl_server=vfl_server,
        data_loader=val_loader,
        num_classes=num_classes,
        device=run_device,
    )
    attack_test_data = collect_attack_test_data(val_loader, max_samples=512)

    save_checkpoint(
        checkpoint_path=best_checkpoint,
        image_client=image_client,
        vfl_server=vfl_server,
        dataset_name=dataset_name,
        model_name=model_name,
        fold_index=0,
        num_classes=num_classes,
        metrics=metrics,
        attack_test_data=attack_test_data,
    )

    del image_client, vfl_server, train_loader, val_loader, attack_test_data
    cleanup_fold()

    return {
        "dataset": dataset_name,
        "model": model_name,
        "num_classes": num_classes,
        "best_checkpoint": str(best_checkpoint),
        "transfer_file": str(transfer_path),
        **metrics,
    }
