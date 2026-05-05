from __future__ import annotations

import gc
from typing import Any, Dict, List, Tuple

import lpips
import torch
import torch.nn as nn
import torchattacks
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.attacks.at_spgd import ATSPGD
from src.attacks.ssa import SSA
from src.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from src.models.split_models import FullVFLModel


def _attack_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _metric_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    return _attack_device()


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    if float(images.min()) >= 0.0 and float(images.max()) <= 1.0:
        return images.clamp(0.0, 1.0)

    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def _clean_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
) -> float:
    correct = 0
    total = int(labels.size(0))

    for start in range(0, total, chunk_size):
        end = start + chunk_size
        logits = model(images[start:end])
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels[start:end]).sum().item())

    return correct / max(total, 1)


def _build_attack_suite(
    model: nn.Module,
    eps: float,
    alpha: float,
    steps: int,
    K: float,
) -> Dict[str, Any]:
    return {
        "PGD": torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps),
        "APGD": torchattacks.APGD(model, eps=eps, steps=steps),
        "MIFGSM": torchattacks.MIFGSM(model, eps=eps, steps=steps),
        "SSA": SSA(model, eps=eps, alpha=alpha, steps=steps),
        "AT-SPGD": ATSPGD(model, eps=eps, alpha_f=alpha, alpha_x=alpha, steps=steps, K=K),
    }


def _evaluate_attack(
    attack_name: str,
    attack: Any,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_accuracy: float,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    lpips_metric: nn.Module,
    metric_device: torch.device,
    chunk_size: int,
    eps: float,
    alpha: float,
    steps: int,
) -> Tuple[Dict[str, float | str], torch.Tensor]:
    correct = 0
    total = int(labels.size(0))
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    adversarial_batches: List[torch.Tensor] = []

    for start in range(0, total, chunk_size):
        end = start + chunk_size
        image_chunk = images[start:end]
        label_chunk = labels[start:end]
        batch_size = int(label_chunk.size(0))

        adversarial = attack(image_chunk, label_chunk)
        adversarial_batches.append(adversarial.detach().cpu())

        with torch.no_grad():
            logits = model(adversarial)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == label_chunk).sum().item())

            clean_metric = image_chunk.to(metric_device)
            adversarial_metric = adversarial.to(metric_device)
            psnr_total += float(psnr_metric(adversarial_metric, clean_metric).item()) * batch_size
            ssim_total += float(ssim_metric(adversarial_metric, clean_metric).item()) * batch_size
            lpips_total += (
                float(lpips_metric(adversarial_metric * 2 - 1, clean_metric * 2 - 1).mean().item())
                * batch_size
            )

        del adversarial, image_chunk, label_chunk, clean_metric, adversarial_metric, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        {
            "attack": attack_name,
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "clean_accuracy": clean_accuracy,
            "asr": 1.0 - (correct / max(total, 1)),
            "psnr": psnr_total / max(total, 1),
            "ssim": ssim_total / max(total, 1),
            "lpips": lpips_total / max(total, 1),
        },
        torch.cat(adversarial_batches, dim=0),
    )


def run_attack_arena(
    client: nn.Module,
    server: nn.Module,
    test_batch: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    K: float = 0.1,
    chunk_size: int = 8,
) -> Tuple[List[Dict[str, float | str]], Dict[str, torch.Tensor]]:
    attack_device = _attack_device()
    metric_device = _metric_device()

    full_model = FullVFLModel(client, server, normalize_inputs=True).to(attack_device).eval()
    images = _denormalize(test_batch.detach().cpu()).to(attack_device)
    targets = labels.detach().long().to(attack_device)

    clean_accuracy = _clean_accuracy(full_model, images, targets, chunk_size)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(metric_device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(metric_device)
    lpips_metric = lpips.LPIPS(net="vgg").to(metric_device).eval()
    attack_suite = _build_attack_suite(full_model, eps, alpha, steps, K)

    rows: List[Dict[str, float | str]] = []
    adversarial_tensors: Dict[str, torch.Tensor] = {
        "clean": images.detach().cpu(),
        "labels": targets.detach().cpu(),
    }
    for attack_name, attack in attack_suite.items():
        row, adversarial = _evaluate_attack(
            attack_name=attack_name,
            attack=attack,
            model=full_model,
            images=images,
            labels=targets,
            clean_accuracy=clean_accuracy,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            lpips_metric=lpips_metric,
            metric_device=metric_device,
            chunk_size=chunk_size,
            eps=eps,
            alpha=alpha,
            steps=steps,
        )
        rows.append(row)
        adversarial_tensors[attack_name] = adversarial

    del full_model, images, targets, psnr_metric, ssim_metric, lpips_metric, attack_suite
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, adversarial_tensors
