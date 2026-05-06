from __future__ import annotations

import gc
import warnings
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
def _clean_predictions(
    model: nn.Module,
    images: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    predictions: List[torch.Tensor] = []
    total = int(images.size(0))

    for start in range(0, total, chunk_size):
        end = start + chunk_size
        logits = model(images[start:end])
        predictions.append(logits.argmax(dim=1).detach())

    return torch.cat(predictions, dim=0)


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
    clean_correct_mask: torch.Tensor,
    clean_correct_count: int,
    clean_accuracy_on_attack_batch: float,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    lpips_metric: nn.Module,
    metric_device: torch.device,
    chunk_size: int,
    eps: float,
    alpha: float,
    steps: int,
) -> Tuple[Dict[str, float | int | str], torch.Tensor]:
    adv_correct_count = 0
    total = int(labels.size(0))
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    psnr_clean_correct_total = 0.0
    ssim_clean_correct_total = 0.0
    lpips_clean_correct_total = 0.0
    attack_success_count = 0
    adversarial_batches: List[torch.Tensor] = []

    for start in range(0, total, chunk_size):
        end = start + chunk_size
        image_chunk = images[start:end]
        label_chunk = labels[start:end]
        mask_chunk = clean_correct_mask[start:end]
        batch_size = int(label_chunk.size(0))
        clean_correct_chunk_count = int(mask_chunk.sum().item())

        adversarial = attack(image_chunk, label_chunk)
        adversarial_batches.append(adversarial.detach().cpu())

        with torch.no_grad():
            logits = model(adversarial)
            predictions = logits.argmax(dim=1)
            adv_correct_count += int((predictions == label_chunk).sum().item())
            attack_success_count += int((mask_chunk & (predictions != label_chunk)).sum().item())

            clean_metric = image_chunk.to(metric_device)
            adversarial_metric = adversarial.to(metric_device)
            psnr_total += float(psnr_metric(adversarial_metric, clean_metric).item()) * batch_size
            ssim_total += float(ssim_metric(adversarial_metric, clean_metric).item()) * batch_size
            lpips_total += (
                float(lpips_metric(adversarial_metric * 2 - 1, clean_metric * 2 - 1).mean().item())
                * batch_size
            )
            if clean_correct_chunk_count > 0:
                metric_mask = mask_chunk.to(metric_device)
                clean_correct_metric = clean_metric[metric_mask]
                adversarial_clean_correct_metric = adversarial_metric[metric_mask]
                psnr_clean_correct_total += (
                    float(psnr_metric(adversarial_clean_correct_metric, clean_correct_metric).item())
                    * clean_correct_chunk_count
                )
                ssim_clean_correct_total += (
                    float(ssim_metric(adversarial_clean_correct_metric, clean_correct_metric).item())
                    * clean_correct_chunk_count
                )
                lpips_clean_correct_total += (
                    float(
                        lpips_metric(
                            adversarial_clean_correct_metric * 2 - 1,
                            clean_correct_metric * 2 - 1,
                        ).mean().item()
                    )
                    * clean_correct_chunk_count
                )

        del adversarial, image_chunk, label_chunk, mask_chunk, clean_metric, adversarial_metric, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    asr = float("nan") if clean_correct_count == 0 else attack_success_count / clean_correct_count
    psnr_clean_correct = float("nan") if clean_correct_count == 0 else psnr_clean_correct_total / clean_correct_count
    ssim_clean_correct = float("nan") if clean_correct_count == 0 else ssim_clean_correct_total / clean_correct_count
    lpips_clean_correct = float("nan") if clean_correct_count == 0 else lpips_clean_correct_total / clean_correct_count

    return (
        {
            "attack": attack_name,
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "clean_accuracy": clean_accuracy_on_attack_batch,
            "psnr": psnr_total / max(total, 1),
            "ssim": ssim_total / max(total, 1),
            "lpips": lpips_total / max(total, 1),
            "total_samples": total,
            "clean_correct_count": clean_correct_count,
            "clean_accuracy_on_attack_batch": clean_accuracy_on_attack_batch,
            "adv_correct_count": adv_correct_count,
            "attack_success_count": attack_success_count,
            # ASR follows standard evasion evaluation: only clean-correct samples form the denominator.
            "asr": asr,
            "psnr_clean_correct": psnr_clean_correct,
            "ssim_clean_correct": ssim_clean_correct,
            "lpips_clean_correct": lpips_clean_correct,
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
) -> Tuple[List[Dict[str, float | int | str]], Dict[str, torch.Tensor]]:
    attack_device = _attack_device()
    metric_device = _metric_device()

    full_model = FullVFLModel(client, server, normalize_inputs=True).to(attack_device).eval()
    images = _denormalize(test_batch.detach().cpu()).to(attack_device)
    targets = labels.detach().long().to(attack_device)

    clean_preds = _clean_predictions(full_model, images, chunk_size)
    clean_correct_mask = clean_preds == targets
    clean_correct_count = int(clean_correct_mask.sum().item())
    clean_accuracy_on_attack_batch = clean_correct_count / max(int(targets.size(0)), 1)
    if clean_correct_count == 0:
        warnings.warn(
            "No clean-correct samples found in the attack batch; ASR and clean-correct fidelity metrics are NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(metric_device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(metric_device)
    lpips_metric = lpips.LPIPS(net="vgg").to(metric_device).eval()
    attack_suite = _build_attack_suite(full_model, eps, alpha, steps, K)

    rows: List[Dict[str, float | int | str]] = []
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
            clean_correct_mask=clean_correct_mask,
            clean_correct_count=clean_correct_count,
            clean_accuracy_on_attack_batch=clean_accuracy_on_attack_batch,
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

    del full_model, images, targets, clean_preds, clean_correct_mask, psnr_metric, ssim_metric, lpips_metric, attack_suite
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, adversarial_tensors
