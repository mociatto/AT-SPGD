# %%
!git clone https://github.com/mociatto/AT-SPGD.git

# %%
%cd AT-SPGD

# %%
!pip install -q matplotlib torchattacks lpips torchmetrics

# %%
from __future__ import annotations

# %%
import io
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchattacks
import torchvision.transforms.functional as TF
from IPython.display import Image as IPythonImage
from IPython.display import display
from matplotlib.figure import Figure
from PIL import Image

from src.attacks.at_spgd import ATSPGD
from src.attacks.ssa import SSA
from src.data.datasets import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from src.models.split_models import EMB_DIM, FullVFLModel, ImageClient, VFLServer
from src.visualization.plots import (
    GAUSSIAN_BLUR_CONFIG,
    GRADCAM_CONTOUR_CONFIG,
    JPEG_COMPRESSION_CONFIG,
    MAGNIFIED_NOISE_CONFIG,
    PARETO_FRONTIER_CONFIG,
    RADIAL_ENERGY_CONFIG,
    plot_average_radial_energy_comparison,
    plot_average_radial_energy_panel,
    plot_gaussian_blur_comparison,
    plot_jpeg_compression_comparison,
    plot_jpeg_compression_panel,
    plot_noise_gradcam_sample,
    plot_pareto_frontier_comparison,
    plot_pareto_frontier_panel,
)

# ==========================================
# 1. CORE SETUP & DYNAMIC ARTIFACT GENERATOR
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("/kaggle/input/notebooks/mostafaanoosha/at-spgd-01-training/AT-SPGD/checkpoints")
FIGURE_DIR = Path.cwd() / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    if float(images.min()) >= 0.0 and float(images.max()) <= 1.0:
        return images.clamp(0.0, 1.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_vfl_model(dataset_name: str, model_name: str) -> FullVFLModel:
    checkpoint = load_checkpoint(CHECKPOINT_DIR / f"01_baseline_{dataset_name}_{model_name}.pth")
    num_classes = int(checkpoint["num_classes"])
    client = ImageClient(model_name=model_name, dim=EMB_DIM)
    server = VFLServer(emb_dim=EMB_DIM, num_classes=num_classes)
    client.load_state_dict(checkpoint["image_client"])
    server.load_state_dict(checkpoint["vfl_server"])
    return FullVFLModel(client, server, normalize_inputs=True).eval().to(device)


@torch.no_grad()
def get_clean_batch(dataset_name: str, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _, test_loader, _ = get_dataloaders(dataset_name=dataset_name, batch_size=num_samples, num_workers=2)
    images, labels = next(iter(test_loader))
    return _denormalize(images[:num_samples]), labels[:num_samples]


def generate_artifacts(dataset_name: str, model_name: str, num_samples: int) -> dict:
    model = build_vfl_model(dataset_name, model_name)
    images, labels = get_clean_batch(dataset_name, num_samples)
    images, labels = images.to(device), labels.to(device)

    eps, alpha, steps = 8.0 / 255.0, 2.0 / 255.0, 10
    attacks = {
        "PGD": torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps),
        "APGD": torchattacks.APGD(model, eps=eps, steps=steps),
        "MIFGSM": torchattacks.MIFGSM(model, eps=eps, steps=steps),
        "SSA": SSA(model, eps=eps, alpha=alpha, steps=steps),
        "AT-SPGD": ATSPGD(model, eps=eps, alpha_f=alpha, alpha_x=alpha, steps=steps, K=0.1),
    }

    vis_dict = {"clean_images": images.cpu(), "labels": labels.cpu()}
    for attack_name, attack in attacks.items():
        vis_dict[f"adv_{attack_name}"] = attack(images, labels).cpu()

    del model, images, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vis_dict


def display_preview(fig: Figure, config: dict) -> None:
    preview_buffer = io.BytesIO()
    fig.savefig(preview_buffer, format="png", bbox_inches="tight", dpi=config.get("figure_dpi", 100))
    preview_buffer.seek(0)
    display(IPythonImage(data=preview_buffer.getvalue(), width=config.get("display_width_px", 800)))


# %%
# ==========================================
# 2. NOISE MAGNIFICATION & GRAD-CAM
# ==========================================
VIS_DATASET = "gtsrb"
VIS_MODEL = "resnet18"
NUM_VIS_SAMPLES = int(MAGNIFIED_NOISE_CONFIG.get("num_samples", 5))

print(f"Generating visual artifacts for {VIS_DATASET} / {VIS_MODEL}...")
vis_dict = generate_artifacts(VIS_DATASET, VIS_MODEL, NUM_VIS_SAMPLES)
full_model = build_vfl_model(VIS_DATASET, VIS_MODEL)

for i in range(NUM_VIS_SAMPLES):
    fig_sample = plot_noise_gradcam_sample(full_model, VIS_MODEL, vis_dict, sample_idx=i)
    fig_sample.savefig(
        FIGURE_DIR / f"03_noise_gradcam_sample_{i}.pdf",
        bbox_inches="tight",
        dpi=max(MAGNIFIED_NOISE_CONFIG.get("save_dpi", 300), GRADCAM_CONTOUR_CONFIG.get("save_dpi", 300)),
    )
    display_preview(fig_sample, MAGNIFIED_NOISE_CONFIG)
    plt.close(fig_sample)

del full_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %%
# ==========================================
# 3. RADIAL ENERGY PROFILES
# ==========================================
ENERGY_DATASETS = RADIAL_ENERGY_CONFIG["datasets"]
ENERGY_MODEL_PAIRS = RADIAL_ENERGY_CONFIG["model_pairs"]
ENERGY_MODEL_LABELS = RADIAL_ENERGY_CONFIG["model_labels"]
ENERGY_DATASET_LABELS = RADIAL_ENERGY_CONFIG["dataset_labels"]
ENERGY_SAMPLES = int(RADIAL_ENERGY_CONFIG.get("samples_per_case", 16))


def radial_panel_label(dataset_name: str, model_name: str) -> str:
    model_label = ENERGY_MODEL_LABELS.get(model_name, model_name)
    dataset_label = ENERGY_DATASET_LABELS.get(dataset_name, dataset_name.upper())
    return f"{model_label} | {dataset_label}"


print(f"Generating Radial Energy rows ({ENERGY_SAMPLES} samples per model/dataset)...")

for dataset_name in ENERGY_DATASETS:
    for row_idx, (cnn_model, transformer_model) in enumerate(ENERGY_MODEL_PAIRS, start=1):
        row_panels = {
            radial_panel_label(dataset_name, cnn_model): [
                generate_artifacts(dataset_name, cnn_model, num_samples=ENERGY_SAMPLES)
            ],
            radial_panel_label(dataset_name, transformer_model): [
                generate_artifacts(dataset_name, transformer_model, num_samples=ENERGY_SAMPLES)
            ],
        }
        fig_energy = plot_average_radial_energy_comparison(row_panels)
        fig_energy.savefig(
            FIGURE_DIR / f"03_radial_energy_{dataset_name}_row_{row_idx}_{cnn_model}_{transformer_model}.pdf",
            bbox_inches="tight",
            dpi=RADIAL_ENERGY_CONFIG.get("save_dpi", 300),
        )
        display_preview(fig_energy, RADIAL_ENERGY_CONFIG)
        plt.close(fig_energy)


# %%
# ==========================================
# 4. PARETO FRONTIER
# ==========================================
PARETO_SAMPLES = 64
PARETO_PANEL_LABELS = PARETO_FRONTIER_CONFIG["panel_labels"]
PARETO_CASES = {
    PARETO_PANEL_LABELS["cnn"]: {
        "panel_key": "cnn",
        "dataset": PARETO_FRONTIER_CONFIG["cnn_dataset"],
        "model": PARETO_FRONTIER_CONFIG["cnn_model"],
    },
    PARETO_PANEL_LABELS["transformer"]: {
        "panel_key": "transformer",
        "dataset": PARETO_FRONTIER_CONFIG["transformer_dataset"],
        "model": PARETO_FRONTIER_CONFIG["transformer_model"],
    },
}


def run_pareto_sweep(dataset_name: str, model_name: str) -> list[dict]:
    print(f"Running Pareto Grid Search for {dataset_name} / {model_name}...")
    model = build_vfl_model(dataset_name, model_name)
    images, labels = get_clean_batch(dataset_name, PARETO_SAMPLES)
    images, labels = images.to(device), labels.to(device)

    results = []
    epsilon = float(PARETO_FRONTIER_CONFIG["epsilon"])
    for k in PARETO_FRONTIER_CONFIG["k_ratios"]:
        for alpha_multiplier in PARETO_FRONTIER_CONFIG["alpha_multipliers"]:
            for steps in PARETO_FRONTIER_CONFIG["steps_sweep"]:
                attack = ATSPGD(
                    model=model,
                    eps=epsilon,
                    alpha_f=(epsilon / steps) * alpha_multiplier,
                    steps=steps,
                    K=k,
                ).eval()
                adversarial = attack(images, labels)

                with torch.no_grad():
                    logits = model(adversarial)
                    asr = (logits.argmax(dim=1) != labels).float().mean().item() * 100.0
                    mse = torch.mean((adversarial - images) ** 2, dim=[1, 2, 3])
                    psnr = (20 * torch.log10(1.0 / torch.sqrt(mse))).mean().item()

                results.append({"k": k, "alpha": alpha_multiplier, "steps": steps, "asr": asr, "psnr": psnr})
                del attack, adversarial, logits

    del model, images, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


pareto_results = {
    panel_label: run_pareto_sweep(case["dataset"], case["model"]) for panel_label, case in PARETO_CASES.items()
}

fig_pareto = plot_pareto_frontier_comparison(pareto_results)
fig_pareto.savefig(
    FIGURE_DIR / "03_pareto_frontier.pdf",
    bbox_inches="tight",
    dpi=PARETO_FRONTIER_CONFIG.get("save_dpi", 300),
)
display_preview(fig_pareto, PARETO_FRONTIER_CONFIG)
plt.close(fig_pareto)

for panel_label, case in PARETO_CASES.items():
    fig_panel = plot_pareto_frontier_panel(pareto_results[panel_label], panel_label=panel_label)
    fig_panel.savefig(
        FIGURE_DIR / f"03_pareto_frontier_{case['panel_key']}_{case['model']}.pdf",
        bbox_inches="tight",
        dpi=PARETO_FRONTIER_CONFIG.get("save_dpi", 300),
    )
    plt.close(fig_panel)


# %%
# ==========================================
# 5. JPEG COMPRESSION
# ==========================================
JPEG_DATASETS = JPEG_COMPRESSION_CONFIG["datasets"]
JPEG_MODELS = JPEG_COMPRESSION_CONFIG["models"]
JPEG_QUALITIES = JPEG_COMPRESSION_CONFIG["jpeg_qualities"]


def apply_jpeg_batch(images: torch.Tensor, quality: int) -> torch.Tensor:
    compressed_images = []
    for image in images:
        image_np = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np.transpose(1, 2, 0))
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_np = np.asarray(Image.open(buffer)).astype(np.float32) / 255.0
        compressed_images.append(torch.from_numpy(compressed_np.transpose(2, 0, 1)))
    return torch.stack(compressed_images).to(device)


def compute_jpeg_asr_curve(dataset_name: str, model_name: str) -> list[float]:
    print(f"Running JPEG Compression for {dataset_name} / {model_name}...")
    model = build_vfl_model(dataset_name, model_name)
    artifact = generate_artifacts(dataset_name, model_name, num_samples=32)

    attack_key = next((key for key in ("adv_AT-SPGD", "adv_ATSPGD", "adv_Adaptive") if key in artifact), None)
    if attack_key is None:
        raise KeyError("No AT-SPGD adversarial tensor found in generated artifact.")

    adversarial_images = artifact[attack_key].to(device)
    labels = artifact["labels"].to(device)

    asr_curve = []
    for quality in JPEG_QUALITIES:
        compressed = apply_jpeg_batch(adversarial_images, quality)
        with torch.no_grad():
            logits = model(compressed)
            asr = (logits.argmax(dim=1) != labels).float().mean().item() * 100.0
        asr_curve.append(asr)
        del compressed, logits

    del model, artifact, adversarial_images, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return asr_curve


jpeg_curves = {
    dataset_name: {model_name: compute_jpeg_asr_curve(dataset_name, model_name) for model_name in JPEG_MODELS}
    for dataset_name in JPEG_DATASETS
}

fig_jpeg = plot_jpeg_compression_comparison(jpeg_curves)
fig_jpeg.savefig(
    FIGURE_DIR / "03_jpeg_compression.pdf",
    bbox_inches="tight",
    dpi=JPEG_COMPRESSION_CONFIG.get("save_dpi", 300),
)
display_preview(fig_jpeg, {**JPEG_COMPRESSION_CONFIG, "display_width_px": 800})
plt.close(fig_jpeg)

for dataset_name, model_curves in jpeg_curves.items():
    fig_panel = plot_jpeg_compression_panel(dataset_name, model_curves)
    fig_panel.savefig(
        FIGURE_DIR / f"03_jpeg_compression_{dataset_name}.pdf",
        bbox_inches="tight",
        dpi=JPEG_COMPRESSION_CONFIG.get("save_dpi", 300),
    )
    plt.close(fig_panel)


# %%
# ==========================================
# 6. GAUSSIAN BLUR DEFENSE
# ==========================================
GAUSSIAN_DATASETS = GAUSSIAN_BLUR_CONFIG["datasets"]
GAUSSIAN_MODEL_GROUPS = GAUSSIAN_BLUR_CONFIG["model_groups"]
GAUSSIAN_ATTACK_ORDER = GAUSSIAN_BLUR_CONFIG["attack_order"]
GAUSSIAN_SAMPLES = int(GAUSSIAN_BLUR_CONFIG["num_samples"])
GAUSSIAN_KERNEL_SIZE = int(GAUSSIAN_BLUR_CONFIG["blur_kernel_size"])
GAUSSIAN_SIGMA = float(GAUSSIAN_BLUR_CONFIG["blur_sigma"])

if GAUSSIAN_KERNEL_SIZE % 2 == 0:
    raise ValueError("GAUSSIAN_BLUR_CONFIG['blur_kernel_size'] must be an odd integer.")


def resolve_attack_tensor_key(artifact: dict, attack_name: str) -> str:
    attack_keys = {
        "AT-SPGD (Ours)": ("adv_AT-SPGD", "adv_ATSPGD", "adv_Adaptive"),
        "AT-SPGD": ("adv_AT-SPGD", "adv_ATSPGD", "adv_Adaptive"),
    }.get(attack_name, (f"adv_{attack_name}",))
    return next(key for key in attack_keys if key in artifact)


def apply_gaussian_blur(images: torch.Tensor) -> torch.Tensor:
    return TF.gaussian_blur(
        images,
        kernel_size=[GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE],
        sigma=[GAUSSIAN_SIGMA, GAUSSIAN_SIGMA],
    )


def compute_gaussian_blur_panel(model_names: list[str]) -> dict[str, dict[str, float]]:
    accumulated = {
        attack_name: {"no_defense": [], "gaussian_blur": []}
        for attack_name in GAUSSIAN_ATTACK_ORDER
    }

    for dataset_name in GAUSSIAN_DATASETS:
        for model_name in model_names:
            print(f"Running Gaussian Blur Defense for {dataset_name} / {model_name}...")
            model = build_vfl_model(dataset_name, model_name)
            artifact = generate_artifacts(dataset_name, model_name, num_samples=GAUSSIAN_SAMPLES)
            labels = artifact["labels"].to(device)

            for attack_name in GAUSSIAN_ATTACK_ORDER:
                adversarial = artifact[resolve_attack_tensor_key(artifact, attack_name)].to(device)
                blurred = apply_gaussian_blur(adversarial)
                with torch.no_grad():
                    no_defense_logits = model(adversarial)
                    blur_logits = model(blurred)
                    no_defense_asr = (no_defense_logits.argmax(dim=1) != labels).float().mean().item() * 100.0
                    blur_asr = (blur_logits.argmax(dim=1) != labels).float().mean().item() * 100.0
                accumulated[attack_name]["no_defense"].append(no_defense_asr)
                accumulated[attack_name]["gaussian_blur"].append(blur_asr)
                del adversarial, blurred, no_defense_logits, blur_logits

            del model, artifact, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        attack_name: {
            metric_name: float(np.mean(values))
            for metric_name, values in metric_values.items()
        }
        for attack_name, metric_values in accumulated.items()
    }


gaussian_blur_metrics = {
    panel_title: compute_gaussian_blur_panel(model_names)
    for panel_title, model_names in GAUSSIAN_MODEL_GROUPS.items()
}

fig_gaussian = plot_gaussian_blur_comparison(gaussian_blur_metrics)
fig_gaussian.savefig(
    FIGURE_DIR / "03_gaussian_blur_defense.pdf",
    bbox_inches="tight",
    dpi=GAUSSIAN_BLUR_CONFIG.get("save_dpi", 300),
)
display_preview(fig_gaussian, GAUSSIAN_BLUR_CONFIG)
plt.close(fig_gaussian)
