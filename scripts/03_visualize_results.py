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
from pathlib import Path
import sys


PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image as IPythonImage
from IPython.display import display
from PIL import Image

from src.attacks.at_spgd import ATSPGD
from src.models.split_models import EMB_DIM, FullVFLModel, ImageClient, VFLServer
from src.visualization.plots import (
    GRADCAM_CONTOUR_CONFIG,
    JPEG_COMPRESSION_CONFIG,
    MAGNIFIED_NOISE_CONFIG,
    PARETO_FRONTIER_CONFIG,
    RADIAL_ENERGY_CONFIG,
    plot_average_radial_energy_comparison,
    plot_average_radial_energy_panel,
    plot_gradcam_contour_row,
    plot_jpeg_compression_comparison,
    plot_jpeg_compression_panel,
    plot_magnified_noise_row,
    plot_pareto_frontier_comparison,
    plot_pareto_frontier_panel,
)

LEGACY_KAGGLE_PATH = "/kaggle/input/notebooks/mostafaanoosha/at-spgd-02-attack"
CHECKPOINT_DIR = Path("/kaggle/input/notebooks/mostafaanoosha/at-spgd-01-training/AT-SPGD/checkpoints")
VIS_DATASET = "gtsrb"
VIS_MODEL = "resnet18"
ARTIFACT_PATH = Path(LEGACY_KAGGLE_PATH) / f"vis_artifacts_{VIS_DATASET}_{VIS_MODEL}.pt"


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


try:
    vis_dict = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)
except TypeError:
    vis_dict = torch.load(ARTIFACT_PATH, map_location="cpu")

# %%
FIGURE_DIR = Path.cwd() / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

num_samples = int(MAGNIFIED_NOISE_CONFIG.get("num_samples", 5))

for i in range(num_samples):
    fig = plot_magnified_noise_row(vis_dict, sample_idx=i)

    pdf_path = FIGURE_DIR / f"03_magnified_noise_sample_{i}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=MAGNIFIED_NOISE_CONFIG.get("save_dpi", 300))

    preview_buffer = io.BytesIO()
    fig.savefig(preview_buffer, format="png", bbox_inches="tight", dpi=MAGNIFIED_NOISE_CONFIG.get("figure_dpi", 100))
    preview_buffer.seek(0)
    display(
        IPythonImage(
            data=preview_buffer.getvalue(),
            width=MAGNIFIED_NOISE_CONFIG.get("display_width_px", 800),
        )
    )

    plt.close(fig)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = load_checkpoint(CHECKPOINT_DIR / f"01_baseline_{VIS_DATASET}_{VIS_MODEL}.pth")
num_classes = int(checkpoint["num_classes"])

client = ImageClient(model_name=VIS_MODEL, dim=EMB_DIM)
server = VFLServer(emb_dim=EMB_DIM, num_classes=num_classes)
client.load_state_dict(checkpoint["image_client"])
server.load_state_dict(checkpoint["vfl_server"])
full_model = FullVFLModel(client, server, normalize_inputs=True).eval().to(device)

num_gradcam_samples = int(GRADCAM_CONTOUR_CONFIG.get("num_samples", 5))

for i in range(num_gradcam_samples):
    fig_cam = plot_gradcam_contour_row(full_model, VIS_MODEL, vis_dict, sample_idx=i)

    pdf_path = FIGURE_DIR / f"03_gradcam_contours_sample_{i}.pdf"
    fig_cam.savefig(pdf_path, bbox_inches="tight", dpi=GRADCAM_CONTOUR_CONFIG.get("save_dpi", 300))

    preview_buffer = io.BytesIO()
    fig_cam.savefig(preview_buffer, format="png", bbox_inches="tight", dpi=GRADCAM_CONTOUR_CONFIG.get("figure_dpi", 100))
    preview_buffer.seek(0)
    display(
        IPythonImage(
            data=preview_buffer.getvalue(),
            width=GRADCAM_CONTOUR_CONFIG.get("display_width_px", 800),
        )
    )

    plt.close(fig_cam)

# %%
ENERGY_DATASETS = RADIAL_ENERGY_CONFIG["datasets"]
ENERGY_CNN_MODEL = RADIAL_ENERGY_CONFIG["cnn_model"]
ENERGY_TRANSFORMER_MODEL = RADIAL_ENERGY_CONFIG["transformer_model"]
ENERGY_PANEL_LABELS = RADIAL_ENERGY_CONFIG["panel_labels"]


def load_energy_artifacts(model_name: str) -> list[dict]:
    artifacts = []
    for dataset_name in ENERGY_DATASETS:
        artifact_path = Path(LEGACY_KAGGLE_PATH) / f"vis_artifacts_{dataset_name}_{model_name}.pt"
        try:
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(artifact_path, map_location="cpu")
        artifacts.append(artifact)
    return artifacts


energy_panels = {
    ENERGY_PANEL_LABELS["cnn"]: load_energy_artifacts(ENERGY_CNN_MODEL),
    ENERGY_PANEL_LABELS["transformer"]: load_energy_artifacts(ENERGY_TRANSFORMER_MODEL),
}

fig_energy = plot_average_radial_energy_comparison(energy_panels)
radial_preview_buffer = io.BytesIO()
fig_energy.savefig(
    radial_preview_buffer,
    format="png",
    bbox_inches="tight",
    dpi=RADIAL_ENERGY_CONFIG.get("figure_dpi", 100),
)
radial_preview_buffer.seek(0)
display(
    IPythonImage(
        data=radial_preview_buffer.getvalue(),
        width=RADIAL_ENERGY_CONFIG.get("display_width_px", 520),
    )
)
fig_energy.savefig(
    FIGURE_DIR / "03_radial_energy.pdf",
    bbox_inches="tight",
    dpi=RADIAL_ENERGY_CONFIG.get("save_dpi", 300),
)
plt.close(fig_energy)

for panel_key, panel_label, model_name in [
    ("cnn", ENERGY_PANEL_LABELS["cnn"], ENERGY_CNN_MODEL),
    ("transformer", ENERGY_PANEL_LABELS["transformer"], ENERGY_TRANSFORMER_MODEL),
]:
    fig_panel = plot_average_radial_energy_panel(energy_panels[panel_label], panel_label=panel_label)
    fig_panel.savefig(
        FIGURE_DIR / f"03_radial_energy_{panel_key}_{model_name}.pdf",
        bbox_inches="tight",
        dpi=RADIAL_ENERGY_CONFIG.get("save_dpi", 300),
    )
    plt.close(fig_panel)

# %%
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


def build_vfl_model(dataset_name: str, model_name: str, device: torch.device) -> FullVFLModel:
    checkpoint = load_checkpoint(CHECKPOINT_DIR / f"01_baseline_{dataset_name}_{model_name}.pth")
    num_classes = int(checkpoint["num_classes"])
    client = ImageClient(model_name=model_name, dim=EMB_DIM)
    server = VFLServer(emb_dim=EMB_DIM, num_classes=num_classes)
    client.load_state_dict(checkpoint["image_client"])
    server.load_state_dict(checkpoint["vfl_server"])
    return FullVFLModel(client, server, normalize_inputs=True).eval().to(device)


def load_visual_artifact(dataset_name: str, model_name: str) -> dict:
    artifact_path = Path(LEGACY_KAGGLE_PATH) / f"vis_artifacts_{dataset_name}_{model_name}.pt"
    try:
        return torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(artifact_path, map_location="cpu")


def run_pareto_sweep(dataset_name: str, model_name: str, device: torch.device) -> list[dict]:
    model = build_vfl_model(dataset_name, model_name, device)
    artifact = load_visual_artifact(dataset_name, model_name)
    images = artifact["clean_images"][:PARETO_SAMPLES].to(device)
    labels = artifact["labels"][:PARETO_SAMPLES].to(device)

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

                results.append(
                    {
                        "k": k,
                        "alpha": alpha_multiplier,
                        "steps": steps,
                        "asr": asr,
                        "psnr": psnr,
                    }
                )

                del attack, adversarial, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del model, artifact, images, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pareto_results = {
    panel_label: run_pareto_sweep(case["dataset"], case["model"], device)
    for panel_label, case in PARETO_CASES.items()
}

fig_pareto = plot_pareto_frontier_comparison(pareto_results)
pareto_preview_buffer = io.BytesIO()
fig_pareto.savefig(
    pareto_preview_buffer,
    format="png",
    bbox_inches="tight",
    dpi=PARETO_FRONTIER_CONFIG.get("figure_dpi", 100),
)
pareto_preview_buffer.seek(0)
display(
    IPythonImage(
        data=pareto_preview_buffer.getvalue(),
        width=PARETO_FRONTIER_CONFIG.get("display_width_px", 800),
    )
)
plt.close(fig_pareto)

for panel_label, case in PARETO_CASES.items():
    panel_key = case["panel_key"]
    model_name = case["model"]
    fig_panel = plot_pareto_frontier_panel(pareto_results[panel_label], panel_label=panel_label)
    fig_panel.savefig(
        FIGURE_DIR / f"03_pareto_frontier_{panel_key}_{model_name}.pdf",
        bbox_inches="tight",
        dpi=PARETO_FRONTIER_CONFIG.get("save_dpi", 300),
    )
    plt.close(fig_panel)

# %%
JPEG_DATASETS = JPEG_COMPRESSION_CONFIG["datasets"]
JPEG_MODELS = JPEG_COMPRESSION_CONFIG["models"]
JPEG_QUALITIES = JPEG_COMPRESSION_CONFIG["jpeg_qualities"]
JPEG_ATTACK_KEYS = ("adv_AT-SPGD", "adv_ATSPGD", "adv_Adaptive")


def get_ours_attack_images(artifact: dict) -> torch.Tensor:
    for attack_key in JPEG_ATTACK_KEYS:
        if attack_key in artifact:
            return artifact[attack_key]
    raise KeyError(f"No AT-SPGD adversarial tensor found. Tried: {JPEG_ATTACK_KEYS}")


def apply_jpeg_batch(images: torch.Tensor, quality: int, device: torch.device) -> torch.Tensor:
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


def compute_jpeg_asr_curve(dataset_name: str, model_name: str, device: torch.device) -> list[float]:
    model = build_vfl_model(dataset_name, model_name, device)
    artifact = load_visual_artifact(dataset_name, model_name)
    adversarial_images = get_ours_attack_images(artifact)
    sample_count = min(adversarial_images.shape[0], artifact["labels"].shape[0])
    adversarial_images = adversarial_images[:sample_count]
    labels = artifact["labels"][:sample_count].to(device)

    asr_curve = []
    for quality in JPEG_QUALITIES:
        compressed = apply_jpeg_batch(adversarial_images, quality, device)
        with torch.no_grad():
            logits = model(compressed)
            asr = (logits.argmax(dim=1) != labels).float().mean().item() * 100.0
        asr_curve.append(asr)
        del compressed, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, artifact, adversarial_images, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return asr_curve


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
jpeg_curves = {
    dataset_name: {
        model_name: compute_jpeg_asr_curve(dataset_name, model_name, device)
        for model_name in JPEG_MODELS
    }
    for dataset_name in JPEG_DATASETS
}

fig_jpeg = plot_jpeg_compression_comparison(jpeg_curves)
jpeg_preview_buffer = io.BytesIO()
fig_jpeg.savefig(
    jpeg_preview_buffer,
    format="png",
    bbox_inches="tight",
    dpi=JPEG_COMPRESSION_CONFIG.get("figure_dpi", 100),
)
jpeg_preview_buffer.seek(0)
display(
    IPythonImage(
        data=jpeg_preview_buffer.getvalue(),
        width=JPEG_COMPRESSION_CONFIG.get("display_width_px", 800),
    )
)
plt.close(fig_jpeg)

for dataset_name, model_curves in jpeg_curves.items():
    fig_panel = plot_jpeg_compression_panel(dataset_name, model_curves)
    fig_panel.savefig(
        FIGURE_DIR / f"03_jpeg_compression_{dataset_name}.pdf",
        bbox_inches="tight",
        dpi=JPEG_COMPRESSION_CONFIG.get("save_dpi", 300),
    )
    plt.close(fig_panel)
