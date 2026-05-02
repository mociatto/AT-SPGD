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
import torch
from IPython.display import Image as IPythonImage
from IPython.display import display

from src.attacks.at_spgd import ATSPGD
from src.models.split_models import EMB_DIM, FullVFLModel, ImageClient, VFLServer
from src.visualization.plots import (
    GRADCAM_CONTOUR_CONFIG,
    MAGNIFIED_NOISE_CONFIG,
    RADIAL_ENERGY_CONFIG,
    plot_average_radial_energy_comparison,
    plot_average_radial_energy_panel,
    plot_gradcam_contour_row,
    plot_magnified_noise_row,
    plot_pareto_frontier,
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
PARETO_DATASET = "gtsrb"
PARETO_MODEL = "swin_tiny_patch4_window7_224"
PARETO_SAMPLES = 64
K_RATIOS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
ALPHA_MULTIPLIERS = [0.1, 0.2, 0.5, 1.0, 1.5]
STEPS_SWEEP = [1, 2, 5, 10]
EPSILON = 8.0 / 255.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pareto_checkpoint = load_checkpoint(CHECKPOINT_DIR / f"01_baseline_{PARETO_DATASET}_{PARETO_MODEL}.pth")
pareto_num_classes = int(pareto_checkpoint["num_classes"])

pareto_client = ImageClient(model_name=PARETO_MODEL, dim=EMB_DIM)
pareto_server = VFLServer(emb_dim=EMB_DIM, num_classes=pareto_num_classes)
pareto_client.load_state_dict(pareto_checkpoint["image_client"])
pareto_server.load_state_dict(pareto_checkpoint["vfl_server"])
pareto_model = FullVFLModel(pareto_client, pareto_server, normalize_inputs=True).eval().to(device)

PARETO_ARTIFACT_PATH = Path(LEGACY_KAGGLE_PATH) / f"vis_artifacts_{PARETO_DATASET}_{PARETO_MODEL}.pt"
try:
    vis_dict_pareto = torch.load(PARETO_ARTIFACT_PATH, map_location="cpu", weights_only=False)
except TypeError:
    vis_dict_pareto = torch.load(PARETO_ARTIFACT_PATH, map_location="cpu")

images = vis_dict_pareto["clean_images"][:PARETO_SAMPLES].to(device)
labels = vis_dict_pareto["labels"][:PARETO_SAMPLES].to(device)

results = []
for k in K_RATIOS:
    for a_mult in ALPHA_MULTIPLIERS:
        for s in STEPS_SWEEP:
            atk = ATSPGD(
                model=pareto_model,
                eps=EPSILON,
                alpha_f=(EPSILON / s) * a_mult,
                steps=s,
                K=k,
            ).eval()
            adv = atk(images, labels)

            with torch.no_grad():
                logits = pareto_model(adv)
                asr = (logits.argmax(dim=1) != labels).float().mean().item() * 100.0
                mse = torch.mean((adv - images) ** 2, dim=[1, 2, 3])
                psnr = (20 * torch.log10(1.0 / torch.sqrt(mse))).mean().item()

            results.append({"k": k, "alpha": a_mult, "steps": s, "asr": asr, "psnr": psnr})
            del atk, adv, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

fig_pareto = plot_pareto_frontier(results, PARETO_DATASET, PARETO_MODEL)
plt.show()
fig_pareto.savefig(
    FIGURE_DIR / "03_pareto_frontier.pdf",
    bbox_inches="tight",
    dpi=300,
)
