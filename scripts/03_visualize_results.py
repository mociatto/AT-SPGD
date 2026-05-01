# %%
!git clone https://github.com/mociatto/AT-SPGD.git

# %%
%cd AT-SPGD

# %%
!pip install -q matplotlib torchattacks lpips torchmetrics

# %%
from __future__ import annotations

# %%
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

from src.models.split_models import EMB_DIM, FullVFLModel, ImageClient, VFLServer
from src.visualization.plots import (
    RADIAL_ENERGY_CONFIG,
    plot_gradcam_contours,
    plot_magnified_noise_grid,
    plot_radial_energy,
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

fig = plot_magnified_noise_grid(vis_dict)
plt.show()
fig.savefig(FIGURE_DIR / "03_magnified_noise.pdf", bbox_inches="tight")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = load_checkpoint(CHECKPOINT_DIR / f"01_baseline_{VIS_DATASET}_{VIS_MODEL}.pth")
num_classes = int(checkpoint["num_classes"])

client = ImageClient(model_name=VIS_MODEL, dim=EMB_DIM)
server = VFLServer(emb_dim=EMB_DIM, num_classes=num_classes)
client.load_state_dict(checkpoint["image_client"])
server.load_state_dict(checkpoint["vfl_server"])
full_model = FullVFLModel(client, server, normalize_inputs=True).eval().to(device)

fig_cam = plot_gradcam_contours(full_model, VIS_MODEL, vis_dict, num_samples=5)
plt.show()
fig_cam.savefig(FIGURE_DIR / "03_gradcam_contours.pdf", bbox_inches="tight")

# %%
ENERGY_DATASET = "gtsrb"
ENERGY_MODEL = "resnet18"
ENERGY_ARTIFACT_PATH = Path(LEGACY_KAGGLE_PATH) / f"vis_artifacts_{ENERGY_DATASET}_{ENERGY_MODEL}.pt"

try:
    vis_dict_energy = torch.load(ENERGY_ARTIFACT_PATH, map_location="cpu", weights_only=False)
except TypeError:
    vis_dict_energy = torch.load(ENERGY_ARTIFACT_PATH, map_location="cpu")

fig_energy = plot_radial_energy(vis_dict_energy)
radial_preview_path = FIGURE_DIR / "03_radial_energy_preview.png"
fig_energy.savefig(
    radial_preview_path,
    bbox_inches="tight",
    dpi=RADIAL_ENERGY_CONFIG.get("figure_dpi", 100),
)
display(
    IPythonImage(
        filename=str(radial_preview_path),
        width=RADIAL_ENERGY_CONFIG.get("display_width_px", 520),
    )
)
fig_energy.savefig(
    FIGURE_DIR / "03_radial_energy.pdf",
    bbox_inches="tight",
    dpi=RADIAL_ENERGY_CONFIG.get("save_dpi", 300),
)
plt.close(fig_energy)
