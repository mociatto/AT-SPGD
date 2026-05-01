# %%
!git clone https://github.com/mociatto/AT-SPGD.git

# %%
%cd AT-SPGD

# %%
!pip install matplotlib

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

from src.models.split_models import EMB_DIM, FullVFLModel, ImageClient, VFLServer
from src.visualization.plots import plot_gradcam_contours, plot_magnified_noise_grid

LEGACY_KAGGLE_PATH = "/kaggle/input/notebooks/mostafaanoosha/at-spgd-02-attack"
CHECKPOINT_DIR = Path("/kaggle/input/notebooks/mostafaanoosha/spectralvfl/AT-SPGD/checkpoints")
VIS_DATASET = "cifar10"
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
