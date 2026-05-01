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

from src.visualization.plots import plot_magnified_noise_grid

LEGACY_KAGGLE_PATH = "/kaggle/input/notebooks/mostafaanoosha/spectralvfl"
ARTIFACT_PATH = Path(LEGACY_KAGGLE_PATH) / "vis_artifacts_cifar10_resnet18.pt"

try:
    vis_dict = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)
except TypeError:
    vis_dict = torch.load(ARTIFACT_PATH, map_location="cpu")

# %%
FIGURE_DIR = Path.cwd() / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

fig = plot_magnified_noise_grid(vis_dict, num_samples=5)
plt.show()
fig.savefig(FIGURE_DIR / "03_magnified_noise.pdf", bbox_inches="tight")
