# %%
!git clone https://github.com/mociatto/AT-SPGD.git

# %%
%cd AT-SPGD

# %%
from __future__ import annotations

# %%
from pathlib import Path
import sys


def find_project_root() -> Path:
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)

    for start in starts:
        for candidate in [start, *start.parents]:
            if (candidate / "src").is_dir():
                return candidate

    for candidate in Path.cwd().iterdir():
        if candidate.is_dir() and (candidate / "src").is_dir():
            return candidate

    raise RuntimeError("Unable to locate project root containing src/.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
from typing import Any, Dict, List

import pandas as pd
import torch

from src.engine.trainer import default_device, run_cross_validation, set_seed

# %%
DATASETS = ["cifar10", "cifar100", "svhn", "gtsrb"]
MODELS = ["swin_tiny_patch4_window7_224", "resnet18", "mobilenet_v2", "vit_base_patch16_224"]

EPOCHS = 10
K_FOLDS = 5
BATCH_SIZE = 128
LR = 1e-3
NUM_WORKERS = 4
SEED = 42

WORK_DIR = Path.cwd()
DATA_ROOT = WORK_DIR / "data"
RESULTS_DIR = WORK_DIR / "results" / "csv"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
BASELINE_CSV = RESULTS_DIR / "01_baseline_metrics.csv"

# %%
def run_experiments() -> pd.DataFrame:
    set_seed(SEED)
    device = default_device()
    rows: List[Dict[str, Any]] = []

    for dataset_name in DATASETS:
        for model_name in MODELS:
            row = run_cross_validation(
                dataset_name=dataset_name,
                model_name=model_name,
                data_root=DATA_ROOT,
                checkpoint_dir=CHECKPOINT_DIR,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                k_folds=K_FOLDS,
                lr=LR,
                num_workers=NUM_WORKERS,
                seed=SEED,
                device=device,
            )
            rows.append(row)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


results_df = run_experiments()

# %%
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(BASELINE_CSV, index=False)
display(results_df)
