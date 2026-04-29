from __future__ import annotations

# %%
from pathlib import Path
import sys


def kaggle_setup() -> None:
    try:
        from IPython import get_ipython
    except ImportError:
        return

    shell = get_ipython()
    if shell is None:
        return

    if Path.cwd().name != "AT-SPGD" and not (Path.cwd() / "src").is_dir():
        shell.system("git clone https://github.com/mociatto/AT-SPGD.git")
        shell.run_line_magic("cd", "AT-SPGD")
    shell.system("pip install torchattacks lpips torchmetrics")


kaggle_setup()

# %%
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from IPython.display import display

from src.data.datasets import get_dataloaders
from src.engine.evaluator import run_attack_arena
from src.models.split_models import EMB_DIM, ImageClient, VFLServer

DATASETS = ["cifar10", "cifar100", "svhn", "gtsrb"]
MODELS = ["swin_tiny_patch4_window7_224", "resnet18", "mobilenet_v2", "vit_base_patch16_224"]
NUM_SAMPLES = 128
BATCH_SIZE = 128
NUM_WORKERS = 4

WORK_DIR = Path.cwd()
DATA_ROOT = WORK_DIR / "data"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
RESULTS_DIR = WORK_DIR / "results" / "csv"
OUTPUT_CSV = RESULTS_DIR / "02_baseline_arena_metrics.csv"


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_samples(dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    _, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=BATCH_SIZE,
        data_root=DATA_ROOT,
        num_workers=NUM_WORKERS,
    )
    image_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []

    for images, labels in test_loader:
        image_batches.append(images)
        label_batches.append(labels)
        if sum(batch.size(0) for batch in image_batches) >= NUM_SAMPLES:
            break

    test_batch = torch.cat(image_batches, dim=0)[:NUM_SAMPLES]
    labels = torch.cat(label_batches, dim=0)[:NUM_SAMPLES]
    return test_batch, labels, num_classes

# %%
def run_evaluation() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for dataset_name in DATASETS:
        test_batch, labels, dataset_num_classes = collect_samples(dataset_name)

        for model_name in MODELS:
            checkpoint_path = CHECKPOINT_DIR / f"01_baseline_{dataset_name}_{model_name}.pth"
            checkpoint = load_checkpoint(checkpoint_path)
            num_classes = int(checkpoint.get("num_classes", dataset_num_classes))

            client = ImageClient(model_name=model_name, dim=EMB_DIM)
            server = VFLServer(emb_dim=EMB_DIM, num_classes=num_classes)
            client.load_state_dict(checkpoint["image_client"])
            server.load_state_dict(checkpoint["vfl_server"])
            client.eval()
            server.eval()

            attack_rows = run_attack_arena(client, server, test_batch, labels)
            for row in attack_rows:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "num_classes": num_classes,
                        **row,
                    }
                )

            del client, server
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del test_batch, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


arena_df = run_evaluation()

# %%
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
arena_df = arena_df.sort_values(["dataset", "model", "attack"]).reset_index(drop=True)
arena_df.to_csv(OUTPUT_CSV, index=False)
display(arena_df)
