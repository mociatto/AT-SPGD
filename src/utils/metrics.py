from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        if num_classes == 2:
            macro_auroc = roc_auc_score(y_true, y_prob[:, 1], average="macro")
        else:
            macro_auroc = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        metrics["macro_auroc"] = float(macro_auroc)
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate_system(
    image_client: nn.Module,
    vfl_server: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    eval_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_client.eval()
    vfl_server.eval()

    labels_buffer: List[np.ndarray] = []
    probability_buffer: List[np.ndarray] = []

    for images, labels in data_loader:
        images = images.to(eval_device, non_blocking=True)
        logits = vfl_server(image_client(images))
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probability_buffer.append(probabilities)
        labels_buffer.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(labels_buffer, axis=0)
    y_prob = np.concatenate(probability_buffer, axis=0)
    return classification_metrics(y_true, y_prob, num_classes)
