from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)

EMB_DIM = 128


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _build_backbone(model_name: str) -> Tuple[nn.Module, int]:
    name = model_name.lower()

    if name == "resnet18":
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        return backbone, 512

    if name == "mobilenet_v2":
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()
        return backbone, 1280

    if name in {"vit_base_patch16_224", "swin_tiny_patch4_window7_224"}:
        backbone = timm.create_model(name, pretrained=True, num_classes=0)
        return backbone, int(backbone.num_features)

    raise ValueError(
        f"Unsupported model_name={model_name!r}. "
        "Expected one of: resnet18, mobilenet_v2, vit_base_patch16_224, "
        "swin_tiny_patch4_window7_224."
    )


class ImageClient(nn.Module):
    def __init__(self, model_name: str, dim: int = EMB_DIM) -> None:
        super().__init__()
        backbone, feature_dim = _build_backbone(model_name)
        _freeze_module(backbone)
        self.model_name = model_name
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() > 2:
            features = features.flatten(1)
        return self.projection(features)


class VFLServer(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class FullVFLModel(nn.Module):
    def __init__(self, image_client: nn.Module, vfl_server: nn.Module) -> None:
        super().__init__()
        self.image_client = image_client
        self.vfl_server = vfl_server

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.image_client(x)
        return self.vfl_server(embedding)
