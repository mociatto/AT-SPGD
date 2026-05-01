from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

try:
    import seaborn as sns
except ImportError:
    sns = None

MAGNIFIED_NOISE_CONFIG = {
    "font_family": "serif",
    "title_fontsize": 10,
    "label_fontsize": 10,
    "figure_dpi": 300,
    "noise_magnification": 10.0,
    "num_samples": 5,
    "figsize_per_sample": (10, 2),
    "image_border_width": 0.0,
    "image_border_color": "black",
}

GRADCAM_CONTOUR_CONFIG = {
    "font_family": "serif",
    "title_fontsize": 10,
    "label_fontsize": 10,
    "figure_dpi": 300,
    "num_samples": 5,
    "contour_levels": 8,
    "contour_min_level": 0.15,
    "contour_max_level": 0.95,
    "contour_color_low": "#3d348b",
    "contour_color_high": "#f18701",
    "contour_smooth_sigma": 8,
    "contour_linewidth": 1.0,
    "contour_alpha": 1.0,
    "back_image_alpha": 0.2,
    "image_border_width": 0.0,
    "image_border_color": "black",
    "figsize_scale": 2.15,
    "target_class": "predicted",
    "use_fallback_input_gradient": True,
}

RADIAL_ENERGY_CONFIG = {
    "font_family": "serif",
    "axes_label_fontsize": 10,
    "tick_label_fontsize": 10,
    "legend_fontsize": 10,
    "legend_loc": "upper right",
    "figure_dpi": 100,
    "save_dpi": 300,
    "display_width_px": 1040,
    "figsize": (4.5, 3.5),
    "line_width": 1.0,
    "smooth_window": 3,
    "fill_alpha": 0.5,
    "y_scale": "linear",
    "x_label": "Spatial Frequency (Low \u2192 High)",
    "y_label": "Mean Adversarial Energy",
    "attack_styles": {
        "PGD": {"color": "#3d348b", "linestyle": "-"},
        "APGD": {"color": "#7678ed", "linestyle": "-"},
        "MIFGSM": {"color": "#a8dadc", "linestyle": "-"},
        "SSA": {"color": "#f7b801", "linestyle": "-"},
        "Adaptive": {"color": "#f18701", "linestyle": "-"},
    },
}

ATTACK_ORDER = ["PGD", "APGD", "MIFGSM", "SSA", "AT-SPGD (Ours)"]
ATTACK_KEY_ALIASES = {
    "AT-SPGD (Ours)": ("adv_AT-SPGD", "adv_ATSPGD", "adv_Adaptive"),
}


def _attack_key(attack_name: str) -> str:
    return f"adv_{attack_name}"


def _attack_keys(attack_name: str) -> Tuple[str, ...]:
    return ATTACK_KEY_ALIASES.get(attack_name, (_attack_key(attack_name),))


def _to_image_array(image: torch.Tensor):
    return image.detach().cpu().float().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)


def _available_attacks(vis_dict: Dict[str, torch.Tensor]) -> List[str]:
    return [attack_name for attack_name in ATTACK_ORDER if any(key in vis_dict for key in _attack_keys(attack_name))]


def _resolve_attack_key(vis_dict: Dict[str, torch.Tensor], attack_name: str) -> str:
    for key in _attack_keys(attack_name):
        if key in vis_dict:
            return key
    raise KeyError(f"No image tensor found for attack {attack_name!r}.")


def plot_magnified_noise_grid(vis_dict: dict, num_samples: int | None = None) -> Figure:
    clean_images = vis_dict["clean_images"]
    attacks = _available_attacks(vis_dict)
    if not attacks:
        raise KeyError("No adversarial image tensors found for the expected attack order.")

    requested_samples = num_samples or int(MAGNIFIED_NOISE_CONFIG["num_samples"])
    sample_count = min(requested_samples, clean_images.shape[0])
    column_count = 1 + len(attacks)
    base_width, base_height = MAGNIFIED_NOISE_CONFIG["figsize_per_sample"]
    figure_size = (base_width, base_height * sample_count)

    plt.rcParams["font.family"] = MAGNIFIED_NOISE_CONFIG["font_family"]
    fig, axes = plt.subplots(
        sample_count,
        column_count,
        figsize=figure_size,
        dpi=MAGNIFIED_NOISE_CONFIG["figure_dpi"],
        squeeze=False,
    )

    magnification = float(MAGNIFIED_NOISE_CONFIG["noise_magnification"])

    for sample_idx in range(sample_count):
        clean_img = clean_images[sample_idx]
        row_images = [clean_img]
        column_titles = ["Clean"]

        for attack_name in attacks:
            adv_img = vis_dict[_resolve_attack_key(vis_dict, attack_name)][sample_idx]
            noise_vis = torch.clamp(clean_img + (adv_img - clean_img) * magnification, 0.0, 1.0)
            row_images.append(noise_vis)
            column_titles.append(attack_name)

        for column_idx, image in enumerate(row_images):
            ax = axes[sample_idx, column_idx]
            ax.imshow(_to_image_array(image))
            ax.set_xticks([])
            ax.set_yticks([])
            border_width = float(MAGNIFIED_NOISE_CONFIG["image_border_width"])
            for spine in ax.spines.values():
                spine.set_visible(border_width > 0)
                spine.set_linewidth(border_width)
                spine.set_edgecolor(MAGNIFIED_NOISE_CONFIG["image_border_color"])
            if sample_idx == 0:
                ax.set_title(
                    column_titles[column_idx],
                    fontsize=MAGNIFIED_NOISE_CONFIG["title_fontsize"],
                )

        axes[sample_idx, 0].set_ylabel(
            f"Sample {sample_idx}",
            fontsize=MAGNIFIED_NOISE_CONFIG["label_fontsize"],
        )

    plt.tight_layout()
    return fig


def get_radial_profile(clean_img: torch.Tensor, adv_img: torch.Tensor) -> np.ndarray:
    """
    Radial average of 2D FFT magnitude of adversarial noise (adv - clean).

    1. noise = adv_img - clean_img
    2. Per-channel 2D FFT, fftshift, magnitude; average over RGB and batch
    3. Radial average vs distance from spectrum center (low → high frequency index)
    """
    noise = adv_img - clean_img
    if noise.dim() == 3:
        noise = noise.unsqueeze(0)
    b, c, h, w = noise.shape
    device = noise.device
    dtype = torch.float64

    mag_sum = torch.zeros(h, w, device=device, dtype=dtype)
    noise_64 = noise.to(dtype=dtype)
    for bi in range(b):
        for ci in range(c):
            x = noise_64[bi, ci]
            spec = torch.fft.fftshift(torch.fft.fft2(x))
            mag_sum += torch.abs(spec)

    mag_avg = (mag_sum / float(b * c)).detach().cpu().numpy()
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w), dtype=np.float64)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = int(np.ceil(float(r.max())))
    r_int = np.minimum(np.round(r).astype(np.int64).ravel(), r_max)
    weights = mag_avg.ravel()
    radial_sum = np.bincount(r_int, weights=weights, minlength=r_max + 1).astype(np.float64)
    counts = np.bincount(r_int, minlength=r_max + 1).astype(np.float64)
    radial = np.divide(
        radial_sum,
        np.maximum(counts, 1.0),
        out=np.zeros_like(radial_sum),
        where=counts > 0,
    )
    return radial


def _smooth_curve(y: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return y
    return np.convolve(y, np.ones(window_size) / window_size, mode="same")


def _radial_style_key(attack_name: str) -> str:
    if attack_name in {"AT-SPGD", "AT-SPGD (Ours)"}:
        return "Adaptive"
    return attack_name


def _radial_label(attack_name: str) -> str:
    if _radial_style_key(attack_name) == "Adaptive":
        return "AT-SPGD (Ours)"
    return attack_name


def plot_radial_energy(vis_dict: dict) -> Figure:
    plt.rcParams["font.family"] = RADIAL_ENERGY_CONFIG["font_family"]
    fig, ax = plt.subplots(
        figsize=RADIAL_ENERGY_CONFIG["figsize"],
        dpi=RADIAL_ENERGY_CONFIG["figure_dpi"],
    )

    clean_images = vis_dict["clean_images"]
    attacks = _available_attacks(vis_dict)
    profiles = {
        attack_name: get_radial_profile(
            clean_images,
            vis_dict[_resolve_attack_key(vis_dict, attack_name)],
        )
        for attack_name in attacks
    }
    if not profiles:
        raise KeyError("No adversarial image tensors found for radial energy plotting.")

    min_len = min(len(profile) for profile in profiles.values())
    freq_axis = np.arange(min_len, dtype=float)
    attack_styles = RADIAL_ENERGY_CONFIG["attack_styles"]
    smooth_window = int(RADIAL_ENERGY_CONFIG["smooth_window"])

    for attack_name, profile in profiles.items():
        style_key = _radial_style_key(attack_name)
        style = attack_styles.get(style_key, {"color": "black", "linestyle": "-"})
        smoothed_profile = _smooth_curve(profile[:min_len], smooth_window)
        ax.plot(
            freq_axis,
            smoothed_profile,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=RADIAL_ENERGY_CONFIG["line_width"],
            label=_radial_label(attack_name),
        )
        ax.fill_between(
            freq_axis,
            0,
            smoothed_profile,
            color=style["color"],
            alpha=RADIAL_ENERGY_CONFIG["fill_alpha"],
        )

    ax.set_yscale(RADIAL_ENERGY_CONFIG["y_scale"])
    ax.set_xlabel(
        RADIAL_ENERGY_CONFIG["x_label"],
        fontsize=RADIAL_ENERGY_CONFIG["axes_label_fontsize"],
    )
    ax.set_ylabel(
        RADIAL_ENERGY_CONFIG["y_label"],
        fontsize=RADIAL_ENERGY_CONFIG["axes_label_fontsize"],
    )
    ax.tick_params(axis="both", labelsize=RADIAL_ENERGY_CONFIG["tick_label_fontsize"])
    ax.legend(
        loc=RADIAL_ENERGY_CONFIG["legend_loc"],
        fontsize=RADIAL_ENERGY_CONFIG["legend_fontsize"],
        frameon=True,
    )

    if sns is not None:
        sns.despine(ax=ax, offset=2, trim=False)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def _client_from_full_model(full_model: nn.Module) -> nn.Module:
    if hasattr(full_model, "image_client"):
        return full_model.image_client
    if hasattr(full_model, "client"):
        return full_model.client
    raise AttributeError("Full model must expose either image_client or client.")


def _resolve_gradcam_target_layer(full_model: nn.Module, model_name: str) -> nn.Module | None:
    backbone = _client_from_full_model(full_model).backbone
    name = model_name.lower()
    if name == "resnet18":
        return backbone.layer4[-1].conv2
    if name == "mobilenet_v2":
        return backbone.features[-1]
    if name == "swin_tiny_patch4_window7_224":
        return backbone.layers[-1].blocks[-1].norm2
    if name == "vit_base_patch16_224":
        return backbone.blocks[-1].norm1
    return None


def _normalize_cam(cam_tensor: torch.Tensor) -> torch.Tensor:
    cam_tensor = cam_tensor.detach()
    cam_min = cam_tensor.amin(dim=(-2, -1), keepdim=True)
    cam_max = cam_tensor.amax(dim=(-2, -1), keepdim=True)
    return (cam_tensor - cam_min) / (cam_max - cam_min + 1e-8)


def _target_class_index(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    target_class = GRADCAM_CONTOUR_CONFIG["target_class"]
    if target_class == "label":
        return label.view(1).long()
    if target_class == "predicted":
        return logits.argmax(dim=1)
    return torch.tensor([int(target_class)], device=logits.device)


def _input_gradient_cam(
    full_model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    x = image.unsqueeze(0).detach().clone().requires_grad_(True)
    logits = full_model(x)
    class_idx = _target_class_index(logits, label)
    score = logits.gather(1, class_idx.view(-1, 1)).sum()
    full_model.zero_grad(set_to_none=True)
    score.backward()
    cam_tensor = x.grad.detach().abs().mean(dim=1, keepdim=True)
    cam_tensor = torch.nn.functional.interpolate(
        cam_tensor,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return _normalize_cam(cam_tensor)[0, 0].cpu()


def _layer_gradcam(
    full_model: nn.Module,
    target_layer: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor | None:
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradients.append(grad_outputs[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        x = image.unsqueeze(0).detach().clone().requires_grad_(True)
        logits = full_model(x)
        class_idx = _target_class_index(logits, label)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        full_model.zero_grad(set_to_none=True)
        score.backward()

        if not activations or not gradients:
            return None

        activation = activations[-1]
        gradient = gradients[-1]
        if activation.dim() == 3:
            token_count = activation.shape[1]
            side = int(np.sqrt(token_count))
            if side * side != token_count:
                return None
            activation = activation.permute(0, 2, 1).reshape(activation.shape[0], activation.shape[2], side, side)
            gradient = gradient.permute(0, 2, 1).reshape(gradient.shape[0], gradient.shape[2], side, side)
        elif activation.dim() == 4 and activation.shape[1] not in (activation.shape[-1], 3):
            pass
        elif activation.dim() == 4:
            activation = activation.permute(0, 3, 1, 2)
            gradient = gradient.permute(0, 3, 1, 2)
        else:
            return None

        weights = gradient.mean(dim=(-2, -1), keepdim=True)
        cam_tensor = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam_tensor = torch.nn.functional.interpolate(
            cam_tensor,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return _normalize_cam(cam_tensor)[0, 0].cpu()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _compute_gradcam_contour_map(
    full_model: nn.Module,
    target_layer: nn.Module | None,
    image: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    cam_tensor = None
    if target_layer is not None:
        cam_tensor = _layer_gradcam(full_model, target_layer, image, label)
    if cam_tensor is None and bool(GRADCAM_CONTOUR_CONFIG["use_fallback_input_gradient"]):
        cam_tensor = _input_gradient_cam(full_model, image, label)
    if cam_tensor is None:
        raise RuntimeError("Unable to compute Grad-CAM map for the selected model/layer.")
    return cam_tensor


def plot_gradcam_contours(
    full_model: nn.Module,
    model_name: str,
    vis_dict: dict,
    num_samples: int | None = None,
) -> Figure:
    clean_images = vis_dict["clean_images"]
    labels = vis_dict["labels"]
    attacks = _available_attacks(vis_dict)
    attack_keys = [_resolve_attack_key(vis_dict, attack_name) for attack_name in attacks]
    requested_samples = num_samples or int(GRADCAM_CONTOUR_CONFIG["num_samples"])
    sample_count = min(requested_samples, clean_images.shape[0], labels.shape[0], *[vis_dict[key].shape[0] for key in attack_keys])
    column_count = 1 + len(attacks)
    scale = float(GRADCAM_CONTOUR_CONFIG["figsize_scale"])
    device = next(full_model.parameters()).device
    target_layer = _resolve_gradcam_target_layer(full_model, model_name)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cam",
        [
            GRADCAM_CONTOUR_CONFIG["contour_color_low"],
            GRADCAM_CONTOUR_CONFIG["contour_color_high"],
        ],
    )
    contour_levels = np.linspace(
        float(GRADCAM_CONTOUR_CONFIG["contour_min_level"]),
        float(GRADCAM_CONTOUR_CONFIG["contour_max_level"]),
        int(GRADCAM_CONTOUR_CONFIG["contour_levels"]),
    )

    plt.rcParams["font.family"] = GRADCAM_CONTOUR_CONFIG["font_family"]
    fig, axes = plt.subplots(
        sample_count,
        column_count,
        figsize=(scale * column_count, scale * sample_count),
        dpi=int(GRADCAM_CONTOUR_CONFIG["figure_dpi"]),
        squeeze=False,
    )

    for sample_idx in range(sample_count):
        label = labels[sample_idx].to(device)
        row_items = [("Clean", clean_images[sample_idx])]
        row_items.extend(
            (attack_name, vis_dict[_resolve_attack_key(vis_dict, attack_name)][sample_idx])
            for attack_name in attacks
        )

        for column_idx, (title, image) in enumerate(row_items):
            ax = axes[sample_idx, column_idx]
            image = image.to(device)
            cam_map = _compute_gradcam_contour_map(full_model, target_layer, image, label)
            cam_np = cam_map.numpy()
            smooth_sigma = float(GRADCAM_CONTOUR_CONFIG["contour_smooth_sigma"])
            if smooth_sigma > 0:
                cam_np = gaussian_filter(cam_np, sigma=smooth_sigma)

            ax.imshow(_to_image_array(image), alpha=float(GRADCAM_CONTOUR_CONFIG["back_image_alpha"]))
            ax.contour(
                cam_np,
                levels=contour_levels,
                cmap=cmap,
                linewidths=float(GRADCAM_CONTOUR_CONFIG["contour_linewidth"]),
                alpha=float(GRADCAM_CONTOUR_CONFIG["contour_alpha"]),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            border_width = float(GRADCAM_CONTOUR_CONFIG["image_border_width"])
            for spine in ax.spines.values():
                spine.set_visible(border_width > 0)
                spine.set_linewidth(border_width)
                spine.set_edgecolor(GRADCAM_CONTOUR_CONFIG["image_border_color"])
            if sample_idx == 0:
                ax.set_title(title, fontsize=GRADCAM_CONTOUR_CONFIG["title_fontsize"])
            if column_idx == 0:
                ax.set_ylabel(
                    f"Sample {sample_idx}",
                    fontsize=GRADCAM_CONTOUR_CONFIG["label_fontsize"],
                )

    plt.tight_layout()
    return fig
