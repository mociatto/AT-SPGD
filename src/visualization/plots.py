from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

try:
    import seaborn as sns
except ImportError:
    sns = None

MAGNIFIED_NOISE_CONFIG = {
    "font_family": "serif",
    "title_fontsize": 10,
    "figure_dpi": 100,
    "save_dpi": 300,
    "display_width_px": 800,
    "noise_magnification": 10.0,
    "num_samples": 10,
    "figsize_per_sample": (10, 2),
    "image_border_width": 0.0,
    "image_border_color": "black",
}

GRADCAM_CONTOUR_CONFIG = {
    "font_family": "serif",
    "title_fontsize": 10,
    "label_fontsize": 10,
    "figure_dpi": 100,
    "save_dpi": 300,
    "display_width_px": 800,
    "num_samples": 10,
    "contour_levels": 8,
    "contour_min_level": 0.15,
    "contour_max_level": 0.95,
    "contour_color_low": "#3A86FF",
    "contour_color_midlow": "#8338EC",
    "contour_color_mid": "#FF006E",
    "contour_color_midhigh": "#FB5607",
    "contour_color_high": "#FFBE0B",
    "contour_smooth_sigma": 8,
    "contour_linewidth": 1.0,
    "contour_alpha": 1.0,
    "back_image_alpha": 0.5,
    "image_border_width": 0.0,
    "image_border_color": "black",
    "figsize_per_sample": (10, 2),
    "figsize_scale": 2.15,
    "target_class": "predicted",
    "use_fallback_input_gradient": True,
}

RADIAL_ENERGY_CONFIG = {
    "font_family": "serif",
    "axes_label_fontsize": 10,
    "title_fontsize": 10,
    "title_pad": 8,
    "tick_label_fontsize": 10,
    "legend_fontsize": 8,
    "legend_loc": "upper right",
    "figure_dpi": 100,
    "save_dpi": 300,
    "display_width_px": 800,
    "figsize": (4.5, 3.5),
    "figsize_scale": 1.0,
    "datasets": ["cifar10", "cifar100", "svhn", "gtsrb"],
    "cnn_model": "resnet18",
    "transformer_model": "swin_tiny_patch4_window7_224",
    "panel_labels": {"cnn": "CNN", "transformer": "Transformer"},
    "model_pairs": [
        ("resnet18", "swin_tiny_patch4_window7_224"),
        ("mobilenet_v2", "vit_base_patch16_224"),
    ],
    "model_labels": {
        "resnet18": "ResNet-18",
        "mobilenet_v2": "MobileNetV2",
        "swin_tiny_patch4_window7_224": "Swin-Tiny",
        "vit_base_patch16_224": "ViT-B/16",
    },
    "dataset_labels": {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "svhn": "SVHN",
        "gtsrb": "GTSRB",
    },
    "samples_per_case": 16,
    "line_width": 1.0,
    "smooth_window": 3,
    "fill_alpha": 0.0,
    "y_scale": "linear",
    "x_label": "Spatial Frequency",
    "y_label": "Mean Adversarial Energy",
    "attack_styles": {
        "PGD": {"color": "#FFBE0B", "linestyle": "-"},
        "APGD": {"color": "#FB5607", "linestyle": "-"},
        "MIFGSM": {"color": "#FF006E", "linestyle": "-"},
        "SSA": {"color": "#8338EC", "linestyle": "-"},
        "Adaptive": {"color": "#3A86FF", "linestyle": "-"},
    },
}

PARETO_FRONTIER_CONFIG = {
    "font_family": "serif",
    "axes_label_fontsize": 10,
    "title_fontsize": 10,
    "title_pad": 8,
    "tick_label_fontsize": 10,
    "legend_fontsize": 10,
    "annotation_fontsize": 8,
    "figure_dpi": 100,
    "save_dpi": 300,
    "figsize": (4.5, 3.5),
    "figsize_scale": 1.0,
    "display_width_px": 800,
    "cnn_dataset": "gtsrb",
    "cnn_model": "resnet18",
    "transformer_dataset": "gtsrb",
    "transformer_model": "swin_tiny_patch4_window7_224",
    "panel_labels": {"cnn": "CNN", "transformer": "Transformer"},
    "datasets": ["cifar10", "cifar100", "svhn", "gtsrb"],
    "model_pairs": [
        ("resnet18", "swin_tiny_patch4_window7_224"),
        ("mobilenet_v2", "vit_base_patch16_224"),
    ],
    "model_labels": {
        "resnet18": "ResNet-18",
        "mobilenet_v2": "MobileNetV2",
        "swin_tiny_patch4_window7_224": "Swin-Tiny",
        "vit_base_patch16_224": "ViT-B/16",
    },
    "dataset_labels": {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "svhn": "SVHN",
        "gtsrb": "GTSRB",
    },
    "k_ratios": [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1],
    "alpha_multipliers": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5],
    "steps_sweep": [1, 2, 3, 8],
    "epsilon": 8.0 / 255.0,
    "suboptimal_color": "lightgray",
    "suboptimal_alpha": 1.0,
    "suboptimal_size": 20,
    "optimal_color": "#3A86FF",
    "optimal_marker": "o",
    "optimal_linewidth": 1.0,
    "optimal_markersize": 6,
    "annotation_x_offset": 8,
    "annotation_y_offsets": [0, 5, -5, 10, -10, 15, -15, 20, -20],
    "annotation_min_gap_px": 10,
    "annotation_arrow_alpha": 0.45,
}

JPEG_COMPRESSION_CONFIG = {
    "font_family": "serif",
    "axes_label_fontsize": 10,
    "tick_label_fontsize": 10,
    "legend_fontsize": 8,
    "legend_loc": "lower left",
    "figure_dpi": 100,
    "save_dpi": 300,
    "figsize": (4.5, 3.5),
    "figsize_scale": 1.0,
    "display_width_px": 800,
    "datasets": ["cifar10", "cifar100", "svhn", "gtsrb"],
    "dataset_labels": {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "svhn": "SVHN",
        "gtsrb": "GTSRB",
    },
    "models": ["swin_tiny_patch4_window7_224", "resnet18", "mobilenet_v2", "vit_base_patch16_224"],
    "model_labels": {
        "swin_tiny_patch16_224": "Swin-Tiny",
        "swin_tiny_patch4_window7_224": "Swin-Tiny",
        "resnet18": "ResNet-18",
        "mobilenet_v2": "MobileNetV2",
        "vit_base_patch16_224": "ViT-B/16",
    },
    "model_colors": {
        "swin_tiny_patch4_window7_224": "#FFBE0B",
        "resnet18": "#FB5607",
        "mobilenet_v2": "#8338EC",
        "vit_base_patch16_224": "#3A86FF",
    },
    "jpeg_qualities": [100, 90, 80, 70, 60, 50, 40, 30],
    "marker": "o",
    "line_width": 1.0,
    "markersize": 6,
    "x_label": "JPEG Quality",
    "y_label": "Attack Success Rate",
}

GAUSSIAN_BLUR_CONFIG = {
    "font_family": "serif",
    "axes_label_fontsize": 10,
    "title_fontsize": 10,
    "title_pad": 8,
    "tick_label_fontsize": 10,
    "legend_fontsize": 8,
    "legend_loc": "lower left",
    "figure_dpi": 100,
    "save_dpi": 300,
    "figsize": (4.5, 3.5),
    "figsize_scale": 1.0,
    "display_width_px": 800,
    "datasets": ["cifar10", "cifar100", "svhn", "gtsrb"],
    "model_groups": {
        "CNN": ["resnet18", "mobilenet_v2"],
        "Transformer": ["swin_tiny_patch4_window7_224", "vit_base_patch16_224"],
    },
    "num_samples": 32,
    "blur_kernel_size": 5,
    "blur_sigma": 1.0,
    "resize_scale": 0.4,
    "attack_order": ["PGD", "APGD", "MIFGSM", "SSA", "AT-SPGD (Ours)"],
    "attack_labels": {
        "PGD": "PGD",
        "APGD": "APGD",
        "MIFGSM": "MI-FGSM",
        "SSA": "SSA",
        "AT-SPGD (Ours)": "AT-SPGD",
    },
    "defense_colors": {
        "CNN": {"blur_asr": "#FFBE0B", "resize_asr": "#FB5607"},
        "Transformer": {"blur_asr": "#8338EC", "resize_asr": "#3A86FF"},
    },
    "blur_label": "Gaussian Blur",
    "resize_label": "Resize Defense",
    "bar_width": 0.34,
    "bar_alpha": 1.0,
    "x_label": "Attack Method",
    "y_label": "Attack Success Rate (%)",
    "y_limit": (0.0, 100.0),
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


def plot_magnified_noise_row(vis_dict: dict, sample_idx: int) -> Figure:
    clean_images = vis_dict["clean_images"]
    attacks = _available_attacks(vis_dict)
    if not attacks:
        raise KeyError("No adversarial image tensors found for the expected attack order.")
    if sample_idx >= len(clean_images):
        raise IndexError(f"sample_idx={sample_idx} is out of range for {len(clean_images)} clean images.")

    column_count = 1 + len(attacks)
    figure_size = MAGNIFIED_NOISE_CONFIG["figsize_per_sample"]

    plt.rcParams["font.family"] = MAGNIFIED_NOISE_CONFIG["font_family"]
    fig, axes = plt.subplots(
        1,
        column_count,
        figsize=figure_size,
        dpi=MAGNIFIED_NOISE_CONFIG["figure_dpi"],
        squeeze=False,
    )

    magnification = float(MAGNIFIED_NOISE_CONFIG["noise_magnification"])
    clean_img = clean_images[sample_idx]
    row_images = [clean_img]
    column_titles = ["Clean"]

    for attack_name in attacks:
        adv_img = vis_dict[_resolve_attack_key(vis_dict, attack_name)][sample_idx]
        noise_vis = torch.clamp(clean_img + (adv_img - clean_img) * magnification, 0.0, 1.0)
        row_images.append(noise_vis)
        column_titles.append(attack_name)

    for column_idx, image in enumerate(row_images):
        ax = axes[0, column_idx]
        ax.imshow(_to_image_array(image))
        ax.set_xticks([])
        ax.set_yticks([])
        border_width = float(MAGNIFIED_NOISE_CONFIG["image_border_width"])
        for spine in ax.spines.values():
            spine.set_visible(border_width > 0)
            spine.set_linewidth(border_width)
            spine.set_edgecolor(MAGNIFIED_NOISE_CONFIG["image_border_color"])
        ax.set_title(
            column_titles[column_idx],
            fontsize=MAGNIFIED_NOISE_CONFIG["title_fontsize"],
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


def _average_radial_profiles(vis_dicts: list[dict]) -> Dict[str, np.ndarray]:
    profile_groups: Dict[str, List[np.ndarray]] = {}

    for vis_dict in vis_dicts:
        clean_images = vis_dict["clean_images"]
        for attack_name in _available_attacks(vis_dict):
            profile = get_radial_profile(
                clean_images,
                vis_dict[_resolve_attack_key(vis_dict, attack_name)],
            )
            profile_groups.setdefault(attack_name, []).append(profile)

    averaged_profiles: Dict[str, np.ndarray] = {}
    for attack_name, profiles in profile_groups.items():
        min_len = min(len(profile) for profile in profiles)
        stacked_profiles = np.stack([profile[:min_len] for profile in profiles], axis=0)
        averaged_profiles[attack_name] = stacked_profiles.mean(axis=0)

    return averaged_profiles


def _draw_average_radial_energy_panel(ax, averaged_profiles: Dict[str, np.ndarray], panel_label: str | None = None) -> None:
    if not averaged_profiles:
        raise KeyError("No adversarial image tensors found for averaged radial energy plotting.")

    min_len = min(len(profile) for profile in averaged_profiles.values())
    freq_axis = np.arange(min_len, dtype=float)
    attack_styles = RADIAL_ENERGY_CONFIG["attack_styles"]
    smooth_window = int(RADIAL_ENERGY_CONFIG["smooth_window"])

    for attack_name, profile in averaged_profiles.items():
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
    if panel_label is not None:
        ax.set_title(
            panel_label,
            fontsize=RADIAL_ENERGY_CONFIG["title_fontsize"],
            pad=RADIAL_ENERGY_CONFIG["title_pad"],
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


def plot_average_radial_energy_panel(vis_dicts: list[dict], panel_label: str) -> Figure:
    plt.rcParams["font.family"] = RADIAL_ENERGY_CONFIG["font_family"]
    scale = float(RADIAL_ENERGY_CONFIG["figsize_scale"])
    width, height = RADIAL_ENERGY_CONFIG["figsize"]
    fig, ax = plt.subplots(
        figsize=(width * scale, height * scale),
        dpi=RADIAL_ENERGY_CONFIG["figure_dpi"],
    )
    _draw_average_radial_energy_panel(ax, _average_radial_profiles(vis_dicts), panel_label=panel_label)
    plt.tight_layout()
    return fig


def plot_average_radial_energy_comparison(panel_vis_dicts: Dict[str, list[dict]]) -> Figure:
    plt.rcParams["font.family"] = RADIAL_ENERGY_CONFIG["font_family"]
    scale = float(RADIAL_ENERGY_CONFIG["figsize_scale"])
    width, height = RADIAL_ENERGY_CONFIG["figsize"]
    fig, axes = plt.subplots(
        1,
        len(panel_vis_dicts),
        figsize=(width * len(panel_vis_dicts) * scale, height * scale),
        dpi=RADIAL_ENERGY_CONFIG["figure_dpi"],
        squeeze=False,
    )

    for ax, (panel_label, vis_dicts) in zip(axes[0], panel_vis_dicts.items()):
        _draw_average_radial_energy_panel(ax, _average_radial_profiles(vis_dicts), panel_label=panel_label)

    plt.tight_layout()
    return fig


def _pareto_split(results: list[dict]) -> Tuple[list[dict], list[dict]]:
    sorted_results = sorted(results, key=lambda item: item["psnr"], reverse=True)
    frontier = []
    max_asr_seen = -1.0

    for result in sorted_results:
        if result["asr"] > max_asr_seen:
            frontier.append(result)
            max_asr_seen = result["asr"]

    frontier_ids = {id(result) for result in frontier}
    suboptimal = [result for result in sorted_results if id(result) not in frontier_ids]
    return frontier, suboptimal


def _draw_pareto_frontier_panel(ax, results: list[dict], panel_label: str | None = None) -> None:
    frontier, suboptimal = _pareto_split(results)

    if suboptimal:
        ax.scatter(
            [result["psnr"] for result in suboptimal],
            [result["asr"] for result in suboptimal],
            color=PARETO_FRONTIER_CONFIG["suboptimal_color"],
            alpha=PARETO_FRONTIER_CONFIG["suboptimal_alpha"],
            s=PARETO_FRONTIER_CONFIG["suboptimal_size"],
            label="Suboptimal States",
        )

    if frontier:
        ax.plot(
            [result["psnr"] for result in frontier],
            [result["asr"] for result in frontier],
            color=PARETO_FRONTIER_CONFIG["optimal_color"],
            marker=PARETO_FRONTIER_CONFIG["optimal_marker"],
            linewidth=PARETO_FRONTIER_CONFIG["optimal_linewidth"],
            markersize=PARETO_FRONTIER_CONFIG["optimal_markersize"],
            label="Pareto Frontier",
        )

        used_label_y = []
        points_to_pixels = ax.figure.dpi / 72.0
        y_offsets = PARETO_FRONTIER_CONFIG["annotation_y_offsets"]
        min_gap = float(PARETO_FRONTIER_CONFIG["annotation_min_gap_px"])

        for result in frontier:
            _, point_y = ax.transData.transform((result["psnr"], result["asr"]))
            selected_y_offset = y_offsets[0]
            for candidate_offset in y_offsets:
                candidate_y = point_y + candidate_offset * points_to_pixels
                if all(abs(candidate_y - used_y) >= min_gap for used_y in used_label_y):
                    selected_y_offset = candidate_offset
                    used_label_y.append(candidate_y)
                    break
            else:
                candidate_y = point_y + selected_y_offset * points_to_pixels
                used_label_y.append(candidate_y)

            ax.annotate(
                f"K={result['k']}",
                (result["psnr"], result["asr"]),
                textcoords="offset points",
                xytext=(PARETO_FRONTIER_CONFIG["annotation_x_offset"], selected_y_offset),
                ha="left",
                va="center",
                fontsize=PARETO_FRONTIER_CONFIG["annotation_fontsize"],
                arrowprops={
                    "arrowstyle": "-",
                    "color": PARETO_FRONTIER_CONFIG["optimal_color"],
                    "alpha": PARETO_FRONTIER_CONFIG["annotation_arrow_alpha"],
                    "linewidth": 0.6,
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
            )

    if panel_label is not None:
        ax.set_title(
            panel_label,
            fontsize=PARETO_FRONTIER_CONFIG["title_fontsize"],
            pad=PARETO_FRONTIER_CONFIG["title_pad"],
        )

    ax.set_xlabel(
        "Stealth (PSNR in dB \u2192 Higher is Better)",
        fontsize=PARETO_FRONTIER_CONFIG["axes_label_fontsize"],
    )
    ax.set_ylabel(
        "Lethality (ASR %)",
        fontsize=PARETO_FRONTIER_CONFIG["axes_label_fontsize"],
    )
    ax.tick_params(axis="both", labelsize=PARETO_FRONTIER_CONFIG["tick_label_fontsize"])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(
        loc="lower right",
        fontsize=PARETO_FRONTIER_CONFIG["legend_fontsize"],
        frameon=True,
    )

    if sns is not None:
        sns.despine(ax=ax, offset=2, trim=False)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_pareto_frontier_panel(results: list[dict], panel_label: str) -> Figure:
    plt.rcParams["font.family"] = PARETO_FRONTIER_CONFIG["font_family"]
    scale = float(PARETO_FRONTIER_CONFIG["figsize_scale"])
    width, height = PARETO_FRONTIER_CONFIG["figsize"]
    fig, ax = plt.subplots(
        figsize=(width * scale, height * scale),
        dpi=PARETO_FRONTIER_CONFIG["figure_dpi"],
    )
    _draw_pareto_frontier_panel(ax, results, panel_label=panel_label)
    plt.tight_layout()
    return fig


def plot_pareto_frontier_comparison(panel_results: Dict[str, list[dict]]) -> Figure:
    plt.rcParams["font.family"] = PARETO_FRONTIER_CONFIG["font_family"]
    scale = float(PARETO_FRONTIER_CONFIG["figsize_scale"])
    width, height = PARETO_FRONTIER_CONFIG["figsize"]
    fig, axes = plt.subplots(
        1,
        len(panel_results),
        figsize=(width * len(panel_results) * scale, height * scale),
        dpi=PARETO_FRONTIER_CONFIG["figure_dpi"],
        squeeze=False,
    )

    for ax, (panel_label, results) in zip(axes[0], panel_results.items()):
        _draw_pareto_frontier_panel(ax, results, panel_label=panel_label)

    plt.tight_layout()
    return fig


def plot_pareto_frontier(results: list[dict], dataset: str, model_name: str) -> Figure:
    return plot_pareto_frontier_panel(results, panel_label=f"{dataset} / {model_name}")


def _draw_jpeg_compression_panel(ax, dataset_name: str, model_curves: Dict[str, list[float]]) -> None:
    qualities = JPEG_COMPRESSION_CONFIG["jpeg_qualities"]
    model_labels = JPEG_COMPRESSION_CONFIG["model_labels"]
    model_colors = JPEG_COMPRESSION_CONFIG["model_colors"]

    for model_name, asr_curve in model_curves.items():
        ax.plot(
            qualities,
            asr_curve,
            color=model_colors[model_name],
            marker=JPEG_COMPRESSION_CONFIG["marker"],
            linewidth=JPEG_COMPRESSION_CONFIG["line_width"],
            markersize=JPEG_COMPRESSION_CONFIG["markersize"],
            label=model_labels.get(model_name, model_name),
        )

    ax.invert_xaxis()
    ax.set_title(
        JPEG_COMPRESSION_CONFIG["dataset_labels"].get(dataset_name, dataset_name),
        fontsize=JPEG_COMPRESSION_CONFIG["axes_label_fontsize"],
        pad=8,
    )
    ax.set_xlabel(
        JPEG_COMPRESSION_CONFIG["x_label"],
        fontsize=JPEG_COMPRESSION_CONFIG["axes_label_fontsize"],
        labelpad=6,
    )
    ax.set_ylabel(
        JPEG_COMPRESSION_CONFIG["y_label"],
        fontsize=JPEG_COMPRESSION_CONFIG["axes_label_fontsize"],
        labelpad=6,
    )
    ax.tick_params(axis="both", labelsize=JPEG_COMPRESSION_CONFIG["tick_label_fontsize"])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(
        loc=JPEG_COMPRESSION_CONFIG["legend_loc"],
        fontsize=JPEG_COMPRESSION_CONFIG["legend_fontsize"],
        frameon=True,
    )

    if sns is not None:
        sns.despine(ax=ax, offset=2, trim=False)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_jpeg_compression_panel(dataset_name: str, model_curves: Dict[str, list[float]]) -> Figure:
    plt.rcParams["font.family"] = JPEG_COMPRESSION_CONFIG["font_family"]
    scale = float(JPEG_COMPRESSION_CONFIG["figsize_scale"])
    width, height = JPEG_COMPRESSION_CONFIG["figsize"]
    fig, ax = plt.subplots(
        figsize=(width * scale, height * scale),
        dpi=JPEG_COMPRESSION_CONFIG["figure_dpi"],
    )
    _draw_jpeg_compression_panel(ax, dataset_name, model_curves)
    plt.tight_layout()
    return fig


def plot_jpeg_compression_comparison(dataset_curves: Dict[str, Dict[str, list[float]]]) -> Figure:
    plt.rcParams["font.family"] = JPEG_COMPRESSION_CONFIG["font_family"]
    scale = float(JPEG_COMPRESSION_CONFIG["figsize_scale"])
    width, height = JPEG_COMPRESSION_CONFIG["figsize"]
    column_count = 2
    row_count = (len(dataset_curves) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(width * column_count * scale, height * row_count * scale),
        dpi=JPEG_COMPRESSION_CONFIG["figure_dpi"],
        squeeze=False,
    )

    flat_axes = axes.ravel()
    for ax, (dataset_name, model_curves) in zip(flat_axes, dataset_curves.items()):
        _draw_jpeg_compression_panel(ax, dataset_name, model_curves)

    for ax in flat_axes[len(dataset_curves) :]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def _draw_gaussian_blur_panel(ax, panel_title: str, attack_metrics: Dict[str, Dict[str, float]]) -> None:
    attack_order = GAUSSIAN_BLUR_CONFIG["attack_order"]
    attack_labels = GAUSSIAN_BLUR_CONFIG["attack_labels"]
    defense_colors = GAUSSIAN_BLUR_CONFIG["defense_colors"][panel_title]
    x_positions = np.arange(len(attack_order))
    bar_width = float(GAUSSIAN_BLUR_CONFIG["bar_width"])
    blur_values = [attack_metrics[attack_name]["blur_asr"] for attack_name in attack_order]
    resize_values = [attack_metrics[attack_name]["resize_asr"] for attack_name in attack_order]

    ax.bar(
        x_positions - bar_width / 2,
        blur_values,
        width=bar_width,
        color=defense_colors["blur_asr"],
        alpha=float(GAUSSIAN_BLUR_CONFIG["bar_alpha"]),
        zorder=2,
        label=GAUSSIAN_BLUR_CONFIG["blur_label"],
    )
    ax.bar(
        x_positions + bar_width / 2,
        resize_values,
        width=bar_width,
        color=defense_colors["resize_asr"],
        alpha=float(GAUSSIAN_BLUR_CONFIG["bar_alpha"]),
        zorder=2,
        label=GAUSSIAN_BLUR_CONFIG["resize_label"],
    )

    ax.set_title(
        panel_title,
        fontsize=GAUSSIAN_BLUR_CONFIG["title_fontsize"],
        pad=GAUSSIAN_BLUR_CONFIG["title_pad"],
    )
    ax.set_xlabel(
        GAUSSIAN_BLUR_CONFIG["x_label"],
        fontsize=GAUSSIAN_BLUR_CONFIG["axes_label_fontsize"],
        labelpad=6,
    )
    ax.set_ylabel(
        GAUSSIAN_BLUR_CONFIG["y_label"],
        fontsize=GAUSSIAN_BLUR_CONFIG["axes_label_fontsize"],
        labelpad=6,
    )
    ax.set_ylim(*GAUSSIAN_BLUR_CONFIG["y_limit"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [attack_labels.get(attack_name, attack_name) for attack_name in attack_order],
        rotation=20,
        ha="right",
    )
    ax.tick_params(axis="both", labelsize=GAUSSIAN_BLUR_CONFIG["tick_label_fontsize"])
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.legend(
        handles=[
            Patch(facecolor=defense_colors["blur_asr"], label=GAUSSIAN_BLUR_CONFIG["blur_label"]),
            Patch(facecolor=defense_colors["resize_asr"], label=GAUSSIAN_BLUR_CONFIG["resize_label"]),
        ],
        loc=GAUSSIAN_BLUR_CONFIG["legend_loc"],
        fontsize=GAUSSIAN_BLUR_CONFIG["legend_fontsize"],
        frameon=True,
    )

    if sns is not None:
        sns.despine(ax=ax, offset=2, trim=False)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_gaussian_blur_comparison(panel_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Figure:
    plt.rcParams["font.family"] = GAUSSIAN_BLUR_CONFIG["font_family"]
    scale = float(GAUSSIAN_BLUR_CONFIG["figsize_scale"])
    width, height = GAUSSIAN_BLUR_CONFIG["figsize"]
    fig, axes = plt.subplots(
        1,
        len(panel_metrics),
        figsize=(width * len(panel_metrics) * scale, height * scale),
        dpi=GAUSSIAN_BLUR_CONFIG["figure_dpi"],
        squeeze=False,
    )

    for ax, (panel_title, attack_metrics) in zip(axes[0], panel_metrics.items()):
        _draw_gaussian_blur_panel(ax, panel_title, attack_metrics)

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
            GRADCAM_CONTOUR_CONFIG["contour_color_midlow"],
            GRADCAM_CONTOUR_CONFIG["contour_color_mid"],
            GRADCAM_CONTOUR_CONFIG["contour_color_midhigh"],
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


def plot_gradcam_contour_row(
    full_model: nn.Module,
    model_name: str,
    vis_dict: dict,
    sample_idx: int,
) -> Figure:
    clean_images = vis_dict["clean_images"]
    labels = vis_dict["labels"]
    attacks = _available_attacks(vis_dict)
    attack_keys = [_resolve_attack_key(vis_dict, attack_name) for attack_name in attacks]
    max_samples = min(clean_images.shape[0], labels.shape[0], *[vis_dict[key].shape[0] for key in attack_keys])
    if sample_idx >= max_samples:
        raise IndexError(f"sample_idx={sample_idx} is out of range for {max_samples} Grad-CAM samples.")

    column_count = 1 + len(attacks)
    device = next(full_model.parameters()).device
    target_layer = _resolve_gradcam_target_layer(full_model, model_name)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cam",
        [
            GRADCAM_CONTOUR_CONFIG["contour_color_low"],
            GRADCAM_CONTOUR_CONFIG["contour_color_midlow"],
            GRADCAM_CONTOUR_CONFIG["contour_color_mid"],
            GRADCAM_CONTOUR_CONFIG["contour_color_midhigh"],
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
        1,
        column_count,
        figsize=GRADCAM_CONTOUR_CONFIG["figsize_per_sample"],
        dpi=int(GRADCAM_CONTOUR_CONFIG["figure_dpi"]),
        squeeze=False,
    )

    label = labels[sample_idx].to(device)
    row_items = [clean_images[sample_idx]]
    row_items.extend(vis_dict[_resolve_attack_key(vis_dict, attack_name)][sample_idx] for attack_name in attacks)

    for column_idx, image in enumerate(row_items):
        ax = axes[0, column_idx]
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

    plt.tight_layout()
    return fig


def plot_noise_gradcam_sample(
    full_model: nn.Module,
    model_name: str,
    vis_dict: dict,
    sample_idx: int,
) -> Figure:
    clean_images = vis_dict["clean_images"]
    labels = vis_dict["labels"]
    attacks = _available_attacks(vis_dict)
    if not attacks:
        raise KeyError("No adversarial image tensors found for the expected attack order.")

    attack_keys = [_resolve_attack_key(vis_dict, attack_name) for attack_name in attacks]
    max_samples = min(clean_images.shape[0], labels.shape[0], *[vis_dict[key].shape[0] for key in attack_keys])
    if sample_idx >= max_samples:
        raise IndexError(f"sample_idx={sample_idx} is out of range for {max_samples} visualization samples.")

    column_count = 1 + len(attacks)
    noise_width, noise_height = MAGNIFIED_NOISE_CONFIG["figsize_per_sample"]
    _, cam_height = GRADCAM_CONTOUR_CONFIG["figsize_per_sample"]
    device = next(full_model.parameters()).device
    target_layer = _resolve_gradcam_target_layer(full_model, model_name)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cam",
        [
            GRADCAM_CONTOUR_CONFIG["contour_color_low"],
            GRADCAM_CONTOUR_CONFIG["contour_color_midlow"],
            GRADCAM_CONTOUR_CONFIG["contour_color_mid"],
            GRADCAM_CONTOUR_CONFIG["contour_color_midhigh"],
            GRADCAM_CONTOUR_CONFIG["contour_color_high"],
        ],
    )
    contour_levels = np.linspace(
        float(GRADCAM_CONTOUR_CONFIG["contour_min_level"]),
        float(GRADCAM_CONTOUR_CONFIG["contour_max_level"]),
        int(GRADCAM_CONTOUR_CONFIG["contour_levels"]),
    )

    plt.rcParams["font.family"] = MAGNIFIED_NOISE_CONFIG["font_family"]
    fig, axes = plt.subplots(
        2,
        column_count,
        figsize=(noise_width, noise_height + cam_height),
        dpi=int(MAGNIFIED_NOISE_CONFIG["figure_dpi"]),
        squeeze=False,
    )

    clean_img = clean_images[sample_idx]
    magnification = float(MAGNIFIED_NOISE_CONFIG["noise_magnification"])
    row_images = [clean_img]
    column_titles = ["Clean"]

    for attack_name in attacks:
        adv_img = vis_dict[_resolve_attack_key(vis_dict, attack_name)][sample_idx]
        noise_vis = torch.clamp(clean_img + (adv_img - clean_img) * magnification, 0.0, 1.0)
        row_images.append(noise_vis)
        column_titles.append(attack_name)

    for column_idx, image in enumerate(row_images):
        ax = axes[0, column_idx]
        ax.imshow(_to_image_array(image))
        ax.set_xticks([])
        ax.set_yticks([])
        border_width = float(MAGNIFIED_NOISE_CONFIG["image_border_width"])
        for spine in ax.spines.values():
            spine.set_visible(border_width > 0)
            spine.set_linewidth(border_width)
            spine.set_edgecolor(MAGNIFIED_NOISE_CONFIG["image_border_color"])
        ax.set_title(column_titles[column_idx], fontsize=MAGNIFIED_NOISE_CONFIG["title_fontsize"])

    label = labels[sample_idx].to(device)
    gradcam_images = [clean_img]
    gradcam_images.extend(vis_dict[_resolve_attack_key(vis_dict, attack_name)][sample_idx] for attack_name in attacks)

    for column_idx, image in enumerate(gradcam_images):
        ax = axes[1, column_idx]
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

    plt.tight_layout()
    return fig
