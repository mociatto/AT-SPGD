from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

MAGNIFIED_NOISE_CONFIG = {
    "font_family": "serif",
    "title_fontsize": 10,
    "label_fontsize": 10,
    "figure_dpi": 300,
    "noise_magnification": 10.0,
    "num_samples": 5,
    "figsize_per_sample": (10, 4),
    "image_border_width": 0.0,
    "image_border_color": "black",
}

ATTACK_ORDER = ["PGD", "APGD", "MIFGSM", "SSA", "AT-SPGD (Ours)"]


def _attack_key(attack_name: str) -> str:
    return f"adv_{attack_name}"


def _to_image_array(image: torch.Tensor):
    return image.detach().cpu().float().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)


def _available_attacks(vis_dict: Dict[str, torch.Tensor]) -> List[str]:
    return [attack_name for attack_name in ATTACK_ORDER if _attack_key(attack_name) in vis_dict]


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
            adv_img = vis_dict[_attack_key(attack_name)][sample_idx]
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
