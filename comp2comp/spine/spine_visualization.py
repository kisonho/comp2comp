"""
@author: louisblankemeier
"""

import os
from pathlib import Path
from typing import Union

import numpy as np

from comp2comp.visualization.detectron_visualizer import Visualizer

def spine_binary_segmentation_overlay(
    img_in: Union[str, Path],
    mask: Union[str, Path],
    base_path: Union[str, Path],
    file_name: str,
    figure_text_key=None,
    spine_hus=None,
    seg_hus=None,
    spine=True,
    model_type=None,
    pixel_spacing=None,
):
    """Save binary segmentation overlay.
    Args:
        img_in (Union[str, Path]): Path to the input image.
        mask (Union[str, Path]): Path to the mask.
        base_path (Union[str, Path]): Path to the output directory.
        file_name (str): Output file name.
        centroids (list, optional): List of centroids. Defaults to None.
        figure_text_key (dict, optional): Figure text key. Defaults to None.
        spine_hus (list, optional): List of HU values. Defaults to None.
        spine (bool, optional): Spine flag. Defaults to True.
        model_type (Models): Model type. Defaults to None.
    """
    spine_levels = [*(f"C{i}" for i in range(1, 8)), *(f"T{i}" for i in range(1, 13)), *(f"L{i}" for i in range(1, 6))]
    level_colors = {}
    hue_steps = len(spine_levels)
    for i, level in enumerate(spine_levels):
        h = i / hue_steps
        c = 0.85
        x = c * (1 - abs((h * 6) % 2 - 1))
        if h < 1 / 6:
            r, g, b = c, x, 0.0
        elif h < 2 / 6:
            r, g, b = x, c, 0.0
        elif h < 3 / 6:
            r, g, b = 0.0, c, x
        elif h < 4 / 6:
            r, g, b = 0.0, x, c
        elif h < 5 / 6:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        level_colors[level] = np.array([r + 0.15, g + 0.15, b + 0.15], dtype=np.float32)

    _ROI_COLOR = np.array([1.000, 0.340, 0.200])

    _SPINE_TEXT_OFFSET_FROM_TOP = 10.0
    _SPINE_TEXT_OFFSET_FROM_RIGHT = 40.0
    _SPINE_TEXT_VERTICAL_SPACING = 14.0

    img_in = np.clip(img_in, -300, 1800)
    img_in = normalize_img(img_in) * 255.0
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = level_colors.get(level, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois
    for i, _ in enumerate(levels):
        color = _ROI_COLOR
        edge_color = color
        vis.draw_binary_mask(
            mask[:, :, num_levels + i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    vis.draw_text(
        text="ROI",
        position=(
            mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT - 35,
            _SPINE_TEXT_OFFSET_FROM_TOP,
        ),
        color=[1, 1, 1],
        font_size=9,
        horizontal_alignment="center",
    )

    vis.draw_text(
        text="Seg",
        position=(
            mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
            _SPINE_TEXT_OFFSET_FROM_TOP,
        ),
        color=[1, 1, 1],
        font_size=9,
        horizontal_alignment="center",
    )

    # draw text and lines
    for i, level in enumerate(levels):
        vis.draw_text(
            text=f"{level}:",
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT - 80,
                _SPINE_TEXT_VERTICAL_SPACING * (i + 1) + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=level_colors.get(level, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            font_size=9,
            horizontal_alignment="left",
        )
        vis.draw_text(
            text=f"{round(float(spine_hus[level]))}",
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT - 35,
                _SPINE_TEXT_VERTICAL_SPACING * (i + 1) + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=level_colors.get(level, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            font_size=9,
            horizontal_alignment="center",
        )
        vis.draw_text(
            text=f"{round(float(seg_hus[level]))}",
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                _SPINE_TEXT_VERTICAL_SPACING * (i + 1) + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=level_colors.get(level, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            font_size=9,
            horizontal_alignment="center",
        )

        """
        vis.draw_line(
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=level_colors.get(level, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            linestyle="dashed",
            linewidth=0.25,
        )
        """

    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())
