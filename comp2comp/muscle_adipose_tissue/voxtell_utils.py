import os
from itertools import permutations
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download


VOXTELL_MODEL_NAME = "voxtell_v1.1"
VOXTELL_REPO_ID = "mrokuss/VoxTell"
VOXTELL_PROMPTS = (
    "oblique muscle",
    "abdominis muscle",
    "psoas muscle",
    "spinalis muscle",
)
VOXTELL_MUSCLE_PROMPT_INDICES = (0, 1, 2, 3)
VOXTELL_COMP2COMP_CHANNELS = {
    "muscle": VOXTELL_MUSCLE_PROMPT_INDICES,
    "vat": (),
    "sat": (),
    "imat": (),
}
VOXTELL_CATEGORIES = {
    "muscle": 0,
    "vat": 1,
    "sat": 2,
    "imat": 3,
}


def is_voxtell_model(model_name: str) -> bool:
    return model_name == VOXTELL_MODEL_NAME


def ensure_voxtell_checkpoint(model_dir: str) -> str:
    cache_root = Path(model_dir) / ".voxtell"
    model_path = cache_root / VOXTELL_MODEL_NAME

    if (model_path / "plans.json").exists():
        print("VoxTell model already downloaded.")
        return str(model_path)

    cache_root.mkdir(parents=True, exist_ok=True)
    print("Downloading VoxTell model...")
    snapshot_download(
        repo_id=VOXTELL_REPO_ID,
        allow_patterns=[f"{VOXTELL_MODEL_NAME}/*", "*.json"],
        local_dir=str(cache_root),
    )
    print("VoxTell model downloaded.")
    return str(model_path)


def select_voxtell_input_path(output_dir: str) -> Tuple[str, str]:
    segmentations_dir = Path(output_dir) / "segmentations"
    multilevel_path = segmentations_dir / "converted_dcm_multilevel.nii.gz"
    full_volume_path = segmentations_dir / "converted_dcm.nii.gz"

    if multilevel_path.exists():
        return str(multilevel_path), str(segmentations_dir / "multilevel_muscle_fat_seg.nii.gz")

    return str(full_volume_path), str(segmentations_dir / "muscle_fat_seg.nii.gz")


def ensure_voxtell_slice_metadata(inference_pipeline, num_slices: int) -> None:
    if hasattr(inference_pipeline, "dicom_file_paths") and inference_pipeline.dicom_file_paths:
        if not hasattr(inference_pipeline, "dicom_file_names"):
            inference_pipeline.dicom_file_names = [
                Path(str(dicom_file_path)).stem
                for dicom_file_path in inference_pipeline.dicom_file_paths
            ]
        return

    inference_pipeline.dicom_file_paths = [f"slice_{idx:04d}" for idx in range(num_slices)]
    inference_pipeline.dicom_file_names = [f"slice_{idx:04d}" for idx in range(num_slices)]


def _transform_image_slice(image_slice: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(image_slice, axis=0), axis=1).T


def _transform_mask_slice(mask_slice: np.ndarray) -> np.ndarray:
    return np.transpose(np.flip(np.flip(mask_slice, axis=0), axis=1), (1, 0, 2))


def align_voxtell_segmentation(
    segmentation: np.ndarray, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    spatial_shape = segmentation.shape[1:]
    if spatial_shape == image_shape:
        return segmentation

    for axes in permutations(range(3)):
        if tuple(spatial_shape[axis] for axis in axes) == image_shape:
            return np.transpose(segmentation, (0,) + tuple(axis + 1 for axis in axes))

    raise ValueError(
        "VoxTell segmentation shape does not match input image shape: "
        f"segmentation spatial shape {spatial_shape}, image shape {image_shape}."
    )


def align_voxtell_segmentation_to_image(
    segmentation: np.ndarray,
    voxtell_shape: Tuple[int, int, int],
    image_shape: Tuple[int, int, int],
) -> np.ndarray:
    segmentation = align_voxtell_segmentation(segmentation, voxtell_shape)
    segmentation = np.transpose(segmentation, (0, 3, 2, 1))

    if segmentation.shape[1:] != image_shape:
        raise ValueError(
            "VoxTell segmentation shape does not match input image shape after "
            f"reorientation: segmentation spatial shape {segmentation.shape[1:]}, "
            f"image shape {image_shape}."
        )

    return segmentation


def combine_voxtell_segmentation(segmentation: np.ndarray) -> np.ndarray:
    combined = np.zeros(segmentation.shape[1:], dtype=np.uint8)
    for idx in range(segmentation.shape[0]):
        combined[segmentation[idx].astype(bool)] = idx + 1
    return combined


def remap_voxtell_segmentation(segmentation: np.ndarray) -> np.ndarray:
    required_channels = max(
        idx
        for prompt_indices in VOXTELL_COMP2COMP_CHANNELS.values()
        for idx in prompt_indices
    ) + 1
    if segmentation.shape[0] < required_channels:
        raise ValueError(
            "VoxTell segmentation does not contain all prompt channels: "
            f"expected at least {required_channels}, got {segmentation.shape[0]}."
        )

    mapped = np.zeros(
        (len(VOXTELL_CATEGORIES),) + segmentation.shape[1:], dtype=np.uint8
    )
    for label, comp2comp_idx in VOXTELL_CATEGORIES.items():
        prompt_indices = VOXTELL_COMP2COMP_CHANNELS[label]
        if not prompt_indices:
            continue
        mapped[comp2comp_idx] = np.any(segmentation[prompt_indices].astype(bool), axis=0)
    return mapped


def save_combined_voxtell_segmentation(
    segmentation: np.ndarray, reader, properties: dict, output_path: str
) -> None:
    combined = combine_voxtell_segmentation(segmentation)

    reader.write_seg(combined, output_path, properties)


def predict_voxtell_volume(
    model_path: str, nifti_path: str, output_path: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[float, float]]]:
    try:
        import torch
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        from voxtell.inference.predictor import VoxTellPredictor
    except ImportError as exc:
        raise ImportError(
            "VoxTell dependencies are not available. Install the package with the "
            "updated Comp2Comp requirements to use muscle_fat_model=voxtell_v1.1."
        ) from exc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_nib = nib.as_closest_canonical(nib.load(nifti_path))
    image = image_nib.get_fdata()
    reader = NibabelIOWithReorient()
    voxtell_image, properties = reader.read_images([nifti_path])
    predictor = VoxTellPredictor(model_dir=model_path, device=device)
    segmentation = predictor.predict_single_image(voxtell_image, list(VOXTELL_PROMPTS))
    segmentation = remap_voxtell_segmentation(segmentation)

    segmentation_for_export = align_voxtell_segmentation(
        segmentation, voxtell_image.shape[1:]
    )
    save_combined_voxtell_segmentation(
        segmentation_for_export, reader, properties, output_path
    )
    segmentation = align_voxtell_segmentation_to_image(
        segmentation, voxtell_image.shape[1:], image.shape
    )

    images = []
    masks = []
    spacings = []
    for slice_idx in range(image.shape[-1]):
        images.append(_transform_image_slice(image[:, :, slice_idx]))
        slice_mask = np.transpose(segmentation[:, :, :, slice_idx], (1, 2, 0)).astype(
            np.uint8
        )
        masks.append(_transform_mask_slice(slice_mask))
        spacings.append(image_nib.header.get_zooms()[0:2])

    return images, masks, spacings
