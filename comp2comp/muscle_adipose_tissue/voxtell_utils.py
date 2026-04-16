import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download


VOXTELL_MODEL_NAME = "voxtell_v1.1"
VOXTELL_REPO_ID = "mrokuss/VoxTell"
VOXTELL_PROMPTS = (
    "skeletal muscle",
    "visceral adipose tissue",
    "subcutaneous adipose tissue",
    "intramuscular adipose tissue",
)
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


def save_combined_voxtell_segmentation(
    segmentation: np.ndarray, image_nib: nib.Nifti1Image, output_path: str
) -> None:
    combined = np.zeros(segmentation.shape[1:], dtype=np.uint8)
    for idx in range(segmentation.shape[0]):
        combined[segmentation[idx].astype(bool)] = idx + 1

    nib.save(nib.Nifti1Image(combined, image_nib.affine, image_nib.header), output_path)


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
    voxtell_image, _ = reader.read_images([nifti_path])
    predictor = VoxTellPredictor(model_dir=model_path, device=device)
    segmentation = predictor.predict_single_image(voxtell_image, list(VOXTELL_PROMPTS))

    save_combined_voxtell_segmentation(segmentation, image_nib, output_path)

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
