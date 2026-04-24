import argparse
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json

from comp2comp.models.models import Models
from comp2comp.muscle_adipose_tissue.data import Dataset, _window, parse_windows, predict
from comp2comp.muscle_adipose_tissue.muscle_adipose_tissue import (
    MuscleAdiposeTissuePostProcessing,
    MuscleAdiposeTissueSegmentation,
    TS_MUSCLE_ONLY_MODELS,
    TS_TOTAL_MODEL,
)
from comp2comp.utils.process import (
    _configure_nnunet_paths,
    _configure_nnunet_predictor_compat,
    _configure_torch_checkpoint_loading,
)


class EvalConfigs(Namespace):
    batch_size: int
    data_dir: str
    model: str
    outputs_dir: str


class NRRDSliceDataset(keras.utils.Sequence):
    def __init__(
        self,
        image: np.ndarray,
        spacing: tuple[float, float],
        /,
        *,
        batch_size: int,
        windows: tuple[str, ...],
    ) -> None:
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        self.image = image.astype(np.float32, copy=False)
        self.spacing = spacing
        self.batch_size = batch_size
        self.windows = windows

    def __len__(self) -> int:
        return (len(self.image) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[dict[str, object]]]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.image))
        xs = self.image[start:stop]
        params = [{"spacing": self.spacing, "image": x} for x in xs]

        if self.windows:
            xs = _window(xs, parse_windows(self.windows))
        else:
            xs = xs[..., np.newaxis]

        return xs, params


def _mask_to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    return np.stack([(mask == (i + 1)) for i in range(num_classes)], axis=-1)


def _extract_labeled_slice(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return mask, 0
    if mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape {mask.shape}")

    nonzero_slices = np.where(np.any(mask > 0, axis=(1, 2)))[0]
    if len(nonzero_slices) == 0:
        slice_idx = mask.shape[0] // 2
    else:
        slice_idx = int(nonzero_slices[len(nonzero_slices) // 2])
    return mask[slice_idx], slice_idx


def _extract_reference_slice(image: sitk.Image, slice_idx: int) -> sitk.Image:
    if image.GetDimension() == 2:
        return image
    return image[:, :, slice_idx]


def _image_geometry(image: sitk.Image) -> dict[str, object]:
    return {
        "size": tuple(int(v) for v in image.GetSize()),
        "spacing": tuple(float(v) for v in image.GetSpacing()),
        "origin": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
    }


def _save_prediction(
    pred_mask: np.ndarray,
    reference_image: sitk.Image,
    output_path: Path,
    *,
    slice_idx: int | None = None,
) -> None:
    pred_seg = np.zeros(pred_mask.shape[:-1], dtype=np.uint8)
    foreground = np.any(pred_mask, axis=-1)
    pred_seg[foreground] = (
        np.argmax(pred_mask.astype(np.uint8), axis=-1)[foreground] + 1
    ).astype(np.uint8)

    ref_size_xyz = tuple(int(v) for v in reference_image.GetSize())
    ref_shape = tuple(reversed(ref_size_xyz))
    if reference_image.GetDimension() == 3:
        if pred_seg.ndim == 2:
            full_pred_seg = np.zeros(ref_shape, dtype=np.uint8)
            target_slice_idx = (
                slice_idx if slice_idx is not None else ref_shape[0] // 2
            )
            full_pred_seg[target_slice_idx] = pred_seg
            pred_seg = full_pred_seg
        elif pred_seg.ndim == 3 and pred_seg.shape != ref_shape:
            full_pred_seg = np.zeros(ref_shape, dtype=np.uint8)
            if pred_seg.shape[1:] != ref_shape[1:]:
                raise ValueError(
                    f"Prediction shape {pred_seg.shape} does not match reference in-plane shape {ref_shape}"
                )

            if pred_seg.shape[0] == 3 and np.array_equal(pred_seg[0], pred_seg[1]) and np.array_equal(pred_seg[1], pred_seg[2]):
                target_slice_idx = (
                    slice_idx if slice_idx is not None else ref_shape[0] // 2
                )
                full_pred_seg[target_slice_idx] = pred_seg[1]
            else:
                start_idx = 0
                if slice_idx is not None:
                    start_idx = max(
                        0,
                        min(
                            ref_shape[0] - pred_seg.shape[0],
                            slice_idx - (pred_seg.shape[0] // 2),
                        ),
                    )
                full_pred_seg[start_idx : start_idx + pred_seg.shape[0]] = pred_seg
            pred_seg = full_pred_seg
    elif pred_seg.ndim == 3 and pred_seg.shape[0] == 1:
        pred_seg = pred_seg[0]

    pred_image = sitk.GetImageFromArray(pred_seg)
    pred_image.CopyInformation(reference_image)
    sitk.WriteImage(pred_image, str(output_path))
    if pred_seg.ndim == 2:
        sitk.WriteImage(
            sitk.GetImageFromArray(pred_seg.astype(np.uint8)),
            str(output_path.with_suffix(".png")),
        )


def _ts_model_folders(model_name: str) -> list[Path]:
    models_dir = Path("models").resolve()
    if model_name == "ts_abdominal_muscles_v0.0.1":
        return [
            models_dir
            / "Dataset952_abdominal_muscles_167subj"
            / "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres_high"
        ]
    if model_name == "ts_headneck_muscles_v0.0.1":
        return [
            models_dir
            / "Dataset778_headneck_muscles_part1_492subj"
            / "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres_high",
            models_dir
            / "Dataset779_headneck_muscles_part2_492subj"
            / "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres_high",
        ]
    raise ValueError(f"Unsupported TS muscle model: {model_name}")


def _predict_with_totalsegmentator_api(
    image_itk: sitk.Image,
    segmentation: MuscleAdiposeTissueSegmentation,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float]], np.ndarray]:
    models_dir = str(Path("models").resolve())
    _configure_torch_checkpoint_loading()
    _configure_nnunet_paths(models_dir)
    _configure_nnunet_predictor_compat()

    task_config = segmentation._totalseg_task_config()
    image_np = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    keep_center_only = image_np.ndim == 2
    volume = image_np if image_np.ndim == 3 else np.repeat(image_np[np.newaxis, ...], 3, axis=0)
    spacing_xy = image_itk.GetSpacing()

    with tempfile.TemporaryDirectory(prefix="comp2comp_eval_ts_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        nifti_path = temp_dir_path / "converted_dcm_multilevel.nii.gz"
        output_path = temp_dir_path / task_config["output_name"]

        if keep_center_only:
            image_3d = sitk.GetImageFromArray(volume.astype(np.float32, copy=False))
            origin_xy = image_itk.GetOrigin()
            direction_xy = image_itk.GetDirection()
            image_3d.SetSpacing((float(spacing_xy[0]), float(spacing_xy[1]), 1.0))
            image_3d.SetOrigin((float(origin_xy[0]), float(origin_xy[1]), 0.0))
            image_3d.SetDirection(
                (
                    float(direction_xy[0]),
                    float(direction_xy[1]),
                    0.0,
                    float(direction_xy[2]),
                    float(direction_xy[3]),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
            )
            sitk.WriteImage(image_3d, str(nifti_path))
        else:
            sitk.WriteImage(image_itk, str(nifti_path))

        pred_volume: np.ndarray | None = None
        if segmentation.model_name == TS_TOTAL_MODEL:
            from totalsegmentator.python_api import totalsegmentator

            predictor_device = "gpu" if torch.cuda.is_available() else "cpu"
            totalsegmentator(
                input=str(nifti_path),
                output=str(output_path),
                ml=True,
                nr_thr_resamp=1,
                nr_thr_saving=1,
                fast=False,
                nora_tag="None",
                preview=False,
                task=task_config["task"],
                roi_subset=None,
                statistics=False,
                radiomics=False,
                crop_path=None,
                body_seg=False,
                force_split=False,
                output_type="nifti",
                quiet=False,
                verbose=False,
                test=0,
                skip_saving=False,
                device=predictor_device,
                license_number=None,
                statistics_exclude_masks_at_border=True,
                no_derived_masks=False,
                v1_order=False,
            )
            pred_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path))).astype(
                np.uint8,
                copy=False,
            )
        else:
            predictor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for model_folder in _ts_model_folders(segmentation.model_name):
                dataset_json = json.loads((model_folder / "dataset.json").read_text())
                reader_writer_cls = determine_reader_writer_from_dataset_json(
                    dataset_json,
                    example_file=str(nifti_path),
                    allow_nonmatching_filename=True,
                    verbose=False,
                )
                data, data_properties = reader_writer_cls().read_images([str(nifti_path)])

                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_device=predictor_device.type == "cuda",
                    device=predictor_device,
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False,
                )
                predictor.initialize_from_trained_model_folder(
                    str(model_folder),
                    use_folds=(0,),
                    checkpoint_name="checkpoint_final.pth",
                )
                pred = cast(
                    np.ndarray,
                    predictor.predict_single_npy_array(data, data_properties),
                ).astype(np.uint8, copy=False)
                pred_volume = pred if pred_volume is None else np.maximum(pred_volume, pred)

    if pred_volume is None:
        raise RuntimeError(f"No nnU-Net prediction produced for {task_config['task']}")

    images = [slc for slc in volume]
    preds = [slc for slc in pred_volume]

    if keep_center_only:
        center_idx = len(images) // 2
        images = [images[center_idx]]
        preds = [preds[center_idx]]
    spacings = [(float(spacing_xy[1]), float(spacing_xy[0])) for _ in images]
    masks = segmentation._build_totalseg_masks(preds, segmentation.model_type.categories)
    return images, masks, spacings, pred_volume


def eval(configs: EvalConfigs) -> dict[str, float]:
    dataset_dir = Path(configs.data_dir)
    image_paths = sorted((dataset_dir / "images").glob("*.nrrd"))
    outputs_dir = Path(configs.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _configure_torch_checkpoint_loading()
    _configure_nnunet_paths(str(Path("models").resolve()))
    _configure_nnunet_predictor_compat()

    model_type = Models.model_from_name(configs.model)
    if model_type is None:
        raise ValueError(f"Unknown comp2comp model: {configs.model}")
    model_windows = cast(tuple[str, ...], getattr(model_type, "windows", ()))
    segmentation = MuscleAdiposeTissueSegmentation(configs.batch_size, configs.model)
    model = (
        None
        if configs.model in (*TS_MUSCLE_ONLY_MODELS, TS_TOTAL_MODEL)
        else cast(keras.Model, model_type.load_model("models"))
    )

    pipeline = SimpleNamespace(
        muscle_adipose_tissue_model_type=model_type,
        muscle_adipose_tissue_model_name=configs.model,
    )
    postprocess = MuscleAdiposeTissuePostProcessing()

    total_dice = 0.0
    num_samples = 0

    for image_path in tqdm(image_paths):
        mask_path = dataset_dir / "masks" / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        image_itk = sitk.ReadImage(str(image_path))
        mask_itk = sitk.ReadImage(str(mask_path))
        image_np = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
        mask_np = sitk.GetArrayFromImage(mask_itk)
        labeled_mask_np, labeled_slice_idx = _extract_labeled_slice(mask_np)
        raw_pred_volume = None

        print(f"\n=== Evaluating {image_path.name} ===")
        print(f"Image geometry: {_image_geometry(image_itk)}")
        print(f"Mask geometry: {_image_geometry(mask_itk)}")
        print(f"Image array shape: {tuple(int(v) for v in image_np.shape)}")
        print(f"Mask array shape: {tuple(int(v) for v in mask_np.shape)}")
        print(f"GT labels present: {np.unique(mask_np).tolist()}")
        print(f"Labeled slice index: {labeled_slice_idx}")
        print(f"GT labels on labeled slice: {np.unique(labeled_mask_np).tolist()}")

        if configs.model in (*TS_MUSCLE_ONLY_MODELS, TS_TOTAL_MODEL):
            images, preds, spacings, raw_pred_volume = _predict_with_totalsegmentator_api(
                image_itk,
                segmentation,
            )
            print(f"TS raw prediction volume shape: {tuple(int(v) for v in raw_pred_volume.shape)}")
            print(f"TS raw labels present: {np.unique(raw_pred_volume).tolist()}")
            if raw_pred_volume.ndim == 3 and labeled_slice_idx < raw_pred_volume.shape[0]:
                print(
                    "TS raw labels on labeled slice: "
                    f"{np.unique(raw_pred_volume[labeled_slice_idx]).tolist()}"
                )
        else:
            dataset = cast(
                Dataset,
                NRRDSliceDataset(
                    image_np,
                    cast(tuple[float, float], image_itk.GetSpacing()[:2]),
                    batch_size=configs.batch_size,
                    windows=model_windows,
                ),
            )
            _, preds, params = predict(
                model,
                dataset,
                batch_size=configs.batch_size,
                num_workers=0,
                use_multiprocessing=False,
            )
            images = [cast(np.ndarray, p["image"]) for p in params]
            spacings = [cast(tuple[float, float], p["spacing"]) for p in params]

        processed = postprocess(
            pipeline,
            images=images,
            preds=preds,
            spacings=spacings,
        )
        pred_mask = np.stack(processed["masks"], axis=0).astype(bool, copy=False)
        print(f"Processed prediction mask shape: {tuple(int(v) for v in pred_mask.shape)}")
        print(
            "Processed foreground voxels by class across volume: "
            f"{pred_mask.sum(axis=(0, 1, 2), dtype=np.int64).tolist()}"
        )
        full_pred_mask = pred_mask
        _save_prediction(
            full_pred_mask,
            image_itk,
            outputs_dir / image_path.name,
            slice_idx=labeled_slice_idx,
        )
        if labeled_slice_idx >= pred_mask.shape[0]:
            raise IndexError(
                f"Labeled slice index {labeled_slice_idx} out of range for prediction shape {pred_mask.shape}"
            )
        pred_mask = pred_mask[labeled_slice_idx : labeled_slice_idx + 1]
        print(
            "Processed foreground voxels by class on labeled slice: "
            f"{pred_mask.sum(axis=(0, 1, 2), dtype=np.int64).tolist()}"
        )
        gt_mask = _mask_to_one_hot(labeled_mask_np, pred_mask.shape[-1])
        print(
            "GT foreground voxels by class on labeled slice: "
            f"{gt_mask.sum(axis=(0, 1, 2), dtype=np.int64).tolist()}"
        )

        intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
        denominator = pred_mask.sum(dtype=np.float64) + gt_mask.sum(dtype=np.float64)
        dice = 1.0 if denominator == 0 else (2.0 * intersection) / denominator

        total_dice += dice
        num_samples += 1

    return {"dice": total_dice / max(num_samples, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("outputs_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    configs = parser.parse_args(namespace=EvalConfigs())
    results = eval(configs)
    print(results)
