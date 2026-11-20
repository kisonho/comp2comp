import os
import zipfile
from pathlib import Path
from time import perf_counter
from typing import List, Union

import cv2
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import wget
from keras import backend as K
from tqdm import tqdm

from comp2comp.inference_class_base import InferenceClass
from comp2comp.metrics.metrics import CrossSectionalArea, HounsfieldUnits
from comp2comp.models.models import Models
from comp2comp.muscle_adipose_tissue.data import Dataset, predict


class MuscleAdiposeTissueSegmentation(InferenceClass):
    """Muscle adipose tissue segmentation class."""

    CERVICAL_LEVELS = {f"C{i}" for i in range(1, 8)}

    def __init__(
        self,
        batch_size: int,
        model_name: str,
        model_dir: str = None,
        backup_model_name: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_type = Models.model_from_name(model_name)
        self.backup_model_name = backup_model_name
        self.backup_model_type = (
            Models.model_from_name(backup_model_name) if backup_model_name else None
        )
        if self.model_type is None:
            raise ValueError(f"Invalid muscle/fat model: {model_name}")
        if backup_model_name and self.backup_model_type is None:
            raise ValueError(f"Invalid backup muscle/fat model: {backup_model_name}")

    def _uses_backup_model(self, level_name):
        return (
            self.backup_model_type is not None
            and str(level_name).upper() in self.CERVICAL_LEVELS
        )

    def forward_pass_2d(self, files, model, model_type, model_name):
        dataset = Dataset(files, windows=model_type.windows)
        num_workers = 1

        print("Computing segmentation masks using {}...".format(model_name))
        start_time = perf_counter()
        _, preds, results = predict(
            model,
            dataset,
            num_workers=num_workers,
            use_multiprocessing=num_workers > 1,
            batch_size=self.batch_size,
        )
        K.clear_session()
        print(
            f"Completed {len(files)} segmentations in {(perf_counter() - start_time):.2f} seconds."
        )
        for i in range(len(results)):
            results[i]["preds"] = preds[i]
        return results

    def download_muscle_adipose_tissue_model(self, model_dir: Union[str, Path]):
        download_dir = Path(
            os.path.join(
                model_dir,
                ".totalsegmentator/nnunet/results/nnUNet/2d/Task927_FatMuscle/nnUNetTrainerV2__nnUNetPlansv2.1",
            )
        )
        all_path = download_dir / "all"
        if not os.path.exists(all_path):
            download_dir.mkdir(parents=True, exist_ok=True)
            wget.download(
                "https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/resolve/main/all.zip",
                out=os.path.join(download_dir, "all.zip"),
            )
            with zipfile.ZipFile(os.path.join(download_dir, "all.zip"), "r") as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(os.path.join(download_dir, "all.zip"))
            wget.download(
                "https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/resolve/main/plans.pkl",
                out=os.path.join(download_dir, "plans.pkl"),
            )
            print("Muscle and adipose tissue model downloaded.")
        else:
            print("Muscle and adipose tissue model already downloaded.")

    def __call__(self, inference_pipeline):
        inference_pipeline.muscle_adipose_tissue_model_type = self.model_type
        inference_pipeline.muscle_adipose_tissue_model_name = self.model_name
        inference_pipeline.backup_muscle_adipose_tissue_model_type = self.backup_model_type
        inference_pipeline.backup_muscle_adipose_tissue_model_name = self.backup_model_name

        dicom_file_paths = list(inference_pipeline.dicom_file_paths)
        if not hasattr(inference_pipeline, "dicom_file_names"):
            inference_pipeline.dicom_file_names = [
                dicom_file_path.stem for dicom_file_path in dicom_file_paths
            ]
        dicom_file_names = list(inference_pipeline.dicom_file_names)
        output_count = len(dicom_file_names)

        backup_indices = [
            i for i, level in enumerate(dicom_file_names) if self._uses_backup_model(level)
        ]
        backup_index_set = set(backup_indices)
        primary_indices = [i for i in range(output_count) if i not in backup_index_set]

        images = [None] * output_count
        preds = [None] * output_count
        spacings = [None] * output_count
        model_types = [None] * output_count
        model_names = [None] * output_count

        if primary_indices:
            primary_images, primary_preds, primary_spacings = self._segment_primary(
                inference_pipeline, primary_indices
            )
            for output_index, image, pred, spacing in zip(
                primary_indices, primary_images, primary_preds, primary_spacings
            ):
                images[output_index] = image
                preds[output_index] = pred
                spacings[output_index] = spacing
                model_types[output_index] = self.model_type
                model_names[output_index] = self.model_name

        if backup_indices:
            backup_files = [dicom_file_paths[i] for i in backup_indices]
            backup_model = self.backup_model_type.load_model(inference_pipeline.model_dir)
            backup_results = self.forward_pass_2d(
                backup_files, backup_model, self.backup_model_type, self.backup_model_name
            )
            for output_index, result in zip(backup_indices, backup_results):
                images[output_index] = result["image"]
                preds[output_index] = result["preds"]
                spacings[output_index] = result["spacing"]
                model_types[output_index] = self.backup_model_type
                model_names[output_index] = self.backup_model_name

        return {
            "images": images,
            "preds": preds,
            "spacings": spacings,
            "model_types": model_types,
            "model_names": model_names,
        }

    def _segment_primary(self, inference_pipeline, selected_indices):
        if self.model_name == "stanford_v0.0.2":
            return self._segment_multilevel_nnunet(inference_pipeline, selected_indices)

        dicom_file_paths = [
            inference_pipeline.dicom_file_paths[i] for i in selected_indices
        ]
        model = self.model_type.load_model(inference_pipeline.model_dir)
        results = self.forward_pass_2d(
            dicom_file_paths, model, self.model_type, self.model_name
        )
        images = [result["image"] for result in results]
        preds = [result["preds"] for result in results]
        spacings = [result["spacing"] for result in results]
        return images, preds, spacings

    def _segment_multilevel_nnunet(self, inference_pipeline, selected_indices):
        self.download_muscle_adipose_tissue_model(inference_pipeline.model_dir)
        nifti_path = os.path.join(
            inference_pipeline.output_dir,
            "segmentations",
            "converted_dcm_multilevel.nii.gz",
        )
        output_path = os.path.join(
            inference_pipeline.output_dir,
            "segmentations",
            "multilevel_muscle_fat_seg.nii.gz",
        )
        if selected_indices != list(range(len(inference_pipeline.dicom_file_paths))):
            source_nifti = nib.load(nifti_path)
            selected_data = source_nifti.get_fdata()[:, :, selected_indices]
            nifti_path = os.path.join(
                inference_pipeline.output_dir,
                "segmentations",
                "converted_dcm_multilevel_primary.nii.gz",
            )
            output_path = os.path.join(
                inference_pipeline.output_dir,
                "segmentations",
                "multilevel_muscle_fat_seg_primary.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(selected_data, source_nifti.affine, source_nifti.header),
                nifti_path,
            )

        from nnunet.inference import predict

        predict.predict_cases(
            model=os.path.join(
                inference_pipeline.model_dir,
                ".totalsegmentator/nnunet/results/nnUNet/2d/Task927_FatMuscle/nnUNetTrainerV2__nnUNetPlansv2.1",
            ),
            list_of_lists=[[nifti_path]],
            output_filenames=[output_path],
            folds="all",
            save_npz=False,
            num_threads_preprocessing=8,
            num_threads_nifti_save=8,
            segs_from_prev_stage=None,
            do_tta=False,
            mixed_precision=True,
            overwrite_existing=False,
            all_in_gpu=False,
            step_size=0.5,
            checkpoint_name="model_final_checkpoint",
            segmentation_export_kwargs=None,
        )

        image_nib = nib.load(nifti_path)
        image_nib = nib.as_closest_canonical(image_nib)
        image = image_nib.get_fdata()
        pred = nib.load(output_path)
        pred = nib.as_closest_canonical(pred)
        pred = pred.get_fdata()

        images = [image[:, :, i] for i in range(image.shape[-1])]
        preds = [pred[:, :, i] for i in range(pred.shape[-1])]

        # flip both axes and transpose
        images = [np.flip(np.flip(image, axis=0), axis=1).T for image in images]
        preds = [np.flip(np.flip(pred, axis=0), axis=1).T for pred in preds]

        spacings = [image_nib.header.get_zooms()[0:2] for i in range(image.shape[-1])]

        categories = self.model_type.categories
        masks = []
        for pred in preds:
            mask = np.zeros((pred.shape[0], pred.shape[1], len(categories)))
            for i, category in enumerate(categories):
                mask[:, :, i] = pred == categories[category]
            masks.append(mask.astype(np.uint8))
        return images, masks, spacings


class MuscleAdiposeTissuePostProcessing(InferenceClass):
    """Post-process muscle and adipose tissue segmentation."""

    def __init__(self):
        super().__init__()

    def preds_to_mask(self, preds, use_softmax):
        """Convert model predictions to a mask.

        Args:
            preds (np.ndarray): Model predictions.

        Returns:
            np.ndarray: Mask.
        """
        if use_softmax:
            # softmax
            labels = np.zeros_like(preds, dtype=np.uint8)
            l_argmax = np.argmax(preds, axis=-1)
            for c in range(labels.shape[-1]):
                labels[l_argmax == c, c] = 1
            return labels.astype(np.bool)
        else:
            # sigmoid
            return preds >= 0.5

    def __call__(
        self, inference_pipeline, images, preds, spacings, model_types, model_names
    ):
        """Post-process muscle and adipose tissue segmentation."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        return self.post_process(images, preds, spacings, model_types, model_names)

    def remove_small_objects(self, mask, min_size=10):
        mask = mask.astype(np.uint8)
        components, output, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        sizes = stats[1:, -1]
        mask = np.zeros((output.shape))
        for i in range(0, components - 1):
            if sizes[i] >= min_size:
                mask[output == i + 1] = 1
        return mask

    def post_process(
        self,
        images,
        preds,
        spacings,
        model_types,
        model_names,
    ):
        start_time = perf_counter()

        masks = []
        for pred, model_type, model_name in zip(preds, model_types, model_names):
            if model_name == "stanford_v0.0.2":
                mask = pred
            else:
                mask = self.preds_to_mask(pred, model_type.use_softmax)
            masks.append(np.squeeze(mask))

        masks = self.fill_holes(masks)

        file_idx = 0
        for mask, image, model_type in tqdm(
            zip(masks, images, model_types), total=len(masks)
        ):
            cats = list(model_type.categories.keys())
            muscle_mask = mask[..., cats.index("muscle")]
            imat_mask = mask[..., cats.index("imat")]
            imat_mask = (
                np.logical_and(
                    (image * muscle_mask) <= -30, (image * muscle_mask) >= -190
                )
            ).astype(int)
            imat_mask = self.remove_small_objects(imat_mask)
            mask[..., cats.index("imat")] += imat_mask
            mask[..., cats.index("muscle")][imat_mask == 1] = 0
            masks[file_idx] = mask
            images[file_idx] = image
            file_idx += 1

        print(
            f"Completed post-processing in {(perf_counter() - start_time):.2f} seconds."
        )

        return {
            "images": images,
            "masks": masks,
            "spacings": spacings,
            "model_types": model_types,
            "model_names": model_names,
        }

    # function that fills in holes in a segmentation mask
    def _fill_holes(self, mask: np.ndarray, mask_id: int):
        """Fill in holes in a segmentation mask.

        Args:
            mask (ndarray): NxHxW
            mask_id (int): Label of the mask.

        Returns:
            ndarray: Filled mask.
        """
        int_mask = ((1 - mask) > 0.5).astype(np.int8)
        components, output, stats, _ = cv2.connectedComponentsWithStats(
            int_mask, connectivity=8
        )
        sizes = stats[1:, -1]
        components = components - 1
        # Larger threshold for SAT
        # TODO make this configurable / parameter
        if mask_id == 2:
            min_size = 50
            # min_size = 0
        else:
            min_size = 5
            # min_size = 0
        img_out = np.ones_like(mask)
        for i in range(0, components):
            if sizes[i] > min_size:
                img_out[output == i + 1] = 0
        return img_out

    def fill_holes(self, ys: List):
        """Take an array of size NxHxWxC and for each channel fill in holes.

        Args:
            ys (list): List of segmentation masks.
        """
        segs = []
        for n in range(len(ys)):
            ys_out = [
                self._fill_holes(ys[n][..., i], i) for i in range(ys[n].shape[-1])
            ]
            segs.append(np.stack(ys_out, axis=2).astype(float))

        return segs


class MuscleAdiposeTissueComputeMetrics(InferenceClass):
    """Compute muscle and adipose tissue metrics."""

    def __init__(self):
        super().__init__()

    def __call__(
        self, inference_pipeline, images, masks, spacings, model_types, model_names
    ):
        """Compute muscle and adipose tissue metrics."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        metrics = self.compute_metrics_all(images, masks, spacings, model_types)
        return metrics

    def compute_metrics_all(self, images, masks, spacings, model_types):
        """Compute metrics for all images and masks.

        Args:
            images (List[np.ndarray]): Images.
            masks (List[np.ndarray]): Masks.

        Returns:
            Dict: Results.
        """
        results = []
        for image, mask, spacing, model_type in zip(images, masks, spacings, model_types):
            results.append(self.compute_metrics(image, mask, spacing, model_type))
        return {"images": images, "results": results}

    def compute_metrics(self, x, mask, spacing, model_type):
        """Compute results for a given segmentation."""
        categories = model_type.categories

        hu = HounsfieldUnits()
        csa_units = "cm^2" if spacing else ""
        csa = CrossSectionalArea(csa_units)

        hu_vals = hu(mask, x, category_dim=-1)
        csa_vals = csa(mask=mask, spacing=spacing, category_dim=-1)

        # check if any values are nan and replace with 0
        hu_vals = np.nan_to_num(hu_vals)
        csa_vals = np.nan_to_num(csa_vals)

        assert mask.shape[-1] == len(
            categories
        ), "{} categories found in mask, " "but only {} categories specified".format(
            mask.shape[-1], len(categories)
        )

        results = {
            cat: {
                "mask": mask[..., idx],
                hu.name(): hu_vals[idx],
                csa.name(): csa_vals[idx],
            }
            for idx, cat in enumerate(categories.keys())
        }
        return results


class MuscleAdiposeTissueH5Saver(InferenceClass):
    """Save results to an HDF5 file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, results):
        """Save results to an HDF5 file."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        self.output_dir = inference_pipeline.output_dir
        self.h5_output_dir = os.path.join(self.output_dir, "segmentations")
        os.makedirs(self.h5_output_dir, exist_ok=True)
        self.dicom_file_paths = inference_pipeline.dicom_file_paths
        self.dicom_file_names = inference_pipeline.dicom_file_names
        self.save_results(results)
        return {"results": results}

    def save_results(self, results):
        """Save results to an HDF5 file."""
        for i, result in enumerate(results):
            cats = list(result.keys())
            file_name = self.dicom_file_names[i]
            with h5py.File(
                os.path.join(self.h5_output_dir, file_name + ".h5"), "w"
            ) as f:
                for cat in cats:
                    mask = result[cat]["mask"]
                    f.create_dataset(name=cat, data=np.array(mask, dtype=np.uint8))


class MuscleAdiposeTissueMetricsSaver(InferenceClass):
    """Save metrics to a CSV file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, results):
        """Save metrics to a CSV file."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        self.output_dir = inference_pipeline.output_dir
        self.csv_output_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(self.csv_output_dir, exist_ok=True)
        self.dicom_file_paths = inference_pipeline.dicom_file_paths
        self.dicom_file_names = inference_pipeline.dicom_file_names
        self.save_results(results)
        return {}

    def save_results(self, results):
        """Save results to a CSV file."""
        self.model_type.categories
        df = pd.DataFrame(
            columns=[
                "Level",
                "Index",
                "Muscle HU",
                "Muscle CSA (cm^2)",
                "SAT HU",
                "SAT CSA (cm^2)",
                "VAT HU",
                "VAT CSA (cm^2)",
                "IMAT HU",
                "IMAT CSA (cm^2)",
            ]
        )

        for i, result in enumerate(results):
            row = []
            row.append(self.dicom_file_names[i])
            row.append(self.dicom_file_paths[i])
            for cat in result:
                row.append(result[cat]["Hounsfield Unit"])
                row.append(result[cat]["Cross-sectional Area (cm^2)"])
            df.loc[i] = row
        df = df.iloc[::-1]
        df.to_csv(
            os.path.join(self.csv_output_dir, "muscle_adipose_tissue_metrics.csv"),
            index=False,
        )
