"""Evaluate Keras models or native nnUNet v1 checkpoints (Comp2Comp FatMuscle).

Examples:
    python eval.py DATASET pretrained.keras --preprocessed --fold 0
    python eval.py DATASET experiments/RUN/checkpoints/best.keras --preprocessed
    python eval.py DATASET finetuned.keras --output results.json
    python eval.py RAW_DATASET NNUNET_TRAINER_DIR --model_type nnunet
    python eval.py PROCESSED_DATASET NNUNET_CHECKPOINT.model --preprocessed \
        --raw_dataset_dir RAW_DATASET

Uses the held-out split and mean Dice metric from train.py. Dice includes all
classes (including background), excluding classes absent in both masks and
predictions across the evaluation split. Native nnUNet inference reports accuracy
and Dice without loss. Its default label map merges FatMuscle SAT/VAT/IMAT into
TCIA adipose=1 and muscle into TCIA muscle=2. Use --nnunet_label_map for other
three-class TCIA-compatible models. A .model file requires its companion
.model.pkl and the trainer directory's plans.pkl; nnUNet v2 .pth is unsupported.
"""

import argparse, json, os, tempfile
from pathlib import Path
from torchmanager.configs import Configs as BaseConfigs
from torchmanager_core import view
from typing import cast, overload

import numpy as np


ArgumentParser = argparse.ArgumentParser | argparse._ArgumentGroup


class EvaluateConfigs(BaseConfigs):
    """Command-line configurations for model evaluation."""

    dataset_dir: str
    model_path: str
    batch_size: int
    fold: int
    preprocessed: bool
    num_workers: int
    output: str | None
    model_type: str
    raw_dataset_dir: str | None
    nnunet_fold: str
    checkpoint_name: str
    nnunet_label_map: list[int]

    def format_arguments(self) -> None:
        """Normalize paths and validate scalar arguments."""
        super().format_arguments()
        self.dataset_dir = os.path.abspath(os.path.expanduser(self.dataset_dir))
        self.model_path = os.path.abspath(os.path.expanduser(self.model_path))
        if self.raw_dataset_dir is not None:
            self.raw_dataset_dir = os.path.abspath(os.path.expanduser(self.raw_dataset_dir))
        if self.output is not None:
            self.output = os.path.abspath(os.path.expanduser(self.output))
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"Number of workers must be nonnegative, got {self.num_workers}.")
        if not 0 <= self.fold < 10:
            raise ValueError(f"Cross validation fold index must be in range [0, 9], got {self.fold}.")

    @staticmethod
    @overload
    def get_arguments(parser: argparse.ArgumentParser = ...) -> argparse.ArgumentParser: ...

    @staticmethod
    @overload
    def get_arguments(parser: argparse._ArgumentGroup) -> argparse._ArgumentGroup: ...

    @staticmethod
    def get_arguments(parser: ArgumentParser = argparse.ArgumentParser()) -> ArgumentParser:
        """Add evaluation arguments to an argument parser."""
        parser.add_argument("dataset_dir", help="TCIA dataset directory")
        parser.add_argument("model_path", help="Keras model, nnUNet v1 trainer directory, fold directory, or .model checkpoint")
        dataset_args = parser.add_argument_group("Dataset Arguments")
        dataset_args.add_argument(
            "-b", "--batch_size", type=int, default=32,
            help="Evaluation batch size (default: 32)",
        )
        dataset_args.add_argument(
            "-f", "--fold", type=int, choices=range(10), default=0,
            help="Held-out fold, matching train.py (default: 0)",
        )
        dataset_args.add_argument("--preprocessed", action="store_true")
        dataset_args.add_argument(
            "--num_workers", type=int, default=min(4, os.cpu_count() or 1),
            help="Number of data-loading workers",
        )
        evaluation_args = parser.add_argument_group("Evaluation Arguments")
        evaluation_args.add_argument(
            "--output", help="Optionally save metrics and settings as JSON",
        )
        evaluation_args.add_argument(
            "--model_type", choices=("auto", "keras", "nnunet"), default="auto",
            help="Auto-detect Keras files or native nnUNet v1 checkpoints",
        )
        evaluation_args.add_argument(
            "--raw_dataset_dir", help="Original TCIA NRRDs; required for preprocessed nnUNet evaluation",
        )
        evaluation_args.add_argument(
            "--nnunet_fold", default="all",
            help="Model training fold (all or an integer); independent of --fold",
        )
        evaluation_args.add_argument(
            "--checkpoint_name", default="model_final_checkpoint",
            help="Checkpoint basename when model_path is a directory",
        )
        evaluation_args.add_argument(
            "--nnunet_label_map", type=int, nargs="+", default=[0, 1, 1, 1, 2],
            help="Output label for each native class; default maps FatMuscle to TCIA",
        )
        BaseConfigs.get_arguments(parser)
        return parser

    def show_settings(self) -> None:
        """Print the resolved evaluation settings."""
        view.logger.info(
            f"Dataset: dataset_dir={self.dataset_dir}, fold={self.fold}, "
            f"preprocessed={self.preprocessed}"
        )
        view.logger.info(f"Model: {self.model_path}, model_type={self.model_type}")
        view.logger.info(
            f"nnUNet: raw_dataset_dir={self.raw_dataset_dir}, model_fold={self.nnunet_fold}, "
            f"checkpoint={self.checkpoint_name}, label_map={self.nnunet_label_map}"
        )
        view.logger.info(f"Output: {self.output}")
        view.logger.info(
            f"Evaluation arguments: batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}"
        )


def evaluate(
    dataset_dir: str | Path,
    model_path: str | Path,
    *,
    batch_size: int = 32,
    fold: int = 0,
    preprocessed: bool = False,
    num_workers: int = 0,
    model_type: str = "auto",
    raw_dataset_dir: str | Path | None = None,
    nnunet_fold: str = "all",
    checkpoint_name: str = "model_final_checkpoint",
    nnunet_label_map: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a Keras or native nnUNet v1 model on the training held-out split.

    nnUNet returns accuracy and Dice only: its inference API exports hard labels.
    The default label map is for Comp2Comp Task927_FatMuscle, with fat labels
    1/2/3 merged into TCIA adipose=1 and muscle=4 mapped to TCIA muscle=2.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if not 0 <= fold < 10:
        raise ValueError("fold must be between 0 and 9")

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if model_type not in ("auto", "keras", "nnunet"):
        raise ValueError(f"Unknown model_type: {model_type}")
    if model_type == "auto":
        model_type = "nnunet" if model_path.is_dir() or model_path.suffix == ".model" else "keras"
    if model_type == "nnunet":
        if preprocessed and raw_dataset_dir is None:
            raise ValueError("nnUNet with --preprocessed requires --raw_dataset_dir")
        return evaluate_nnunet(
            dataset_dir, model_path, fold=fold, preprocessed=preprocessed,
            raw_dataset_dir=raw_dataset_dir, model_fold=nnunet_fold,
            checkpoint_name=checkpoint_name, label_map=nnunet_label_map,
            num_workers=num_workers,
        )

    from keras import activations, layers, losses, metrics, models, Model
    from torch.utils.data import DataLoader
    import data
    from comp2comp.finetune.metrics import SparseMeanDice

    _, dataset = data.load(
        dataset_dir, preprocessed=preprocessed, fold_idx=fold
    )
    if not len(dataset):
        raise ValueError("The selected evaluation split is empty")

    # Ignore saved compile state so custom training optimizers are unnecessary.
    model = cast(Model, models.load_model(model_path, compile=False))
    output_layer = model.layers[-1]
    if not isinstance(output_layer, layers.Conv2D):
        raise TypeError(
            "Expected the segmentation output layer to be Conv2D, got "
            f"{type(output_layer).__name__}."
        )
    if output_layer.activation not in (activations.linear, activations.softmax):
        raise ValueError("Expected a linear or softmax segmentation output")
    # Match train.py's evaluation: use logits for both original softmax models
    # and fine-tuned checkpoints. This only changes the in-memory model.
    output_layer.activation = activations.linear
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMeanDice(num_classes=output_layer.filters),
        ],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    results = model.evaluate(loader, return_dict=True)
    return {name: float(value) for name, value in results.items()}


def resolve_nnunet_checkpoint(
    model_path: Path, model_fold: str, checkpoint_name: str,
) -> tuple[Path, str | list[int], str]:
    """Resolve v1 trainer/fold directories or a checkpoint and its companion files."""
    if model_path.suffix == ".model":
        checkpoint_name = model_path.stem
        model_path = model_path.parent
    if model_path.name == "all" or model_path.name.startswith("fold_"):
        model_fold = "all" if model_path.name == "all" else model_path.name.removeprefix("fold_")
        model_path = model_path.parent
    if model_fold == "all":
        folds: str | list[int] = "all"
        fold_dir = model_path / "all"
    else:
        try:
            fold_index = int(model_fold)
        except ValueError as error:
            raise ValueError("nnunet_fold must be 'all' or a nonnegative integer") from error
        if fold_index < 0:
            raise ValueError("nnunet_fold must be nonnegative")
        folds = [fold_index]
        fold_dir = model_path / f"fold_{fold_index}"
    for path in (
        model_path / "plans.pkl", fold_dir / f"{checkpoint_name}.model",
        fold_dir / f"{checkpoint_name}.model.pkl",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required nnUNet v1 model file does not exist: {path}")
    return model_path, folds, checkpoint_name


def nrrd_to_nifti(image: np.ndarray, header: dict):
    """Keep source voxel axes while converting NRRD physical coordinates to RAS."""
    from nibabel.nifti1 import Nifti1Image

    signs = {
        "left-posterior-superior": (-1, -1, 1), "lps": (-1, -1, 1),
        "right-anterior-superior": (1, 1, 1), "ras": (1, 1, 1),
        "left-anterior-superior": (-1, 1, 1), "las": (-1, 1, 1),
        "right-posterior-superior": (1, -1, 1), "rps": (1, -1, 1),
    }
    space = str(header.get("space", "")).lower()
    if space not in signs:
        raise ValueError(f"Unsupported NRRD coordinate space: {space!r}")
    directions = np.asarray(header["space directions"], dtype=float)
    origin = np.asarray(header["space origin"], dtype=float)
    if image.ndim != 3 or directions.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("Expected a spatial 3D NRRD with directions and origin")
    affine = np.eye(4)
    affine[:3, :3] = directions.T
    affine[:3, 3] = origin
    affine = np.diag([*signs[space], 1]) @ affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid NRRD spatial geometry")
    return Nifti1Image(image.astype(np.float32), affine)


def update_confusion(counts: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> None:
    """Accumulate class counts across all pixels, matching SparseMeanDice."""
    if truth.shape != prediction.shape:
        raise ValueError(f"Mask shape mismatch: {truth.shape} != {prediction.shape}")
    for mask in (truth, prediction):
        if not np.isin(mask, (0, 1, 2)).all():
            raise ValueError("Expected TCIA labels 0=background, 1=adipose, 2=muscle")
    counts += np.bincount(
        truth.astype(np.int64).ravel() * 3 + prediction.astype(np.int64).ravel(),
        minlength=9,
    ).reshape(3, 3)


def confusion_metrics(counts: np.ndarray) -> dict[str, float]:
    """Return global accuracy and Dice, excluding jointly absent classes."""
    if not counts.sum():
        raise ValueError("No pixels were evaluated")
    intersection = np.diag(counts)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    dice = np.divide(2 * intersection, denominator, out=np.zeros(3), where=denominator > 0)
    return {
        "accuracy": float(intersection.sum() / counts.sum()),
        "dice": float(dice[denominator > 0].mean()),
        **{f"dice_{name}": float(score) for name, score in zip(
            ("background", "adipose", "muscle"), dice
        )},
    }


def evaluate_nnunet(
    dataset_dir: Path, model_path: Path, *, fold: int, preprocessed: bool,
    raw_dataset_dir: str | Path | None, model_fold: str, checkpoint_name: str,
    label_map: list[int] | None, num_workers: int,
) -> dict[str, float]:
    """Run Comp2Comp's nnUNet v1 inference on raw CT and score held-out slices."""
    model_path, folds, checkpoint_name = resolve_nnunet_checkpoint(
        model_path, model_fold, checkpoint_name
    )
    mapping = np.asarray([0, 1, 1, 1, 2] if label_map is None else label_map)
    if mapping.ndim != 1 or not len(mapping) or not np.isin(mapping, (0, 1, 2)).all() or mapping[0] != 0:
        raise ValueError("nnunet_label_map must map class 0 to background and all classes to 0, 1, or 2")

    from nibabel.loadsave import load, save
    from nibabel.processing import resample_from_to
    import nrrd
    import torch
    from torch.nn import functional as F
    from nnunet.inference import predict
    import data

    _, dataset = data.load(dataset_dir, preprocessed=preprocessed, fold_idx=fold)
    if not len(dataset):
        raise ValueError("The selected evaluation split is empty")
    raw_root = Path(raw_dataset_dir or dataset_dir).expanduser().resolve()
    patients: dict[Path, list[tuple[int, int]]] = {}
    source_indices: dict[Path, np.ndarray] = {}
    for subset_index, parent_index in enumerate(dataset.indices):
        if preprocessed:
            processed_dataset = cast(data.ProcessedTCIA, dataset.dataset)
            sample = processed_dataset.slices[parent_index]
            patient_dir = sample.patient_dir
            if patient_dir not in source_indices:
                source_indices[patient_dir] = np.load(patient_dir / "slice_indices.npy", allow_pickle=False)
            slice_index = int(source_indices[patient_dir][sample.index])
            patient_dir = raw_root / patient_dir.name
        else:
            raw_dataset = cast(data.TCIA, dataset.dataset)
            raw_sample = raw_dataset._slices[parent_index]
            patient_dir, slice_index = raw_sample.sample_dir, raw_sample.index
        patients.setdefault(patient_dir, []).append((subset_index, slice_index))

    view.logger.info(
        f"nnUNet v1: model={model_path}, model_folds={folds}, checkpoint={checkpoint_name}, "
        f"label_map={mapping.tolist()}. Reporting hard-label metrics; loss is unavailable."
    )
    counts = np.zeros((3, 3), dtype=np.int64)
    for patient_dir, slices in patients.items():
        view.logger.info(f"Evaluating {patient_dir.name}: {len(slices)} held-out slices")
        image, header = nrrd.read(str(patient_dir / data.TCIA.IMAGE_FILENAME), index_order="F")
        source = nrrd_to_nifti(image, header)
        with tempfile.TemporaryDirectory(prefix="eval_nnunet_") as tmp:
            input_path = str(Path(tmp) / "ct.nii.gz")
            output_path = str(Path(tmp) / "seg.nii.gz")
            save(source, input_path)
            # Same inference settings used by Comp2Comp's FatMuscle pipeline.
            predict.predict_cases(
                model=str(model_path), list_of_lists=[[input_path]],
                output_filenames=[output_path], folds=folds, save_npz=False,
                num_threads_preprocessing=max(1, num_workers),
                num_threads_nifti_save=max(1, num_workers),
                segs_from_prev_stage=None, do_tta=False, mixed_precision=True,
                overwrite_existing=False, all_in_gpu=False, step_size=0.5,
                checkpoint_name=checkpoint_name,
            )
            # Restore original voxel axes; pipeline display flips do not apply here.
            result = resample_from_to(load(output_path), source, order=0, mode="nearest")
            labels = np.asarray(result.dataobj)
            if not np.isin(labels, np.arange(len(mapping))).all():
                raise ValueError("nnUNet predicted a class absent from nnunet_label_map")
            for subset_index, slice_index in slices:
                if not 0 <= slice_index < image.shape[2]:
                    raise ValueError(f"Source slice {slice_index} out of bounds in {patient_dir}")
                _, truth = dataset[subset_index]
                truth = np.asarray(truth)
                mask = mapping[labels[:, :, slice_index].astype(np.int64)]
                if mask.shape != truth.shape:
                    mask = F.interpolate(
                        torch.from_numpy(mask.astype(np.float32))[None, None],
                        size=truth.shape, mode="nearest",
                    )[0, 0].numpy()
                update_confusion(counts, truth, mask)
    return confusion_metrics(counts)


def main() -> None:
    args = cast(EvaluateConfigs, EvaluateConfigs.from_arguments())
    results = evaluate(
        args.dataset_dir,
        args.model_path,
        batch_size=args.batch_size,
        fold=args.fold,
        preprocessed=args.preprocessed,
        num_workers=args.num_workers,
        model_type=args.model_type,
        raw_dataset_dir=args.raw_dataset_dir,
        nnunet_fold=args.nnunet_fold,
        checkpoint_name=args.checkpoint_name,
        nnunet_label_map=args.nnunet_label_map,
    )
    view.logger.info("Evaluation results: " + ", ".join(
        f"{name}={value:.6f}" for name, value in results.items()
    ))
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "dataset_dir": args.dataset_dir,
            "model_path": args.model_path,
            "fold": args.fold,
            "preprocessed": args.preprocessed,
            "batch_size": args.batch_size,
            "model_type": args.model_type,
            "raw_dataset_dir": args.raw_dataset_dir,
            "nnunet_fold": args.nnunet_fold,
            "checkpoint_name": args.checkpoint_name,
            "nnunet_label_map": args.nnunet_label_map,
            "metrics": results,
        }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
