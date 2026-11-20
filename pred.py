"""Save matching prediction, input, and ground-truth PNGs for each TCIA test slice.

Example: python pred.py DATASET MODEL.keras OUTPUT --preprocessed --fold 0
Files <sample_folder>_slice_<slice_index>.png in preds/, inputs/, and gt/
identify original source slices.
"""

import argparse, json, numpy as np, os
from keras import models, Model
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import cast

import data


def predict(
    dataset_dir: Path,
    model_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 32,
    fold: int = 0,
    preprocessed: bool = False,
    num_workers: int = 0,
) -> None:
    """Save prediction, input, and ground-truth plots using notebook display settings."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    _, dataset = data.load(
        dataset_dir, preprocessed=preprocessed, fold_idx=fold
    )
    if not len(dataset):
        raise ValueError("The selected TCIA test split is empty")

    # Inference does not need the custom training optimizer or metrics.
    model = cast(Model, models.load_model(model_path, compile=False))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    for folder in ("preds", "inputs", "gt"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    offset = 0
    source_slice_indices: dict[Path, np.ndarray] = {}
    for images, labels in tqdm(loader, desc="Predicting TCIA"):
        image_batch = images.numpy()
        label_batch = labels.numpy()
        scores = np.asarray(model.predict_on_batch(image_batch))
        masks = scores.argmax(axis=-1)
        for batch_index, mask in enumerate(masks):
            dataset_index = int(dataset.indices[offset + batch_index])
            sample = dataset.dataset._slices[dataset_index]  # type: ignore
            if preprocessed:
                sample_dir = sample.patient_dir
                if sample_dir not in source_slice_indices:
                    source_slice_indices[sample_dir] = np.load(
                        sample_dir / "slice_indices.npy", allow_pickle=False
                    )
                slice_index = int(source_slice_indices[sample_dir][sample.index])
            else:
                sample_dir = sample.sample_dir
                slice_index = sample.index
            filename = f"{sample_dir.name}_slice_{slice_index}.png"
            # Match the notebook's spatial transpose and input normalization.
            input_image = image_batch[batch_index].swapaxes(0, 1)
            input_image = input_image - input_image.min()
            intensity_range = input_image.max()
            if intensity_range > 0:
                input_image = input_image / intensity_range
            for folder, display in (
                ("preds", mask.T),
                ("inputs", input_image),
                ("gt", label_batch[batch_index].T),
            ):
                figure = Figure(figsize=(6, 6), layout="constrained")
                FigureCanvasAgg(figure)
                try:
                    axes = figure.subplots()
                    cmap = "gray" if folder == "inputs" and display.ndim == 2 else None
                    axes.imshow(display, cmap=cmap)
                    axes.set_title(f"{sample_dir.name} — slice {slice_index}")
                    axes.set_axis_off()
                    figure.savefig(output_dir / folder / filename, dpi=150)
                finally:
                    figure.clear()
        offset += len(masks)
    np.save(
        output_dir / "dataset_indices.npy",
        np.asarray(dataset.indices, dtype=np.int64),
        allow_pickle=False,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir.resolve()),
                "model_path": str(model_path.resolve()),
                "fold": fold,
                "preprocessed": preprocessed,
                "count": offset,
                "prediction_format": "PNG plots of transposed class labels matching eval.ipynb",
                "output_folders": {"predictions": "preds", "inputs": "inputs", "ground_truth": "gt"},
                "prediction_filename": "<sample_folder>_slice_<slice_index>.png",
            },
            indent=2,
        ) + "\n"
    )


class PredictionConfigs(argparse.Namespace):
    dataset_dir: Path
    model_path: Path
    output_dir: Path
    batch_size: int
    fold: int
    preprocessed: bool
    num_workers: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-f", "--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--preprocessed", action="store_true")
    parser.add_argument(
        "--num_workers", type=int, default=min(4, os.cpu_count() or 1)
    )
    args: PredictionConfigs = parser.parse_args(namespace=PredictionConfigs())
    predict(
        args.dataset_dir.expanduser(),
        args.model_path.expanduser(),
        args.output_dir.expanduser(),
        batch_size=args.batch_size,
        fold=args.fold,
        preprocessed=args.preprocessed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
