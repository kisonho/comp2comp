"""Preprocess TCIA NRRD volumes into memory-mappable NumPy arrays."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import nrrd
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from tqdm import tqdm

from data import TCIA


BATCH_SIZE = 16


def _sample_dirs(dataset_dir: Path) -> list[Path]:
    """Return input patient directories containing both required volumes."""
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir()
        and (path / TCIA.IMAGE_FILENAME).is_file()
        and (path / TCIA.SEGMENTATION_FILENAME).is_file()
    )


def _process_patient(sample_dir: Path, output_dir: Path, device: torch.device) -> None:
    """Preprocess and save all labeled slices for one patient."""
    image, _ = nrrd.read(str(sample_dir / TCIA.IMAGE_FILENAME))
    segmentation, _ = nrrd.read(str(sample_dir / TCIA.SEGMENTATION_FILENAME))
    if image.ndim != 3 or segmentation.ndim != 3:
        raise ValueError(f"Expected 3D volumes in {sample_dir}")
    if image.shape != segmentation.shape:
        raise ValueError(
            f"Image and segmentation shapes differ in {sample_dir}: "
            f"{image.shape} != {segmentation.shape}"
        )
    if segmentation.min() < 0 or segmentation.max() > np.iinfo(np.uint8).max:
        raise ValueError(f"Segmentation labels do not fit in uint8: {sample_dir}")

    slice_indices = np.flatnonzero(np.any(segmentation != 0, axis=(0, 1)))
    if len(slice_indices) < TCIA.LEAST_LABELED_SLICES_PER_SAMPLE:
        raise ValueError(f"No labeled slices found in {sample_dir}")

    patient_dir = output_dir / sample_dir.name
    patient_dir.mkdir(parents=True, exist_ok=True)
    images = open_memmap(
        patient_dir / "images.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(slice_indices), *TCIA.MODEL_SIZE, len(TCIA.MODEL_WINDOWS)),
    )
    masks = open_memmap(
        patient_dir / "masks.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(slice_indices), *TCIA.MODEL_SIZE),
    )

    for start in range(0, len(slice_indices), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(slice_indices))
        batch_indices = slice_indices[start:stop]
        image_batch = torch.as_tensor(
            np.ascontiguousarray(image[:, :, batch_indices].transpose(2, 0, 1)),
            dtype=torch.float32,
            device=device,
        )
        channels = []
        for window in TCIA.MODEL_WINDOWS:
            width, level = TCIA.WINDOW_SETTINGS[window]
            channels.append(
                image_batch.clamp(level - width / 2, level + width / 2)
            )
        image_batch = torch.stack(channels, dim=1)
        mask_batch = torch.as_tensor(
            np.ascontiguousarray(
                segmentation[:, :, batch_indices].transpose(2, 0, 1)
            ),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        if tuple(image_batch.shape[-2:]) != TCIA.MODEL_SIZE:
            image_batch = F.interpolate(
                image_batch,
                size=TCIA.MODEL_SIZE,
                mode="bilinear",
                align_corners=False,
            )
            mask_batch = F.interpolate(
                mask_batch,
                size=TCIA.MODEL_SIZE,
                mode="nearest",
            )

        images[start:stop] = (
            image_batch.permute(0, 2, 3, 1).half().cpu().numpy()
        )
        masks[start:stop] = mask_batch.squeeze(1).byte().cpu().numpy()

    images.flush()
    masks.flush()
    np.save(patient_dir / "slice_indices.npy", slice_indices.astype(np.int32))
    with (patient_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_shape": list(image.shape),
                "processed_shape": list(images.shape),
                "image_dtype": "float16",
                "mask_dtype": "uint8",
            },
            file,
            indent=2,
        )


def process(dataset_dir: Path, output_dir: Path) -> None:
    """Preprocess every patient beneath ``dataset_dir``."""
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_dir}")
    if dataset_dir == output_dir:
        raise ValueError("output_dir must differ from dataset_dir")
    samples = _sample_dirs(dataset_dir)
    if not samples:
        raise FileNotFoundError(f"No TCIA patient directories found in {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing {len(samples)} patients on {device}")
    for sample_dir in tqdm(samples, desc="Processing TCIA patients"):
        _process_patient(sample_dir, output_dir, device)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the command-line preprocessing entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Input TCIA dataset directory")
    parser.add_argument("output_dir", type=Path, help="Processed output directory")
    arguments = parser.parse_args(argv)
    process(arguments.dataset_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
