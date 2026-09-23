"""Memory-mapped dataset for preprocessed TCIA slices."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _Slice:
    patient_dir: Path
    index: int


class ProcessedTCIA(Dataset):
    """Labeled TCIA slices produced by ``process.py``."""
    return_path: bool

    IMAGE_FILENAME = "images.npy"
    MASK_FILENAME = "masks.npy"
    SLICE_INDICES_FILENAME = "slice_indices.npy"
    IMAGE_SHAPE = (512, 512, 3)

    def __init__(self, root_dir: str | Path, *, return_path: bool = False) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        if not self._root_dir.is_dir():
            raise NotADirectoryError(
                f"Processed TCIA directory does not exist: {self._root_dir}"
            )

        self._slices: list[_Slice] = []
        self._sample_ranges: list[range] = []
        self.return_path = return_path
        patient_dirs = self._find_patient_dirs()
        if not patient_dirs:
            raise FileNotFoundError(
                f"No processed TCIA patient directories found in {self._root_dir}"
            )
        for patient_dir in patient_dirs:
            start = len(self._slices)
            slice_count = self._validate_patient(patient_dir)
            self._slices.extend(
                _Slice(patient_dir, index) for index in range(slice_count)
            )
            self._sample_ranges.append(range(start, len(self._slices)))

    @property
    def root_dir(self) -> Path:
        """Root directory containing processed patient directories."""
        return self._root_dir

    @property
    def sample_ranges(self) -> tuple[range, ...]:
        """Dataset index ranges grouped by patient."""
        return tuple(self._sample_ranges)

    def _find_patient_dirs(self) -> list[Path]:
        filenames = (
            self.IMAGE_FILENAME,
            self.MASK_FILENAME,
            self.SLICE_INDICES_FILENAME,
        )
        patient_dirs = []
        for path in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            existing = [(path / filename).is_file() for filename in filenames]
            if any(existing) and not all(existing):
                missing = [
                    filename for filename, exists in zip(filenames, existing) if not exists
                ]
                raise FileNotFoundError(f"Missing {', '.join(missing)} in {path}")
            if all(existing):
                patient_dirs.append(path)
        return patient_dirs

    def _validate_patient(self, patient_dir: Path) -> int:
        images, masks = self._load_arrays(patient_dir)
        slice_indices = np.load(
            patient_dir / self.SLICE_INDICES_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        if images.ndim != 4 or tuple(images.shape[1:]) != self.IMAGE_SHAPE:
            raise ValueError(f"Invalid processed image shape in {patient_dir}: {images.shape}")
        if masks.shape != images.shape[:3]:
            raise ValueError(
                f"Image and mask shapes differ in {patient_dir}: "
                f"{images.shape} != {masks.shape}"
            )
        if slice_indices.shape != (len(images),):
            raise ValueError(
                f"Slice index count differs from image count in {patient_dir}"
            )
        if images.dtype != np.float16:
            raise TypeError(f"Expected float16 images in {patient_dir}, got {images.dtype}")
        if masks.dtype != np.uint8:
            raise TypeError(f"Expected uint8 masks in {patient_dir}, got {masks.dtype}")
        return len(images)

    @staticmethod
    def _load_arrays(patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        images = np.load(
            patient_dir / ProcessedTCIA.IMAGE_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        masks = np.load(
            patient_dir / ProcessedTCIA.MASK_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        return images, masks

    @lru_cache(maxsize=1)
    def _load_patient(self, patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        return self._load_arrays(patient_dir)

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:
        sample = self._slices[index]
        images, masks = self._load_patient(sample.patient_dir)
        image = torch.from_numpy(
            np.array(images[sample.index], dtype=np.float32, copy=True)
        )
        mask = torch.from_numpy(
            np.array(masks[sample.index], dtype=np.int64, copy=True)
        )
        return (image, mask, str(sample.patient_dir)) if self.return_path else (image, mask)
