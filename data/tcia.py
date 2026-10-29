"""TCIA tissue-segmentation dataset."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import nrrd, numpy as np, torch
from monai.data.dataset import Dataset
from monai.data.image_reader import NrrdReader
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import Resize
from monai.transforms.intensity.array import ScaleIntensityRange
from tqdm import tqdm


@dataclass(frozen=True)
class _Slice:
    sample_dir: Path
    index: int


class TCIA(Dataset):
    """Labeled axial slices from paired TCIA NRRD volumes."""
    _load_image: LoadImage
    return_path: bool
    IMAGE_FILENAME = "CT_IMAGE.nrrd"
    SEGMENTATION_FILENAME = "FinalReviewed.nrrd"
    # SEGMENTATION_FILENAME = "segmented_tissues_final.nrrd"
    LEAST_LABELED_SLICES_PER_SAMPLE = 1
    WINDOW_SETTINGS = {
        "soft": (400.0, 50.0),
        "bone": (1800.0, 400.0),
        "custom": (500.0, 50.0),
    }
    MODEL_WINDOWS = ("soft", "bone", "custom")
    MODEL_SIZE = (512, 512)

    def __init__(self, root_dir: str | Path, *, return_path: bool = False) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        if not self._root_dir.is_dir():
            raise NotADirectoryError(
                f"TCIA root directory does not exist: {self._root_dir}"
            )

        reader = NrrdReader(dtype=np.float32, index_order="F")
        self._load_image = LoadImage(reader=reader, image_only=True, dtype=np.float32)
        self._resize_image = Resize(
            self.MODEL_SIZE, mode="bilinear", align_corners=False
        )
        self._resize_segmentation = Resize(self.MODEL_SIZE, mode="nearest")
        self._window_transforms = tuple(
            ScaleIntensityRange(
                a_min=level - width / 2,
                a_max=level + width / 2,
                b_min=level - width / 2,
                b_max=level + width / 2,
                clip=True,
            )
            for width, level in (
                self.WINDOW_SETTINGS[window] for window in self.MODEL_WINDOWS
            )
        )

        self._slices: list[_Slice] = []
        self._sample_ranges: list[range] = []
        sample_dirs = self._find_sample_dirs()
        
        if return_path:
            raise NotImplementedError("TCIA dataset does not support return the patient path.")

        for sample_dir in tqdm(sample_dirs, desc="Indexing TCIA samples"):
            start = len(self._slices)
            self._index_sample(sample_dir)
            self._sample_ranges.append(range(start, len(self._slices)))
        super().__init__(data=self._slices, transform=self._transform_slice)

    @property
    def root_dir(self) -> Path:
        """Root directory containing the TCIA sample folders."""
        return self._root_dir

    @property
    def sample_ranges(self) -> tuple[range, ...]:
        """Dataset index ranges grouped by source patient."""
        return tuple(self._sample_ranges)

    def _find_sample_dirs(self) -> list[Path]:
        filenames = (self.IMAGE_FILENAME, self.SEGMENTATION_FILENAME)
        return sorted(
            path
            for path in self.root_dir.iterdir()
            if path.is_dir() and any((path / filename).is_file() for filename in filenames)
        )

    def _index_sample(self, sample_dir: Path) -> None:
        image_path = sample_dir / self.IMAGE_FILENAME
        segmentation_path = sample_dir / self.SEGMENTATION_FILENAME
        missing = [
            path.name
            for path in (image_path, segmentation_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {', '.join(missing)} in TCIA sample {sample_dir}"
            )

        segmentation = cast(torch.Tensor, self._load_image(segmentation_path))
        if segmentation.ndim != 3:
            raise ValueError(
                f"Expected a 3D segmentation in {segmentation_path}, "
                f"got shape {segmentation.shape}"
            )

        image_shape = tuple(
            int(size) for size in nrrd.read_header(str(image_path))["sizes"]
        )
        if image_shape != segmentation.shape:
            raise ValueError(
                f"Image and segmentation shapes differ in {sample_dir}: "
                f"{image_shape} != {segmentation.shape}"
            )

        labeled = torch.any(segmentation != 0, dim=(0, 1))
        slice_indices = torch.nonzero(labeled, as_tuple=False).flatten().tolist()
        if len(slice_indices) < self.LEAST_LABELED_SLICES_PER_SAMPLE:
            raise ValueError(
                f"Expected at least {self.LEAST_LABELED_SLICES_PER_SAMPLE} labeled slices in "
                f"{sample_dir}, found {len(slice_indices)}"
            )

        self._slices.extend(
            _Slice(sample_dir, int(index)) for index in slice_indices
        )

    @lru_cache(maxsize=1)
    def _load_sample(self, sample_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
        image = cast(torch.Tensor, self._load_image(sample_dir / self.IMAGE_FILENAME))
        segmentation = cast(torch.Tensor, self._load_image(sample_dir / self.SEGMENTATION_FILENAME))
        return image, segmentation

    def _transform_slice(self, sample: _Slice) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and transform one labeled axial slice with MONAI."""
        image, segmentation = self._load_sample(sample.sample_dir)
        image_tensor = image[:, :, sample.index].unsqueeze(0)
        # image_tensor = torch.cat(
        #     [transform(image_slice) for transform in self._window_transforms], dim=0
        # )
        segmentation_tensor = segmentation[:, :, sample.index].unsqueeze(0)
        if tuple(image_tensor.shape[1:]) != self.MODEL_SIZE:
            image_tensor = self._resize_image(image_tensor)
            segmentation_tensor = self._resize_segmentation(segmentation_tensor)
        image_tensor = image_tensor.movedim(0, -1).contiguous()
        segmentation_tensor = segmentation_tensor.squeeze(0).long()
        return image_tensor, segmentation_tensor
