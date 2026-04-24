from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureTyped
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from torchmanager.data import Dataset
from torchmanager_core import torch
from torchmanager_core.typing import Any, TypedDict, cast


class TCIAData(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor


class TCIA(Dataset[torch.Tensor]):
    data: list[dict[str, Path]]
    root_dir: Path
    images_dir: Path
    masks_dir: Path
    transform: Compose

    def __init__(self, root_dir: str | Path, /, batch_size: int, *, drop_last: bool = False, img_size: int | tuple[int, int] | None = None) -> None:
        super().__init__(batch_size, drop_last=drop_last)
        # initialize dirs
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"

        # ensure images and masks dir exists
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # read image paths
        image_paths = sorted(self.images_dir.glob("*.nrrd"))
        if not image_paths:
            raise FileNotFoundError(f"No NRRD images found in: {self.images_dir}")

        # zip the data path by image paths
        self.data: list[dict[str, Path]] = []
        for image_path in image_paths:
            mask_path = self.masks_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")
            self.data.append({"image": image_path, "mask": mask_path})

        # initialize transform
        transforms = [
            _LoadNRRDD(keys=("image", "mask")),
            EnsureTyped(keys=("image", "mask")),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
        ]
        if img_size is not None:
            img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
            transforms.append(Resized(keys=("image", "mask"), spatial_size=img_size))
        self.transform = Compose(transforms)

    @property
    def unbatched_len(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        sample = cast(TCIAData, self.transform(sample))
        return sample["image"], sample["mask"]


class _LoadNRRDD:
    keys: tuple[str, ...]

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys

    @staticmethod
    def _load(path: Path) -> np.ndarray[Any, Any]:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        if array.ndim < 4:
            array = np.expand_dims(array, axis=0)
        return array

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        for key in self.keys:
            output[key] = self._load(Path(output[key]))
        return output
