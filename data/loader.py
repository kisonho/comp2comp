from pathlib import Path
from typing import Union

from torch.utils.data import Subset

from .processed_tcia import ProcessedTCIA
from .tcia import TCIA


def load(
    dataset_dir: Union[str, Path],
    *,
    preprocessed: bool = False,
    split: float = 0.2,
    fold_idx: int = 0,
    return_path: bool = False,
) -> tuple[Subset[ProcessedTCIA], Subset[ProcessedTCIA]]:
    dataset_cls = ProcessedTCIA if preprocessed else TCIA
    dataset = dataset_cls(dataset_dir, return_path=return_path)
    if not 0 <= fold_idx < 10:
        raise ValueError(f"fold_idx must be between 0 and 9, got {fold_idx}")

    test_size = int(len(dataset) * split)
    test_start = fold_idx * test_size
    test_stop = test_start + test_size
    if test_stop > len(dataset):
        raise ValueError(
            f"fold_idx={fold_idx} with split={split} exceeds dataset length {len(dataset)}"
        )

    train_indices = list(range(0, test_start)) + list(range(test_stop, len(dataset)))
    return Subset(dataset, train_indices), Subset(dataset, range(test_start, test_stop))
