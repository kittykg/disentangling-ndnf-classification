from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from data_utils_generic import GenericUCIDataset

NPZ_FILE_NAME = "zoo.npz"


class ZooDataset(GenericUCIDataset):
    """
    Zoo dataset
    """


def get_zoo_data_np_from_path(
    data_dir_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(data_dir_path / NPZ_FILE_NAME)
    return data["X"], data["y"], data["feature_names"]


def get_zoo_dataloaders(
    dataset: ZooDataset,
    train_index: npt.NDArray[np.int64],
    test_index: npt.NDArray[np.int64],
    batch_size: int,
    loader_num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    def _get_dataloader(
        index: npt.NDArray[np.int64],
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=loader_num_workers,
            pin_memory=pin_memory,
            sampler=torch.utils.data.SubsetRandomSampler(index.tolist()),
        )

    return _get_dataloader(train_index), _get_dataloader(test_index)


def get_x_and_y_zoo(
    data: list[Tensor], device: torch.device, use_ndnf: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Get ground truth x and y for classifier

    Args:
        data: a list of tensors
        device: device to move data to
        use_ndnf: whether the data is used by neural DNF based models or not

    Returns:
        x: attribute label/attribute score
        y: class label
    """
    x = data[0]
    if use_ndnf:
        x = 2 * data[0] - 1

    y = data[1]

    x = x.to(device).float()
    y = y.to(device)

    return x, y
