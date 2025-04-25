from pathlib import Path
import sys

from omegaconf import DictConfig
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.utils.data import Dataset

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from bn.data_processing.common import BNDatasetType, BNDatasetSubType


def get_boolean_network_data_np_from_path(
    data_preprocess_cfg: DictConfig, is_test: bool = False
) -> npt.NDArray[np.int64]:
    dataset_name = data_preprocess_cfg["dataset_name"]
    dataset = BNDatasetType.from_str(dataset_name)
    dataset_dir = (
        Path(data_preprocess_cfg["save_file_base_dir"]) / dataset.value
    )
    if not dataset_dir.is_dir() or not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if (
        "hold_out" in data_preprocess_cfg
        and data_preprocess_cfg["hold_out"]["create_hold_out"]
    ):
        if is_test:
            file_name = f"{dataset_name}_hold_out_test.npy"
        else:
            file_name = f"{dataset_name}_train.npy"
    else:
        # If there is no hold out, return the full dataset
        file_name = f"{dataset_name}_raw_data.npy"

    return np.load(dataset_dir / file_name)


def get_boolean_network_full_data_np_from_path(
    data_preprocess_cfg: DictConfig,
) -> npt.NDArray[np.int64]:
    dataset_name = data_preprocess_cfg["dataset_name"]
    dataset = BNDatasetType.from_str(dataset_name)
    dataset_dir = (
        Path(data_preprocess_cfg["save_file_base_dir"]) / dataset.value
    )
    if not dataset_dir.is_dir() or not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    return np.load(dataset_dir / f"{dataset_name}_raw_data.npy")


def split_boolean_network_data(
    data: npt.NDArray[np.int64],
    test_size: float = 0.1,
    random_state: int = 73,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    indices = np.arange(len(data))
    train_idx, hold_out_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )

    return data[train_idx], data[hold_out_idx]


class BooleanNetworkDataset(Dataset):
    dataset_type: BNDatasetType
    subtype: BNDatasetSubType | None  # we only support normal for now

    data: npt.NDArray[np.int64]

    def __init__(
        self,
        dataset_type: BNDatasetType | str,
        subtype: BNDatasetSubType | str | None,
        data: npt.NDArray[np.int64],
    ):
        if isinstance(dataset_type, str):
            self.dataset_type = BNDatasetType.from_str(dataset_type)
        else:
            self.dataset_type = dataset_type

        if subtype is str:
            self.subtype = BNDatasetSubType.from_str(subtype)
        elif isinstance(subtype, BNDatasetSubType):
            self.subtype = subtype
        else:
            self.subtype = None

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        transition = self.data[idx]
        s1 = torch.Tensor(transition[0])
        s2 = torch.Tensor(transition[1])

        return s1.float(), s2.float()


def get_x_and_y_boolean_network(
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
    y = y.to(device).float()

    return x, y
