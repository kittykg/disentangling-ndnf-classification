from pathlib import Path
import sys

from omegaconf import DictConfig
import numpy as np
import torch
from torch import Tensor

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from data_utils import GenericUCIDataset


class MushroomDataset(GenericUCIDataset):
    """
    Mushroom dataset
    """


def get_mushroom_data_np_from_path(
    data_preprocess_cfg: DictConfig, is_test: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert data_preprocess_cfg["hold_out"][
        "create_hold_out"
    ], "Hold out test data must be created"

    impute_method = data_preprocess_cfg.get("impute", None)
    if impute_method is None:
        file_name = "mushroom_no_imputation.npz"
    else:
        file_name = f"mushroom_{impute_method}.npz"

    if is_test:
        file_name = f"hold_out_test_{file_name}"
    else:
        file_name = f"train_{file_name}"

    npz_path = Path(data_preprocess_cfg["save_dir"]) / file_name

    data = np.load(npz_path)
    return data["X"], data["y"], data["feature_names"]


def get_x_and_y_mushroom(
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
