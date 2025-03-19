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

# The first 6 features are real valued, and the rest are binary (converted from
# 2 categorical features)
NUM_REAL_VALUED_FEATURES = 6


class CoverTypeDataset(GenericUCIDataset):
    """
    CoverType dataset
    """


def get_covertype_data_np_from_path(
    data_preprocess_cfg: DictConfig, is_test: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    assert data_preprocess_cfg["hold_out"][
        "create_hold_out"
    ], "Hold out test data must be created"

    undersample = data_preprocess_cfg.get("undersample", False)
    if undersample:
        file_name = "covertype_undersampled.npz"
    else:
        file_name = f"covertype.npz"

    if is_test:
        file_name = f"hold_out_test_{file_name}"
    else:
        file_name = f"train_{file_name}"

    npz_path = Path(data_preprocess_cfg["save_dir"]) / file_name
    data = np.load(npz_path)
    return data["X"], data["y"]


def get_x_and_y_covertype(
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
        # Convert the binary attributes in {0, 1} to be in {-1, 1}
        x[:, NUM_REAL_VALUED_FEATURES:] = (
            2 * x[:, NUM_REAL_VALUED_FEATURES:] - 1
        )

    y = data[1]

    x = x.to(device).float()
    y = y.to(device)

    return x, y
