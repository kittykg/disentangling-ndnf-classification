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

NUM_REAL_VALUED_FEATURES = 1
NUM_VOLATILE_FEATURES = 3


class CDCDataset(GenericUCIDataset):
    """
    CDC dataset
    """


def get_cdc_data_np_from_path(
    data_preprocess_cfg: DictConfig, is_test: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert data_preprocess_cfg["hold_out"][
        "create_hold_out"
    ], "Hold out test data must be created"

    undersample = data_preprocess_cfg.get("undersample", False)
    if undersample:
        file_name = "cdc_undersampled.npz"
    else:
        file_name = f"cdc.npz"

    if is_test:
        file_name = f"hold_out_test_{file_name}"
    else:
        file_name = f"train_{file_name}"

    npz_path = Path(data_preprocess_cfg["save_dir"]) / file_name

    data = np.load(npz_path)
    return data["X"], data["y"], data["feature_names"]


def get_scaler_mean_var_from_path(
    data_preprocess_cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(data_preprocess_cfg["save_dir"]) / "scaler.npz")
    return data["mean"], data["var"]


def get_x_and_y_cdc(
    data: list[Tensor],
    device: torch.device,
    use_ndnf: bool = False,
    include_volatile_as_numeric: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Get ground truth x and y for classifier

    Args:
        data: a list of tensors
        device: device to move data to
        use_ndnf: whether the data is used by neural DNF based models or not
        include_volatile_as_numeric: whether to volatile features are included
            as numeric features when going through preprocessing. Default is
            True, which means that the volatile features are included as numeric
            features. This flag only matters when use_ndnf is True.

    Returns:
        x: attribute label/attribute score
        y: class label
    """
    x = data[0]
    if use_ndnf:
        # Convert the binary attributes in {0, 1} to be in {-1, 1}
        actual_num_real_value_features = NUM_REAL_VALUED_FEATURES
        if include_volatile_as_numeric:
            actual_num_real_value_features += NUM_VOLATILE_FEATURES
        x[:, actual_num_real_value_features:] = (
            2 * x[:, actual_num_real_value_features:] - 1
        )

    y = data[1]

    x = x.to(device).float()
    y = y.to(device).float()

    return x, y
