import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo, dotdict

CATEGORICAL_FEATURES = ["a1", "a2", "a4", "a5"]
BINARY_FEATURES = ["a3", "a6"]

log = logging.getLogger()


def convert_finite_integer_to_categorical(
    feature_encoding: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Convert a finite integer column to a set of categorical columns.
    Args:
        feature_encoding: A 1D numpy array of integers.
    Returns:
        A tuple of two numpy arrays:
        - The first array is the unique values of the input array.
        - The second array is the one-hot encoding of the input array.
    """
    unique_values = np.sort(np.unique(feature_encoding))
    one_hot = np.zeros((len(feature_encoding), len(unique_values)), dtype=int)
    for i, f in enumerate(feature_encoding):
        one_hot[i, np.where(unique_values == f)[0]] = 1

    return unique_values, one_hot


def log_relevant_metadata(uci_dataset: dotdict) -> None:
    log.info(f"Dataset name: {uci_dataset.metadata.name}")  # type: ignore
    log.info(f"Number of instances: {uci_dataset.metadata.num_instances}")  # type: ignore
    log.info(f"Number of features: {uci_dataset.metadata.num_features}")  # type: ignore


def split_hold_out_data(
    X: np.ndarray, y: np.ndarray, hold_out_cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_size = hold_out_cfg.get("test_size", 0.2)
    random_state = hold_out_cfg.get("random_state", 73)
    stratify = hold_out_cfg.get("stratify", False)

    log.info(
        f"Creating hold out test set: test_size={test_size}, "
        f"random_state={random_state}, stratify={stratify}"
    )

    X_train, X_hold_out_test, y_train, y_hold_out_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    return X_train, X_hold_out_test, y_train, y_hold_out_test


@hydra.main(
    version_base=None, config_path="../conf/dataset", config_name="monk"
)
def convert_to_discrete_and_save(cfg: DictConfig) -> None:
    uci_dataset = fetch_ucirepo(id=cfg["ucimlrepo_id"])
    log_relevant_metadata(uci_dataset)

    data_pd = uci_dataset.data

    feature_names = []
    X = data_pd.features
    X_np = None

    for col in X.columns:
        if col in CATEGORICAL_FEATURES:
            unique_values, col_one_hot = convert_finite_integer_to_categorical(
                X[col].copy().to_numpy(dtype=np.int64)
            )
            for val in unique_values:
                feature_names.append(f"{col}_{val}")
            if X_np is None:
                X_np = col_one_hot
            else:
                X_np = np.column_stack((X_np, col_one_hot))
        elif col in BINARY_FEATURES:
            col_np = X[col].copy().to_numpy(dtype=np.int64)
            col_np -= 1  # make it 0-based
            feature_names.append(f"{col}_2")
            if X_np is None:
                X_np = col_np
            else:
                X_np = np.column_stack((X_np, col_np))

    assert X_np is not None
    log.info(f"Processed dataset: number of features: {X_np.shape[1]}")

    y_np = data_pd.targets.to_numpy().flatten()  # type: ignore

    # Save the two numpy arrays into a compressed numpy file
    file_name = "monk.npz"
    output_file_path = Path(cfg["save_dir"]) / file_name
    np.savez_compressed(
        output_file_path, X=X_np, y=y_np, feature_names=feature_names
    )

    log.info(f"Saved the processed dataset to {output_file_path}")

    if cfg["hold_out"]["create_hold_out"] is not None:
        log.info("===============================================")
        X_train, X_hold_out_test, y_train, y_hold_out_test = (
            split_hold_out_data(X_np, y_np, cfg["hold_out"])
        )
        np.savez_compressed(
            Path(cfg["save_dir"]) / f"train_{file_name}",
            X=X_train,
            y=y_train,
            feature_names=feature_names,
        )
        np.savez_compressed(
            Path(cfg["save_dir"]) / f"hold_out_test_{file_name}",
            X=X_hold_out_test,
            y=y_hold_out_test,
            feature_names=feature_names,
        )
        log.info("Hold out split data saved")
        log.info(f"Train data shape: {X_train.shape}")
        log.info(f"Train label balance: {np.bincount(y_train)}")
        log.info(
            f"Train label percentage: {np.bincount(y_train) / len(y_train)}"
        )
        log.info(f"Hold out test data shape: {X_hold_out_test.shape}")
        log.info(f"Hold out test label balance: {np.bincount(y_hold_out_test)}")
        log.info(
            "Hold out test label percentage: "
            f"{np.bincount(y_hold_out_test) / len(y_hold_out_test)}"
        )


if __name__ == "__main__":
    convert_to_discrete_and_save()
