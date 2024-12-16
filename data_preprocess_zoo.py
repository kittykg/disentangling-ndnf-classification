import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import numpy.typing as npt
from ucimlrepo import fetch_ucirepo, dotdict


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


@hydra.main(config_path="conf/dataset", config_name="zoo", version_base="1.2")
def convert_to_discrete_and_save(cfg: DictConfig) -> None:
    uci_dataset = fetch_ucirepo(id=cfg["ucimlrepo_id"])
    log_relevant_metadata(uci_dataset)

    data_pd = uci_dataset.data
    feature_names = []
    X_np = None

    for v in uci_dataset.variables.to_numpy():  # type: ignore
        if v[1] != "Feature":
            # Skip non-feature rows
            continue

        feature_name = v[0]
        feature_type = v[2]

        col: npt.NDArray[np.int64] = data_pd.features[feature_name].to_numpy()  # type: ignore

        if feature_type == "Categorical":
            unique_vals, col = convert_finite_integer_to_categorical(col)
            for val in unique_vals:
                feature_names.append(f"{feature_name}_{val}")
        else:
            feature_names.append(feature_name)

        if X_np is None:
            X_np = col
        else:
            X_np = np.column_stack((X_np, col))
    assert X_np is not None

    log.info(f"Processed dataset: number of features: {X_np.shape[1]}")

    y_np = data_pd.targets.to_numpy().flatten() - 1  # type: ignore

    # Save the two numpy arrays into a compressed numpy file
    output_file_path = Path(cfg["save_dir"]) / "zoo.npz"
    np.savez_compressed(
        output_file_path, X=X_np, y=y_np, feature_names=feature_names
    )

    log.info(f"Saved the processed dataset to {output_file_path}")


if __name__ == "__main__":
    convert_to_discrete_and_save()
