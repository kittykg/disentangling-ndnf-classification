import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo, dotdict

log = logging.getLogger()

# Features to be removed before the model, based on the analysis in the notebook
# https://www.kaggle.com/code/roshanchoudhary/forest-cover-walkthrough-in-python-knn-96-51/notebook
# Hillshade_3pm: high correlation with Hillshade_9am and Hillshade_Noon
# Soil type 7, 15, 37: only one target cover type
# Other soil types: low standard deviation
REMOVED_FEATURES = [
    "Hillshade_3pm",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type21",
    "Soil_Type25",
    "Soil_Type28",
    "Soil_Type36",
    "Soil_Type37",
]


def log_relevant_metadata(uci_dataset: dotdict) -> None:
    log.info(f"Dataset name: {uci_dataset.metadata.name}")  # type: ignore
    log.info(f"Number of instances: {uci_dataset.metadata.num_instances}")  # type: ignore
    log.info(f"Number of features: {uci_dataset.metadata.num_features}")  # type: ignore


def get_undersampling_strategy(
    X: np.ndarray,
    y: np.ndarray,
    nm_cfg: dict[str, Any] = {
        "version": 1,
        "n_neighbors": 10,
    },
) -> tuple[np.ndarray, np.ndarray]:
    nm = NearMiss(**nm_cfg)
    X_resampled, y_resampled = nm.fit_resample(X, y)  # type: ignore
    log.info(
        f"Undersampling using Near Miss: {nm_cfg}, "
        f"original shape: {X.shape}, resampled shape: {X_resampled.shape}"
    )
    return X_resampled, y_resampled  # type: ignore


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


def standardise_numerical_features(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    # Soil type and wilderness area are binary features, so we don't need to
    # standardise them
    to_exclude_indices = [
        col
        for col in X.columns
        if col.startswith("Soil_Type") or col.startswith("Wilderness_Area")
    ]
    to_scale_indices = [
        col for col in X.columns if col not in to_exclude_indices
    ]

    # apply the standard scaler to the columns that are not soil type
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(X[to_scale_indices])

    # combine the scaled columns with the columns that were excluded
    X_scaled = pd.concat(
        [
            pd.DataFrame(scaled_cols, columns=to_scale_indices),
            X[to_exclude_indices],
        ],
        axis=1,
    )

    return X_scaled, scaler


def convert_categorical_to_binary_encoding(X: pd.DataFrame) -> pd.DataFrame:
    # covert the wilderness area and soil type to binary encoding.
    # e.g. We convert the wilderness area to 2 bits, where 00 is wilderness area
    # 1, 01 is wilderness area 2, 10 is wilderness area 3, and 11 is wilderness
    # area 4. For an example, if wilderness area 1 is present (and by definition
    # of categorical feature only one of the wilderness areas can be present),
    # the feature vector will be [0 0]
    wa_indices = [col for col in X.columns if col.startswith("Wilderness_Area")]
    wa_columns = X[wa_indices].values
    new_wa_columns = np.zeros((wa_columns.shape[0], 2), dtype=int)
    for i in range(wa_columns.shape[0]):
        one_location = np.where(wa_columns[i] == 1)[0]
        assert len(one_location) == 1
        one_location = one_location[0]
        new_wa_columns[i] = np.array(
            [int(x) for x in np.binary_repr(one_location, width=2)]
        )

    st_indices = [col for col in X.columns if col.startswith("Soil_Type")]
    # after the removal of some soil types, there are 31 types left, and we have
    # 32 unique combinations since it's possible to have all 31 types not true.
    # So we need log_2(32) = 5 bits to represent the soil types.
    new_st_columns = np.zeros((wa_columns.shape[0], 5), dtype=int)
    st_columns = X[st_indices].values
    for i in range(st_columns.shape[0]):
        # there is maximum one place where the value is 1, and the rest are 0.
        # we take the location of the 1 as its int value and convert it to
        # binary If there is no 1, we assign it to '31'
        one_location = np.where(st_columns[i] == 1)[0]
        assert len(one_location) <= 1
        if len(one_location) == 0:
            one_location = 31
        else:
            one_location = one_location[0]

        new_st_columns[i] = np.array(
            [int(x) for x in np.binary_repr(one_location, width=5)]
        )

    # Drop the original columns and add the new columns
    X_new = X.drop(wa_indices + st_indices, axis=1)
    X_new = pd.concat(
        [
            X_new,
            pd.DataFrame(
                new_wa_columns,
                columns=["Wilderness_Area_Bit_1", "Wilderness_Area_Bit_0"],
            ),
            pd.DataFrame(
                new_st_columns,
                columns=[f"Soil_Type_Bit{i}" for i in range(4, -1, -1)],
            ),
        ],
        axis=1,
    )
    return X_new


def data_preprocess(uci_dataset: dotdict, cfg: DictConfig) -> dict[str, Any]:
    X: pd.DataFrame = uci_dataset.data.features  # type: ignore
    y: pd.DataFrame = uci_dataset.data.targets  # type: ignore

    # wilderness area 2, 3, and 4 are not in the right position, so we need to
    # move them
    wrong_location_keys = [f"Wilderness_Area{i}" for i in [2, 3, 4]]
    for k in wrong_location_keys:
        X = X.drop(k, axis=1)
    for k, i in zip(wrong_location_keys, [11, 12, 13]):
        X.insert(i, k, uci_dataset.data.features[k])  # type: ignore

    # remove the features that are not useful
    X_clean = X.drop(REMOVED_FEATURES, axis=1)

    if cfg.get("convert_categorical_to_binary_encoding", False):
        X_clean = convert_categorical_to_binary_encoding(X_clean)

    # standardise the numerical features
    X_scaled, scaler = standardise_numerical_features(X_clean)

    return {
        "X_no_scaling": X_clean.to_numpy(),
        "X": X_scaled.to_numpy(),
        "y": y.to_numpy().flatten() - 1,
        "feature_names": X_scaled.columns.to_list(),
        "scaler": scaler,
    }


@hydra.main(
    version_base=None, config_path="../conf/dataset", config_name="covertype"
)
def preprocess_and_save(cfg: DictConfig) -> None:
    uci_dataset = fetch_ucirepo(id=cfg["ucimlrepo_id"])
    log_relevant_metadata(uci_dataset)

    hold_out = cfg["hold_out"]["create_hold_out"]

    ret_dict = data_preprocess(uci_dataset, cfg)
    X_np = ret_dict["X"]
    y_np = ret_dict["y"]
    feature_names = ret_dict["feature_names"]
    log.info("===============================================")
    log.info(f"Processed dataset: number of features: {X_np.shape[1]}")

    # Save the two numpy arrays into a compressed numpy file
    file_base_name = "covertype"
    if cfg.get("convert_categorical_to_binary_encoding", False):
        file_base_name = "covertype_c2b"
    output_file_path = Path(cfg["save_dir"]) / f"{file_base_name}.npz"
    np.savez_compressed(
        output_file_path,
        X=X_np,
        y=y_np,
        feature_names=feature_names,
    )
    log.info(f"Saved the processed dataset to {output_file_path}")

    # Also save the unscaled data
    unscaled_output_path = (
        Path(cfg["save_dir"]) / f"{file_base_name}_no_scaling.npz"
    )
    np.savez_compressed(
        unscaled_output_path,
        X=ret_dict["X_no_scaling"],
        y=y_np,
        feature_names=feature_names,
    )
    log.info(f"Saved the unscaled data to {unscaled_output_path}")

    # save the scaler
    scaler = ret_dict["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_ is not None and scaler.var_ is not None
    np.savez_compressed(
        Path(cfg["save_dir"]) / "scaler.npz",
        mean=scaler.mean_,
        var=scaler.var_,
    )
    log.info(f"Saved the scaler to {Path(cfg['save_dir']) / 'scaler.npz'}")

    undersample = cfg.get("undersample", False)
    if undersample:
        log.info("===============================================")
        log.info("Undersampling the dataset")
        X_np, y_np = get_undersampling_strategy(
            X_np,
            y_np,
            cfg.get(
                "undersample_cfg",
                {
                    "version": 1,
                    "n_neighbors": 10,
                },
            ),
        )
        file_base_name += "_undersampled"
        log.info(f"Undersampled dataset: {np.bincount(y_np)}")

        output_file_path = Path(cfg["save_dir"]) / f"{file_base_name}.npz"
        np.savez_compressed(
            output_file_path,
            X=X_np,
            y=y_np,
            feature_names=feature_names,
        )
        log.info(f"Saved the undersampled dataset to {output_file_path}")

    # create hold out test set if needed
    if hold_out is not None:
        log.info("===============================================")
        X_train, X_hold_out_test, y_train, y_hold_out_test = (
            split_hold_out_data(X_np, y_np, cfg["hold_out"])
        )
        np.savez_compressed(
            Path(cfg["save_dir"]) / f"train_{file_base_name}.npz",
            X=X_train,
            y=y_train,
            feature_names=feature_names,
        )
        np.savez_compressed(
            Path(cfg["save_dir"]) / f"hold_out_test_{file_base_name}.npz",
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
    preprocess_and_save()
