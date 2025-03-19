import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo, dotdict

log = logging.getLogger()

# Features to be removed before the model, based on the analysis in the notebook
# https://www.kaggle.com/code/abdallahsaadelgendy/diabetes-prediction-eda-preprocessing-models#Data-Splitting
# These features are not selected when using sklearn's feature selection
REMOVED_FEATURES = ["Fruits", "Veggies", "Sex", "CholCheck", "AnyHealthcare"]

REAL_VALUE_FEATURES = ["BMI"]  # 84 distinct values

# These features below can be considered as categorical or real value features
VOLATILE_FEATURES = [
    "Age",  # 13 distinct values
    "MentHlth",  # 31 distinct values
    "PhysHlth",  # 31 distinct values
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


def standardise_numerical_features_and_convert_one_hot(
    X: pd.DataFrame, include_volatile_as_numeric: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    to_scale_indices = REAL_VALUE_FEATURES.copy()
    if include_volatile_as_numeric:
        to_scale_indices += VOLATILE_FEATURES

    to_exclude_indices = [
        col for col in X.columns if col not in to_scale_indices
    ]

    # apply the standard scaler to the numeric columns
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(X[to_scale_indices])
    scaled_pd = pd.DataFrame(scaled_cols, columns=to_scale_indices)

    # apply one hot encoding to the categorical columns
    X_numeric_pd = X[to_exclude_indices].copy()
    for col in to_exclude_indices:
        X_numeric_pd[col] = X_numeric_pd[col].astype("category").cat.codes

    def custom_combiner(feature: str, category):
        if feature == "feature":
            return "test"
        feature_og_name_dict = dict(
            enumerate(X[feature].astype("category").cat.categories)
        )
        return f"{feature}={feature_og_name_dict[category]}"

    ohe = OneHotEncoder(
        drop="if_binary",
        feature_name_combiner=custom_combiner,  # type: ignore
    )
    oh_np: np.ndarray = ohe.fit_transform(X_numeric_pd.to_numpy()).toarray()  # type: ignore
    feature_names: list[str] = ohe.get_feature_names_out(
        input_features=X_numeric_pd.columns.to_list()
    ).tolist()
    oh_pd = pd.DataFrame(oh_np, columns=feature_names)

    # combine the scaled columns with the columns that were excluded
    X_processed = pd.concat([scaled_pd, oh_pd], axis=1)

    # also return the unscaled columns with the one hot encoded columns
    X_no_scale = pd.concat([X[to_exclude_indices], oh_pd], axis=1)

    return X_processed, X_no_scale, scaler


def data_preprocess(
    uci_dataset: dotdict, include_volatile_as_numeric: bool
) -> dict[str, Any]:
    X: pd.DataFrame = uci_dataset.data.features  # type: ignore
    y: pd.DataFrame = uci_dataset.data.targets  # type: ignore

    # remove the features that are not useful
    X_clean = X.drop(REMOVED_FEATURES, axis=1)

    # standardise the numerical features and convert the categorical features to
    # one hot encoding
    X_processed, X_no_scale, scaler = (
        standardise_numerical_features_and_convert_one_hot(
            X_clean, include_volatile_as_numeric=include_volatile_as_numeric
        )
    )

    return {
        "X_no_scaling": X_no_scale.to_numpy(),
        "X": X_processed.to_numpy(),
        "y": y.to_numpy().flatten(),
        "feature_names": X_processed.columns.to_list(),
        "scaler": scaler,
    }


@hydra.main(version_base=None, config_path="../conf/dataset", config_name="cdc")
def preprocess_and_save(cfg: DictConfig) -> None:
    uci_dataset = fetch_ucirepo(id=cfg["ucimlrepo_id"])
    log_relevant_metadata(uci_dataset)

    hold_out = cfg["hold_out"]["create_hold_out"]

    ret_dict = data_preprocess(
        uci_dataset, cfg.get("include_volatile_as_numeric", True)
    )
    X_np = ret_dict["X"]
    y_np = ret_dict["y"]
    feature_names = ret_dict["feature_names"]
    log.info("===============================================")
    log.info(f"Processed dataset: number of features: {X_np.shape[1]}")

    # Save the two numpy arrays into a compressed numpy file
    file_name = "cdc.npz"
    output_file_path = Path(cfg["save_dir"]) / file_name
    np.savez_compressed(
        output_file_path,
        X=X_np,
        y=y_np,
        feature_names=feature_names,
    )
    log.info(f"Saved the processed dataset to {output_file_path}")

    # Also save the unscaled data
    np.savez_compressed(
        Path(cfg["save_dir"]) / f"cdc_no_scaling.npz",
        X=ret_dict["X_no_scaling"],
        y=y_np,
        feature_names=feature_names,
    )
    log.info(
        f"Saved the unscaled data to {Path(cfg['save_dir']) / 'cdc_no_scaling.npz'}"
    )

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

    undersample = cfg.get("undersample", True)
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
        file_name = "cdc_undersampled.npz"
        log.info(f"Undersampled dataset: {np.bincount(y_np)}")

        output_file_path = Path(cfg["save_dir"]) / file_name
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
    preprocess_and_save()
