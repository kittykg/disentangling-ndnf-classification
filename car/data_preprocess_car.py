import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ucimlrepo import fetch_ucirepo, dotdict


log = logging.getLogger()


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


def data_preprocess(
    data_pd: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    X: pd.DataFrame = data_pd.features  # type: ignore
    y: pd.DataFrame = data_pd.targets  # type: ignore

    X_numeric_pd = X.copy()
    for col in X.columns:
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
    X_oh_np: np.ndarray = ohe.fit_transform(X_numeric_pd.to_numpy()).toarray()  # type: ignore
    feature_names: list[str] = ohe.get_feature_names_out(
        input_features=X_numeric_pd.columns.to_list()
    ).tolist()

    y_str_np = y.to_numpy().flatten()
    le = LabelEncoder()
    y_np: np.ndarray = le.fit_transform(y_str_np)  # type: ignore

    return X_oh_np, y_np, feature_names, le.classes_.tolist()


@hydra.main(version_base=None, config_path="../conf/dataset", config_name="car")
def convert_to_discrete_and_save(cfg: DictConfig) -> None:
    uci_dataset = fetch_ucirepo(id=cfg["ucimlrepo_id"])
    log_relevant_metadata(uci_dataset)

    hold_out = cfg["hold_out"]["create_hold_out"]

    X_np, y_np, feature_names, class_names = data_preprocess(uci_dataset.data)  # type: ignore
    log.info("===============================================")
    log.info(f"Processed dataset: number of features: {X_np.shape[1]}")

    # Save the two numpy arrays into a compressed numpy file
    file_name = "car.npz"
    output_file_path = Path(cfg["save_dir"]) / file_name
    np.savez_compressed(
        output_file_path,
        X=X_np,
        y=y_np,
        feature_names=feature_names,
        class_names=class_names,
    )

    log.info(f"Saved the processed dataset to {output_file_path}")

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
    convert_to_discrete_and_save()
