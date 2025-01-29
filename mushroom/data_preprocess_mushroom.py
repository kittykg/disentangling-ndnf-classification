import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo, dotdict


log = logging.getLogger()


def log_relevant_metadata(uci_dataset: dotdict) -> None:
    log.info(f"Dataset name: {uci_dataset.metadata.name}")  # type: ignore
    log.info(f"Number of instances: {uci_dataset.metadata.num_instances}")  # type: ignore
    log.info(f"Number of features: {uci_dataset.metadata.num_features}")  # type: ignore


def impute_data(X: np.ndarray, method: str) -> np.ndarray:  # type: ignore
    # TODO: Unuser if we will need this. Leave it for now.
    pass


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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X: pd.DataFrame = data_pd.features  # type: ignore
    y: pd.DataFrame = data_pd.targets  # type: ignore

    cols_with_nan = []
    X_numeric_pd = X.copy()
    for col in X.columns:
        X_numeric_pd[col] = X_numeric_pd[col].astype("category").cat.codes
        # if there is nan in this column, replace it with max + 1
        if X[col].isna().sum() > 0:
            X_numeric_pd[col] = X_numeric_pd[col].replace(
                -1, max(X_numeric_pd[col]) + 1
            )
            cols_with_nan.append(col)

    for e in cols_with_nan:
        log.info(f"Column {e} had NaN.")
        log.info(f"Now the unique values are: {X_numeric_pd[e].unique()}")

        # check the frequency of each value in the column, including NaN
        log.info(f"Frequency of each value in the column: ")
        log.info(X_numeric_pd["stalk-root"].value_counts(dropna=False))

        log.info("The categories are: ")
        log.info(X[e].astype("category").cat.categories)

    def custom_combiner(feature: str, category):
        if feature == "feature":
            return "test"
        feature_og_name_dict = dict(
            enumerate(X[feature].astype("category").cat.categories)
        )
        if feature in cols_with_nan and category not in feature_og_name_dict:
            return f"{feature}=nan"
        return f"{feature}={feature_og_name_dict[category]}"

    ohe = OneHotEncoder(
        drop="if_binary",
        feature_name_combiner=custom_combiner,  # type: ignore
    )
    X_oh_np: np.ndarray = ohe.fit_transform(X_numeric_pd.to_numpy()).toarray()  # type: ignore
    feature_names: list[str] = ohe.get_feature_names_out(
        input_features=X_numeric_pd.columns.to_list()
    ).tolist()

    y_np = np.where(y == "p", 1, 0).flatten()

    return X_oh_np, y_np, feature_names


@hydra.main(
    version_base=None, config_path="../conf/dataset", config_name="mushroom"
)
def convert_to_discrete_and_save(cfg: DictConfig) -> None:
    uci_dataset = fetch_ucirepo(id=cfg["ucimlrepo_id"])
    log_relevant_metadata(uci_dataset)

    impute_method = cfg.get("impute", None)
    hold_out = cfg["hold_out"]["create_hold_out"]

    X_np, y_np, feature_names = data_preprocess(uci_dataset.data)  # type: ignore
    log.info("===============================================")
    log.info(f"Processed dataset: number of features: {X_np.shape[1]}")

    # Save the two numpy arrays into a compressed numpy file
    if impute_method is None:
        file_name = "mushroom_no_imputation.npz"
    else:
        file_name = f"mushroom_{impute_method}.npz"
        X_np = impute_data(X_np, impute_method)
    output_file_path = Path(cfg["save_dir"]) / file_name
    np.savez_compressed(
        output_file_path, X=X_np, y=y_np, feature_names=feature_names
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
        log.info(f"Hold out test data shape: {X_hold_out_test.shape}")
        log.info(f"Hold out test label balance: {np.bincount(y_hold_out_test)}")


if __name__ == "__main__":
    convert_to_discrete_and_save()
