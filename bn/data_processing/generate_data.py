from copy import deepcopy
import logging
from pathlib import Path
import random
import sys

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import RepeatedKFold, train_test_split

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from data_processing.common import (
    MUTATION_RATIO_LIST,
    INCOMPLETE_RATIO_LIST,
    LOGIC_PROGRAM_DIR,
    BNDatasetSubType,
)
from data_processing.logic_program import Transition, LogicProgram


log = logging.getLogger()


def generate_raw_transition_data(logic_program_path: Path) -> list[Transition]:
    if not logic_program_path.exists():
        raise FileNotFoundError(f"File {logic_program_path} does not exist")
    logic_program = LogicProgram.load_from_file(logic_program_path)

    return logic_program.generate_all_transitions()


def gen_normal_data(
    raw_data: list[Transition],
    save_file_dir: Path,
    repeated_time: int = 2,
    k_fold: int = 5,
    random_state: int = 73,
) -> None:
    rkf_and_save(
        np.array(raw_data),
        k_fold,
        repeated_time,
        random_state,
        save_file_dir / BNDatasetSubType.NORMAL.value,
    )


def gen_data_fuzzy(
    raw_data: list[Transition],
    save_file_dir: Path,
    repeated_time: int = 2,
    k_fold: int = 5,
    random_state: int = 73,
    complete_random_flip: bool = False,
) -> None:
    # D_LFIT version of fuzzy only creates 1-to-2 mapping since it only changes
    # the input bits at random
    # To create a version where both 1-to-2 and 2-to-1 mapping are possible,
    # please set `complete_random_flip` to True
    num_data = len(raw_data)
    num_variables = len(raw_data[0][0])

    def _add_noise(mutation_ratio: int) -> list[Transition]:
        total_number = num_data * num_variables
        if complete_random_flip:
            total_number *= 2

        mutation_length = int(total_number * mutation_ratio * 0.01)
        mutation_index = random.sample(range(total_number), mutation_length)

        new_data = deepcopy(raw_data)
        for i in mutation_index:
            # Decode i to either [x][0][z] or [x][y][z]
            # i = x * num_variables + z if not `complete_random_flip`
            # i = x * num_variables * 2 + y * num_variables + z if
            # `complete_random_flip`
            if complete_random_flip:
                x = i // (2 * num_variables)
                y = (i - 2 * num_variables * x) // num_variables
                z = i - 2 * num_variables * x - num_variables * y
            else:
                x = i // num_variables
                y = 0
                z = i - num_variables * x

            value = new_data[x][y][z]
            new_data[x][y][z] = 1 - value
        return new_data

    for mr in MUTATION_RATIO_LIST:
        rkf_and_save(
            np.array(_add_noise(mr)),
            k_fold,
            repeated_time,
            random_state,
            save_file_dir
            / BNDatasetSubType.FUZZY.value
            / f"{BNDatasetSubType.FUZZY.subtype_folder_prefix()}_{mr}",
        )


def gen_data_incomplete(
    raw_data: list[Transition],
    save_file_dir: Path,
    repeated_time: int = 2,
    k_fold: int = 5,
    random_state: int = 73,
) -> None:
    num_data = len(raw_data)
    num_variables = len(raw_data[0][0])
    total_number = num_data * num_variables

    def _leave_out_data(incomplete_ratio: int) -> list[Transition]:
        new_data = deepcopy(raw_data)
        random.Random(random_state).shuffle(new_data)
        remaining_length = int(total_number * incomplete_ratio * 0.01)
        return new_data[:remaining_length]

    for ir in INCOMPLETE_RATIO_LIST:
        rkf_and_save(
            np.array(_leave_out_data(ir)),
            k_fold,
            repeated_time,
            random_state,
            save_file_dir
            / BNDatasetSubType.INCOMPLETE.value
            / f"{BNDatasetSubType.INCOMPLETE.subtype_folder_prefix()}{ir}",
        )


def rkf_and_save(
    data: np.ndarray,
    k_fold: int,
    repeated_time: int,
    random_state: int,
    save_file_dir: Path,
):
    rkf = RepeatedKFold(
        n_splits=k_fold,
        n_repeats=repeated_time,
        random_state=random_state,
    )
    try:
        for i, (train_idx, test_idx) in enumerate(rkf.split(data)):
            train_data = data[train_idx]
            test_data = data[test_idx]

            main_dir = save_file_dir / f"fold_{i}"
            if not main_dir.exists() or not main_dir.is_dir():
                main_dir.mkdir(parents=True, exist_ok=True)

            train_file = main_dir / "train.npy"
            test_file = main_dir / "test.npy"
            with open(train_file, "wb") as f:
                np.save(f, train_data)
            with open(test_file, "wb") as f:
                np.save(f, test_data)
    except ValueError as e:
        log.info(e)
        log.info("Continuing...")


def split_hold_out_data(
    raw_data: np.ndarray, hold_out_cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    test_size = hold_out_cfg.get("test_size", 0.2)
    random_state = hold_out_cfg.get("random_state", 73)

    log.info(
        f"Creating hold out test set: test_size={test_size}, "
        f"random_state={random_state}"
    )

    indices = np.arange(len(raw_data))
    train_idx, hold_out_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )

    return raw_data[train_idx], raw_data[hold_out_idx]


@hydra.main(
    version_base=None, config_path="../../conf/dataset", config_name="bn"
)
def main(cfg: DictConfig):
    dataset_name = cfg["dataset_name"]
    repeated_time = cfg["repeated_time"]
    k_fold = cfg["k_fold"]
    random_state = cfg["random_state"]

    logic_program_path = LOGIC_PROGRAM_DIR / f"{dataset_name}.lp"
    if not logic_program_path.exists():
        raise FileNotFoundError(f"File {logic_program_path} does not exist")

    save_file_dir = Path(cfg["save_file_base_dir"]) / dataset_name
    save_file_dir.mkdir(parents=True, exist_ok=True)

    raw_data = generate_raw_transition_data(logic_program_path)

    with open(save_file_dir / f"{dataset_name}_raw_data.npy", "wb") as f:
        np.save(f, np.array(raw_data))

    if "hold_out" in cfg and cfg["hold_out"]["create_hold_out"]:
        train_data, hold_out_data = split_hold_out_data(
            np.array(raw_data), cfg["hold_out"]
        )
        with open(save_file_dir / f"{dataset_name}_train.npy", "wb") as f:
            np.save(f, train_data)
        with open(
            save_file_dir / f"{dataset_name}_hold_out_test.npy", "wb"
        ) as f:
            np.save(f, hold_out_data)
        log.info(f"Train data shape: {train_data.shape}")
        log.info(f"Hold out test data shape: {hold_out_data.shape}")

    gen_normal_data(
        raw_data, save_file_dir, repeated_time, k_fold, random_state
    )
    gen_data_fuzzy(raw_data, save_file_dir, repeated_time, k_fold, random_state)
    gen_data_incomplete(
        raw_data, save_file_dir, repeated_time, k_fold, random_state
    )


if __name__ == "__main__":
    main()
