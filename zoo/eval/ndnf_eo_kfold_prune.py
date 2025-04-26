"""
This script prunes the plain NDNF part of its corresponding NDNF-EO model. The
input NDNF-EO models are strictly after training and without any post-training
processing. The pruned NDNF model are stored and evaluated. The evaluation
metrics include accuracy, sample Jaccard, and macro Jaccard.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from neural_dnf.post_training import prune_neural_dnf
from neural_dnf import NeuralDNFEO, NeuralDNF

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import synthesize
from utils import post_to_discord_webhook
from zoo.data_utils_zoo import *
from zoo.eval.ndnf_eval_common import (
    ndnf_based_model_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    AFTER_TRAIN_MODEL_BASE_NAME,
    FIRST_PRUNE_MODEL_BASE_NAME,
)
from zoo.train_zoo import construct_model


log = logging.getLogger()


def comparison_fn(og_parsed_eval_log, new_parsed_eval_log):
    # check accuracy, if it is less than the original, then no prune
    if new_parsed_eval_log["accuracy"] < og_parsed_eval_log["accuracy"]:
        return False

    # check sample jaccard, if it is less than the original, then no prune
    if (
        new_parsed_eval_log["sample_jaccard"]
        < og_parsed_eval_log["sample_jaccard"]
    ):
        return False

    # check macro jaccard, if it is less than the original, then no prune
    if (
        new_parsed_eval_log["macro_jaccard"]
        < og_parsed_eval_log["macro_jaccard"]
    ):
        return False

    # error dict
    og_error_dict = og_parsed_eval_log["error_dict"]
    new_error_dict = new_parsed_eval_log["error_dict"]

    # check overall error class count, if it is greater than the original,
    # then no prune
    if (
        new_error_dict["overall_error_class_count"]
        > og_error_dict["overall_error_class_count"]
    ):
        return False

    for k in [
        "missing_predictions",
        "multiple_predictions",
        "wrong_predictions",
    ]:
        # for each of the error, check if the set of classes in the new
        # dict is a subset of the classes in the original dict
        og_error_classes = set([int(c) for c in og_error_dict[k].keys()])
        new_error_classes = set([int(c) for c in new_error_dict[k].keys()])
        if not new_error_classes.issubset(og_error_classes):
            return False

    return True


def multiround_prune(
    model: NeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    eval_log_fn: Callable[[dict[str, Any]], dict[str, float]],
) -> int:
    prune_eval_function = lambda: parse_eval_return_meters_with_logging(
        eval_meters=ndnf_based_model_eval(model, device, train_loader),
        model_name="Prune (intermediate)",
        do_logging=False,
    )

    prune_iteration = 1
    while True:
        log.info(f"Pruning iteration: {prune_iteration }")
        start_time = datetime.now()

        prune_result_dict = prune_neural_dnf(
            model,
            prune_eval_function,
            {},
            comparison_fn,
            options={
                "skip_prune_disj_with_empty_conj": True,
                "skip_last_prune_disj": True,
            },
        )

        important_keys = [
            "disj_prune_count_1",
            "unused_conjunctions_2",
            "conj_prune_count_3",
            # "prune_disj_with_empty_conj_count_4",
        ]

        end_time = datetime.now()
        log.info(f"\tTime taken: {end_time - start_time}")
        log.info(
            f"\tPruned disjunction count: {prune_result_dict['disj_prune_count_1']}"
        )
        log.info(
            f"\tRemoved unused conjunction count: {prune_result_dict['unused_conjunctions_2']}"
        )
        log.info(
            f"\tPruned conjunction count: {prune_result_dict['conj_prune_count_3']}"
        )
        # log.info(
        #     f"\tPruned disj with empty conj: {prune_result_dict['prune_disj_with_empty_conj_count_4']}"
        # )

        eval_log_fn(
            {"model_name": f"Plain NDNF - (Prune iteration: {prune_iteration})"}
        )
        log.info("..................................")
        # If any of the important keys has the value not 0, then we should
        # continue pruning
        if any([prune_result_dict[k] != 0 for k in important_keys]):
            prune_iteration += 1
        else:
            break

    return prune_iteration


def single_model_prune(
    fold_id: int,
    model: NeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_dir: Path,
) -> dict[str, float]:
    def _eval_with_log_wrapper(model_name: str) -> dict[str, float]:
        eval_meters = ndnf_based_model_eval(model, device, val_loader)
        return parse_eval_return_meters_with_logging(
            eval_meters, model_name, check_error_meter=False
        )

    # Stage 1: Evaluate the model post-training
    _eval_with_log_wrapper("Plain NDNF (after training)")
    log.info("------------------------------------------")

    # Stage 2: Prune the model / load pruned checkpoint
    # First check for checkpoints. If the model is already pruned, then we load
    # the pruned model Otherwise, we prune the model and save the pruned model
    model_path = model_dir / f"{FIRST_PRUNE_MODEL_BASE_NAME}_fold_{fold_id}.pth"
    if model_path.exists():
        log.info("Loading the pruned model...")
        pruned_state = torch.load(
            model_path, map_location=device, weights_only=True
        )
        model.load_state_dict(pruned_state)
    else:
        log.info("Pruning the model...")
        multiround_prune(
            model,
            device,
            train_loader,
            lambda x: _eval_with_log_wrapper(x["model_name"]),
        )
        torch.save(model.state_dict(), model_path)

    prune_eval_log = _eval_with_log_wrapper("Plain NDNF pruned")
    log.info("============================================================")

    return prune_eval_log


def post_train_prune(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]
    full_experiment_name = f"{eval_cfg['experiment_name']}_{eval_cfg['seed']}"
    run_dir_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    # Set up device
    use_cuda = torch.cuda.is_available() and eval_cfg["use_cuda"]
    use_mps = (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and eval_cfg.get("use_mps", False)
    )
    assert not (use_cuda and use_mps), "Cannot use both CUDA and MPS"
    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
    log.info(f"Device: {device}")

    # Load data
    X, y, feature_names = get_zoo_data_np_from_path(
        data_dir_path=Path(cfg["dataset"]["save_dir"])
    )
    dataset = ZooDataset(X, y)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=eval_cfg["k_folds"],
        shuffle=True,
        random_state=eval_cfg["seed"],
    )

    ret_dicts: list[dict[str, float]] = []
    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y)):
        log.info(f"Fold {fold_id} starts")
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
        )
        model = construct_model(eval_cfg)
        assert isinstance(model, NeuralDNFEO)
        model.to(device)
        model_state = torch.load(
            model_dir / f"{AFTER_TRAIN_MODEL_BASE_NAME}_fold_{fold_id}.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(model_state)

        # Data loaders
        train_loader, val_loader = get_zoo_dataloaders(
            dataset=dataset,
            train_index=train_index,
            test_index=test_index,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            loader_num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
        )

        plain_model = model.to_ndnf()
        plain_model.to(device)
        plain_model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        prune_eval_log = single_model_prune(
            fold_id,
            plain_model,
            device,
            train_loader,
            val_loader,
            model_dir,
        )
        ret_dicts.append(prune_eval_log)
        with open(model_dir / f"fold_{fold_id}_mr_prune_result.json", "w") as f:
            json.dump(prune_eval_log, f, indent=4)
        log.info("============================================================")

    # Synthesize the results
    relevant_keys = list(ret_dicts[0].keys())
    return_dict = {}
    for k in relevant_keys:
        synth_dict = synthesize(np.array([d[k] for d in ret_dicts]))
        for sk, sv in synth_dict.items():
            return_dict[f"{k}/{sk}"] = sv

    log.info("Synthesized results:")
    for k, v in return_dict.items():
        log.info(f"{k}: {v:.3f}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    torch.autograd.set_detect_anomaly(True)

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_prune(cfg)
        if use_discord_webhook:
            msg_body = "Success!"
    except BaseException as e:
        if use_discord_webhook:
            if isinstance(e, KeyboardInterrupt):
                keyboard_interrupt = True
            else:
                msg_body = "Check the logs for more details."

        print(traceback.format_exc())
        errored = True
    finally:
        if use_discord_webhook:
            if msg_body is None:
                msg_body = ""
            webhook_url = cfg["webhook"]["discord_webhook_url"]
            post_to_discord_webhook(
                webhook_url=webhook_url,
                experiment_name=f"{cfg['eval']['experiment_name']} Kfold Prune",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
