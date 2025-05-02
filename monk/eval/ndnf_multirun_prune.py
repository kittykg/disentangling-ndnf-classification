"""
This script prunes the MonkNDNF model. The input models are strictly after
training and without any post-training processing. The pruned NDNF models are
stored and evaluated. The evaluation metrics include accuracy, precision,
recall, F1 score and MCC.
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
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from neural_dnf.post_training import prune_neural_dnf

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

from monk.data_utils_monk import (
    MonkDataset,
    get_monk_data_np_from_path,
)
from monk.eval.eval_common import (
    monk_classifier_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    AFTER_TRAIN_MODEL_BASE_NAME,
    FIRST_PRUNE_MODEL_BASE_NAME,
)
from monk.models import MonkNeuralDNF, construct_model


log = logging.getLogger()


def multiround_prune(
    model: MonkNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    eval_log_fn: Callable[[dict[str, Any]], dict[str, float]],
) -> int:
    def comparison_fn(og_parsed_eval_log, new_parsed_eval_log):
        for k in ["accuracy", "precision", "recall", "f1", "mcc"]:
            if new_parsed_eval_log[k] < og_parsed_eval_log[k]:
                return False
        return True

    prune_iteration = 1

    prune_eval_function = lambda: parse_eval_return_meters_with_logging(
        eval_meters=monk_classifier_eval(model, device, train_loader),
        model_name="Prune (intermediate)",
        do_logging=False,
    )

    while True:
        log.info(f"Pruning iteration: {prune_iteration }")
        start_time = datetime.now()

        prune_result_dict = prune_neural_dnf(
            model.ndnf,
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
    model: MonkNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_dir: Path,
) -> dict[str, float]:
    def _eval_with_log_wrapper(
        model_name: str, data_loader: DataLoader = val_loader
    ) -> dict[str, float]:
        eval_meters = monk_classifier_eval(model, device, data_loader)
        return parse_eval_return_meters_with_logging(eval_meters, model_name)

    # Stage 1: Evaluate the model post-training
    _eval_with_log_wrapper("Monk NDNF (after training, test)", test_loader)
    log.info("------------------------------------------")

    # Stage 2: Prune the model / load pruned checkpoint
    # First check for checkpoints. If the model is already pruned, then we load
    # the pruned model Otherwise, we prune the model and save the pruned model
    model_path = model_dir / f"{FIRST_PRUNE_MODEL_BASE_NAME}.pth"
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

    prune_eval_log = _eval_with_log_wrapper(
        "Monk NDNF pruned (test)", test_loader
    )
    log.info("============================================================")

    return prune_eval_log


def multirun_prune(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

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

    # Load test data
    hold_out_test_X, hold_out_test_y, _ = get_monk_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = MonkDataset(hold_out_test_X, hold_out_test_y)
    test_loader = DataLoader(
        test_dataset,
        batch_size=DEFAULT_LOADER_BATCH_SIZE,
        num_workers=DEFAULT_LOADER_NUM_WORKERS,
        pin_memory=device == torch.device("cuda"),
    )

    # Experiment name
    experiment_name = eval_cfg["experiment_name"]
    caps_experiment_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(experiment_name.split("_"))
        ]
    )

    ret_dicts: list[dict[str, float]] = []
    for s in eval_cfg["multirun_seeds"]:
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"])
            / caps_experiment_name
            / f"{caps_experiment_name}-{s}"
        )
        model = construct_model(eval_cfg, num_features=hold_out_test_X.shape[1])
        assert isinstance(model, MonkNeuralDNF)
        model.to(device)
        model_state = torch.load(
            model_dir / f"{AFTER_TRAIN_MODEL_BASE_NAME}.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(model_state)
        model.eval()

        # Data loaders
        X, y, _ = get_monk_data_np_from_path(cfg["dataset"], is_test=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=eval_cfg.get("val_size", 0.2),
            random_state=eval_cfg.get("val_seed", 73),
        )
        train_dataset = MonkDataset(X_train, y_train)
        val_dataset = MonkDataset(X_val, y_val)
        train_loader = DataLoader(
            train_dataset,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
        )

        log.info(f"Experiment {model_dir.name} loaded!")
        prune_eval_log = single_model_prune(
            model, device, train_loader, val_loader, test_loader, model_dir
        )
        ret_dicts.append(prune_eval_log)

        with open(model_dir / f"model_mr_prune_result.json", "w") as f:
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

    with open("multirun_prune_result.json", "w") as f:
        json.dump(return_dict, f, indent=4)


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
        multirun_prune(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Prune",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
