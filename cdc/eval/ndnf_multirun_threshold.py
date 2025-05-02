"""
This script thresholds pruned CDCNeuralDNF model. The input NDNF models are
strictly after pruning stage in the post-training processing pipeline. The
thresholded CDCNeuralDNF models are stored and evaluated. The evaluation
metrics include accuracy, precision, recall, F1 score and MCC.
"""

import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from neural_dnf.post_training import (
    get_thresholding_upper_bound,
    apply_threshold,
)

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import AccuracyMeter, synthesize
from utils import post_to_discord_webhook

from cdc.data_utils_cdc import (
    CDCDataset,
    get_cdc_data_np_from_path,
)
from cdc.eval.eval_common import (
    cdc_classifier_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    FIRST_PRUNE_MODEL_BASE_NAME,
    THRESHOLD_MODEL_BASE_NAME,
    THRESHOLD_RESULT_JSON_BASE_NAME,
)
from cdc.eval.ndnf_multirun_prune import multiround_prune
from cdc.models import CDCNeuralDNF, construct_model


log = logging.getLogger()


def single_model_threshold(
    model: CDCNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_dir: Path,
) -> dict[str, Any]:
    def _eval_with_log_wrapper(
        model_name: str, data_loader: DataLoader = val_loader
    ) -> dict[str, Any]:
        eval_meters = cdc_classifier_eval(model, device, data_loader)
        return parse_eval_return_meters_with_logging(eval_meters, model_name)

    # Stage 1: Evaluate the pruned model
    prune_log = _eval_with_log_wrapper(
        "Pruned CDC NDNF model (test)", test_loader
    )
    log.info("------------------------------------------")

    # Stage 2: Threshold + after threshold prune
    def threshold(do_logging: bool = False) -> dict[str, Any]:
        log.info("Thresholding the model...")

        og_conj_weight = model.ndnf.conjunctions.weights.data.clone()
        og_disj_weight = model.ndnf.disjunctions.weights.data.clone()

        threshold_upper_bound = get_thresholding_upper_bound(model.ndnf)
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts_with_t_val = []
        for v in t_vals:
            apply_threshold(model.ndnf, og_conj_weight, og_disj_weight, v)
            threshold_eval_dict = cdc_classifier_eval(
                model, device, train_loader
            )
            acc_meter: AccuracyMeter = threshold_eval_dict["acc_meter"]  # type: ignore
            accuracy = acc_meter.get_average()
            other_metrics = acc_meter.get_other_classification_metrics()
            precision = other_metrics["precision"]
            recall = other_metrics["recall"]
            f1 = other_metrics["f1"]
            mcc = other_metrics["mcc"]

            result_dicts_with_t_val.append(
                {
                    "t_val": v.item(),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                }
            )

        sorted_result_dicts: list[dict[str, float]] = sorted(
            result_dicts_with_t_val,
            key=lambda x: (
                x["mcc"],
                x["recall"],  # maximise recall to minimise false negatives
            ),
            reverse=True,
        )

        if do_logging:
            log.info("Top 5 thresholding candidates:")
            for i, d in enumerate(sorted_result_dicts[:5]):
                log.info(
                    f"-- Candidate {i + 1} --\n"
                    f"\tt_val: {d['t_val']:.2f}  "
                    f"Acc: {d['accuracy']:.3f}  "
                    f"Precision: {d['precision']:.3f}  "
                    f"Recall: {d['recall']:.3f}  "
                    f"F1: {d['f1']:.3f}  "
                    f"MCC: {d['mcc']:.3f}"
                )

        # Apply the best threshold
        best_t_val = sorted_result_dicts[0]["t_val"]
        apply_threshold(model.ndnf, og_conj_weight, og_disj_weight, best_t_val)
        intermediate_log = _eval_with_log_wrapper(
            f"Thresholded model (t={best_t_val})"
        )
        log.info("------------------------------------------")

        # Prune the model after thresholding
        multiround_prune(
            model,
            device,
            train_loader,
            lambda x: _eval_with_log_wrapper(x["model_name"]),
        )
        threshold_final_log = _eval_with_log_wrapper(
            f"Thresholded NDNF model (t={best_t_val}) after final prune (test)",
            test_loader,
        )

        return {
            "threshold_val": best_t_val,
            "intermediate_log": intermediate_log,
            "threshold_final_log": threshold_final_log,
        }

    # Check for checkpoints
    # If the model is already thresholded, then we load the thresholded model
    # Otherwise, we threshold the model and save
    model_path = model_dir / f"{THRESHOLD_MODEL_BASE_NAME}.pth"
    threshold_result_json = (
        model_dir / f"{THRESHOLD_RESULT_JSON_BASE_NAME}.json"
    )
    if model_path.exists() and threshold_result_json.exists():
        threshold_state = torch.load(
            model_path, map_location=device, weights_only=True
        )
        model.load_state_dict(threshold_state)
        threshold_eval_log = _eval_with_log_wrapper("Thresholded NDNF model")
    else:
        threshold_ret_dict = threshold(do_logging=True)
        torch.save(model.state_dict(), model_path)
        with open(threshold_result_json, "w") as f:
            json.dump(
                {"prune_log": prune_log, **threshold_ret_dict},
                f,
                indent=4,
            )
        threshold_eval_log = threshold_ret_dict["threshold_final_log"]
    log.info("============================================================")

    return threshold_eval_log


def multirun_threshold(cfg: DictConfig) -> None:
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
    hold_out_test_X, hold_out_test_y, _ = get_cdc_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = CDCDataset(hold_out_test_X, hold_out_test_y)
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
        model = construct_model(eval_cfg)
        assert isinstance(model, CDCNeuralDNF)
        model.to(device)
        model_state = torch.load(
            model_dir / f"{FIRST_PRUNE_MODEL_BASE_NAME}.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(model_state)
        model.eval()

        # Data loaders
        X, y, _ = get_cdc_data_np_from_path(cfg["dataset"], is_test=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=eval_cfg.get("val_size", 0.2),
            random_state=eval_cfg.get("val_seed", 73),
        )
        train_dataset = CDCDataset(X_train, y_train)
        val_dataset = CDCDataset(X_val, y_val)
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
        ret_dicts.append(
            single_model_threshold(
                model, device, train_loader, val_loader, test_loader, model_dir
            )
        )
        log.info("============================================================")

    # Synthesize the results
    relevant_keys = [
        k for k, v in ret_dicts[0].items() if isinstance(v, (int, float))
    ]
    return_dict = {}
    for k in relevant_keys:
        synth_dict = synthesize(np.array([d[k] for d in ret_dicts]))
        for sk, sv in synth_dict.items():
            return_dict[f"{k}/{sk}"] = sv

    log.info("Synthesized results:")
    for k, v in return_dict.items():
        log.info(f"{k}: {v:.3f}")

    with open("threshold_result.json", "w") as f:
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
        multirun_threshold(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Threshold",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
