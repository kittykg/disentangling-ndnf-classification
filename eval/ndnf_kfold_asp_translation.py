"""
This script translates a plain NDNF with weights only in the set {±6, 0} to its
logically-equivalent ASP program. The input NDNF models are strictly after
pruning stage and a discretisation step (either thresholding/disentanglement) in
the post-training processing pipeline. The ASP programs are stored and
evaluated, with the relevant information stored in a json. The evaluation
metrics include accuracy, sample Jaccard, macro Jaccard, and error metrics.
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
from sklearn.model_selection import StratifiedKFold
import torch

from neural_dnf import NeuralDNF
from neural_dnf.post_training import extract_asp_rules

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import synthesize
from data_utils_zoo import *
from eval.asp_eval_common import (
    asp_eval,
    ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME,
    ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME,
)
from eval.ndnf_eval_common import (
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    THRESHOLD_MODEL_BASE_NAME,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from utils import post_to_discord_webhook


log = logging.getLogger()


def load_discretised_ndnf_model(
    fold_id: int, model_dir: Path, eval_cfg: DictConfig
) -> NeuralDNF:
    discretisation_method = eval_cfg["discretisation_method"]
    assert discretisation_method in [
        "threshold",
        "disentangle",
    ], "Invalid discretisation method"

    if discretisation_method == "threshold":
        threshold_sd_path = (
            model_dir / f"{THRESHOLD_MODEL_BASE_NAME}_fold_{fold_id}.pth"
        )
        assert (
            threshold_sd_path.exists()
        ), "Thresholded model not found! Run thresholding first."

        threshold_sd = torch.load(
            threshold_sd_path, map_location="cpu", weights_only=True
        )
        model = NeuralDNF(
            eval_cfg["model_architecture"]["n_in"],
            eval_cfg["model_architecture"]["n_conjunctions"],
            eval_cfg["model_architecture"]["num_classes"],
            delta=1.0,
        )
        model.load_state_dict(threshold_sd)

    else:
        disentangle_json_path = (
            model_dir
            / f"fold_{fold_id}_{DISENTANGLED_RESULT_JSON_BASE_NAME}.json"
        )
        assert disentangle_json_path.exists(), (
            "Disentanglement result JSON not found. Run the disentanglement "
            "script first."
        )
        disentangle_sd_path = (
            model_dir / f"{DISENTANGLED_MODEL_BASE_NAME}_fold_{fold_id}.pth"
        )
        assert disentangle_sd_path.exists(), (
            "Disentangled model not found. Run the disentanglement script "
            "first."
        )
        with open(disentangle_json_path, "r") as f:
            disentangle_result = json.load(f)
        model = NeuralDNF(
            disentangle_result["disentangled_model_n_in"],
            disentangle_result["disentangled_model_n_conjunctions"],
            disentangle_result["disentangled_model_n_out"],
            delta=1.0,
        )
        disentangle_sd = torch.load(
            disentangle_sd_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(disentangle_sd)

    model.eval()
    return model


def single_model_translate(
    cfg: DictConfig,
    test_data: np.ndarray,
    fold_id: int,
    run_dir_name: str,
) -> dict[str, Any]:
    num_classes = cfg["eval"]["model_architecture"]["num_classes"]
    discretisation_method = cfg["eval"]["discretisation_method"]

    model_dir = (
        Path(cfg["eval"]["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
    )
    model = load_discretised_ndnf_model(fold_id, model_dir, cfg["eval"])
    log.info(f"Experiment {model_dir.name} loaded!")

    # Check for checkpoints
    # If checkpoint exists, load it, else extract the rules from the model
    translation_json_path = (
        model_dir
        / f"fold_{fold_id}_{ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME if discretisation_method == 'threshold' else ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME}.json"
    )
    if translation_json_path.exists():
        # The model has been disentangled, pruned and condensed
        with open(translation_json_path, "r") as f:
            asp_translate_json = json.load(f)
        asp_eval_log = parse_eval_return_meters_with_logging(
            eval_meters=asp_eval(
                test_data, asp_translate_json["rules"], num_classes
            ),
            model_name="ASP Translation",
        )
    else:
        rules_dict = extract_asp_rules(model.state_dict(), return_as_dict=True)
        assert isinstance(rules_dict, dict)
        asp_eval_log = parse_eval_return_meters_with_logging(
            eval_meters=asp_eval(test_data, rules_dict["rules"], num_classes),
            model_name="ASP Translation",
        )
        with open(translation_json_path, "w") as f:
            json.dump(
                {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in {**rules_dict, **asp_eval_log}.items()
                },
                f,
                indent=4,
            )

    return asp_eval_log


def post_train_asp_translate(cfg: DictConfig):
    eval_cfg = cfg["eval"]
    full_experiment_name = f"{eval_cfg['experiment_name']}_{eval_cfg['seed']}"
    run_dir_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    # Load data
    X, y, feature_names = get_zoo_data_np_from_path(
        data_dir_path=Path(cfg["dataset"]["save_dir"])
    )

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=eval_cfg["k_folds"],
        shuffle=True,
        random_state=eval_cfg["seed"],
    )

    ret_dicts: list[dict[str, float]] = []

    for fold_id, (_, test_index) in enumerate(skf.split(X, y)):
        test_data_with_label = np.column_stack((X[test_index], y[test_index]))
        ret_dicts.append(
            single_model_translate(
                cfg, test_data_with_label, fold_id, run_dir_name
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_asp_translate(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Disentangle",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
