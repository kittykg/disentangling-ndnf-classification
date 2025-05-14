"""
This script translates Car NeuralDNF-MT's conjunctive layer with weights only in
the set {±6, 0} to its logically-equivalent ASP program. The input model is
strictly after pruning stage and soft-extraction (either
thresholding/disentanglement) in the post-training processing pipeline. This
script only translates and does not evaluate.
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
import torch

from neural_dnf import NeuralDNF, SemiSymbolicMutexTanh, SemiSymbolicLayerType
from neural_dnf.post_training import extract_asp_rules

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

from car.data_utils_car import get_car_data_np_from_path
from car.eval.eval_common import DEFAULT_GEN_SEED
from car.eval.ndnf_mt_multirun_soft_extraction import (
    SOFT_EXTRCT_THRESHOLD_MODEL_BASE_NAME,
    SOFT_EXTRCT_THRESHOLD_RESULT_JSON_BASE_NAME,
    SOFT_EXTRCT_DISENTANGLED_MODEL_BASE_NAME,
    SOFT_EXTRCT_DISENTANGLED_RESULT_JSON_BASE_NAME,
    BaseChainedNeuralDNF,
    MutexTanhChainedNeuralDNF,
)
from car.models import CarNeuralDNFMT

log = logging.getLogger()

MT_CONJ_ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME = (
    "mt_conj_translation_thresholded"
)
MT_CONJ_ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME = (
    "mt_conj_translation_disentangled"
)


def load_discretised_model(
    input_features: int,
    model_dir: Path,
    eval_cfg: DictConfig,
) -> CarNeuralDNFMT | BaseChainedNeuralDNF:
    discretisation_method = eval_cfg["discretisation_method"]
    assert discretisation_method in [
        "threshold",
        "disentangle",
    ], "Invalid discretisation method"

    if discretisation_method == "threshold":
        model_path = model_dir / f"{SOFT_EXTRCT_THRESHOLD_MODEL_BASE_NAME}.pth"
        threshold_result_json = (
            model_dir / f"{SOFT_EXTRCT_THRESHOLD_RESULT_JSON_BASE_NAME}.json"
        )
        assert (
            model_path.exists() and threshold_result_json.exists()
        ), "Threshold-based soft-extracted model not found! Run soft-extraction first."

        model = CarNeuralDNFMT(
            num_features=input_features,
            num_conjunctions=eval_cfg["model_architecture"]["n_conjunctions"],
        )
        threshold_state = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(threshold_state)
        model.eval()
        return model

    # Conjunction disentanglement
    model_path = model_dir / f"{SOFT_EXTRCT_DISENTANGLED_MODEL_BASE_NAME}.pth"
    disentangled_result_json = (
        model_dir / f"{SOFT_EXTRCT_DISENTANGLED_RESULT_JSON_BASE_NAME}.json"
    )
    assert (
        model_path.exists() and disentangled_result_json.exists()
    ), "Disentanglement-based soft-extracted model not found! Run soft-extraction first."

    with open(disentangled_result_json, "r") as f:
        stats = json.load(f)

    chained_model = MutexTanhChainedNeuralDNF(
        sub_ndnf=NeuralDNF(
            stats["sub_ndnf_in"],
            stats["sub_ndnf_n_conunctions"],
            stats["sub_ndnf_out"],
            1.0,
        ),
        disjunctive_layer=SemiSymbolicMutexTanh(
            stats["disjunctive_layer_in"],
            stats["disjunctive_layer_out"],
            SemiSymbolicLayerType.DISJUNCTION,
            1.0,
        ),
    )
    chained_model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    chained_model.eval()

    return chained_model


def single_model_translate(
    cfg: DictConfig,
    input_features: int,
    model_dir: Path,
    format_options: dict[str, str] = {},
) -> dict[str, Any]:
    discretisation_method = cfg["eval"]["discretisation_method"]

    model = load_discretised_model(input_features, model_dir, cfg["eval"])
    log.info(f"Experiment {model_dir.name} loaded!")

    # Check for checkpoints
    # If checkpoint exists, load it, else extract the rules from the model
    if discretisation_method == "threshold":
        translation_json_path = (
            model_dir
            / f"{MT_CONJ_ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME}.json"
        )
        log.info("Discretisation method: thresholding")
    else:
        # Conjunction disentanglement
        cd_version = cfg["eval"].get("cd_version", None)
        cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
        translation_json_path = (
            model_dir
            / f"{MT_CONJ_ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME}{cd_version_str}.json"
        )
        log.info(f"Discretisation method: disentanglement{cd_version_str}")

    rules_as_dict = extract_asp_rules(
        (
            model.ndnf.state_dict()
            if isinstance(model, CarNeuralDNFMT)
            else model.sub_ndnf.state_dict()
        ),
        format_options=format_options,
        return_as_dict=True,
    )

    assert isinstance(rules_as_dict, dict)
    with open(translation_json_path, "w") as f:
        json.dump(
            {k: list(v) if isinstance(v, set) else v for k, v in rules_as_dict.items()},
            f,
            indent=4,
        )

    num_effective_rules = 0
    rule_body_length = []

    for r in rules_as_dict["rules"]:
        if r.startswith("disj_"):
            num_effective_rules += 1
        else:
            rule_body_length.append(len(r.split(",")))

    avg_rule_body_length = sum(rule_body_length) / len(rule_body_length)
    max_rule_body_length = max(rule_body_length)

    log.info(f"Number of effective rules: {num_effective_rules}")
    log.info(f"Average rule body length: {avg_rule_body_length}")
    log.info(f"Max rule body length: {max_rule_body_length}")

    return {
        "num_effective_rules": num_effective_rules,
        "avg_rule_body_length": avg_rule_body_length,
        "max_rule_body_length": max_rule_body_length,
    }


def multirun_asp_translate(cfg: DictConfig):
    eval_cfg = cfg["eval"]

    # Load test data
    hold_out_test_X, _ = get_car_data_np_from_path(cfg["dataset"], is_test=True)
    input_features = hold_out_test_X.shape[1]

    # Experiment name
    experiment_name = eval_cfg["experiment_name"]
    caps_experiment_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(experiment_name.split("_"))
        ]
    )

    ret_dicts: list[dict[str, Any]] = []

    for s in eval_cfg["multirun_seeds"]:

        model_dir = (
            Path(eval_cfg["storage_dir"])
            / caps_experiment_name
            / f"{caps_experiment_name}-{s}"
        )
        ret_dicts.append(single_model_translate(cfg, input_features, model_dir))
        log.info("============================================================")

    # Synthesize the results
    return_dict = {}
    for k in [
        "num_effective_rules",
        "avg_rule_body_length",
        "max_rule_body_length",
    ]:
        synth_dict = synthesize(np.array([d[k] for d in ret_dicts]))
        for sk, sv in synth_dict.items():
            return_dict[f"{k}/{sk}"] = sv

    log.info("Synthesized results:")
    for k, v in return_dict.items():
        log.info(f"{k}: {v:.3f}")

    with open(
        "mt_soft_extraction_asp_translation_stats_results.json", "w"
    ) as f:
        json.dump(return_dict, f, indent=4)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
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
        multirun_asp_translate(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Soft-Extraction ASP Translation + Stats",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
