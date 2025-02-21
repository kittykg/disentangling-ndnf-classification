"""
This script translates a plain NDNF with weights only in the set {±6, 0} to its
logically-equivalent ASP program. The input NDNF models are strictly after
pruning stage and a disentanglement v3. The ASP programs are stored and
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
from zoo.eval.asp_eval_common import (
    ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME,
    asp_eval,
)
from zoo.eval.ndnf_eval_common import (
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from zoo.eval.ndnf_eo_kfold_disentangle_v3 import ChainedNDNF


log = logging.getLogger()


def load_discretised_ndnf_model(
    fold_id: int, model_dir: Path, eval_cfg: DictConfig
) -> ChainedNDNF:
    discretisation_method = eval_cfg["discretisation_method"]
    assert discretisation_method == "disentangle", "Only disentanglement v3!"

    cd_version = eval_cfg.get("cd_version", None)
    assert cd_version == 3, "Only disentanglement v3!"
    cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
    disentangle_json_path = (
        model_dir
        / f"fold_{fold_id}_{DISENTANGLED_RESULT_JSON_BASE_NAME}{cd_version_str}.json"
    )
    assert disentangle_json_path.exists(), (
        "Disentanglement result JSON not found. Run the disentanglement "
        "script first."
    )
    disentangle_sd_path = (
        model_dir
        / f"{DISENTANGLED_MODEL_BASE_NAME}{cd_version_str}_fold_{fold_id}.pth"
    )
    assert disentangle_sd_path.exists(), (
        "Disentangled model not found. Run the disentanglement script " "first."
    )
    with open(disentangle_json_path, "r") as f:
        disentangle_result = json.load(f)
    sub_ndnf_conj = NeuralDNF(
        disentangle_result["sub_ndnf_conj_in"],
        disentangle_result["sub_ndnf_conj_n_conjunctions"],
        disentangle_result["sub_ndnf_conj_out"],
        1.0,
    )
    sub_ndnf_disj = NeuralDNF(
        disentangle_result["sub_ndnf_disj_in"],
        disentangle_result["sub_ndnf_disj_n_conjunctions"],
        disentangle_result["sub_ndnf_disj_out"],
        1.0,
    )
    chained_ndnf = ChainedNDNF(sub_ndnf_conj, sub_ndnf_disj)
    disentangle_sd = torch.load(
        disentangle_sd_path, map_location="cpu", weights_only=True
    )
    chained_ndnf.load_state_dict(disentangle_sd)
    chained_ndnf.eval()

    return chained_ndnf


def extract_asp_rules_from_chained_ndnf(
    model: ChainedNDNF,
) -> dict[str, dict[str, Any]]:
    sub_ndnf_conj_rules = extract_asp_rules(
        model.sub_ndnf.state_dict(),
        format_options={
            "input_name": "a",
            "input_syntax": "PRED",
            "conjunction_name": "aux_conj",
            "conjunction_syntax": "PRED",
            "disjunction_name": "conj",
            "disjunction_syntax": "PRED",
        },
        return_as_dict=True,
    )
    assert isinstance(sub_ndnf_conj_rules, dict)
    sub_ndnf_disj_rules = extract_asp_rules(
        model.sub_ndnf_disj.state_dict(),
        format_options={
            "input_name": "conj",
            "input_syntax": "PRED",
            "conjunction_name": "aux_disj",
            "conjunction_syntax": "PRED",
            "disjunction_name": "disj",
            "disjunction_syntax": "PRED",
        },
        return_as_dict=True,
    )
    assert isinstance(sub_ndnf_disj_rules, dict)

    return {
        "sub_ndnf_conj": sub_ndnf_conj_rules,
        "sub_ndnf_disj": sub_ndnf_disj_rules,
    }


def single_model_translate(
    cfg: DictConfig,
    test_data: np.ndarray,
    fold_id: int,
    run_dir_name: str,
) -> dict[str, Any]:
    num_classes = cfg["eval"]["model_architecture"]["num_classes"]
    discretisation_method = cfg["eval"]["discretisation_method"]
    assert discretisation_method == "disentangle", "Only disentanglement v3!"

    model_dir = (
        Path(cfg["eval"]["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
    )
    model = load_discretised_ndnf_model(fold_id, model_dir, cfg["eval"])
    log.info(f"Experiment {model_dir.name} loaded!")

    cd_version = cfg["eval"].get("cd_version", None)
    assert cd_version == 3, "Only disentanglement v3!"
    cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
    translation_json_path = (
        model_dir
        / f"fold_{fold_id}_{ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME}{cd_version_str}.json"
    )
    log.info(f"Discretisation method: disentanglement{cd_version_str}")
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
        rules_ret = extract_asp_rules_from_chained_ndnf(model)
        sub_ndnf_conj_rules = rules_ret["sub_ndnf_conj"]
        sub_ndnf_disj_rules = rules_ret["sub_ndnf_disj"]
        overall_rules = (
            sub_ndnf_conj_rules["rules"] + sub_ndnf_disj_rules["rules"]
        )
        asp_eval_log = parse_eval_return_meters_with_logging(
            eval_meters=asp_eval(test_data, overall_rules, num_classes),
            model_name="ASP Translation",
        )

        json_dict = {}
        json_dict["sub_ndnf_conj_rules"] = {
            k: list(v) if isinstance(v, set) else v
            for k, v in sub_ndnf_conj_rules.items()
        }
        json_dict["sub_ndnf_disj_rules"] = {
            k: list(v) if isinstance(v, set) else v
            for k, v in sub_ndnf_disj_rules.items()
        }
        json_dict["rules"] = overall_rules
        for k, v in asp_eval_log.items():
            json_dict[k] = list(v) if isinstance(v, set) else v

        with open(translation_json_path, "w") as f:
            json.dump(json_dict, f, indent=4)

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
    X, y, _ = get_zoo_data_np_from_path(
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
                experiment_name=f"{cfg['eval']['experiment_name']} Kfold ASP Translation + Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
