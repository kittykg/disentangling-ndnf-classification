"""
This script measures the interpretability in terms of body length of an ASP
extracted from its plain NDNF. The ASP programs should be translated from the
NDNF already and stored (via running ndnf_kfold_asp_translation).
"""

import json
import logging
from pathlib import Path
import sys
import traceback

import hydra
import numpy as np
from omegaconf import DictConfig

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
from bcc.eval.ndnf_kfold_asp_translation import (
    ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME,
    ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME,
)


log = logging.getLogger()


def load_asp_translation(
    fold_id: int, model_dir: Path, eval_cfg: DictConfig
) -> list[str]:
    discretisation_method = eval_cfg["discretisation_method"]
    assert discretisation_method in [
        "threshold",
        "disentangle",
    ], "Invalid discretisation method"

    if discretisation_method == "threshold":
        translation_json_path = (
            model_dir
            / f"fold_{fold_id}_{ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME}.json"
        )
    else:
        # Conjunction disentanglement
        cd_version = eval_cfg.get("cd_version", None)
        cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
        translation_json_path = (
            model_dir
            / f"fold_{fold_id}_{ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME}{cd_version_str}.json"
        )
    assert (
        translation_json_path.exists()
    ), "Translation result json doesn't exist. Run the translation script first."

    with open(translation_json_path, "r") as f:
        return json.load(f)["rules"]


def single_asp_program_stats(
    cfg: DictConfig, fold_id: int, run_dir_name: str
) -> dict[str, float]:
    model_dir = (
        Path(cfg["eval"]["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
    )
    rules = load_asp_translation(fold_id, model_dir, cfg["eval"])
    log.info(f"ASP rules from {model_dir.name} loaded!")

    num_effective_rules = 0
    rule_body_length = []

    for r in rules:
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


def post_train_asp_stats(cfg: DictConfig):
    eval_cfg = cfg["eval"]
    full_experiment_name = f"{eval_cfg['experiment_name']}_{eval_cfg['seed']}"
    run_dir_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    ret_dicts: list[dict[str, float]] = []
    for i in range(eval_cfg["k_folds"]):
        ret_dicts.append(single_asp_program_stats(cfg, i, run_dir_name))
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
    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_asp_stats(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Kfold ASP Stats",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
