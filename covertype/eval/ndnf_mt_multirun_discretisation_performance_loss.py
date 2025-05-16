"""
This script compares the performance of Covertype NDNF-MT model after train vs
after post-training process (threshold/disentangle). The input models are
strictly after the full post-training processing pipeline. The evaluation
metrics include accuracy and f1 score.
"""

import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

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

log = logging.getLogger()

AFTER_TRAIN_JSON = "after_train_eval_result.json"
AFTER_THRESHOLD_JSON = "soft_extract_threshold_result.json"
AFTER_DISENTANGLE_JSON = "soft_extract_disentangled_result.json"


def single_model_comparison(
    model_dir: Path,
) -> dict[str, Any]:
    train_result_json = model_dir / AFTER_TRAIN_JSON
    threshold_result_json = model_dir / AFTER_THRESHOLD_JSON
    disentangle_result_json = model_dir / AFTER_DISENTANGLE_JSON

    assert all(
        p.exists()
        for p in [
            train_result_json,
            threshold_result_json,
            disentangle_result_json,
        ]
    ), (
        f"For {model_dir.name}, all result files must exist, missing: "
        + ", ".join(
            [
                p.name
                for p in [
                    train_result_json,
                    threshold_result_json,
                    disentangle_result_json,
                ]
                if not p.exists()
            ]
        )
    )
    with open(train_result_json, "r") as f:
        train_result = json.load(f)
    with open(threshold_result_json, "r") as f:
        threshold_result = json.load(f)
    with open(disentangle_result_json, "r") as f:
        disentangle_result = json.load(f)

    performance_loss_dict = {}
    for k in ["accuracy", "f1"]:
        performance_loss_dict[f"train_threshold_{k}"] = (
            train_result[k] - threshold_result["threshold_final_log"][k]
        )
        performance_loss_dict[f"train_disentangle_{k}"] = (
            train_result[k] - disentangle_result["pruned_chained_ndnf_log"][k]
        )

    return performance_loss_dict


def multirun_perf_loss_comp(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

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
        ret_dicts.append(single_model_comparison(model_dir))

    # Synthesize the results
    return_dict = {}
    for k in ret_dicts[0].keys():
        synth_dict = synthesize(np.array([d[k] for d in ret_dicts]))
        for sk, sv in synth_dict.items():
            return_dict[f"{k}/{sk}"] = sv

    log.info("Synthesized results:")
    for k, v in return_dict.items():
        log.info(f"{k}: {v:.3f}")

    with open("performance_loss_comparison_result.json", "w") as f:
        json.dump(return_dict, f, indent=4)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        multirun_perf_loss_comp(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Performance Loss Comparison",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
