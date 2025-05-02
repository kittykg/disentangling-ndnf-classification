"""
This script translates Monk NeuralDNF model with weights only in the set
{±6, 0} to its logically-equivalent ASP program. The input model is strictly
after pruning stage and a discretisation step (either
thresholding/disentanglement) in the post-training processing pipeline. The ASP
program is stored and evaluated, with the relevant information stored in a json.
The evaluation metrics include accuracy, precision, recall, F1 score and MCC.
"""

import json
import logging
from pathlib import Path
import random
import re
import sys
import traceback
from typing import Any

import clingo
import hydra
import numpy as np
from omegaconf import DictConfig
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

from analysis import Meter, AccuracyMeter, synthesize
from utils import post_to_discord_webhook

from monk.data_utils_monk import get_monk_data_np_from_path
from monk.eval.eval_common import (
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    THRESHOLD_MODEL_BASE_NAME,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from monk.models import MonkNeuralDNF

log = logging.getLogger()

ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME = "asp_translation_thresholded"
ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME = "asp_translation_disentangled"


def load_discretised_model(
    input_features: int,
    model_dir: Path,
    eval_cfg: DictConfig,
) -> MonkNeuralDNF:
    discretisation_method = eval_cfg["discretisation_method"]
    assert discretisation_method in [
        "threshold",
        "disentangle",
    ], "Invalid discretisation method"

    model = MonkNeuralDNF(
        num_features=input_features,
        num_conjunctions=eval_cfg["model_architecture"]["n_conjunctions"],
    )

    if discretisation_method == "threshold":
        threshold_sd_path = model_dir / f"{THRESHOLD_MODEL_BASE_NAME}.pth"
        assert (
            threshold_sd_path.exists()
        ), "Thresholded model not found! Run thresholding first."

        threshold_sd = torch.load(
            threshold_sd_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(threshold_sd)

    else:
        # Conjunction disentanglement
        cd_version = eval_cfg.get("cd_version", None)
        cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
        disentangle_json_path = (
            model_dir
            / f"{DISENTANGLED_RESULT_JSON_BASE_NAME}{cd_version_str}.json"
        )
        assert disentangle_json_path.exists(), (
            "Disentanglement result JSON not found. Run the disentanglement "
            "script first."
        )
        disentangle_sd_path = (
            model_dir / f"{DISENTANGLED_MODEL_BASE_NAME}{cd_version_str}.pth"
        )
        assert disentangle_sd_path.exists(), (
            "Disentangled model not found. Run the disentanglement script "
            "first."
        )
        with open(disentangle_json_path, "r") as f:
            disentangle_result = json.load(f)
        new_ndnf = NeuralDNF(
            disentangle_result["disentangled_ndnf_n_in"],
            disentangle_result["disentangled_ndnf_n_conjunctions"],
            disentangle_result["disentangled_ndnf_n_out"],
            delta=1.0,
        )
        model.change_ndnf(new_ndnf)
        disentangle_sd = torch.load(
            disentangle_sd_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(disentangle_sd)

    model.eval()
    return model


def asp_eval(
    test_data: np.ndarray,
    model: MonkNeuralDNF,
    format_options: dict[str, str] = {},
    debug: bool = False,
) -> dict[str, Meter]:
    # Assume each row of the test_data is first the attributes present and then
    # the label
    ground_truth = test_data[:, -1].tolist()

    input_name = format_options.get("input_name", "a")
    input_syntax = format_options.get("input_syntax", "PRED")
    disjunction_name = format_options.get("disjunction_name", "disj")
    disjunction_syntax = format_options.get("disjunction_syntax", "PRED")

    rules = extract_asp_rules(
        model.ndnf.state_dict(), format_options=format_options
    )
    predictions = []
    for d in test_data:
        asp_base = []
        for i, a in enumerate(d[:-1]):
            if a == 1:
                asp_base.append(
                    f"{input_name}_{i}."
                    if input_syntax == "PRED"
                    else f"{input_name}({i})"
                )

        asp_base += rules
        if disjunction_syntax == "PRED":
            # The rules have only one disjunctive head "disj_0" (by default),
            # which represents the target being true. We create a new
            # disjunctive head "disj_1" which represents the target being false,
            # and it is true if not disj_0.
            asp_base.append(
                f"{disjunction_name}_1 :- not {disjunction_name}_0."
            )
            asp_base += [f"#show {disjunction_name}_{i}/0." for i in range(2)]
        else:
            asp_base.append(f"#show {disjunction_name}/1.")

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], " ".join(asp_base))
        ctl.ground([("base", [])])

        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

        if debug:
            log.info(f"AS: {all_answer_sets}")

        if len(all_answer_sets) != 1:
            # No model or multiple answer sets, should not happen
            log.warning(
                f"No model or multiple answer sets when evaluating rules."
            )
            continue

        # Use regex to extract the output classes number
        pattern_string = (
            f"{disjunction_name}_(\d+)"
            if disjunction_syntax == "PRED"
            else f"{disjunction_name}\((\d+)\)"
        )
        pattern = re.compile(pattern_string)
        output_classes = [int(i) for i in pattern.findall(all_answer_sets[0])]

        # Should only have one output class
        assert len(output_classes) == 1, "Invalid output classes"

        # For binary classification, if disj_0 is present it means the target is
        # true, and disj_1 is present it means the target is false. The ground
        # truth value is 0 if the target is false and 1 if the target is true,
        # which is the opposite of the ASP output. So we need to reverse the
        # ASP prediction to match with the ground truth
        predictions.append(1 - output_classes[0])

    # Convert to tensors
    prediction_tensor = torch.zeros(len(predictions), dtype=torch.float32)
    for i, p in enumerate(predictions):
        prediction_tensor[i] = p
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.int)

    acc_meter = AccuracyMeter(output_to_prediction_fn=lambda x: x)
    acc_meter.update(prediction_tensor.int(), ground_truth_tensor)

    return {"acc_meter": acc_meter}


def single_model_translate(
    cfg: DictConfig,
    model_dir: Path,
    test_data: np.ndarray,
) -> dict[str, Any]:
    discretisation_method = cfg["eval"]["discretisation_method"]

    model = load_discretised_model(
        test_data.shape[1] - 1, model_dir, cfg["eval"]
    )
    log.info(f"Experiment {model_dir.name} loaded!")

    # Check for checkpoints
    # If checkpoint exists, load it, else extract the rules from the model
    if discretisation_method == "threshold":
        translation_json_path = (
            model_dir / f"{ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME}.json"
        )
        log.info("Discretisation method: thresholding")
    else:
        # Conjunction disentanglement
        cd_version = cfg["eval"].get("cd_version", None)
        cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
        translation_json_path = (
            model_dir
            / f"{ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME}{cd_version_str}.json"
        )
        log.info(f"Discretisation method: disentanglement{cd_version_str}")

    asp_eval_log = parse_eval_return_meters_with_logging(
        eval_meters=asp_eval(test_data, model),
        model_name="ASP Translation",
    )

    with open(translation_json_path, "w") as f:
        json.dump(
            {
                k: list(v) if isinstance(v, set) else v
                for k, v in {
                    **(
                        extract_asp_rules(
                            model.ndnf.state_dict(), return_as_dict=True
                        )
                    ),  # type: ignore
                    **asp_eval_log,
                }.items()
            },
            f,
            indent=4,
        )

    return asp_eval_log


def multirun_asp_translate(cfg: DictConfig):
    eval_cfg = cfg["eval"]

    # Load test data
    hold_out_test_X, hold_out_test_y, _ = get_monk_data_np_from_path(
        cfg["dataset"], is_test=True
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
        test_data_with_label = np.column_stack(
            (hold_out_test_X, hold_out_test_y)
        )
        model_dir = (
            Path(eval_cfg["storage_dir"])
            / caps_experiment_name
            / f"{caps_experiment_name}-{s}"
        )
        ret_dicts.append(
            single_model_translate(cfg, model_dir, test_data_with_label)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun ASP Translation + Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
