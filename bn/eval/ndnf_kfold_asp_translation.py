"""
This script translates the NDNF in BooleanNetwork NeuralDNF model with weights
only in the set {±6, 0} to its logically-equivalent ASP program. It also
translates the invented predicates into a readable format. The input model is
strictly after pruning stage and a discretisation step (either
thresholding/disentanglement) in the post-training processing pipeline. The ASP
program and the predicate translation are stored and evaluated, with the
relevant information stored in a json. The evaluation metrics include accuracy,
precision, recall and F1 score.

The program is evaluated on the full dataset.
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

from analysis import Meter, AccuracyMeter, JaccardScoreMeter, synthesize
from utils import post_to_discord_webhook

from bn.data_utils_bn import (
    get_boolean_network_full_data_np_from_path,
    BooleanNetworkDataset,
)
from bn.eval.eval_common import (
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    THRESHOLD_MODEL_BASE_NAME,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from bn.models import BooleanNetworkNeuralDNF, construct_model


log = logging.getLogger()

ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME = "asp_translation_thresholded"
ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME = "asp_translation_disentangled"


def load_discretised_model(
    fold_id: int,
    num_genes: int,
    model_dir: Path,
    eval_cfg: DictConfig,
) -> BooleanNetworkNeuralDNF:
    discretisation_method = eval_cfg["discretisation_method"]
    assert discretisation_method in [
        "threshold",
        "disentangle",
    ], "Invalid discretisation method"

    model = construct_model(eval_cfg, num_genes)
    assert isinstance(model, BooleanNetworkNeuralDNF)

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
        model.load_state_dict(threshold_sd)

    else:
        # Conjunction disentanglement
        cd_version = eval_cfg.get("cd_version", None)
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
    num_genes: int,
    model: BooleanNetworkNeuralDNF,
    format_options: dict[str, str] = {},
    debug: bool = False,
) -> dict[str, Meter]:
    # Assume each row of the test_data is first the attributes present and then
    # the label
    ground_truth = test_data[:, 1, :].tolist()

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
        for i, a in enumerate(d[0]):
            if a == 1:
                asp_base.append(
                    f"{input_name}_{i}."
                    if input_syntax == "PRED"
                    else f"{input_name}({i})"
                )

        asp_base += rules
        if disjunction_syntax == "PRED":
            asp_base += [
                f"#show {disjunction_name}_{i}/0." for i in range(num_genes)
            ]
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
        predictions.append(output_classes)

    # Convert to tensors
    prediction_tensor = torch.zeros(
        len(predictions), num_genes, dtype=torch.float32
    )
    for i, p in enumerate(predictions):
        prediction_tensor[i, p] = 1.0
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.int)

    acc_meter = AccuracyMeter(output_to_prediction_fn=lambda x: x)
    jacc_meter = JaccardScoreMeter()
    acc_meter.update(prediction_tensor.int(), ground_truth_tensor)
    jacc_meter.update(prediction_tensor.int(), ground_truth_tensor)

    return {"acc_meter": acc_meter, "jacc_meter": jacc_meter}


def single_model_translate(
    cfg: DictConfig,
    test_data: np.ndarray,
    fold_id: int,
    num_genes: int,
    run_dir_name: str,
) -> dict[str, Any]:
    discretisation_method = cfg["eval"]["discretisation_method"]

    model_dir = (
        Path(cfg["eval"]["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
    )
    model = load_discretised_model(fold_id, num_genes, model_dir, cfg["eval"])
    log.info(f"Experiment {model_dir.name} loaded!")

    # Check for checkpoints
    # If checkpoint exists, load it, else extract the rules from the model
    if discretisation_method == "threshold":
        translation_json_path = (
            model_dir
            / f"fold_{fold_id}_{ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME}.json"
        )
        log.info("Discretisation method: thresholding")
    else:
        # Conjunction disentanglement
        cd_version = cfg["eval"].get("cd_version", None)
        cd_version_str = f"_v{cd_version}" if cd_version is not None else ""
        translation_json_path = (
            model_dir
            / f"fold_{fold_id}_{ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME}{cd_version_str}.json"
        )
        log.info(f"Discretisation method: disentanglement{cd_version_str}")

    asp_eval_log = parse_eval_return_meters_with_logging(
        eval_meters=asp_eval(test_data, num_genes, model),
        model_name="ASP Translation",
        log_confusion_matrix=True,
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


def post_train_asp_translate(cfg: DictConfig):
    eval_cfg = cfg["eval"]
    full_experiment_name = f"{eval_cfg['experiment_name']}_{eval_cfg['seed']}"
    log.info(f"Full experiment name: {full_experiment_name}")
    run_dir_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    # Load data
    bn_dataset = BooleanNetworkDataset(
        dataset_type=cfg["dataset"]["dataset_name"],
        subtype=None,
        data=get_boolean_network_full_data_np_from_path(cfg["dataset"]),
    )
    num_genes = bn_dataset.data.shape[2]

    ret_dicts: list[dict[str, float]] = []
    for fold_id in range(eval_cfg["k_folds"]):
        test_data = np.array(bn_dataset)
        ret_dicts.append(
            single_model_translate(
                cfg, test_data, fold_id, num_genes, run_dir_name
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
    import warnings

    warnings.filterwarnings("ignore")

    run_eval()
