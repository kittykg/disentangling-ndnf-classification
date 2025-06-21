"""
This script disentangles pruned plain NDNF from the CoverType MLP PI NDNF-EO
model. The input models are strictly after pruning stage in the post-training
processing pipeline. The disentangled NDNF models are stored and evaluated. The
evaluation metrics include accuracy, sample Jaccard, macro Jaccard and error
metrics.
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

from neural_dnf.neural_dnf import NeuralDNF
from neural_dnf.post_training import (
    split_entangled_conjunction,
    condense_neural_dnf_model,
)

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import synthesize, AccuracyMeter, JaccardScoreMeter, ErrorMeter
from utils import post_to_discord_webhook

from covertype.data_utils_covertype import (
    get_covertype_data_np_from_path,
    CoverTypeDataset,
)
from covertype.eval.eval_common import (
    EVAL_RELEVANT_KEYS,
    EVAL_ERROR_DICT_RELEVANT_KEYS,
    covertype_classifier_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_NUM_WORKERS,
    FIRST_PRUNE_MODEL_BASE_NAME,
    INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from covertype.eval.ndnf_multirun_prune import multiround_prune, comparison_fn
from covertype.models import (
    CoverTypeMLPPINeuralDNFEO,
    CoverTypeMLPPINeuralDNF,
    construct_model,
)

log = logging.getLogger()


DEFAULT_LOADER_BATCH_SIZE = 2**11
DEFAULT_COMPUTE_ERROR_DICT = False


def create_intermediate_ndnf(model: NeuralDNF) -> NeuralDNF:
    # Remember the disjunction-conjunction mapping
    conj_w = model.conjunctions.weights.data.clone().detach().cpu()
    disj_w = model.disjunctions.weights.data.clone().detach().cpu()

    og_disj_conj_mapping = dict()
    for disj_id, w in enumerate(disj_w):
        non_zeros = torch.where(w != 0)[0]
        og_disj_conj_mapping[disj_id] = non_zeros.tolist()

    new_disj_conj_mapping = dict()
    new_conj_list = []

    for disj_id, conjs in og_disj_conj_mapping.items():
        acc_pairs = []

        for conj_id in conjs:
            og_sign = int(torch.sign(disj_w[disj_id][conj_id]).item())
            ret = split_entangled_conjunction(conj_w[conj_id], sign=og_sign)

            if ret is None:
                continue

            for c in ret:
                non_zero_count = torch.sum(c != 0).item()
                # Since the new conjunction is returned ready to be used in
                # positive form, always add the sign as 1. We add the tuple
                # of (non_zero_count, sign, og_weight, conjunction)
                acc_pairs.append(
                    (non_zero_count, 1, disj_w[disj_id][conj_id], c)
                )

        # Arrange the combinations from most general (more 0-weights) to most
        # specific (less 0-weights)
        acc_pairs.sort(key=lambda x: x[0])
        new_disj_conj_mapping[disj_id] = [
            {
                "sign": x[1],
                "conj_id": i + len(new_conj_list),
                "og_weight": x[2],
            }
            for i, x in enumerate(acc_pairs)
        ]
        new_conj_list.extend([x[3] for x in acc_pairs])

    new_conj_w = torch.stack(new_conj_list)

    # We need to update the disjunction weights
    new_disj_w = torch.zeros((disj_w.shape[0], len(new_conj_list)))
    for disj_id, vs in new_disj_conj_mapping.items():
        for v in vs:
            sign = v["sign"]
            conj_id = v["conj_id"]
            og_weight = v["og_weight"]
            new_disj_w[disj_id][conj_id] = sign * torch.abs(og_weight)

    # Create an intermediate model
    intermediate_model = NeuralDNF(
        conj_w.shape[1], len(new_conj_list), disj_w.shape[0], 1.0
    )
    intermediate_model.conjunctions.weights.data = new_conj_w.clone()
    intermediate_model.disjunctions.weights.data = new_disj_w.clone()

    return intermediate_model


def threshold_intermediate_model_disjunctive_layer(
    intermediate_model: CoverTypeMLPPINeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    do_logging: bool = False,
) -> float:
    intermediate_disj_weight = (
        intermediate_model.ndnf.disjunctions.weights.data.clone()
    )
    threshold_upper_bound = round(
        (
            torch.Tensor(
                [
                    intermediate_disj_weight.min(),
                    intermediate_disj_weight.max(),
                ]
            )
            .abs()
            .max()
            + 0.01
        ).item(),
        2,
    )
    log.info(f"Threshold upper bound: {threshold_upper_bound}")

    t_vals = torch.arange(0, threshold_upper_bound, 0.01)
    result_dicts_with_t_val = []
    for v in t_vals:
        intermediate_model.ndnf.disjunctions.weights.data = (
            (torch.abs(intermediate_disj_weight) > v)
            * torch.sign(intermediate_disj_weight)
            * 6
        )
        threshold_eval_dict = covertype_classifier_eval(
            intermediate_model, device, train_loader
        )
        acc_meter = threshold_eval_dict["acc_meter"]
        jacc_meter = threshold_eval_dict["jacc_meter"]
        error_meter = threshold_eval_dict["error_meter"]
        assert isinstance(acc_meter, AccuracyMeter)
        assert isinstance(jacc_meter, JaccardScoreMeter)
        assert isinstance(error_meter, ErrorMeter)
        accuracy = acc_meter.get_average()
        f1 = acc_meter.get_other_classification_metrics("weighted")["f1"]
        assert isinstance(f1, float)
        sample_jacc = jacc_meter.get_average("samples")
        macro_jacc = jacc_meter.get_average("macro")
        assert isinstance(sample_jacc, float)
        assert isinstance(macro_jacc, float)
        if DEFAULT_COMPUTE_ERROR_DICT:
            error_meter = threshold_eval_dict["error_meter"]
            assert isinstance(error_meter, ErrorMeter)
            overall_error_rate = error_meter.get_average()["overall_error_rate"]

        result_dicts_with_t_val.append(
            {
                "t_val": v.item(),
                "accuracy": accuracy,
                "f1": f1,
                "sample_jacc": sample_jacc,
                "macro_jacc": macro_jacc,
            }
        )
        if DEFAULT_COMPUTE_ERROR_DICT:
            assert isinstance(overall_error_rate, float)
            result_dicts_with_t_val[-1][
                "overall_error_rate"
            ] = overall_error_rate

    if DEFAULT_COMPUTE_ERROR_DICT:
        sort_fn = lambda x: (
            x["f1"],
            -x["overall_error_rate"],
            x["sample_jacc"],
            x["accuracy"],
        )
    else:
        sort_fn = lambda x: (x["f1"], x["accuracy"], x["sample_jacc"])
    sorted_result_dicts: list[dict[str, float]] = sorted(
        result_dicts_with_t_val,
        key=sort_fn,
        reverse=True,
    )

    if do_logging:
        log.info("Top 5 thresholding candidates:")
        for i, d in enumerate(sorted_result_dicts[:5]):
            log_info_str = (
                f"-- Candidate {i + 1} --\n"
                f"\tt_val: {d['t_val']:.2f}  "
                f"Acc: {d['accuracy']:.3f}  "
                f"F1: {d['f1']:.3f}  "
                f"Sample Jacc: {d['sample_jacc']:.3f}  "
                f"Macro Jacc: {d['macro_jacc']:.3f}  "
            )
            if DEFAULT_COMPUTE_ERROR_DICT:
                log_info_str += (
                    f"  Overall Error Rate: {d['overall_error_rate']:.3f}"
                )
            log.info(log_info_str)

    # Apply the best threshold
    best_t_val = sorted_result_dicts[0]["t_val"]
    log.info(f"Applying t_val: {best_t_val:.2f}")
    intermediate_model.ndnf.disjunctions.weights.data = (
        (torch.abs(intermediate_disj_weight) > best_t_val)
        * torch.sign(intermediate_disj_weight)
        * 6
    )

    return best_t_val


def single_model_disentangle(
    model: CoverTypeMLPPINeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_dir: Path,
) -> dict[str, Any]:
    def _eval_with_log_wrapper(
        model_name: str, data_loader: DataLoader = val_loader
    ) -> dict[str, float]:
        eval_meters = covertype_classifier_eval(model, device, data_loader)
        return parse_eval_return_meters_with_logging(eval_meters, model_name)

    # Stage 1: Evaluate the pruned model
    prune_log = _eval_with_log_wrapper(
        "Pruned CoverType NDNF (test)", test_loader
    )
    log.info("------------------------------------------")

    # Stage 2: Disentangle the model
    def disentangle(model: CoverTypeMLPPINeuralDNF) -> dict[str, Any]:
        log.info("Disentangling the model...")

        # Create an intermediate model
        intermediate_ndnf = create_intermediate_ndnf(model.ndnf)
        model.change_ndnf(intermediate_ndnf)
        model.to(device)
        model.eval()
        intermediate_pre_threshold_log = _eval_with_log_wrapper(
            "Intermediate disentangled Covertype NDNF model (val)"
        )
        log.info("------------------------------------------")

        # Threshold the intermediate model's disjunctive layer
        best_t_val = threshold_intermediate_model_disjunctive_layer(
            model, device, train_loader, do_logging=True
        )
        intermediate_thresholded_log = _eval_with_log_wrapper(
            f"Thresholded disentangled Covertype NDNF (t={best_t_val}) (val)",
        )
        log.info("------------------------------------------")

        # Prune the intermediate disentangled model
        multiround_prune(
            model,
            device,
            train_loader,
            lambda x: _eval_with_log_wrapper(x["model_name"]),
        )
        intermediate_pruned_log = _eval_with_log_wrapper(
            "Pruned disentangled Covertype NDNF (test)", test_loader
        )
        log.info("------------------------------------------")

        # Condense the model
        model.to(torch.device("cpu"))
        condensed_ndnf = condense_neural_dnf_model(model.ndnf)
        model.change_ndnf(condensed_ndnf)
        model.to(device)
        model.eval()
        condensed_model_log = _eval_with_log_wrapper(
            "Disentangled Covertype NDNF model (condensed, test)",
            test_loader,
        )

        return {
            "intermediate_pre_threshold_log": intermediate_pre_threshold_log,
            "intermediate_thresholded_log": intermediate_thresholded_log,
            "intermediate_pruned_log": intermediate_pruned_log,
            "intermediate_ndnf": intermediate_ndnf,
            "condensed_ndnf": condensed_ndnf,
            "condensed_model": model,
            "condensed_model_log": condensed_model_log,
        }

    # Check for checkpoints
    model_path = model_dir / f"{DISENTANGLED_MODEL_BASE_NAME}_v2.pth"
    disentangle_result_json = (
        model_dir / f"{DISENTANGLED_RESULT_JSON_BASE_NAME}_v2.json"
    )
    if model_path.exists() and disentangle_result_json.exists():
        # The model has been disentangled, pruned and condensed
        with open(disentangle_result_json, "r") as f:
            stats = json.load(f)

        disentangled_ndnf = NeuralDNF(
            stats["disentangled_ndnf_n_in"],
            stats["disentangled_ndnf_n_conjunctions"],
            stats["disentangled_ndnf_n_out"],
            1.0,
        )
        model.change_ndnf(disentangled_ndnf)
        model.to(device)
        disentangled_state = torch.load(
            model_path, map_location=device, weights_only=True
        )
        model.load_state_dict(disentangled_state)
        model.eval()
        condensed_model_log = _eval_with_log_wrapper(
            "Disentangled Covertype NDNF model (condensed, test)",
            test_loader,
        )
    else:
        ret = disentangle(model)
        intermediate_ndnf: NeuralDNF = ret["intermediate_ndnf"]
        disentangled_ndnf: NeuralDNF = ret["condensed_ndnf"]
        disentangled_model: CoverTypeMLPPINeuralDNF = ret["condensed_model"]
        condensed_model_log = ret["condensed_model_log"]

        torch.save(disentangled_model.state_dict(), model_path)
        torch.save(
            intermediate_ndnf.state_dict(),
            model_dir
            / f"{INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME}_ndnf_v2.pth",
        )

        disentanglement_result = {
            "disentangled_ndnf_n_in": disentangled_ndnf.conjunctions.in_features,
            "disentangled_ndnf_n_conjunctions": disentangled_ndnf.conjunctions.out_features,
            "disentangled_ndnf_n_out": disentangled_ndnf.disjunctions.out_features,
            "intermediate_ndnf_n_conjunctions": intermediate_ndnf.conjunctions.out_features,
            "intermediate_pre_threshold_log": ret[
                "intermediate_pre_threshold_log"
            ],
            "intermediate_thresholded_log": ret["intermediate_thresholded_log"],
            "intermediate_pruned_log": ret["intermediate_pruned_log"],
            "condensed_model_log": condensed_model_log,
        }

        with open(disentangle_result_json, "w") as f:
            json.dump(disentanglement_result, f, indent=4)

    log.info("======================================")

    return condensed_model_log


def multirun_disentangle(cfg: DictConfig) -> None:
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
    hold_out_test_X, hold_out_test_y = get_covertype_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = CoverTypeDataset(hold_out_test_X, hold_out_test_y)
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

    ret_dicts: list[dict[str, Any]] = []
    for s in eval_cfg["multirun_seeds"]:
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"])
            / caps_experiment_name
            / f"{caps_experiment_name}-{s}"
        )
        model = construct_model(eval_cfg)
        assert isinstance(model, CoverTypeMLPPINeuralDNFEO)

        model = model.to_ndnf_model()
        model.to(device)
        model.eval()
        model_state = torch.load(
            model_dir / f"{FIRST_PRUNE_MODEL_BASE_NAME}.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(model_state)

        # Data loaders
        X, y = get_covertype_data_np_from_path(cfg["dataset"], is_test=False)
        X_train_og, X_val, y_train_og, y_val = train_test_split(
            X,
            y,
            test_size=eval_cfg.get("val_size", 0.2),
            random_state=eval_cfg.get("val_seed", 73),
        )
        # Since covertype has a lot of instances of data entries, we further
        # split the train dataset to use a smaller subset to speed up the
        # pruning process
        _, X_train, _, y_train = train_test_split(
            X_train_og,
            y_train_og,
            test_size=0.4,
            random_state=eval_cfg.get("val_seed", 73),
        )
        train_dataset = CoverTypeDataset(X_train, y_train)
        val_dataset = CoverTypeDataset(X_val, y_val)
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
            single_model_disentangle(
                model, device, train_loader, val_loader, test_loader, model_dir
            )
        )
        log.info("============================================================")

    # Synthesize the results
    return_dict = {}
    for k in EVAL_RELEVANT_KEYS:
        synth_dict = synthesize(np.array([d[k] for d in ret_dicts]))
        for sk, sv in synth_dict.items():
            return_dict[f"{k}/{sk}"] = sv
    for k in EVAL_ERROR_DICT_RELEVANT_KEYS:
        synth_dict = synthesize(
            np.array([d["error_dict"][k] for d in ret_dicts])
        )
        for sk, sv in synth_dict.items():
            return_dict[f"error_dict/{k}/{sk}"] = sv

    log.info("Synthesized results:")
    for k, v in return_dict.items():
        log.info(f"{k}: {v:.3f}")

    with open("disentangle_result_v2.json", "w") as f:
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
        multirun_disentangle(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Disentangle v2",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
