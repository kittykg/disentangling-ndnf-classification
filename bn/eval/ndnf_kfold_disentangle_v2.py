"""
This script disentangle pruned NDNF in a BooleanNetwork NeuralDNF model. The
input model's NDNF models are strictly after pruning stage in the post-training
processing pipeline. The BooleanNetwork NeuralDNF models with their NDNF models
disentangled are stored and evaluated, with the relevant information stored in a
json. The evaluation metrics include accuracy, precision, recall, and F1 score.

The difference between this script and the previous version is that instead of
re-wiring all the new conjunctions to corresponding disjunctive nodes with
weights of 6, we re-wire them to the disjunctive nodes with the absolute value
of the original weight, then threshold them and prune it.
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
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from neural_dnf import NeuralDNF
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

from analysis import AccuracyMeter, JaccardScoreMeter, synthesize
from utils import post_to_discord_webhook

from bn.data_utils_bn import (
    get_boolean_network_full_data_np_from_path,
    BooleanNetworkDataset,
)
from bn.eval.eval_common import (
    boolean_network_classifier_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    FIRST_PRUNE_MODEL_BASE_NAME,
    INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from bn.eval.ndnf_kfold_prune import multiround_prune
from bn.models import BooleanNetworkNeuralDNF, construct_model


log = logging.getLogger()


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
    intermediate_bn_model: BooleanNetworkNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    do_logging: bool = False,
) -> float:
    intermediate_disj_weight = (
        intermediate_bn_model.ndnf.disjunctions.weights.data.clone()
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
        intermediate_bn_model.ndnf.disjunctions.weights.data = (
            (torch.abs(intermediate_disj_weight) > v)
            * torch.sign(intermediate_disj_weight)
            * 6
        )
        threshold_eval_dict = boolean_network_classifier_eval(
            intermediate_bn_model, device, train_loader
        )
        acc_meter = threshold_eval_dict["acc_meter"]
        assert isinstance(acc_meter, AccuracyMeter)
        jacc_meter = threshold_eval_dict["jacc_meter"]
        assert isinstance(jacc_meter, JaccardScoreMeter)

        accuracy = acc_meter.get_average()
        other_metrics = acc_meter.get_other_classification_metrics(
            average="micro", compute_hamming=True
        )
        precision = other_metrics["precision"]
        recall = other_metrics["recall"]
        f1 = other_metrics["f1"]
        hamming = other_metrics["hamming"]
        avg_sample_jacc = jacc_meter.get_average()
        avg_macro_jacc = jacc_meter.get_average("macro")
        avg_micro_jacc = jacc_meter.get_average("micro")
        assert isinstance(avg_macro_jacc, float)
        assert isinstance(avg_sample_jacc, float)
        assert isinstance(avg_micro_jacc, float)

        result_dicts_with_t_val.append(
            {
                "t_val": v.item(),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hamming": hamming,
                "avg_sample_jacc": avg_sample_jacc,
                "avg_macro_jacc": avg_macro_jacc,
                "avg_micro_jacc": avg_micro_jacc,
            }
        )

    sorted_result_dicts: list[dict[str, float]] = sorted(
        result_dicts_with_t_val,
        key=lambda x: (
            x["f1"],
            -x["hamming"],
            x["avg_sample_jacc"],
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
                f"Hamming: {d['hamming']:.3f}  "
                f"Avg Sample Jacc: {d['avg_sample_jacc']:.3f}  "
                f"Avg Macro Jacc: {d['avg_macro_jacc']:.3f}  "
                f"Avg Micro Jacc: {d['avg_micro_jacc']:.3f}"
            )

    # Apply the best threshold
    best_t_val = sorted_result_dicts[0]["t_val"]
    log.info(f"Applying t_val: {best_t_val:.2f}")
    intermediate_bn_model.ndnf.disjunctions.weights.data = (
        (torch.abs(intermediate_disj_weight) > best_t_val)
        * torch.sign(intermediate_disj_weight)
        * 6
    )

    return best_t_val


def single_model_disentangle(
    fold_id: int,
    model: BooleanNetworkNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_dir: Path,
) -> dict[str, Any]:
    def _eval_with_log_wrapper(
        model: BooleanNetworkNeuralDNF, model_name: str
    ) -> dict[str, Any]:
        eval_meters = boolean_network_classifier_eval(model, device, val_loader)
        return parse_eval_return_meters_with_logging(
            eval_meters, model_name, log_confusion_matrix=True
        )

    # Stage 1: Evaluate the pruned model
    _eval_with_log_wrapper(model, "Pruned NDNF model")
    log.info("------------------------------------------")

    # Stage 2: Disentangle the pruned model
    def disentangle(model: BooleanNetworkNeuralDNF) -> dict[str, Any]:
        log.info("Disentangling the model...")

        # Create an intermediate model
        intermediate_ndnf = create_intermediate_ndnf(model.ndnf)
        model.change_ndnf(intermediate_ndnf)
        model.to(device)
        model.eval()
        intermediate_pre_threshold_log = _eval_with_log_wrapper(
            model, "BooleanNetwork with intermediate disentangled NDNF model"
        )
        log.info("------------------------------------------")

        # Threshold the intermediate model's disjunctive layer
        best_t_val = threshold_intermediate_model_disjunctive_layer(
            model, device, train_loader, do_logging=True
        )
        intermediate_thresholed_log = _eval_with_log_wrapper(
            model,
            f"BooleanNetwork with thresholded disentangled NDNF (t={best_t_val})",
        )
        log.info("------------------------------------------")

        # Prune the intermediate disentangled model
        multiround_prune(
            model,
            device,
            train_loader,
            lambda x: _eval_with_log_wrapper(model, x["model_name"]),
        )
        intermediate_pruned_log = _eval_with_log_wrapper(
            model, "BooleanNetwork with pruned disentangled NDNF model"
        )
        log.info("------------------------------------------")

        # Condense the model
        model.to(torch.device("cpu"))
        condensed_ndnf = condense_neural_dnf_model(model.ndnf)
        model.change_ndnf(condensed_ndnf)
        model.to(device)
        model.eval()
        condensed_model_log = _eval_with_log_wrapper(
            model, "Condensed disentangled NDNF model"
        )

        return {
            "intermediate_pre_threshold_log": intermediate_pre_threshold_log,
            "intermediate_thresholed_log": intermediate_thresholed_log,
            "intermediate_pruned_log": intermediate_pruned_log,
            "intermediate_ndnf": intermediate_ndnf,
            "condensed_ndnf": condensed_ndnf,
            "condensed_model": model,
            "condensed_model_log": condensed_model_log,
        }

    # Check for checkpoints
    model_path = (
        model_dir / f"{DISENTANGLED_MODEL_BASE_NAME}_v2_fold_{fold_id}.pth"
    )
    disentangle_result_json = (
        model_dir
        / f"fold_{fold_id}_{DISENTANGLED_RESULT_JSON_BASE_NAME}_v2.json"
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
            model, "BooleanNetwork with condensed disentangled NDNF model"
        )
    else:
        ret = disentangle(model)
        intermediate_ndnf: NeuralDNF = ret["intermediate_ndnf"]
        disentangled_ndnf: NeuralDNF = ret["condensed_ndnf"]
        disentangled_model: BooleanNetworkNeuralDNF = ret["condensed_model"]
        condensed_model_log = ret["condensed_model_log"]

        torch.save(disentangled_model.state_dict(), model_path)
        torch.save(
            intermediate_ndnf.state_dict(),
            model_dir
            / f"{INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME}_v2_ndnf_fold_{fold_id}.pth",
        )

        disentanglement_result = {
            "disentangled_ndnf_n_in": disentangled_ndnf.conjunctions.in_features,
            "disentangled_ndnf_n_conjunctions": disentangled_ndnf.conjunctions.out_features,
            "disentangled_ndnf_n_out": disentangled_ndnf.disjunctions.out_features,
            "intermediate_ndnf_n_conjunctions": intermediate_ndnf.conjunctions.out_features,
            "intermediate_pre_threshold_log": ret[
                "intermediate_pre_threshold_log"
            ],
            "intermediate_thresholed_log": ret["intermediate_thresholed_log"],
            "intermediate_pruned_log": ret["intermediate_pruned_log"],
            "condensed_model_log": condensed_model_log,
        }

        with open(disentangle_result_json, "w") as f:
            json.dump(disentanglement_result, f, indent=4)

    log.info("======================================")

    return condensed_model_log


def post_train_disentangle(cfg: DictConfig):
    eval_cfg = cfg["eval"]
    full_experiment_name = f"{eval_cfg['experiment_name']}_{eval_cfg['seed']}"
    run_dir_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

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

    # Load data
    bn_dataset = BooleanNetworkDataset(
        dataset_type=cfg["dataset"]["dataset_name"],
        subtype=None,
        data=get_boolean_network_full_data_np_from_path(cfg["dataset"]),
    )

    # K-Fold
    kf = KFold(
        n_splits=eval_cfg["k_folds"],
        shuffle=True,
        random_state=eval_cfg["seed"],
    )

    ret_dicts: list[dict[str, Any]] = []

    for fold_id, (train_index, test_index) in enumerate(
        kf.split(np.arange(len(bn_dataset)))
    ):
        log.info(f"Fold {fold_id} starts")
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
        )
        pruned_pth = (
            model_dir / f"{FIRST_PRUNE_MODEL_BASE_NAME}_fold_{fold_id}.pth"
        )
        assert pruned_pth.exists(), f"Model {model_dir.name} not pruned!"

        model = construct_model(cfg["eval"], bn_dataset.data.shape[2])
        assert isinstance(model, BooleanNetworkNeuralDNF)
        model.to(device)
        model_state = torch.load(
            pruned_pth, map_location=device, weights_only=True
        )
        model.load_state_dict(model_state)
        model.eval()

        # Data loaders
        train_loader = torch.utils.data.DataLoader(
            bn_dataset,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
            sampler=torch.utils.data.SubsetRandomSampler(train_index),  # type: ignore
        )
        val_loader = torch.utils.data.DataLoader(
            bn_dataset,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
            sampler=torch.utils.data.SubsetRandomSampler(test_index),  # type: ignore
        )

        log.info(f"Experiment {model_dir.name} loaded!")
        ret_dicts.append(
            single_model_disentangle(
                fold_id, model, device, train_loader, val_loader, model_dir
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

    torch.autograd.set_detect_anomaly(True)

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_disentangle(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Kfold Disentangle V2",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    run_eval()
