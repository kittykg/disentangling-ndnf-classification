"""
This script thresholds prunedCarNDNFMT model. The input models are strictly
after pruning stage in the post-training processing pipeline. The thresholed
NDNF models are stored and evaluated. The evaluation metrics include accuracy,
sample Jaccard and macro Jaccard.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from neural_dnf.semi_symbolic import (
    BaseSemiSymbolic,
    SemiSymbolicLayerType,
    SemiSymbolic,
    SemiSymbolicMutexTanh,
)
from neural_dnf.neural_dnf import NeuralDNF
from neural_dnf.post_training import (
    prune_neural_dnf,
    split_entangled_conjunction,
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

from car.data_utils_car import get_car_data_np_from_path, CarDataset
from car.eval.eval_common import (
    EVAL_RELEVANT_KEYS,
    EVAL_ERROR_DICT_RELEVANT_KEYS,
    car_classifier_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    FIRST_PRUNE_MODEL_BASE_NAME,
)
from car.eval.ndnf_multirun_prune import multiround_prune, comparison_fn
from car.models import (
    CarBaseNeuralDNF,
    CarNeuralDNFMT,
    CarNeuralDNF,
    construct_model,
)

log = logging.getLogger()


SOFT_EXTRCT_THRESHOLD_MODEL_BASE_NAME = "model_soft_extract_thresholded"
SOFT_EXTRCT_THRESHOLD_RESULT_JSON_BASE_NAME = "soft_extract_threshold_result"
SOFT_EXTRCT_DISENTANGLED_MODEL_BASE_NAME = "model_soft_extract_disentangled"
SOFT_EXTRCT_DISENTANGLED_RESULT_JSON_BASE_NAME = (
    "soft_extract_disentangled_result"
)


class BaseChainedNeuralDNF(CarBaseNeuralDNF):
    sub_ndnf: NeuralDNF
    disjunctive_layer: BaseSemiSymbolic

    def __init__(
        self, sub_ndnf: NeuralDNF, disjunctive_layer: BaseSemiSymbolic
    ):
        assert disjunctive_layer.layer_type == SemiSymbolicLayerType.DISJUNCTION
        assert (
            disjunctive_layer.in_features == sub_ndnf.disjunctions.out_features
        )
        super().__init__(
            num_conjunctions=sub_ndnf.conjunctions.out_features,
            num_features=sub_ndnf.conjunctions.in_features,
        )
        self.sub_ndnf = sub_ndnf
        self.disjunctive_layer = disjunctive_layer

    def get_conjunction(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "This model does not support get_conjunction method."
        )

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError(
            "This model does not support get_weight_reg_loss method."
        )

    def _create_ndnf_model(self) -> None:
        # The `ndnf` attribute is not used in this class, so we will return None
        # to avoid confusion. The `sub_ndnf` is used instead.
        return None

    def forward(self, x: Tensor) -> Tensor:
        x = self.sub_ndnf(x)
        x = torch.tanh(x)
        return self.disjunctive_layer(x)


# class TanhChainedNeuralDNF(BaseChainedNeuralDNF):
#     sub_ndnf: NeuralDNF
#     disjunctive_layer: SemiSymbolic


class MutexTanhChainedNeuralDNF(BaseChainedNeuralDNF):
    sub_ndnf: NeuralDNF
    disjunctive_layer: SemiSymbolicMutexTanh


def threshold_conjunctive_layer(
    model: CarNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    do_logging: bool = False,
) -> float:
    log.info("Thresholding the conjunctive layer...")

    og_conj_weight = model.ndnf.conjunctions.weights.data.clone()

    conj_min = torch.min(model.ndnf.conjunctions.weights.data)
    conj_max = torch.max(model.ndnf.conjunctions.weights.data)
    threshold_upper_bound = round(
        (torch.Tensor([conj_min, conj_max]).abs().max() + 0.01).item(),
        2,
    )
    log.info(f"Threshold upper bound: {threshold_upper_bound}")

    t_vals = torch.arange(0, threshold_upper_bound, 0.01)
    result_dicts_with_t_val = []

    for v in t_vals:
        model.ndnf.conjunctions.weights.data = (
            (torch.abs(og_conj_weight) > v) * torch.sign(og_conj_weight) * 6.0
        )
        threshold_eval_dict = car_classifier_eval(model, device, train_loader)
        acc_meter = threshold_eval_dict["acc_meter"]
        jacc_meter = threshold_eval_dict["jacc_meter"]
        error_meter = threshold_eval_dict["error_meter"]
        assert isinstance(acc_meter, AccuracyMeter)
        assert isinstance(jacc_meter, JaccardScoreMeter)
        assert isinstance(error_meter, ErrorMeter)
        accuracy = acc_meter.get_average()
        sample_jacc = jacc_meter.get_average("samples")
        macro_jacc = jacc_meter.get_average("macro")
        assert isinstance(sample_jacc, float)
        assert isinstance(macro_jacc, float)
        overall_error_rate = error_meter.get_average()["overall_error_rate"]

        result_dicts_with_t_val.append(
            {
                "t_val": v.item(),
                "accuracy": accuracy,
                "sample_jacc": sample_jacc,
                "macro_jacc": macro_jacc,
                "overall_error_rate": overall_error_rate,
            }
        )

    sorted_result_dicts: list[dict[str, float]] = sorted(
        result_dicts_with_t_val,
        key=lambda x: (
            -x["overall_error_rate"],
            x["sample_jacc"],
            x["accuracy"],
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
                f"Sample Jacc: {d['sample_jacc']:.3f}  "
                f"Macro Jacc: {d['macro_jacc']:.3f}  "
                f"Overall Error Rate: {d['overall_error_rate']:.3f}"
            )

    # Apply the best threshold
    best_t_val = sorted_result_dicts[0]["t_val"]
    model.ndnf.conjunctions.weights.data = (
        (torch.abs(og_conj_weight) > best_t_val)
        * torch.sign(og_conj_weight)
        * 6.0
    )

    return best_t_val


def disentangle_conjunctive_layer(model: CarNeuralDNF) -> BaseChainedNeuralDNF:
    log.info("Disentangling the model...")

    # Remember the disjunction-conjunction mapping
    conj_w = model.ndnf.conjunctions.weights.data.clone().detach().cpu()
    disj_w = model.ndnf.disjunctions.weights.data.clone().detach().cpu()

    og_disj_conj_mapping = dict()
    for disj_id, w in enumerate(disj_w):
        non_zeros = torch.where(w != 0)[0]
        og_disj_conj_mapping[disj_id] = non_zeros.tolist()

    # ======================================================================== #
    # Step 1: Split the conjunctions and create conjunctive sub NDNF
    # ======================================================================== #
    # 1.a split the conjunctions
    # set of unique conjunctions
    unique_conj_ids = set()
    for _, conj_ids in og_disj_conj_mapping.items():
        unique_conj_ids.update(conj_ids)

    # split the conjunctions
    splitted_conj: dict[int, list[Tensor]] = dict()
    for conj_id in unique_conj_ids:
        ret = split_entangled_conjunction(conj_w[conj_id])

        if ret is None:
            log.info(f"Conj {conj_id} is skipped")
            continue

        acc_pairs = []
        for c in ret:
            non_zero_count = torch.sum(c != 0).item()
            acc_pairs.append((non_zero_count, c))
        acc_pairs.sort(key=lambda x: x[0])
        splitted_conj[conj_id] = [c for _, c in acc_pairs]

    # 1.b convert the splitted conjunctions to a NDNF
    # log info of the split
    sorted_used_conj_ids = sorted(splitted_conj.keys())
    total_number_conj_splits = 0
    for conj_id in sorted_used_conj_ids:
        len_split = len(splitted_conj[conj_id])
        log.info(f"Conjunction {conj_id} with {len_split} splits")
        total_number_conj_splits += len_split
    log.info(f"Total number of aux conjunctions: {total_number_conj_splits}")

    # create the sub model
    sub_ndnf_conj = NeuralDNF(
        conj_w.shape[1],
        total_number_conj_splits,
        len(sorted_used_conj_ids),
        1.0,
    )
    # clear the weights
    sub_ndnf_conj.conjunctions.weights.data.fill_(0)
    sub_ndnf_conj.disjunctions.weights.data.fill_(0)

    # assign the weights
    conj_assign_counter = 0
    for i, conj_id in enumerate(sorted_used_conj_ids):
        for split in splitted_conj[conj_id]:
            sub_ndnf_conj.conjunctions.weights.data[conj_assign_counter] = split
            sub_ndnf_conj.disjunctions.weights.data[i, conj_assign_counter] = 6
            conj_assign_counter += 1

    # ======================================================================== #
    # Step 2: add disjunctive layer
    # ======================================================================== #
    disjunctive_layer = SemiSymbolic(
        in_features=len(sorted_used_conj_ids),
        out_features=disj_w.shape[0],
        layer_type=SemiSymbolicLayerType.DISJUNCTION,
        delta=1.0,
    )
    disjunctive_layer.weights.data.fill_(0)
    # assign the weights
    for disj_id, conj_ids in og_disj_conj_mapping.items():
        for cid in conj_ids:
            if cid not in sorted_used_conj_ids:
                continue

            disjunctive_layer.weights.data[
                disj_id, sorted_used_conj_ids.index(cid)
            ] = disj_w[disj_id, cid]

    chained_ndnf = MutexTanhChainedNeuralDNF(
        sub_ndnf=sub_ndnf_conj,
        disjunctive_layer=disjunctive_layer,
    )

    return chained_ndnf


def prune_chained_ndnf(
    chained_ndnf: BaseChainedNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    eval_log_fn: Callable[[dict[str, Any]], dict[str, float]],
) -> int:
    prune_eval_fn = lambda: parse_eval_return_meters_with_logging(
        car_classifier_eval(chained_ndnf, device, train_loader),
        model_name="Prune (intermediate)",
        do_logging=False,
    )

    prune_iteration = 1
    continue_pruning = True

    def prune_disjunctive_layer() -> int:
        curr_weight = chained_ndnf.disjunctive_layer.weights.data.clone()
        comparison_dict = prune_eval_fn()

        prune_count = 0
        flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))

        for i in range(flatten_weight_len):
            curr_weight_flatten = torch.reshape(curr_weight, (-1,))

            if curr_weight_flatten[i] == 0:
                continue

            mask = torch.ones(flatten_weight_len, device=curr_weight.device)
            mask[i] = 0
            mask = mask.reshape(curr_weight.shape)

            masked_weight = curr_weight * mask
            chained_ndnf.disjunctive_layer.weights.data = masked_weight
            new_result_dict = prune_eval_fn()

            if comparison_fn(comparison_dict, new_result_dict):
                prune_count += 1
                curr_weight *= mask

        chained_ndnf.disjunctive_layer.weights.data = curr_weight
        return prune_count

    while continue_pruning:
        log.info(f"Prune iteraiton {prune_iteration}")
        start_time = datetime.now()

        # Prune the disjunctive layer first
        disj_pruned_count = prune_disjunctive_layer()

        # Prune the conjunctive DNF next
        prune_result_dict_conj = prune_neural_dnf(
            chained_ndnf.sub_ndnf,
            prune_eval_fn,
            {},
            comparison_fn,
            options={
                "skip_prune_disj_with_empty_conj": False,
                "skip_last_prune_conj": True,
            },
        )

        important_keys = [
            "disj_prune_count_1",
            "unused_conjunctions_2",
            "conj_prune_count_3",
        ]

        end_time = datetime.now()
        log.info(f"\tTime taken: {end_time - start_time}")
        log.info(f"\tDisjunctive prune count: {disj_pruned_count}")
        log.info(
            f"\tPruned disjunction count: {prune_result_dict_conj['disj_prune_count_1']}"
        )
        log.info(
            f"\tRemoved unused conjunction count: {prune_result_dict_conj['unused_conjunctions_2']}"
        )
        log.info(
            f"\tPruned conjunction count: {prune_result_dict_conj['conj_prune_count_3']}"
        )

        eval_log_fn(
            {"model_name": f"Pruned chained NDNF - (Iter: {prune_iteration})"}
        )
        log.info("..................................")
        # If any of the important keys has the value not 0, then we should
        # continue pruning
        if (
            any([prune_result_dict_conj[k] != 0 for k in important_keys])
            or disj_pruned_count != 0
        ):
            prune_iteration += 1
        else:
            break

    return prune_iteration


def single_model_soft_extract(
    model: CarNeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_dir: Path,
    conjunctive_layer_process_method: str = "threshold",
) -> dict[str, Any]:
    def _eval_with_log_wrapper(
        model_name: str, data_loader: DataLoader = val_loader
    ) -> dict[str, float]:
        eval_meters = car_classifier_eval(model, device, data_loader)
        return parse_eval_return_meters_with_logging(eval_meters, model_name)

    # Stage 1: Evaluate the pruned model
    prune_log = _eval_with_log_wrapper("Pruned Car NDNF (test)", test_loader)
    log.info("------------------------------------------")

    # Stage 2: Process the conjunctive layer
    # Threshold method
    def process_conjunctive_layer_threshold(
        do_logging: bool = False,
    ) -> dict[str, Any]:
        # ================================================================ #
        # Step 2.1a: Threshold the conjunctive layer
        # ================================================================ #
        best_t_val = threshold_conjunctive_layer(
            model, device, train_loader, do_logging=do_logging
        )
        intermediate_log = _eval_with_log_wrapper(
            f"Thresholded model (t={best_t_val})"
        )
        log.info("------------------------------------------")

        # ================================================================ #
        # Step 2.2: Prune the model
        # ================================================================ #
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
        log.info("------------------------------------------")

        return {
            "threshold_val": best_t_val,
            "intermediate_log": intermediate_log,
            "threshold_final_log": threshold_final_log,
        }

    # Disentangle method
    def process_conjunctive_layer_disentangle() -> dict[str, Any]:
        # ==================================================================== #
        # Step 2.1b: Disentangle the conjunctive layer
        # ==================================================================== #
        chained_model = disentangle_conjunctive_layer(model)
        print(chained_model)
        chained_model.to(device)
        chained_model.eval()

        intermediate_pre_prune_log = parse_eval_return_meters_with_logging(
            car_classifier_eval(chained_model, device, val_loader),
            model_name="Pre-prune chained NDNF model (val)",
        )
        log.info("------------------------------------------")

        # ==================================================================== #
        # Step 2.2: Prune the model
        # ==================================================================== #
        prune_chained_ndnf(
            chained_model,
            device,
            train_loader,
            lambda x: parse_eval_return_meters_with_logging(
                car_classifier_eval(chained_model, device, val_loader),
                model_name=x["model_name"],
            ),
        )

        pruned_chained_ndnf_log = parse_eval_return_meters_with_logging(
            car_classifier_eval(chained_model, device, val_loader),
            model_name="Pruned chained NDNF model (val)",
        )
        log.info("------------------------------------------")

        return {
            "intermediate_pre_prune_log": intermediate_pre_prune_log,
            "pruned_chained_ndnf_log": pruned_chained_ndnf_log,
            "chained_model": chained_model,
        }

    # Check for checkpoints
    # If the model is already processed, then we load the processed model
    # Otherwise, we process the model and save
    if conjunctive_layer_process_method == "threshold":
        model_path = model_dir / f"{SOFT_EXTRCT_THRESHOLD_MODEL_BASE_NAME}.pth"
        threshold_result_json = (
            model_dir / f"{SOFT_EXTRCT_THRESHOLD_RESULT_JSON_BASE_NAME}.json"
        )
        if model_path.exists() and threshold_result_json.exists():
            threshold_state = torch.load(
                model_path, map_location=device, weights_only=True
            )
            model.load_state_dict(threshold_state)
            threshold_eval_log = _eval_with_log_wrapper(
                "Thresholded NDNF model"
            )
        else:
            threshold_ret_dict = process_conjunctive_layer_threshold(
                do_logging=True
            )
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

    model_path = model_dir / f"{SOFT_EXTRCT_DISENTANGLED_MODEL_BASE_NAME}.pth"
    disentangled_result_json = (
        model_dir / f"{SOFT_EXTRCT_DISENTANGLED_RESULT_JSON_BASE_NAME}.json"
    )
    if model_path.exists() and disentangled_result_json.exists():
        with open(disentangled_result_json, "r") as f:
            stats = json.load(f)

        chained_model = MutexTanhChainedNeuralDNF(
            sub_ndnf=NeuralDNF(
                stats["sub_ndnf_in"],
                stats["sub_ndnf_n_conunctions"],
                stats["sub_ndnf_out"],
                1.0,
            ),
            disjunctive_layer=SemiSymbolic(
                stats["disjunctive_layer_in"],
                stats["disjunctive_layer_out"],
                SemiSymbolicLayerType.DISJUNCTION,
                1.0,
            ),
        )
        chained_model.to(device)
        chained_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        chained_model.eval()

        pruned_chained_ndnf_log = parse_eval_return_meters_with_logging(
            car_classifier_eval(chained_model, device, test_loader),
            "Pruned chained NDNF model(test)",
        )
    else:
        ret = process_conjunctive_layer_disentangle()
        chained_model = ret["chained_model"]
        assert isinstance(chained_model, BaseChainedNeuralDNF)

        torch.save(chained_model.state_dict(), model_path)
        pruned_chained_ndnf_log = parse_eval_return_meters_with_logging(
            car_classifier_eval(chained_model, device, test_loader),
            "Pruned chained NDNF model (test)",
        )

        disentanglement_result = {
            "intermediate_pre_prune_log": ret["intermediate_pre_prune_log"],
            "pruned_chained_ndnf_log_val": ret["pruned_chained_ndnf_log"],
            "pruned_chained_ndnf_log": pruned_chained_ndnf_log,
            "sub_ndnf_in": chained_model.sub_ndnf.conjunctions.in_features,
            "sub_ndnf_n_conunctions": (
                chained_model.sub_ndnf.conjunctions.out_features
            ),
            "sub_ndnf_out": (chained_model.sub_ndnf.disjunctions.out_features),
            "disjunctive_layer_in": (
                chained_model.disjunctive_layer.in_features
            ),
            "disjunctive_layer_out": (
                chained_model.disjunctive_layer.out_features
            ),
        }
        with open(disentangled_result_json, "w") as f:
            json.dump(disentanglement_result, f, indent=4)

    log.info("============================================================")
    return pruned_chained_ndnf_log


def multirun_soft_extraction(cfg: DictConfig) -> None:
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
    hold_out_test_X, hold_out_test_y = get_car_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = CarDataset(hold_out_test_X, hold_out_test_y)
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
        model = construct_model(eval_cfg, num_features=hold_out_test_X.shape[1])
        assert isinstance(model, CarNeuralDNFMT)

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
        X, y = get_car_data_np_from_path(cfg["dataset"], is_test=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=eval_cfg.get("val_size", 0.2),
            random_state=eval_cfg.get("val_seed", 73),
        )
        train_dataset = CarDataset(X_train, y_train)
        val_dataset = CarDataset(X_val, y_val)
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
            single_model_soft_extract(
                model,
                device,
                train_loader,
                val_loader,
                test_loader,
                model_dir,
                conjunctive_layer_process_method=eval_cfg[
                    "discretisation_method"
                ],
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

    with open("soft_extraction_result.json", "w") as f:
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
        multirun_soft_extraction(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun Soft Extraction",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
