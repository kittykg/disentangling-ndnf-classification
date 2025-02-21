"""
This script disentangle pruned plain NDNF. The input NDNF models are strictly
after pruning stage in the post-training processing pipeline. The disentangled
NDNF model are stored and evaluated, with the relevant information stored in a
json. The evaluation metrics include accuracy, sample Jaccard, macro Jaccard,
and error metrics.

The difference between this script and the previous version is that instead of
re-wiring all the new conjunctions to corresponding disjunctive nodes with
weights of 6, we re-wire them to the disjunctive nodes with the absolute value
of the original weight, then threshold them and prune it.
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
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from neural_dnf import NeuralDNF, NeuralDNFEO
from neural_dnf.post_training import (
    split_entangled_conjunction,
    split_entangled_disjunction,
    prune_neural_dnf,
)

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
from zoo.eval.ndnf_eval_common import (
    ndnf_based_model_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    FIRST_PRUNE_MODEL_BASE_NAME,
    INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_MODEL_BASE_NAME,
    DISENTANGLED_RESULT_JSON_BASE_NAME,
)
from zoo.eval.ndnf_eo_kfold_prune import comparison_fn
from zoo.train_zoo import construct_model


log = logging.getLogger()


class ChainedNDNF(torch.nn.Module):
    sub_ndnf: NeuralDNF
    sub_ndnf_disj: NeuralDNF

    def __init__(self, sub_ndnf, sub_ndnf_disj):
        super().__init__()
        self.sub_ndnf = sub_ndnf
        self.sub_ndnf_disj = sub_ndnf_disj

    def forward(self, x):
        return self.sub_ndnf_disj(self.sub_ndnf(x))


def create_sub_models(model: NeuralDNF) -> tuple[NeuralDNF, NeuralDNF]:
    # Remember the disjunction-conjunction mapping
    conj_w = model.conjunctions.weights.data.clone().detach().cpu()
    disj_w = model.disjunctions.weights.data.clone().detach().cpu()

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
    # Step 2: Split the disjunctions and create disjunctive sub NDNF
    # ======================================================================== #
    # 2.a split the disjunction
    splitted_disj: dict[int, list[Tensor]] = dict()
    for disj_id in og_disj_conj_mapping.keys():
        ret = split_entangled_disjunction(disj_w[disj_id])

        if ret is None:
            log.info(f"Disj {disj_id} is skipped")
            continue

        acc_pairs = []
        for t, flag in ret:
            if flag:
                # This is a disjunction already
                # Add each individual non-zero item as a valid candidate
                for c in torch.where(t != 0)[0]:
                    mask = torch.zeros_like(t)
                    mask[c] = 1
                    acc_pairs.append((1, mask * t))
            else:
                non_zero_count = torch.sum(t != 0).item()
                acc_pairs.append((non_zero_count, t))
        acc_pairs.sort(key=lambda x: x[0])
        splitted_disj[disj_id] = [c for _, c in acc_pairs]

    # 2.b convert the splitted disjunctions to a NDNF
    # log info of the split
    sorted_used_disj_ids = sorted(splitted_disj.keys())
    total_number_disj_splits = 0
    for disj_id in sorted_used_disj_ids:
        len_split = len(splitted_disj[disj_id])
        log.info(f"Disjunction {disj_id} with {len_split} splits")
        total_number_disj_splits += len_split
    log.info(f"Total number of aux disjunctions: {total_number_disj_splits}")

    # create the sub model
    sub_ndnf_disj = NeuralDNF(
        len(sorted_used_conj_ids),
        total_number_disj_splits,
        len(sorted_used_disj_ids),
        1.0,
    )

    # clear the weights
    sub_ndnf_disj.conjunctions.weights.data.fill_(0)
    sub_ndnf_disj.disjunctions.weights.data.fill_(0)

    # assign the weights
    disj_assign_counter = 0
    for disj_id in sorted_used_disj_ids:
        for split in splitted_disj[disj_id]:
            aux_w = torch.zeros_like(
                sub_ndnf_disj.conjunctions.weights.data[disj_assign_counter]
            )
            for old_conj_id in torch.where(split != 0)[0]:
                if old_conj_id.item() not in sorted_used_conj_ids:
                    continue
                new_conj_id = sorted_used_conj_ids.index(
                    int(old_conj_id.item())
                )
                aux_w[new_conj_id] = split[old_conj_id]

            sub_ndnf_disj.conjunctions.weights.data[disj_assign_counter] = aux_w
            sub_ndnf_disj.disjunctions.weights.data[
                disj_id, disj_assign_counter
            ] = 6
            disj_assign_counter += 1

    return sub_ndnf_conj, sub_ndnf_disj


def prune_chained_ndnf(
    chained_ndnf: ChainedNDNF,
    device: torch.device,
    train_loader: DataLoader,
    eval_log_fn: Callable[[dict[str, Any]], dict[str, float]],
) -> int:
    prune_eval_fn = lambda: parse_eval_return_meters_with_logging(
        ndnf_based_model_eval(chained_ndnf, device, train_loader),  # type: ignore
        model_name="Prune (intermediate)",
        do_logging=False,
    )

    prune_iteration = 1
    continue_pruning = True
    while continue_pruning:
        print(f"Prune iteraiton {prune_iteration}")
        start_time = datetime.now()

        # Prune the disjunctive DNF first
        prune_result_dict_disj = prune_neural_dnf(
            chained_ndnf.sub_ndnf_disj,
            prune_eval_fn,
            {},
            comparison_fn,
            options={
                "skip_prune_disj_with_empty_conj": True,
                "skip_last_prune_disj": True,
            },
        )

        # Prune the conjunctive DNF next
        prune_result_dict_conj = prune_neural_dnf(
            chained_ndnf.sub_ndnf,
            prune_eval_fn,
            {},
            comparison_fn,
            options={
                "skip_prune_conj_with_empty_disj": True,
                "skip_last_prune_conj": True,
            },
        )

        important_keys = [
            "disj_prune_count_1",
            "unused_conjunctions_2",
            "conj_prune_count_3",
        ]

        end_time = datetime.now()
        print(f"\tTime taken: {end_time - start_time}")
        for d in [prune_result_dict_disj, prune_result_dict_conj]:
            print(f"\tPruned disjunction count: {d['disj_prune_count_1']}")
            print(
                f"\tRemoved unused conjunction count: {d['unused_conjunctions_2']}"
            )
            print(f"\tPruned conjunction count: {d['conj_prune_count_3']}")
            print()

        eval_log_fn(
            {"model_name": f"Pruned chained NDNF - (Iter: {prune_iteration})"}
        )
        print("..................................")
        # If any of the important keys has the value not 0, then we should
        # continue pruning
        continue_pruning = False
        for d in [prune_result_dict_disj, prune_result_dict_conj]:
            if any([d[k] != 0 for k in important_keys]):
                continue_pruning = True
                break
        prune_iteration += 1

    return prune_iteration


def single_model_disentangle(
    fold_id: int,
    model: NeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_dir: Path,
) -> dict[str, Any]:
    # Stage 1: Evaluate the pruned model
    parse_eval_return_meters_with_logging(
        ndnf_based_model_eval(model, device, val_loader),
        model_name="Pruned NDNF model (val)",
    )
    log.info("------------------------------------------")

    # Stage 2: Disentangle the pruned model
    def disentangle(model: NeuralDNF) -> dict[str, Any]:
        log.info("Disentangling the model...")

        # Create the sub models
        sub_ndnf_conj, sub_ndnf_disj = create_sub_models(model)

        # Create the chained model
        chained_ndnf = ChainedNDNF(sub_ndnf_conj, sub_ndnf_disj)
        chained_ndnf.to(device)
        chained_ndnf.eval()
        intermediate_pre_prune_log = parse_eval_return_meters_with_logging(
            ndnf_based_model_eval(model, device, val_loader),
            model_name="Pre-prune chained NDNF model (val)",
        )
        log.info("------------------------------------------")

        # Prune the chained model
        prune_chained_ndnf(
            chained_ndnf,
            device,
            train_loader,
            lambda x: parse_eval_return_meters_with_logging(
                ndnf_based_model_eval(chained_ndnf, device, val_loader),  # type: ignore
                model_name=x["model_name"],
            ),
        )

        pruned_chained_ndnf_log = parse_eval_return_meters_with_logging(
            ndnf_based_model_eval(chained_ndnf, device, val_loader),  # type: ignore
            model_name="Pruned chained NDNF model (val)",
        )
        log.info("------------------------------------------")

        return {
            "intermediate_pre_prune_log": intermediate_pre_prune_log,
            "pruned_chained_ndnf_log": pruned_chained_ndnf_log,
            "sub_ndnf_conj": chained_ndnf.sub_ndnf,
            "sub_ndnf_conj_in": chained_ndnf.sub_ndnf.conjunctions.in_features,
            "sub_ndnf_conj_n_conjunctions": chained_ndnf.sub_ndnf.conjunctions.out_features,
            "sub_ndnf_conj_out": chained_ndnf.sub_ndnf.disjunctions.out_features,
            "sub_ndnf_disj": chained_ndnf.sub_ndnf_disj,
            "sub_ndnf_disj_in": chained_ndnf.sub_ndnf_disj.conjunctions.in_features,
            "sub_ndnf_disj_n_conjunctions": chained_ndnf.sub_ndnf_disj.conjunctions.out_features,
            "sub_ndnf_disj_out": chained_ndnf.sub_ndnf_disj.disjunctions.out_features,
            "chained_ndnf": chained_ndnf,
        }

    # Check for checkpoints
    model_path = (
        model_dir / f"{DISENTANGLED_MODEL_BASE_NAME}_v3_fold_{fold_id}.pth"
    )
    disentangle_result_json = (
        model_dir
        / f"fold_{fold_id}_{DISENTANGLED_RESULT_JSON_BASE_NAME}_v3.json"
    )
    if model_path.exists() and disentangle_result_json.exists():
        # The model has been disentangled, pruned and condensed
        with open(disentangle_result_json, "r") as f:
            stats = json.load(f)

        sub_ndnf_conj = NeuralDNF(
            stats["sub_ndnf_conj_in"],
            stats["sub_ndnf_conj_n_conjunctions"],
            stats["sub_ndnf_conj_out"],
            1.0,
        )
        sub_ndnf_disj = NeuralDNF(
            stats["sub_ndnf_disj_in"],
            stats["sub_ndnf_disj_n_conjunctions"],
            stats["sub_ndnf_disj_out"],
            1.0,
        )

        chained_ndnf = ChainedNDNF(sub_ndnf_conj, sub_ndnf_disj)
        chained_ndnf.to(device)
        chained_ndnf.eval()
        chanied_ndnf_state = torch.load(
            model_path, map_location=device, weights_only=True
        )
        chained_ndnf.load_state_dict(chanied_ndnf_state)

        final_chained_ndnf_log = parse_eval_return_meters_with_logging(
            ndnf_based_model_eval(chained_ndnf, device, val_loader),  # type: ignore
            model_name="Final chained NDNF model (val)",
        )
    else:
        ret = disentangle(model)
        sub_ndnf_conj = ret["sub_ndnf_conj"]
        sub_ndnf_disj = ret["sub_ndnf_disj"]
        chained_ndnf = ret["chained_ndnf"]

        torch.save(
            sub_ndnf_conj.state_dict(),
            model_dir
            / f"{INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME}_v3_sub_conj_fold_{fold_id}.pth",
        )
        torch.save(
            sub_ndnf_disj.state_dict(),
            model_dir
            / f"{INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME}_v3_sub_disj_fold_{fold_id}.pth",
        )
        torch.save(chained_ndnf.state_dict(), model_path)

        disentanglement_result = {
            "intermediate_pre_prune_log": ret["intermediate_pre_prune_log"],
            "pruned_chained_ndnf_log": ret["pruned_chained_ndnf_log"],
            "sub_ndnf_conj_in": ret["sub_ndnf_conj_in"],
            "sub_ndnf_conj_n_conjunctions": ret["sub_ndnf_conj_n_conjunctions"],
            "sub_ndnf_conj_out": ret["sub_ndnf_conj_out"],
            "sub_ndnf_disj_in": ret["sub_ndnf_disj_in"],
            "sub_ndnf_disj_n_conjunctions": ret["sub_ndnf_disj_n_conjunctions"],
            "sub_ndnf_disj_out": ret["sub_ndnf_disj_out"],
        }

        with open(disentangle_result_json, "w") as f:
            json.dump(disentanglement_result, f, indent=4)
        final_chained_ndnf_log = ret["pruned_chained_ndnf_log"]

    log.info("======================================")

    return final_chained_ndnf_log


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
    X, y, _ = get_zoo_data_np_from_path(
        data_dir_path=Path(cfg["dataset"]["save_dir"])
    )
    dataset = ZooDataset(X, y)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=eval_cfg["k_folds"],
        shuffle=True,
        random_state=eval_cfg["seed"],
    )

    ret_dicts: list[dict[str, float]] = []
    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y)):
        log.info(f"Fold {fold_id} starts")
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
        )
        pruned_pth = (
            model_dir / f"{FIRST_PRUNE_MODEL_BASE_NAME}_fold_{fold_id}.pth"
        )
        assert pruned_pth.exists(), f"Model {model_dir.name} not pruned!"

        model = construct_model(eval_cfg)
        assert isinstance(model, NeuralDNFEO)

        model = model.to_ndnf()
        model.to(device)
        model.eval()
        model_state = torch.load(
            pruned_pth, map_location=device, weights_only=True
        )
        model.load_state_dict(model_state)

        # Data loaders
        train_loader, val_loader = get_zoo_dataloaders(
            dataset=dataset,
            train_index=train_index,
            test_index=test_index,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            loader_num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
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
    run_eval()
