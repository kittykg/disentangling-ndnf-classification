"""
This script disentangle pruned plain NDNF. The input NDNF models are strictly
after pruning stage in the post-training processing pipeline. The disentangled
NDNF model are stored and evaluated, with the relevant information stored in a
json. The evaluation metrics include accuracy, sample Jaccard, macro Jaccard,
and error metrics.
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
from torch.utils.data import DataLoader

from neural_dnf import NeuralDNF, NeuralDNFEO
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

from analysis import synthesize
from utils import construct_ndnf_based_model, post_to_discord_webhook
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
from zoo.eval.ndnf_eo_kfold_prune import multiround_prune


log = logging.getLogger()


def single_model_disentangle(
    fold_id: int,
    model: NeuralDNF,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_dir: Path,
) -> dict[str, Any]:
    def _eval_with_log_wrapper(
        model: NeuralDNF, model_name: str
    ) -> dict[str, Any]:
        eval_meters = ndnf_based_model_eval(model, device, val_loader)
        return parse_eval_return_meters_with_logging(eval_meters, model_name)

    # Stage 1: Evaluate the pruned model
    _eval_with_log_wrapper(model, "Pruned NDNF model")
    log.info("------------------------------------------")

    # Stage 2: Disentangle the pruned model
    def disentangle(model: NeuralDNF) -> dict[str, Any]:
        log.info("Disentangling the model...")

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
                    # positive form, always add the sign as 1
                    acc_pairs.append((non_zero_count, 1, c))

            # Arrange the combinations from negation to normal, and from most
            # general (more 0-weights) to most specific (less 0-weights)
            acc_pairs.sort(key=lambda x: (x[1], x[0]))
            new_disj_conj_mapping[disj_id] = [
                {"sign": x[1], "conj_id": i + len(new_conj_list)}
                for i, x in enumerate(acc_pairs)
            ]
            new_conj_list.extend([x[2] for x in acc_pairs])

        new_conj_w = torch.stack(new_conj_list)

        # We need to update the disjunction weights
        new_disj_w = torch.zeros((disj_w.shape[0], len(new_conj_list)))
        for disj_id, vs in new_disj_conj_mapping.items():
            for v in vs:
                sign = v["sign"]
                conj_id = v["conj_id"]
                new_disj_w[disj_id][conj_id] = sign * 6

        # Create an intermediate model
        intermediate_model = NeuralDNF(
            conj_w.shape[1], len(new_conj_list), disj_w.shape[0], 1.0
        )
        intermediate_model.conjunctions.weights.data = new_conj_w.clone()
        intermediate_model.disjunctions.weights.data = new_disj_w.clone()
        intermediate_model.to(device)
        intermediate_model.eval()
        _eval_with_log_wrapper(
            intermediate_model,
            "Disentangled NDNF model (intermediate, no prune)",
        )
        log.info("------------------------------------------")

        # Prune the intermediate disentangled model
        multiround_prune(
            intermediate_model,
            device,
            train_loader,
            lambda x: _eval_with_log_wrapper(
                intermediate_model, x["model_name"]
            ),
        )
        pruned_inter_model_log = _eval_with_log_wrapper(
            intermediate_model, "Disentangled NDNF model (intermediate, pruned)"
        )
        log.info("------------------------------------------")

        # Condense the model
        intermediate_model.to(torch.device("cpu"))
        condensed_model = condense_neural_dnf_model(intermediate_model)
        condensed_model.to(device)
        condensed_model.eval()
        condensed_model_log = _eval_with_log_wrapper(
            condensed_model, "Condensed disentangled NDNF model"
        )

        return {
            "intermediate_model": intermediate_model,
            "intermediate_model_log": pruned_inter_model_log,
            "condensed_model": condensed_model,
            "condensed_model_log": condensed_model_log,
        }

    # Check for checkpoints
    model_path = (
        model_dir / f"{DISENTANGLED_MODEL_BASE_NAME}_fold_{fold_id}.pth"
    )
    disentangle_result_json = (
        model_dir / f"fold_{fold_id}_{DISENTANGLED_RESULT_JSON_BASE_NAME}.json"
    )
    if model_path.exists() and disentangle_result_json.exists():
        # The model has been disentangled, pruned and condensed
        with open(disentangle_result_json, "r") as f:
            stats = json.load(f)

        disentangled_model = NeuralDNF(
            stats["disentangled_model_n_in"],
            stats["disentangled_model_n_conjunctions"],
            stats["disentangled_model_n_out"],
            1.0,
        )
        disentangled_model.to(device)
        disentangled_state = torch.load(
            model_path, map_location=device, weights_only=True
        )
        disentangled_model.load_state_dict(disentangled_state)
        disentangled_model.eval()
        condensed_model_log = _eval_with_log_wrapper(
            disentangled_model, "Condensed disentangled NDNF model"
        )
    else:
        ret = disentangle(model)
        disentangled_model: NeuralDNF = ret["condensed_model"]
        intermediate_model: NeuralDNF = ret["intermediate_model"]
        condensed_model_log = ret["condensed_model_log"]

        torch.save(disentangled_model.state_dict(), model_path)
        torch.save(
            intermediate_model.state_dict(),
            model_dir
            / f"{INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME}_fold_{fold_id}.pth",
        )

        disentanglement_result = {
            "disentangled_model_n_in": disentangled_model.conjunctions.in_features,
            "disentangled_model_n_conjunctions": disentangled_model.conjunctions.out_features,
            "disentangled_model_n_out": disentangled_model.disjunctions.out_features,
            "intermediate_model_n_conjunctions": intermediate_model.conjunctions.out_features,
            "intermediate_model_log": ret["intermediate_model_log"],
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
    X, y, feature_names = get_zoo_data_np_from_path(
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

        model = construct_ndnf_based_model(eval_cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Kfold Disentangle",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
