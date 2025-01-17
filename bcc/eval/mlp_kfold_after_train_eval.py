"""
This script evaluates a BCC NeuralDNF model. The input models are strictly after
training and without any post-training processing. The evaluation metrics
include accuracy, precision, recall, and F1 score.
"""

import json
import logging
from pathlib import Path
import random
import sys
import traceback

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import synthesize
from data_utils import GenericUCIDataset
from utils import post_to_discord_webhook

from bcc.data_utils_bcc import get_bcc_data
from bcc.eval.eval_common import (
    bcc_classifier_eval,
    parse_eval_return_meters_with_logging,
    DEFAULT_GEN_SEED,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    AFTER_TRAIN_MODEL_BASE_NAME,
)
from bcc.models import BCCMLP


log = logging.getLogger()


def post_train_prune(cfg: DictConfig) -> None:
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
    X, y, _ = get_bcc_data(eval_cfg["standardise"])
    bcc_dataset = GenericUCIDataset(X, y)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=eval_cfg["k_folds"],
        shuffle=True,
        random_state=eval_cfg["seed"],
    )

    ret_dicts: list[dict[str, float]] = []
    for fold_id, (_, test_index) in enumerate(skf.split(X, y)):
        log.info(f"Fold {fold_id} starts")
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"]) / run_dir_name / f"fold_{fold_id}"
        )
        model = BCCMLP(num_features=bcc_dataset.X.shape[1])
        model.to(device)
        model_state = torch.load(
            model_dir / f"{AFTER_TRAIN_MODEL_BASE_NAME}_fold_{fold_id}.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(model_state)
        model.eval()

        # Data loaders
        val_loader = torch.utils.data.DataLoader(
            bcc_dataset,
            batch_size=DEFAULT_LOADER_BATCH_SIZE,
            num_workers=DEFAULT_LOADER_NUM_WORKERS,
            pin_memory=device == torch.device("cuda"),
            sampler=torch.utils.data.SubsetRandomSampler(test_index),  # type: ignore
        )

        log.info(f"Experiment {model_dir.name} loaded!")
        eval_log = parse_eval_return_meters_with_logging(
            bcc_classifier_eval(model, device, val_loader), "BCC-MLP"
        )
        ret_dicts.append(eval_log)

        with open(
            model_dir / f"fold_{fold_id}_after_train_eval_result.json", "w"
        ) as f:
            json.dump(eval_log, f, indent=4)

        log.info("============================================================")

    # Synthesize the results
    relevant_keys = list(ret_dicts[0].keys())
    return_dict = {}
    for k in relevant_keys:
        for sk, sv in synthesize(np.array([d[k] for d in ret_dicts])).items():
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
        post_train_prune(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Kfold After Train Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
