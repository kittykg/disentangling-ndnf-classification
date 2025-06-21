"""
This script evaluates a CoverTypeClassifier model. The input models are strictly
after training and without any post-training processing (if applicable). The
evaluation metrics include accuracy, sample Jaccard and macro Jaccard.
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
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_LOADER_NUM_WORKERS,
    AFTER_TRAIN_MODEL_BASE_NAME,
)
from covertype.models import (
    CoverTypeThresholdPINeuralDNFEO,
    CoverTypeThresholdPINeuralDNFMT,
    CoverTypeMLPPINeuralDNFEO,
    CoverTypeMLPPINeuralDNFMT,
    CoverTypeBaseNeuralDNF,
    construct_model,
)


log = logging.getLogger()


def multirun_after_train_eval(cfg: DictConfig) -> None:
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

    single_eval_results: list[dict[str, Any]] = []
    for s in eval_cfg["multirun_seeds"]:
        # Load model
        model_dir = (
            Path(eval_cfg["storage_dir"])
            / caps_experiment_name
            / f"{caps_experiment_name}-{s}"
        )
        model = construct_model(eval_cfg)
        model.to(device)
        model_state = torch.load(
            model_dir / f"{AFTER_TRAIN_MODEL_BASE_NAME}.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(model_state)
        if isinstance(
            model,
            (
                CoverTypeThresholdPINeuralDNFEO,
                CoverTypeThresholdPINeuralDNFMT,
                CoverTypeMLPPINeuralDNFEO,
                CoverTypeMLPPINeuralDNFMT,
            ),
        ):
            model = model.to_ndnf_model()
            model.to(device)
        model.eval()

        log.info(f"Evaluation of {experiment_name}_{s} starts")

        raw_eval_dict = covertype_classifier_eval(model, device, test_loader)
        single_eval_results.append(
            parse_eval_return_meters_with_logging(
                raw_eval_dict,
                f"CoverType-{'NDNF' if isinstance(model, CoverTypeBaseNeuralDNF) else 'MLP'}",
                do_logging=False,
            )
        )
        eval_log = parse_eval_return_meters_with_logging(
            raw_eval_dict,
            f"CoverType-{'NDNF' if isinstance(model, CoverTypeBaseNeuralDNF) else 'MLP'}",
            do_logging=True,
        )

        with open(model_dir / f"after_train_eval_result.json", "w") as f:
            json.dump(eval_log, f, indent=4)

        log.info("============================================================")

    # Synthesize the results
    return_dict = {}
    for k in EVAL_RELEVANT_KEYS:
        synth_dict = synthesize(np.array([d[k] for d in single_eval_results]))
        for sk, sv in synth_dict.items():
            return_dict[f"{k}/{sk}"] = sv
    for k in EVAL_ERROR_DICT_RELEVANT_KEYS:
        synth_dict = synthesize(
            np.array([d["error_dict"][k] for d in single_eval_results])
        )
        for sk, sv in synth_dict.items():
            return_dict[f"error_dict/{k}/{sk}"] = sv

    log.info("Synthesized results:")
    for k, v in return_dict.items():
        log.info(f"{k}: {v:.3f}")

    with open("after_train_eval_synthesized_result.json", "w") as f:
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
        multirun_after_train_eval(cfg)
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
                experiment_name=f"{cfg['eval']['experiment_name']} Multirun After Train Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
