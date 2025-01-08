import logging
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from neural_dnf.neural_dnf import BaseNeuralDNF

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import Meter, AccuracyMeter, JaccardScoreMeter, ErrorMeter
from zoo.data_utils_zoo import get_x_and_y_zoo


log = logging.getLogger()


DEFAULT_GEN_SEED = 2
DEFAULT_LOADER_BATCH_SIZE = 64
DEFAULT_LOADER_NUM_WORKERS = 0

AFTER_TRAIN_MODEL_BASE_NAME = "model"
FIRST_PRUNE_MODEL_BASE_NAME = "model_mr_pruned"
THRESHOLD_MODEL_BASE_NAME = "model_thresholded"
THRESHOLD_RESULT_JSON_BASE_NAME = "threshold_result"
INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME = "intermediate_model_disentangled"
DISENTANGLED_MODEL_BASE_NAME = "model_disentangled"
DISENTANGLED_RESULT_JSON_BASE_NAME = "disentangled_result"


def ndnf_based_model_eval(
    model: BaseNeuralDNF,
    device: torch.device,
    data_loader: DataLoader,
    do_logging: bool = False,
) -> dict[str, Meter]:
    model.eval()
    jacc_meter = JaccardScoreMeter()
    acc_meter = AccuracyMeter()
    error_meter = ErrorMeter()
    iter_jacc_meter = JaccardScoreMeter()
    iter_acc_meter = AccuracyMeter()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            x, y = get_x_and_y_zoo(data, device, use_ndnf=True)
            y_hat = model(x)

        iter_acc_meter.update(y_hat, y)
        acc_meter.update(y_hat, y)

        # To get the jaccard score, we need to threshold the tanh activation
        # to get the binary prediction of each class
        y_hat = (torch.tanh(y_hat) > 0).long()
        iter_jacc_meter.update(y_hat, y)
        jacc_meter.update(y_hat, y)
        error_meter.update(y_hat, y)

        if do_logging:
            log.info(
                "[%3d] Test -- avg acc: %.3f -- avg jacc: %.3f"
                % (
                    i + 1,
                    iter_acc_meter.get_average(),
                    iter_jacc_meter.get_average(),
                )
            )

        iter_acc_meter.reset()
        iter_jacc_meter.reset()

    if do_logging:
        log.info(
            "Overall Test -- avg acc: %.3f -- avg jacc: %.3f"
            % (acc_meter.get_average(), jacc_meter.get_average())
        )

    return {
        "acc_meter": acc_meter,
        "jacc_meter": jacc_meter,
        "error_meter": error_meter,
    }


def parse_eval_return_meters_with_logging(
    eval_meters: dict[str, Meter],
    model_name: str,
    check_error_meter: bool = True,
    do_logging: bool = True,
) -> dict[str, Any]:
    return_dict = {"accuracy": eval_meters["acc_meter"].get_average()}
    jacc_meter = eval_meters["jacc_meter"]
    return_dict["sample_jaccard"] = jacc_meter.get_average("samples")  # type: ignore
    return_dict["macro_jaccard"] = jacc_meter.get_average("macro")  # type: ignore
    log_info_str = (
        f"{model_name}\n"
        f"\tAccuracy: {return_dict['accuracy']:.3f}\n"
        f"\tSample Jaccard: {return_dict['sample_jaccard']:.3f}\n"
        f"\tMacro Jaccard: {return_dict['macro_jaccard']:.3f}"
    )

    if check_error_meter:
        error_dict = eval_meters["error_meter"].get_average()
        assert isinstance(error_dict, dict)
        overall_error_rate = error_dict["overall_error_rate"]
        overall_error_class_count = error_dict["overall_error_class_count"]
        return_dict["error_dict"] = error_dict
        log_info_str += (
            f"\n\tOverall error rate: {overall_error_rate:.3f}\n"
            f"\tOverall error class count: {overall_error_class_count}"
        )

    if do_logging:
        log.info(log_info_str)

    return return_dict
