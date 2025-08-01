import logging
from pathlib import Path
import sys
from typing import Any

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

from analysis import Meter, AccuracyMeter, JaccardScoreMeter, ErrorMeter
from car.data_utils_car import get_x_and_y_car
from car.models import CarClassifier, CarBaseNeuralDNF, CAR_NUM_CLASSES


log = logging.getLogger()


DEFAULT_GEN_SEED = 2
DEFAULT_LOADER_BATCH_SIZE = 256
DEFAULT_LOADER_NUM_WORKERS = 0

AFTER_TRAIN_MODEL_BASE_NAME = "model"
FIRST_PRUNE_MODEL_BASE_NAME = "model_mr_pruned"
THRESHOLD_MODEL_BASE_NAME = "model_thresholded"
THRESHOLD_RESULT_JSON_BASE_NAME = "threshold_result"
INTERMEDIATE_DISENTANGLED_MODEL_BASE_NAME = "intermediate_model_disentangled"
DISENTANGLED_MODEL_BASE_NAME = "model_disentangled"
DISENTANGLED_RESULT_JSON_BASE_NAME = "disentangled_result"


EVAL_RELEVANT_KEYS = ["accuracy", "sample_jaccard", "macro_jaccard", "f1"]
EVAL_ERROR_DICT_RELEVANT_KEYS = [
    "missing_error_class_count",
    "missing_overall_error_count",
    "multiple_error_class_count",
    "multiple_overall_error_count",
    "wrong_error_class_count",
    "wrong_overall_error_count",
    "overall_error_count",
    "overall_error_rate",
    "overall_error_class_count",
]


def car_classifier_eval(
    model: CarClassifier,
    device: torch.device,
    data_loader: DataLoader,
    do_logging: bool = False,
) -> dict[str, Meter]:
    model.eval()
    is_ndnf = isinstance(model, CarBaseNeuralDNF)

    jacc_meter = JaccardScoreMeter()
    acc_meter = AccuracyMeter()
    error_meter = ErrorMeter()

    if do_logging:
        iter_jacc_meter = JaccardScoreMeter()
        iter_acc_meter = AccuracyMeter()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            x, y = get_x_and_y_car(data, device, use_ndnf=is_ndnf)
            y_hat = model(x)

        if is_ndnf:
            # To get the jaccard score, we need to threshold the tanh activation
            # to get the binary prediction of each class
            y_hat_prime = (torch.tanh(y_hat) > 0).long()
        else:
            # To get the jaccard score for MLP, we need to take the argmax
            argmax_y_hat = torch.argmax(y_hat, dim=1)
            y_hat_prime = torch.zeros(len(y), CAR_NUM_CLASSES).long()
            y_hat_prime[range(len(y)), argmax_y_hat] = 1

        acc_meter.update(y_hat, y)
        jacc_meter.update(y_hat_prime, y)
        error_meter.update(y_hat_prime, y)

        if do_logging:
            iter_acc_meter.update(y_hat, y)
            iter_jacc_meter.update(y_hat_prime, y)
            log.info(
                "[%3d] Test -- acc: %.3f -- jacc: %.3f -- weighted f1: %.3f"
                % (
                    i + 1,
                    iter_acc_meter.get_average(),
                    iter_jacc_meter.get_average(),
                    iter_acc_meter.get_other_classification_metrics("weighted")[
                        "f1"
                    ],
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
    filter_out_list: bool = False,  # enable this when logging into wandb
) -> dict[str, Any]:
    acc_meter = eval_meters["acc_meter"]
    assert isinstance(acc_meter, AccuracyMeter)
    return_dict = {
        "accuracy": acc_meter.get_average(),
        "f1": acc_meter.get_other_classification_metrics("weighted")["f1"],
    }

    jacc_meter = eval_meters["jacc_meter"]
    assert isinstance(jacc_meter, JaccardScoreMeter)
    sample_jacc = jacc_meter.get_average("samples")
    macro_jacc = jacc_meter.get_average("macro")
    assert isinstance(sample_jacc, float)
    assert isinstance(macro_jacc, float)
    return_dict["sample_jaccard"] = sample_jacc
    return_dict["macro_jaccard"] = macro_jacc

    log_info_str = (
        f"{model_name}\n"
        f"\tAccuracy: {return_dict['accuracy']:.3f}\n"
        f"\tF1: {return_dict['f1']:.3f}\n"
        f"\tSample Jaccard: {return_dict['sample_jaccard']:.3f}\n"
        f"\tMacro Jaccard: {return_dict['macro_jaccard']:.3f}"
    )

    if check_error_meter:
        error_dict = eval_meters["error_meter"].get_average()
        assert isinstance(error_dict, dict)
        overall_error_rate = error_dict["overall_error_rate"]
        overall_error_class_count = error_dict["overall_error_class_count"]
        if not filter_out_list:
            return_dict["error_dict"] = error_dict
        else:
            return_dict["overall_error_rate"] = overall_error_rate
            return_dict["overall_error_class_count"] = overall_error_class_count
        log_info_str += (
            f"\n\tOverall error rate: {overall_error_rate:.3f}\n"
            f"\tOverall error class count: {overall_error_class_count}"
        )

    if do_logging:
        log.info(log_info_str)

    return return_dict
