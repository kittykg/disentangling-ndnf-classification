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

from analysis import Meter, AccuracyMeter
from bcc.models import BCCClassifier, BCCNeuralDNF

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


def bcc_classifier_eval(
    model: BCCClassifier,
    device: torch.device,
    data_loader: DataLoader,
    do_logging: bool = False,
) -> dict[str, Meter]:
    model.eval()
    if isinstance(model, BCCNeuralDNF):
        acc_meter_conversion_fn = lambda y_hat: y_hat > 0.5
    else:
        acc_meter_conversion_fn = lambda y_hat: torch.sigmoid(y_hat) > 0.5

    acc_meter = AccuracyMeter(output_to_prediction_fn=acc_meter_conversion_fn)
    if do_logging:
        iter_acc_meter = AccuracyMeter(
            output_to_prediction_fn=acc_meter_conversion_fn
        )

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            x = data[0].to(device)
            y = data[1].to(device)

            y_hat = model(x).squeeze()

        if isinstance(model, BCCNeuralDNF):
            # For NeuralDNF, we need to take the tanh of the logit and
            # then scale it to (0, 1)
            y_hat = (torch.tanh(y_hat) + 1) / 2

        acc_meter.update(y_hat, y)

        if do_logging:
            iter_acc_meter.update(y_hat, y)

            iter_acc = iter_acc_meter.get_average()
            other_metrics = iter_acc_meter.get_other_classification_metrics()
            iter_precision = other_metrics["precision"]
            iter_recall = other_metrics["recall"]
            iter_f1 = other_metrics["f1"]

            log.info(
                f"[{i + 1:3d}] Test -- Acc: {iter_acc:.3f} -- "
                f"Precision: {iter_precision:.3f} -- "
                f"Recall: {iter_recall:.3f} -- "
                f"F1: {iter_f1:.3f}"
            )

            iter_acc_meter.reset()

    if do_logging:
        acc = acc_meter.get_average()
        other_metrics = acc_meter.get_other_classification_metrics()
        precision = other_metrics["precision"]
        recall = other_metrics["recall"]
        f1 = other_metrics["f1"]

        log.info(
            f"Overall Test -- Acc: {acc:.3f} -- "
            f"Precision: {precision:.3f} -- "
            f"Recall: {recall:.3f} -- "
            f"F1: {f1:.3f}"
        )

    return {
        "acc_meter": acc_meter,
    }


def parse_eval_return_meters_with_logging(
    eval_meters: dict[str, Meter],
    model_name: str,
    do_logging: bool = True,
) -> dict[str, Any]:
    acc_meter = eval_meters["acc_meter"]
    assert isinstance(acc_meter, AccuracyMeter)

    accuracy = acc_meter.get_average()
    other_metrics = acc_meter.get_other_classification_metrics()
    precision = other_metrics["precision"]
    recall = other_metrics["recall"]
    f1 = other_metrics["f1"]

    return_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    log_info_str = (
        f"{model_name}\n\tAccuracy: {accuracy:.3f}\n"
        f"\tPrecision: {precision:.3f}\n\tRecall: {recall:.3f}\n\tF1: {f1:.3f}"
    )

    if do_logging:
        log.info(log_info_str)

    return return_dict
