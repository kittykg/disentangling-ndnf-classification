from collections import Counter, OrderedDict
from typing import Any, Callable

from sklearn.metrics import accuracy_score, jaccard_score
import numpy as np
import torch
from torch import Tensor


def collate(result_dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Collate a list of dictionaries into a dictionary of lists.
    """
    return_dict: dict[str, list[Any]] = {}
    for key in result_dicts[0].keys():
        return_dict[key] = [d[key] for d in result_dicts]
    return return_dict


def synthesize(array, compute_ste: bool = True) -> OrderedDict[str, float]:
    d = OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    if compute_ste:
        d["ste"] = np.std(array) / np.sqrt(len(array))
    return d


class Meter:
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_average(self) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, float]:
        raise NotImplementedError


class MetricValueMeter(Meter):
    metric_name: str
    vals: list[float]

    def __init__(self, metric_name: str):
        super(MetricValueMeter, self).__init__()
        self.metric_name = metric_name
        self.vals = []

    def update(self, val: float) -> None:
        self.vals.append(val)

    def get_average(self) -> float:
        return float(np.mean(self.vals))

    def reset(self) -> None:
        self.vals = []

    def to_dict(self) -> dict[str, float]:
        return {
            f"{self.metric_name}_mean": self.get_average(),
        }


class AccuracyMeter(Meter):
    outputs: torch.Tensor
    targets: torch.Tensor

    output_to_prediction_fn: Callable[[Tensor], Tensor]

    def __init__(
        self,
        output_to_prediction_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super(AccuracyMeter, self).__init__()
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

        if output_to_prediction_fn is None:
            self.output_to_prediction_fn = lambda y_hat: torch.max(
                y_hat, 1
            ).indices
        else:
            self.output_to_prediction_fn = output_to_prediction_fn

    def update(self, output: Tensor, target: Tensor) -> None:
        """
        Accumulate the output and target. The output will be converted using
        `output_tensor_to_prediction_fn`.
        """
        y_pred = self.output_to_prediction_fn(output)
        self.targets = torch.cat([self.targets, target.detach().cpu()], dim=0)
        self.outputs = torch.cat([self.outputs, y_pred.detach().cpu()], dim=0)

    def get_average(self) -> float:
        return float(accuracy_score(self.targets.int(), self.outputs.int()))

    def reset(self) -> None:
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

    def to_dict(self) -> dict[str, float]:
        return {"accuracy": self.get_average()}


class JaccardScoreMeter(Meter):
    outputs: torch.Tensor
    targets: torch.Tensor

    def __init__(self) -> None:
        super(JaccardScoreMeter, self).__init__()
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

    def update(self, output: Tensor, target: Tensor) -> None:
        """
        Accumulate the output and target. The output should be a binary tensor
        with 1s in the predicted classes (N x C). Target should be a multi-class
        tensor of correct class indices (shape N). The target tensor will be
        converted to one-hot encoding (N x C).
        """
        y = torch.zeros(output.shape, dtype=torch.long)
        y[range(output.shape[0]), target.long()] = 1
        self.targets = torch.cat([self.targets, y.detach().cpu()], dim=0)
        self.outputs = torch.cat([self.outputs, output.detach().cpu()], dim=0)

    def get_average(self, average="samples") -> float | np.ndarray:
        return jaccard_score(self.targets, self.outputs, average=average)  # type: ignore

    def reset(self) -> None:
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

    def to_dict(self) -> dict[str, float]:
        return {
            "samples_jaccard": self.get_average("samples"),  # type: ignore
            "macro_jaccard": self.get_average("macro"),  # type: ignore
        }


class ErrorMeter(Meter):
    targets: list[int]
    missing_predictions: dict[int, list[int]]
    multiple_predictions: dict[int, list[int]]
    wrong_predictions: dict[int, list[int]]

    def __init__(self):
        super(ErrorMeter, self).__init__()
        self.targets = []
        self.missing_predictions = {}
        self.multiple_predictions = {}
        self.wrong_predictions = {}

    def update(self, output: Tensor, target: Tensor) -> None:
        target_list = target.detach().cpu().numpy().tolist()
        initial_len = len(self.targets)

        for i, target_class in enumerate(target_list):
            sample_id = initial_len + i
            tanh_prediction = torch.where(output[i].cpu() > 0)[0]

            if len(tanh_prediction) == 0:
                if target_class not in self.missing_predictions:
                    self.missing_predictions[target_class] = []
                self.missing_predictions[target_class].append(sample_id)
            elif len(tanh_prediction) > 1:
                if target_class not in self.multiple_predictions:
                    self.multiple_predictions[target_class] = []
                self.multiple_predictions[target_class].append(sample_id)
            else:
                tanh_prediction = tanh_prediction.item()
                if tanh_prediction != target_class:
                    if target_class not in self.wrong_predictions:
                        self.wrong_predictions[target_class] = []
                    self.wrong_predictions[target_class].append(sample_id)

        self.targets += target_list

    def get_average(self) -> dict[str, dict[int, Any] | int | float]:
        return_dict: dict[str, dict[int, Any] | int | float] = {
            "missing_predictions": self.missing_predictions,
            "multiple_predictions": self.multiple_predictions,
            "wrong_predictions": self.wrong_predictions,
        }

        class_counts = Counter(self.targets)
        num_samples = len(self.targets)

        overall_error_count = 0
        overall_error_class_count = 0

        for dict_name, d in zip(
            ["missing", "multiple", "wrong"],
            [
                self.missing_predictions,
                self.multiple_predictions,
                self.wrong_predictions,
            ],
        ):
            error_counts: dict[int, int] = {}
            error_rates: dict[int, float] = {}
            type_overall_error_count = 0

            for errored_class, samples in d.items():
                error_count = len(samples)
                type_overall_error_count += error_count
                error_counts[errored_class] = error_count
                error_rates[errored_class] = (
                    error_count / class_counts[errored_class]
                )

            return_dict[f"{dict_name}_class_error_count_dict"] = error_counts
            return_dict[f"{dict_name}_class_error_rate_dict"] = error_rates

            return_dict[f"{dict_name}_error_class_count"] = len(d)
            return_dict[f"{dict_name}_overall_error_count"] = (
                type_overall_error_count
            )

            overall_error_count += type_overall_error_count
            overall_error_class_count += len(d)

        return_dict["overall_error_count"] = overall_error_count
        return_dict["overall_error_rate"] = overall_error_count / num_samples
        return_dict["overall_error_class_count"] = overall_error_class_count

        return return_dict

    def reset(self) -> None:
        self.targets = []
        self.missing_predictions = {}
        self.multiple_predictions = {}
        self.wrong_predictions = {}

    def to_dict(self) -> dict[str, dict[int, Any] | int | float]:
        return self.get_average()
