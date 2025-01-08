import logging
from pathlib import Path
import re
import sys

import clingo
import numpy as np
import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import Meter, AccuracyMeter, JaccardScoreMeter, ErrorMeter


log = logging.getLogger()


ASP_TRANSLATION_THRESHOLDED_JSON_BASE_NAME = "asp_translation_thresholded"
ASP_TRANSLATION_DISENTANGLED_JSON_BASE_NAME = "asp_translation_disentangled"


def asp_eval(
    test_data: np.ndarray,
    rules: list[str],
    num_classes: int,
    format_options: dict[str, str] = {},
    debug: bool = False,
) -> dict[str, Meter]:
    predictions = []
    ground_truth = []

    input_name = format_options.get("input_name", "a")
    input_syntax = format_options.get("input_syntax", "PRED")
    disjunction_name = format_options.get("disjunction_name", "disj")
    disjunction_syntax = format_options.get("disjunction_syntax", "PRED")

    # Evaluate each data item
    # Assume each row of the test_data is first the attributes present and then
    # the label
    for d in test_data:
        asp_base = []
        for i, a in enumerate(d[:-1]):
            if a == 1:
                asp_base.append(
                    f"{input_name}_{i}."
                    if input_syntax == "PRED"
                    else f"{input_name}({i})"
                )

        asp_base += rules
        if disjunction_syntax == "PRED":
            asp_base += [
                f"#show {disjunction_name}_{i}/0." for i in range(num_classes)
            ]
        else:
            asp_base.append(f"#show {disjunction_name}/1.")

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], " ".join(asp_base))
        ctl.ground([("base", [])])

        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

        ground_truth.append(d[-1])

        if debug:
            log.info(f"ground truth: {d[-1]}\tAS: {all_answer_sets}")

        if len(all_answer_sets) != 1:
            # No model or multiple answer sets, should not happen
            log.warning(
                f"No model or multiple answer sets when evaluating rules."
            )
            continue

        # Use regex to extract the output classes number
        pattern_string = (
            f"{disjunction_name}_(\d+)"
            if disjunction_syntax == "PRED"
            else f"{disjunction_name}\((\d+)\)"
        )
        pattern = re.compile(pattern_string)
        output_classes = [int(i) for i in pattern.findall(all_answer_sets[0])]
        predictions.append(output_classes)

    # Convert to tensors
    prediction_tensor = torch.zeros(
        len(predictions), num_classes, dtype=torch.float32
    )
    for i, p in enumerate(predictions):
        prediction_tensor[i, p] = 1.0
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.long)

    jacc_meter = JaccardScoreMeter()
    acc_meter = AccuracyMeter()
    err_meter = ErrorMeter()

    jacc_meter.update(prediction_tensor, ground_truth_tensor)
    acc_meter.update(prediction_tensor, ground_truth_tensor)
    err_meter.update(prediction_tensor, ground_truth_tensor)

    return {
        "acc_meter": acc_meter,
        "jacc_meter": jacc_meter,
        "error_meter": err_meter,
    }
