import logging
from pathlib import Path
import sys
import traceback

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from utils import post_to_discord_webhook

from zoo.data_utils_zoo import get_zoo_data_np_from_path

DEFAULT_LAS_OUTPUT_DIR = file.parent / "las_output"
if not DEFAULT_LAS_OUTPUT_DIR.exists():
    # create the dir
    DEFAULT_LAS_OUTPUT_DIR.mkdir()
LAS_FILE_PREFIX = "zoo"

log = logging.getLogger()


def gen_las_example_and_background(cfg: DictConfig) -> None:
    log.info("Generating LAS example and background files...")
    X, y, _ = get_zoo_data_np_from_path(Path(cfg["dataset"]["save_dir"]))
    num_classes = len(np.unique(y))

    skf = StratifiedKFold(
        n_splits=cfg["las"]["k_folds"],
        shuffle=True,
        random_state=cfg["las"]["seed"],
    )

    for fold_id, (train_idx, _) in enumerate(skf.split(X, y)):
        train_X, train_y = X[train_idx], y[train_idx]

        log.info(
            f"Fold {fold_id} - total number of training examples: {train_X.shape[0]}"
        )
        train_data_with_label = np.column_stack((train_X, train_y))

        output_source = cfg["las"]["output_source"]
        if output_source == "stdout":
            example_file = sys.stdout
        else:
            example_file = open(
                DEFAULT_LAS_OUTPUT_DIR
                / f"{LAS_FILE_PREFIX}_examples_f{fold_id}.las",
                "w",
            )

        # Examples
        for sample_id, sample in enumerate(train_data_with_label):
            class_label = sample[-1]
            # Penalty and inclusion set
            print(
                f"#pos(eg_{sample_id}@{10}, {{ class({(int(class_label))}) }}, {{ ",
                file=example_file,
            )

            # Exclusion set
            print(
                ",\n".join(
                    filter(
                        lambda j: j != "",
                        map(
                            lambda k: (
                                f"    class({k})" if k != class_label else ""
                            ),
                            range(num_classes),
                        ),
                    )
                ),
                file=example_file,
            )
            print("}, {", file=example_file)

            # Context
            for i, a in enumerate(sample[:-1]):
                if a == 1:
                    print(f"    has_attr_{i}.", file=example_file)
            print("}).\n", file=example_file)

        if output_source != "stdout":
            example_file.close()

        # Background knowledge
        is_ilasp = cfg["las"]["is_ilasp"]
        if output_source == "stdout":
            bk_file = sys.stdout
        else:
            bk_file_name = (
                f"{LAS_FILE_PREFIX}_bk_ilasp_f{fold_id}.las"
                if is_ilasp
                else f"{LAS_FILE_PREFIX}_bk_f{fold_id}.las"
            )
            bk_file = open(DEFAULT_LAS_OUTPUT_DIR / bk_file_name, "w")

        print(f"class_id(0..{num_classes -1}).", file=bk_file)
        print(":- class(X),  class(Y),  X < Y.", file=bk_file)
        print("#modeh(class(const(class_id))).", file=bk_file)
        for i in range(train_X.shape[1]):
            print(f"#modeb(has_attr_{i}).", file=bk_file)
            if not is_ilasp:
                # FastLas requires explicit 'not' to include in hypothesis space
                print(f"#modeb(not has_attr_{i}).", file=bk_file)
        if not is_ilasp:
            # FastLas scoring function
            print('#bias("penalty(1, head).").', file=bk_file)
            print('#bias("penalty(1, body(X)) :- in_body(X).").', file=bk_file)

        if output_source != "stdout":
            bk_file.close()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_las_gen(cfg: DictConfig) -> None:
    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        gen_las_example_and_background(cfg)
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
                experiment_name=f"Mushroom LAS Gen",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_las_gen()
