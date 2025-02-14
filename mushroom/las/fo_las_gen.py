import logging
from pathlib import Path
import sys
import traceback

import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parent.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from utils import post_to_discord_webhook


DEFAULT_LAS_OUTPUT_DIR = file.parent / "las_output"
if not DEFAULT_LAS_OUTPUT_DIR.exists():
    # create the dir
    DEFAULT_LAS_OUTPUT_DIR.mkdir()

log = logging.getLogger()


def gen_las_example_and_background(cfg: DictConfig) -> None:
    log.info("Generating LAS example and background files...")
    uci_dataset = fetch_ucirepo(id=cfg["dataset"]["ucimlrepo_id"])
    # data (as pandas dataframes)
    X: pd.DataFrame = uci_dataset.data.features  # type: ignore
    y: pd.DataFrame = uci_dataset.data.targets  # type: ignore

    assert cfg["dataset"]["hold_out"][
        "create_hold_out"
    ], "Holdout test set is required"
    train_X, _, train_y, _ = train_test_split(
        X,
        y,
        test_size=cfg["dataset"]["hold_out"]["test_size"],
        random_state=cfg["dataset"]["hold_out"]["random_state"],
    )

    if cfg["las"]["subset"]["use_subset"]:
        train_X, _, train_y, _ = train_test_split(
            train_X,
            train_y,
            train_size=cfg["las"]["subset"]["subset_proportion"],
            random_state=cfg["las"]["subset"]["subset_random_state"],
            stratify=train_y,
        )

    log.info(f"Total number of training examples: {train_X.shape[0]}")

    output_source = cfg["las"]["output_source"]
    if output_source == "stdout":
        example_file = sys.stdout
    else:
        example_file = open(
            DEFAULT_LAS_OUTPUT_DIR / "mushroom_fo_examples.las", "w"
        )

    # Examples
    for i in range(train_X.shape[0]):
        target = train_y.values[i][0]
        exclusion = "e" if target == "p" else "p"

        print(
            f"#pos(eg{i}@1, {{ {target} }}, {{ {exclusion} }}, {{",
            file=example_file,
        )
        for j in range(train_X.shape[1]):
            col_name = train_X.columns[j].replace("-", "_")
            feature_name = train_X.iloc[i, j]
            print(
                f"has_{col_name}({col_name}_{feature_name}).", file=example_file
            )
        print("}).\n", file=example_file)

    if output_source != "stdout":
        example_file.close()

    # Background knowledge
    is_ilasp = cfg["las"]["is_ilasp"]
    if output_source == "stdout":
        bk_file = sys.stdout
    else:
        bk_file_name = (
            "mushroom_fo_bk_ilasp.las" if is_ilasp else "mushroom_fo_bk.las"
        )
        bk_file = open(DEFAULT_LAS_OUTPUT_DIR / bk_file_name, "w")

    # Implicitly, e is true when p is false, and we hope to learn p
    print(f"e :- not p.", file=bk_file)
    # Type the attributes and generate the mode biases
    # For each column, get the feature name and all unique values
    mode_biases = []
    for col in X.columns:
        # mode biases
        type_name = col.replace("-", "_")
        mode_biases.append(f"#modeb(has_{type_name}(var({type_name}))).")
        if not is_ilasp:
            # FastLas requires explicit 'not' to include in hypothesis space
            mode_biases.append(
                f"#modeb(not has_{type_name}(var({type_name})))."
            )

        for val in X[col].unique():
            # Typing
            # replace "-" with "_"
            var_name = f"{col}-{val}".replace("-", "_")
            print(f"{type_name}({var_name}).", file=bk_file)

    print("#modeh(p).", file=bk_file)
    for m in mode_biases:
        print(m, file=bk_file)

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
