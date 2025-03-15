from datetime import datetime
import logging
from pathlib import Path
import random
import sys
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch
import wandb

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from covertype.train_covertype import train
from utils import post_to_discord_webhook

log = logging.getLogger()


def train_wrapper(cfg: DictConfig):
    # Randomly select a seed based on the current time
    ts = datetime.now().timestamp()
    random.seed(ts)
    seed = random.randrange(10000)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg["training"]["seed"] = seed

    run = wandb.init(dir=HydraConfig.get().run.dir, sync_tensorboard=True)
    assert run is not None, "Wandb run is not initialized"

    use_ndnf = "ndnf" in cfg["training"]["experiment_name"]
    assert use_ndnf, "Must be NDNF based experiment"
    assert cfg["training"]["model_type"] in ["eo", "mt"]

    full_experiment_name = (
        f"{cfg['training']['experiment_name']}_{seed}_{int(ts)}"
    )

    # Override the model architecture parameters
    cfg["training"]["model_architecture"][
        "invented_predicate_per_input"
    ] = wandb.config.invented_predicate_per_input
    cfg["training"]["model_architecture"][
        "n_conjunctions"
    ] = wandb.config.n_conjunctions

    # Override training parameters
    cfg["training"]["optimiser_lr"] = wandb.config.optimiser_lr
    cfg["training"][
        "optimiser_weight_decay"
    ] = wandb.config.optimiser_weight_decay
    cfg["training"]["scheduler_step"] = int(wandb.config.scheduler_step)
    cfg["training"]["batch_size"] = wandb.config.batch_size

    # Override delta and tau scheduler parameters
    cfg["training"]["dds"]["delta_decay_rate"] = wandb.config.delta_decay_rate
    cfg["training"]["dds"]["delta_decay_steps"] = int(
        wandb.config.delta_decay_steps
    )
    cfg["training"]["dds"]["delta_decay_delay"] = int(
        wandb.config.delta_decay_delay
    )
    cfg["training"]["pi_tau"]["tau_decay_rate"] = wandb.config.tau_decay_rate
    cfg["training"]["pi_tau"]["tau_decay_steps"] = int(
        wandb.config.tau_decay_steps
    )
    cfg["training"]["pi_tau"]["tau_decay_delay"] = int(
        wandb.config.tau_decay_delay
    )

    # Override the lambda parameters
    cfg["training"]["aux_loss"][
        "weight_l1_mod_lambda"
    ] = wandb.config.weight_l1_mod_lambda
    cfg["training"]["aux_loss"][
        "tanh_conj_lambda"
    ] = wandb.config.tanh_conj_lambda
    cfg["training"]["aux_loss"]["pi_lambda"] = wandb.config.pi_lambda
    if cfg["training"]["model_type"] == "mt":
        cfg["training"]["aux_loss"][
            "mt_reg_lambda"
        ] = wandb.config.mt_reg_lambda

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False

    try:
        eval_result = train(cfg, Path(HydraConfig.get().run.dir), is_sweep=True)
        combined_metric = (
            eval_result["sample_jaccard"]
            + eval_result["accuracy"]
            + eval_result["macro_jaccard"]
        )
        wandb.log({"combined_metric": combined_metric})

        if use_discord_webhook:
            msg_body = "Success!"
    except BaseException as e:
        if use_discord_webhook:
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
                experiment_name=full_experiment_name,
                message_body=msg_body,
                errored=errored,
            )
            wandb.finish()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    use_wandb = cfg["wandb"]["use_wandb"]
    assert use_wandb, "Must use wandb for hyperparameter search"
    train_wrapper(cfg)


if __name__ == "__main__":
    run_experiment()
