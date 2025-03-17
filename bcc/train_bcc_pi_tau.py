"""
This script is identical to train_bcc.py, except that it is for BCCNeuralDNF
classifier and enables tau update for the predicate inventor.
"""

import json
import logging
from pathlib import Path
import random
import sys
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
import torch
from torch import Tensor, nn
import wandb

from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import MetricValueMeter, AccuracyMeter, collate, synthesize
from data_utils import GenericUCIDataset
from predicate_invention import DelayedExponentialTauDecayScheduler
from utils import post_to_discord_webhook, generate_weight_histogram

from bcc.data_utils_bcc import get_bcc_data
from bcc.models import BCCNeuralDNF


log = logging.getLogger()


def loss_calculation(
    criterion: torch.nn.Module,
    y_hat: Tensor,
    y: Tensor,
    model: BCCNeuralDNF,
    conj_out: Tensor,
    invented_predicates: Tensor,
) -> dict[str, Tensor]:
    return {
        "base_loss": criterion(y_hat, y),
        "weight_reg_loss": model.get_weight_reg_loss(),
        "conj_reg_loss": (1 - conj_out.abs()).mean(),
        "invented_predicates_reg_loss": (1 - invented_predicates.abs()).mean(),
    }


def train_fold(
    fold_id: int,
    train_index: npt.NDArray[np.int64],
    test_index: npt.NDArray[np.int64],
    bcc_dataset: GenericUCIDataset,
    training_cfg: DictConfig,
    device: torch.device,
    use_wandb: bool,
) -> tuple[BCCNeuralDNF, dict[str, float]]:

    # Model
    assert (
        training_cfg["model_type"] == "ndnf"
    ), "This training script only supports NDNF"

    model = BCCNeuralDNF(
        num_features=bcc_dataset.X.shape[1],
        invented_predicate_per_input=training_cfg["model_architecture"][
            "invented_predicate_per_input"
        ],
        num_conjunctions=training_cfg["model_architecture"]["n_conjunctions"],
        predicate_inventor_tau=training_cfg["pi_tau"]["initial_tau"],
    )
    model.to(device)

    # Data loaders
    # Sample elements randomly from a given list of ids, no replacement.
    train_loader = torch.utils.data.DataLoader(
        bcc_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
        sampler=torch.utils.data.SubsetRandomSampler(train_index),  # type: ignore
    )
    val_loader = torch.utils.data.DataLoader(
        bcc_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
        sampler=torch.utils.data.SubsetRandomSampler(test_index),  # type: ignore
    )

    # Optimiser and scheduler
    lr = training_cfg["optimiser_lr"]
    weight_decay = training_cfg["optimiser_weight_decay"]
    optimiser_key = training_cfg["optimiser"]
    if optimiser_key == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=training_cfg["scheduler_step"], gamma=0.1
    )

    # Loss function
    criterion = (
        nn.BCELoss() if training_cfg["loss_func"] == "bce" else nn.MSELoss()
    )

    # Delta scheduler and tau scheduler
    dds = DeltaDelayedExponentialDecayScheduler(
        initial_delta=training_cfg["dds"]["initial_delta"],
        delta_decay_delay=training_cfg["dds"]["delta_decay_delay"],
        delta_decay_steps=training_cfg["dds"]["delta_decay_steps"],
        delta_decay_rate=training_cfg["dds"]["delta_decay_rate"],
        target_module_type=model.ndnf.__class__.__name__,
    )
    model.ndnf.set_delta_val(training_cfg["dds"]["initial_delta"])
    delta_one_counter = 0

    tau_scheduler = DelayedExponentialTauDecayScheduler(
        initial_tau=training_cfg["pi_tau"]["initial_tau"],
        tau_decay_delay=training_cfg["pi_tau"]["tau_decay_delay"],
        tau_decay_steps=training_cfg["pi_tau"]["tau_decay_steps"],
        tau_decay_rate=training_cfg["pi_tau"]["tau_decay_rate"],
        min_tau=training_cfg["pi_tau"]["min_tau"],
    )

    # Other training settings
    gen_weight_hist = training_cfg.get("gen_weight_hist", False)
    log_interval = training_cfg.get("log_interval", 100)
    acc_meter_conversion_fn = lambda y_hat: y_hat > 0.5

    for epoch in range(training_cfg["epochs"]):
        # -------------------------------------------------------------------- #
        #  1. Training
        # -------------------------------------------------------------------- #
        train_loss_meters = {
            "overall_loss": MetricValueMeter("overall_loss_meter"),
            "base_loss": MetricValueMeter("base_loss_meter"),
            "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
            "conj_reg_loss": MetricValueMeter("conj_reg_loss_meter"),
            "invented_predicates_reg_loss": MetricValueMeter(
                "invented_predicates_reg_loss_meter"
            ),
        }
        train_acc_meter = AccuracyMeter(acc_meter_conversion_fn)

        model.train()

        for data in train_loader:
            optimiser.zero_grad()

            x = data[0].to(device)
            y = data[1].to(device)  # y \in {0, 1}

            y_hat = model(x).squeeze()
            # For NeuralDNF, we need to take the tanh of the logit and
            # then scale it to (0, 1)
            y_hat = (torch.tanh(y_hat) + 1) / 2
            conj_out = model.get_conjunction(x)
            invented_predicates = model.get_invented_predicates(x)

            loss_dict = loss_calculation(
                criterion, y_hat, y, model, conj_out, invented_predicates
            )

            loss = (
                loss_dict["base_loss"]
                + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                * loss_dict["weight_reg_loss"]
                + training_cfg["aux_loss"]["tanh_conj_lambda"]
                * loss_dict["conj_reg_loss"]
                + training_cfg["aux_loss"]["pi_lambda"]
                * loss_dict["invented_predicates_reg_loss"]
            )

            loss.backward()
            optimiser.step()

            # Update meters
            for key, loss_val in loss_dict.items():
                train_loss_meters[key].update(loss_val.item())
            train_loss_meters["overall_loss"].update(loss.item())
            train_acc_meter.update(y_hat, y)

        # Update delta value
        delta_dict = dds.step(model.ndnf)
        new_delta = delta_dict["new_delta_vals"][0]
        old_delta = delta_dict["old_delta_vals"][0]

        # Update tau value
        tau_dict = tau_scheduler.step(model.predicate_inventor)
        old_tau = tau_dict["old_tau"]

        if new_delta == 1.0:
            # The first time where new_delta_val becomes 1, the network isn't
            # train with delta being 1 for that epoch. So delta_one_counter
            # starts with -1, and when new_delta_val is first time being 1,
            # the delta_one_counter becomes 0.
            # We do not use the delta_one_counter for now, but it can be used
            # to customise when to add auxiliary loss
            delta_one_counter += 1

        # Log average performance for train
        avg_loss = train_loss_meters["overall_loss"].get_average()
        avg_acc = train_acc_meter.get_average()

        if epoch % log_interval == 0:
            log.info(
                f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] Train  "
                f"Delta: {old_delta:.3f}  Tau:{old_tau:.3f}  "
                f"avg loss: {avg_loss:.3f}  avg acc: {avg_acc:.3f}"
            )

        # -------------------------------------------------------------------- #
        # 2. Evaluate performance on val
        # -------------------------------------------------------------------- #
        epoch_val_loss_meter = MetricValueMeter("val_loss_meter")
        epoch_val_acc_meter = AccuracyMeter(
            output_to_prediction_fn=acc_meter_conversion_fn
        )

        model.eval()

        for data in val_loader:
            with torch.no_grad():
                # Get model output and compute loss
                x = data[0].to(device)
                y = data[1].to(device)

                y_hat = model(x).squeeze()
                y_hat = (torch.tanh(y_hat) + 1) / 2
                conj_out = model.get_conjunction(x)
                invented_predicates = model.get_invented_predicates(x)

                loss_dict = loss_calculation(
                    criterion, y_hat, y, model, conj_out, invented_predicates
                )

                loss = (
                    loss_dict["base_loss"]
                    + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                    * loss_dict["weight_reg_loss"]
                    + training_cfg["aux_loss"]["tanh_conj_lambda"]
                    * loss_dict["conj_reg_loss"]
                    + training_cfg["aux_loss"]["pi_lambda"]
                    * loss_dict["invented_predicates_reg_loss"]
                )

                # Update meters
                epoch_val_loss_meter.update(loss.item())
                epoch_val_acc_meter.update(y_hat, y)

        val_avg_loss = epoch_val_loss_meter.get_average()
        val_avg_acc = epoch_val_acc_meter.get_average()
        if epoch % log_interval == 0:
            log.info(
                f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] "
                f"Val                  avg loss: {val_avg_loss:.3f}  "
                f"avg acc: {val_avg_acc:.3f}"
            )

        # -------------------------------------------------------------------- #
        # 3. Let scheduler update optimiser at end of epoch
        # -------------------------------------------------------------------- #
        scheduler.step()

        # -------------------------------------------------------------------- #
        # 4. (Optional) WandB logging
        # -------------------------------------------------------------------- #
        if use_wandb:
            wandb_log_dict = {
                f"fold_{fold_id}/epoch": epoch,
                f"fold_{fold_id}/train/loss": avg_loss,
                f"fold_{fold_id}/train/accuracy": avg_acc,
                f"fold_{fold_id}/train/delta": old_delta,
                f"fold_{fold_id}/train/tau": old_tau,
                f"fold_{fold_id}/val/loss": val_avg_loss,
                f"fold_{fold_id}/val/accuracy": val_avg_acc,
            }

            for key, meter in train_loss_meters.items():
                if key == "overall_loss":
                    continue
                wandb_log_dict[f"fold_{fold_id}/train/{key}"] = (
                    meter.get_average()
                )
            if gen_weight_hist and isinstance(model, BCCNeuralDNF):
                # Generate weight histogram
                f1, f2 = generate_weight_histogram(model.ndnf)
                wandb_log_dict[f"fold_{fold_id}/conj_w_hist"] = wandb.Image(f1)
                wandb_log_dict[f"fold_{fold_id}/disj_w_hist"] = wandb.Image(f2)
            wandb.log(wandb_log_dict)

    return model, {
        "train_loss": avg_loss,
        "train_accuracy": avg_acc,
        "val_loss": val_avg_loss,
        "val_accuracy": val_avg_acc,
    }


def train(cfg: DictConfig, run_dir: Path) -> dict[str, float]:
    training_cfg = cfg["training"]
    use_wandb = cfg["wandb"]["use_wandb"]

    # Set up device
    use_cuda = torch.cuda.is_available() and training_cfg["use_cuda"]
    use_mps = (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and training_cfg.get("use_mps", False)
    )
    assert not (use_cuda and use_mps), "Cannot use both CUDA and MPS"
    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
    log.info(f"Device: {device}")

    # Get data
    X, y, _ = get_bcc_data(standardise=training_cfg["standardise"])
    bcc_dataset = GenericUCIDataset(X, y)

    # Fold results
    models = []
    fold_results = []

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=training_cfg["k_folds"],
        shuffle=True,
        random_state=training_cfg["seed"],
    )

    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y)):
        log.info(f"Fold {fold_id} starts")
        model, fold_result = train_fold(
            fold_id,
            train_index,
            test_index,
            bcc_dataset,
            training_cfg,
            device,
            use_wandb,
        )

        fold_dir = run_dir / f"fold_{fold_id}"
        if not fold_dir.exists():
            fold_dir.mkdir()
        model_path = fold_dir / f"model_fold_{fold_id}.pth"
        torch.save(model.state_dict(), model_path)
        if use_wandb:
            wandb.save(glob_str=str(model_path.absolute()))

        models.append(model)
        fold_results.append(fold_result)

        with open(fold_dir / f"fold_{fold_id}_result.json", "w") as f:
            json.dump(fold_result, f, indent=4)

    # Average results
    avg_results = dict()
    for k, v in collate(fold_results).items():
        synth_dict = synthesize(v)
        for kk, vv in synth_dict.items():
            avg_results[f"aggregated/{k}/{kk}"] = vv
            log.info(f"{k}/{kk}: {vv:.3f}")

    if use_wandb:
        wandb.log(avg_results)
    return avg_results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # We expect the experiment name to be in the format of:
    # bcc_{mlp/ndnf}_...
    experiment_name = cfg["training"]["experiment_name"]

    seed = cfg["training"]["seed"]
    if seed is None:
        seed = random.randint(0, 10000)

    full_experiment_name = f"{experiment_name}_{seed}"
    use_wandb = cfg["wandb"]["use_wandb"]

    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    run_dir_name = "-".join(
        [
            (s.upper() if i in [0] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )
    hydra_run_dir = Path(HydraConfig.get().run.dir)

    if use_wandb:
        # Set up wandb
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=OmegaConf.to_container(cfg["training"]),  # type: ignore
            dir=hydra_run_dir,
            name=run_dir_name,
            tags=cfg["wandb"]["tags"] if "tags" in cfg["wandb"] else [],
            group=cfg["wandb"]["group"] if "group" in cfg["wandb"] else None,
        )

    torch.autograd.set_detect_anomaly(True)

    log.info(f"{experiment_name} starts, seed: {seed}")

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        final_log_dict = train(cfg, hydra_run_dir)

        if use_discord_webhook:
            msg_body = f"Training of {full_experiment_name} completed.\n"
            for k, v in final_log_dict.items():
                msg_body += f"\tFinal {k}: {v:.3f}\n"
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
                experiment_name=full_experiment_name,
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )
        if use_wandb:
            wandb.finish()
        if not errored:
            hydra_run_dir.rename(hydra_run_dir.absolute().parent / run_dir_name)


if __name__ == "__main__":
    run_experiment()
