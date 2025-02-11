import json
import logging
from pathlib import Path
import random
import sys
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
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

from analysis import MetricValueMeter, AccuracyMeter
from utils import post_to_discord_webhook, generate_weight_histogram

from mushroom.data_utils_mushroom import (
    MushroomDataset,
    get_mushroom_data_np_from_path,
    get_x_and_y_mushroom,
)
from mushroom.models import (
    MushroomNeuralDNF,
    construct_model,
)
from mushroom.eval.eval_common import (
    mushroom_classifier_eval,
    parse_eval_return_meters_with_logging,
)


log = logging.getLogger()


def compute_lagrangian(
    criterion: torch.nn.Module,
    model: MushroomNeuralDNF,
    data: list[Tensor],
    device: torch.device,
    lambda_: Tensor | None = None,
    max_wrt_lambda: bool = False,
) -> dict[str, Tensor]:
    x, y = get_x_and_y_mushroom(
        data, device, use_ndnf=isinstance(model, MushroomNeuralDNF)
    )
    y_hat = model(x).squeeze()
    y_hat = (torch.tanh(y_hat) + 1) / 2

    # Compute the Lagrangian: L(theta, lambda) = f(y_hat, y) + g(theta, lambda)
    f_ = criterion(y_hat, y)
    g_ = model.get_weight_reg_loss()
    L = f_

    if lambda_ is not None:
        if lambda_.shape != g_.shape:
            g_ = (model.get_weight_reg_loss(take_mean=False) * lambda_).mean()
            L += g_
        else:
            L += lambda_ * g_

    if max_wrt_lambda:
        L = -L

    return {
        "L": L,
        "f_": f_,
        "g_": g_,
        "y_hat": y_hat,
        "y": y,
    }


def get_model_weights_mean(model: MushroomNeuralDNF) -> Tensor:
    return torch.cat(
        [
            parameter.view(-1)
            for parameter in model.ndnf.parameters()
            if parameter.requires_grad
        ]
    ).mean()


def _train(
    training_cfg: DictConfig,
    model: MushroomNeuralDNF,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_wandb: bool,
) -> dict[str, float]:
    # Loss function
    loss_func_key = training_cfg["loss_func"]
    if loss_func_key == "bce":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Lambda for the Lagrangian
    multi_lambda = training_cfg.get("use_multi_lambda", False)
    if multi_lambda:
        lambda_ = torch.ones(
            torch.cat(
                [
                    parameter.view(-1)
                    for parameter in model.ndnf.parameters()
                    if parameter.requires_grad
                ]
            ).shape,
        ).to(device)
        lambda_.requires_grad = True
    else:
        lambda_ = torch.tensor(1.0, requires_grad=True)

    # Optimiser and scheduler
    optimiser_key = training_cfg["optimiser"]
    if optimiser_key == "sgd":
        opt_theta = torch.optim.SGD(
            model.parameters(),
            lr=training_cfg["opt_theta_lr"],
            momentum=0.9,
            weight_decay=training_cfg["opt_theta_weight_decay"],
        )
        opt_lambda = torch.optim.SGD(
            [lambda_],
            lr=training_cfg["opt_lambda_lr"],
            momentum=0.9,
        )
    else:
        opt_theta = torch.optim.Adam(
            model.parameters(),
            lr=training_cfg["opt_theta_lr"],
            weight_decay=training_cfg["opt_theta_weight_decay"],
        )
        opt_lambda = torch.optim.Adam(
            [lambda_],
            lr=training_cfg["opt_lambda_lr"],
        )
    opt_theta_scheduler = torch.optim.lr_scheduler.StepLR(
        opt_theta, step_size=training_cfg["opt_theta_scheduler_step"], gamma=0.1
    )
    opt_lambda_scheduler = torch.optim.lr_scheduler.StepLR(
        opt_lambda,
        step_size=training_cfg["opt_lambda_scheduler_step"],
        gamma=training_cfg["opt_lambda_scheduler_gamma"],
    )

    # Delta delay scheduler
    dds = DeltaDelayedExponentialDecayScheduler(
        initial_delta=training_cfg["dds"]["initial_delta"],
        delta_decay_delay=training_cfg["dds"]["delta_decay_delay"],
        delta_decay_steps=training_cfg["dds"]["delta_decay_steps"],
        delta_decay_rate=training_cfg["dds"]["delta_decay_rate"],
        target_module_type=model.ndnf.__class__.__name__,
    )
    model.ndnf.set_delta_val(training_cfg["dds"]["initial_delta"])
    delta_one_counter = 0
    delta_one_counter_threshold = training_cfg["delta_one_counter_threshold"]

    # Other training settings
    gen_weight_hist = training_cfg.get("gen_weight_hist", False)
    log_interval = training_cfg.get("log_interval", 100)
    acc_meter_conversion_fn = lambda y_hat: y_hat > 0.5

    # Meters
    train_loss_meters = {
        "overall_loss": MetricValueMeter("overall_loss_meter"),
        "base_loss": MetricValueMeter("base_loss_meter"),
        "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
    }

    train_acc_meter = AccuracyMeter(acc_meter_conversion_fn)
    epoch_val_loss_meter = MetricValueMeter("val_loss_meter")
    epoch_val_acc_meter = AccuracyMeter(
        output_to_prediction_fn=acc_meter_conversion_fn
    )

    for epoch in range(training_cfg["epochs"]):
        # -------------------------------------------------------------------- #
        #  1. Training
        # -------------------------------------------------------------------- #
        for m in train_loss_meters.values():
            m.reset()
        train_acc_meter.reset()
        model.train()

        for data in train_loader:
            # Act to minimise w.r.t. theta
            if delta_one_counter > delta_one_counter_threshold:
                loss_dict = compute_lagrangian(
                    criterion, model, data, device, lambda_
                )
            else:
                loss_dict = compute_lagrangian(criterion, model, data, device)

            L_theta = loss_dict["L"]
            f_ = loss_dict["f_"]
            g_ = loss_dict["g_"]
            y_hat = loss_dict["y_hat"]
            y = loss_dict["y"]

            opt_theta.zero_grad()
            L_theta.backward()
            opt_theta.step()

            if delta_one_counter > delta_one_counter_threshold:
                # Act to maximise w.r.t. lambda
                L_lambda = compute_lagrangian(
                    criterion, model, data, device, lambda_, max_wrt_lambda=True
                )["L"]
                opt_lambda.zero_grad()
                L_lambda.backward()
                opt_lambda.step()

            # Update meters
            train_loss_meters["overall_loss"].update(L_theta.item())
            train_loss_meters["base_loss"].update(f_.item())
            train_loss_meters["weight_reg_loss"].update(g_.item())
            train_acc_meter.update(y_hat, y)

        # Update delta value
        delta_dict = dds.step(model.ndnf)
        new_delta = delta_dict["new_delta_vals"][0]
        old_delta = delta_dict["old_delta_vals"][0]

        # Log average performance for train
        avg_loss = train_loss_meters["overall_loss"].get_average()
        avg_acc = train_acc_meter.get_average()

        if epoch % log_interval == 0:
            if isinstance(model, MushroomNeuralDNF):
                log_info_str = (
                    f"  [{epoch + 1:3d}] Train  Delta: {old_delta:.3f}  "
                    f"avg loss: {avg_loss:.3f}  avg acc: {avg_acc:.3f}"
                )
            else:
                log_info_str = (
                    f"  [{epoch + 1:3d}] Train                "
                    f"avg loss: {avg_loss:.3f}  avg acc: {avg_acc:.3f}"
                )
            log.info(log_info_str)

        # -------------------------------------------------------------------- #
        # 2. Evaluate performance on val
        # -------------------------------------------------------------------- #
        epoch_val_loss_meter.reset()
        epoch_val_acc_meter.reset()
        model.eval()

        for data in val_loader:
            with torch.no_grad():
                # Get model output and compute loss
                if delta_one_counter > delta_one_counter_threshold:
                    loss_dict = compute_lagrangian(
                        criterion, model, data, device, lambda_
                    )
                else:
                    loss_dict = compute_lagrangian(
                        criterion, model, data, device
                    )

                # Update meters
                epoch_val_loss_meter.update(loss_dict["L"].item())
                epoch_val_acc_meter.update(loss_dict["y_hat"], loss_dict["y"])

        val_avg_loss = epoch_val_loss_meter.get_average()
        val_avg_acc = epoch_val_acc_meter.get_average()
        if epoch % log_interval == 0:
            log.info(
                f"  [{epoch + 1:3d}] Val                  "
                f"avg loss: {val_avg_loss:.3f}  avg acc: {val_avg_acc:.3f}"
            )

        # -------------------------------------------------------------------- #
        # 3. Let scheduler update optimiser at end of epoch
        # -------------------------------------------------------------------- #
        opt_theta_scheduler.step()
        if delta_one_counter > delta_one_counter_threshold:
            opt_lambda_scheduler.step()

        if new_delta == 1.0:
            # The first time where new_delta_val becomes 1, the network isn't
            # train with delta being 1 for that epoch. So delta_one_counter
            # starts with -1, and when new_delta_val is first time being 1,
            # the delta_one_counter becomes 0.
            # We do not use the delta_one_counter for now, but it can be used
            # to customise when to add auxiliary loss
            delta_one_counter += 1

        # -------------------------------------------------------------------- #
        # 4. (Optional) WandB logging
        # -------------------------------------------------------------------- #
        if use_wandb:
            wandb_log_dict = {
                "train/epoch": epoch,
                "train/loss": avg_loss,
                "train/accuracy": avg_acc,
                "train/delta": old_delta,
                "train/model_weights_maen": get_model_weights_mean(
                    model
                ).item(),
                "train/opt_theta_lr": opt_theta.param_groups[0]["lr"],
                "val/loss": val_avg_loss,
                "val/accuracy": val_avg_acc,
            }
            # Log lambda
            if delta_one_counter > delta_one_counter_threshold:
                wandb_log_dict["train/lambda"] = (
                    lambda_.mean().item() if multi_lambda else lambda_.item()
                )
                wandb_log_dict["train/opt_lambda_lr"] = opt_lambda.param_groups[
                    0
                ]["lr"]

            # Log losses
            for key, meter in train_loss_meters.items():
                if key == "overall_loss":
                    continue
                wandb_log_dict[f"train/{key}"] = meter.get_average()

            if gen_weight_hist:
                # Generate weight histogram
                f1, f2 = generate_weight_histogram(model.ndnf)
                wandb_log_dict["conj_w_hist"] = wandb.Image(f1)
                wandb_log_dict["disj_w_hist"] = wandb.Image(f2)
            wandb.log(wandb_log_dict)

    return {
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
    X, y, _ = get_mushroom_data_np_from_path(cfg["dataset"], is_test=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=training_cfg.get("val_size", 0.2),
        random_state=training_cfg["seed"],
    )
    train_dataset = MushroomDataset(X_train, y_train)
    val_dataset = MushroomDataset(X_val, y_val)
    hold_out_test_X, hold_out_test_y, _ = get_mushroom_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = MushroomDataset(hold_out_test_X, hold_out_test_y)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
    )

    # Model
    model = construct_model(training_cfg, X_train.shape[1])
    model.to(device)
    assert isinstance(model, MushroomNeuralDNF)

    _train(training_cfg, model, train_loader, val_loader, device, use_wandb)

    if isinstance(model, MushroomNeuralDNF):
        # Set delta to 1.0 for evaluation
        model.ndnf.set_delta_val(1.0)

    model_path = run_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    # Evaluate on test
    test_eval_result = parse_eval_return_meters_with_logging(
        mushroom_classifier_eval(model, device, test_loader),
        model_name="Model after training",
        do_logging=True,
    )

    if use_wandb:
        wandb.log(test_eval_result)
        wandb.save(glob_str=str(model_path.absolute()))

    with open("train_result.json", "w") as f:
        json.dump(test_eval_result, f, indent=4)
    return test_eval_result


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # We expect the experiment name to be in the format of:
    # mushroom_ndnf_...
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
