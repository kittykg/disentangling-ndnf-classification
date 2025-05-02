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

from neural_dnf.utils import (
    DeltaDelayedDecayScheduler,  # base DDS class
    DeltaDelayedExponentialDecayScheduler,
    DeltaDelayedLinearDecayScheduler,
    DeltaDelayedMonotonicFunctionScheduler,
    DeltaDelayedMonitoringDecayScheduler,  # monitoring base DDS class
    DeltaDelayedMonitoringExponentialDecayScheduler,
    DeltaDelayedMonitoringLinearDecayScheduler,
)

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from analysis import MetricValueMeter, AccuracyMeter
from predicate_invention import DelayedExponentialTauDecayScheduler

from utils import post_to_discord_webhook, generate_weight_histogram

from cdc.data_utils_cdc import (
    CDCDataset,
    get_cdc_data_np_from_path,
    get_x_and_y_cdc,
)
from cdc.models import (
    CDCClassifier,
    CDCMLP,
    CDCNeuralDNF,
    construct_model,
)
from cdc.eval.eval_common import (
    cdc_classifier_eval,
    parse_eval_return_meters_with_logging,
)

NON_TRAIN_LOADER_BATCH_SIZE = 4096
log = logging.getLogger()


def loss_calculation(
    criterion: torch.nn.Module,
    y_hat: Tensor,
    y: Tensor,
    model: CDCClassifier,
    conj_out: Tensor | None = None,
    invented_predicates: Tensor | None = None,
) -> dict[str, Tensor]:
    loss_dict = {
        "base_loss": criterion(y_hat, y),
        "weight_reg_loss": model.get_weight_reg_loss(),
    }

    if conj_out is not None and invented_predicates is not None:
        # Conjunction regularisation loss (push to ±1)
        loss_dict["conj_reg_loss"] = (1 - conj_out.abs()).mean()
        # Invented predicate regularisation loss (push to ±1)
        loss_dict["invented_predicates_reg_loss"] = (
            1 - invented_predicates.abs()
        ).mean()

    return loss_dict


def _train(
    training_cfg: DictConfig,
    model: CDCClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_wandb: bool,
) -> dict[str, float]:
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
    loss_func_key = training_cfg["loss_func"]
    # MLP -> BCEWithLogitsLoss, NDNF -> BCELoss / MSELoss
    if isinstance(model, CDCMLP):
        criterion = nn.BCEWithLogitsLoss()
    elif loss_func_key == "bce":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Delta delay scheduler if using NeuralDNF
    if isinstance(model, CDCNeuralDNF):
        # Delta and tau delay scheduler if using NeuralDNF based model
        dds_type = {
            "linear": DeltaDelayedLinearDecayScheduler,
            "exponential": DeltaDelayedExponentialDecayScheduler,
            "monotonic": DeltaDelayedMonotonicFunctionScheduler,
            "monitoring_linear": DeltaDelayedMonitoringLinearDecayScheduler,
            "monitoring_exponential": DeltaDelayedMonitoringExponentialDecayScheduler,
        }[training_cfg["dds"]["type"]]
        dds_params = {
            "initial_delta": training_cfg["dds"]["initial_delta"],
            "delta_decay_delay": training_cfg["dds"]["delta_decay_delay"],
            "delta_decay_steps": training_cfg["dds"]["delta_decay_steps"],
            "delta_decay_rate": training_cfg["dds"]["delta_decay_rate"],
            "target_module_type": model.ndnf.__class__.__name__,
        }
        if training_cfg["dds"]["type"].startswith("monitoring"):
            dds_params["performance_offset"] = training_cfg["dds"].get(
                "performance_offset", 1e-2
            )

        dds: DeltaDelayedDecayScheduler = dds_type(**dds_params)
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
    if isinstance(model, CDCNeuralDNF):
        acc_meter_conversion_fn = lambda y_hat: y_hat > 0.5
    else:
        acc_meter_conversion_fn = lambda y_hat: torch.sigmoid(y_hat) > 0.5

    # Meters
    train_loss_meters = {
        "overall_loss": MetricValueMeter("overall_loss_meter"),
        "base_loss": MetricValueMeter("base_loss_meter"),
        "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
    }
    if isinstance(model, CDCNeuralDNF):
        train_loss_meters["conj_reg_loss"] = MetricValueMeter(
            "conj_reg_loss_meter"
        )
        train_loss_meters["invented_predicates_reg_loss"] = MetricValueMeter(
            "invented_predicates_reg_loss_meter"
        )

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
            optimiser.zero_grad()

            x, y = get_x_and_y_cdc(
                data, device, use_ndnf=isinstance(model, CDCNeuralDNF)
            )  # y \in {0, 1}

            y_hat = model(x).squeeze()
            conj_out, invented_predicates = None, None
            if isinstance(model, CDCNeuralDNF):
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
            )
            if isinstance(model, CDCNeuralDNF):
                loss += (
                    training_cfg["aux_loss"]["tanh_conj_lambda"]
                    * loss_dict["conj_reg_loss"]
                    + training_cfg["aux_loss"]["pi_lambda"]
                    * loss_dict["invented_predicates_reg_loss"]
                )

            loss.backward()

            # If using NDNF model, check if some weights are manually set to be
            # sparse. If so, after backward, we need to set the gradients of
            # those weights to 0, so that they are not updated. The NDNF module
            # has a `conj_weight_mask` that stores the mask for the weights that
            # are manually set to be sparse.
            if (
                isinstance(model, CDCNeuralDNF)
                and model.manually_spare_conj_layer_k is not None
                and model.manually_spare_conj_layer_k > 0
            ):
                assert model.ndnf.conjunctions.weights.grad is not None
                with torch.no_grad():
                    model.ndnf.conjunctions.weights.grad *= (
                        model.ndnf.conj_weight_mask.to(device)
                    )

            optimiser.step()

            # Update meters
            for key, loss_val in loss_dict.items():
                train_loss_meters[key].update(loss_val.item())
            train_loss_meters["overall_loss"].update(loss.item())
            train_acc_meter.update(y_hat, y)

        # Log average performance for train
        avg_loss = train_loss_meters["overall_loss"].get_average()
        avg_acc = train_acc_meter.get_average()

        if epoch % log_interval == 0:
            if isinstance(model, CDCNeuralDNF):
                log_info_str = (
                    f"  [{epoch + 1:3d}] "
                    f"Train  Delta: {model.ndnf.get_delta_val()[0]:.3f}  "
                    f"Tau: {model.predicate_inventor.tau:.3f}  "
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
                x, y = get_x_and_y_cdc(
                    data, device, use_ndnf=isinstance(model, CDCNeuralDNF)
                )

                y_hat = model(x).squeeze()
                conj_out, invented_predicates = None, None
                if isinstance(model, CDCNeuralDNF):
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
                )
                if isinstance(model, CDCNeuralDNF):
                    loss += (
                        training_cfg["aux_loss"]["tanh_conj_lambda"]
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
                f"  [{epoch + 1:3d}] Val                  "
                f"avg loss: {val_avg_loss:.3f}  avg acc: {val_avg_acc:.3f}"
            )

        # -------------------------------------------------------------------- #
        # 3. Schedulers step
        # -------------------------------------------------------------------- #
        scheduler.step()

        if isinstance(model, CDCNeuralDNF):
            # Update delta value
            if not isinstance(dds, DeltaDelayedMonitoringDecayScheduler):
                delta_dict = dds.step(model.ndnf)
            else:
                delta_dict = dds.step(model.ndnf, val_avg_acc)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

            # Update tau value
            tau_dict = tau_scheduler.step(model.predicate_inventor)
            old_tau = tau_dict["old_tau"]

            if new_delta == 1.0:
                # The first time where new_delta_val becomes 1, the network
                # isn't train with delta being 1 for that epoch. So
                # delta_one_counter starts with -1, and when new_delta_val
                # is first time being 1, the delta_one_counter becomes 0. We
                # do not use the delta_one_counter for now, but it can be
                # used to customise when to add auxiliary loss
                delta_one_counter += 1

        # -------------------------------------------------------------------- #
        # 4. (Optional) WandB logging
        # -------------------------------------------------------------------- #
        if use_wandb:
            wandb_log_dict = {
                "train/epoch": epoch,
                "train/loss": avg_loss,
                "train/accuracy": avg_acc,
                "val/loss": val_avg_loss,
                "val/accuracy": val_avg_acc,
            }
            if isinstance(model, CDCNeuralDNF):
                wandb_log_dict["train/delta"] = old_delta
                wandb_log_dict["train/tau"] = old_tau
            for key, meter in train_loss_meters.items():
                if key == "overall_loss":
                    continue
                wandb_log_dict[f"train/{key}"] = meter.get_average()
            if gen_weight_hist and isinstance(model, CDCNeuralDNF):
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


def train(
    cfg: DictConfig, run_dir: Path, is_sweep: bool = False
) -> dict[str, float]:
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
    X, y, _ = get_cdc_data_np_from_path(cfg["dataset"], is_test=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=training_cfg.get("val_size", 0.2),
        random_state=training_cfg.get("val_seed", 73),
    )
    train_dataset = CDCDataset(X_train, y_train)
    val_dataset = CDCDataset(X_val, y_val)
    hold_out_test_X, hold_out_test_y, _ = get_cdc_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = CDCDataset(hold_out_test_X, hold_out_test_y)

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
        batch_size=NON_TRAIN_LOADER_BATCH_SIZE,
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=NON_TRAIN_LOADER_BATCH_SIZE,
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
    )

    # Model
    model = construct_model(training_cfg)
    model.to(device)
    log.info(f"Model: {model}")

    _train(training_cfg, model, train_loader, val_loader, device, use_wandb)

    if isinstance(model, CDCNeuralDNF):
        # Set delta to 1.0 for evaluation
        model.ndnf.set_delta_val(1.0)

    model_path = run_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    # Evaluate
    final_test_loader = val_loader if is_sweep else test_loader
    test_eval_result_nipd = parse_eval_return_meters_with_logging(
        cdc_classifier_eval(model, device, final_test_loader, False),
        model_name="Model after training (No Invented Predicate Discretisation)",
        do_logging=True,
        metric_prefix="nipd/",
    )
    test_eval_result = test_eval_result_nipd
    if isinstance(model, CDCNeuralDNF):
        test_eval_result_ipd = parse_eval_return_meters_with_logging(
            cdc_classifier_eval(model, device, final_test_loader, True),
            model_name="Model after training (Invented Predicate Discretisation)",
            do_logging=True,
            metric_prefix="ipd/",
        )
        test_eval_result = {**test_eval_result_nipd, **test_eval_result_ipd}

    if use_wandb:
        wandb.log(test_eval_result)
        wandb.save(glob_str=str(model_path.absolute()))

    with open("train_result.json", "w") as f:
        json.dump(test_eval_result, f, indent=4)
    return test_eval_result


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # We expect the experiment name to be in the format of:
    # cdc_{mlp/ndnf}_...
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
