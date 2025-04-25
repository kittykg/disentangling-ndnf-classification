import json
import logging
import math
from pathlib import Path
import random
import sys
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
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

from analysis import MetricValueMeter, AccuracyMeter, JaccardScoreMeter
from utils import post_to_discord_webhook, generate_weight_histogram

from bn.models import (
    BooleanNetworkClassifier,
    BooleanNetworkMLP,
    BooleanNetworkNeuralDNF,
    construct_model,
)
from bn.data_utils_bn import (
    BooleanNetworkDataset,
    get_boolean_network_data_np_from_path,
    get_boolean_network_full_data_np_from_path,
    get_x_and_y_boolean_network,
    split_boolean_network_data,
)
from bn.eval.eval_common import (
    boolean_network_classifier_eval,
    parse_eval_return_meters_with_logging,
)


log = logging.getLogger()
NON_TRAIN_LOADER_BATCH_SIZE = 512


def cosine_with_exponential_decay_and_warmup_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    num_examples: int,
    batch_size: int,
    warmup_steps: int,
    decay_rate=0.9,
    decay_steps=1000,
    min_lr=1e-6,
) -> LRScheduler:
    num_steps_per_epoch = math.ceil(num_examples / batch_size)
    total_steps = num_epochs * num_steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            cosine_decay = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (current_step - warmup_steps)
                    / (total_steps - warmup_steps)
                )
            )
            exponential_decay = decay_rate ** (
                (current_step - warmup_steps) / decay_steps
            )
            return max(min_lr, cosine_decay * exponential_decay)

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def loss_calculation(
    criterion: torch.nn.Module,
    y_hat: Tensor,
    y: Tensor,
    model: BooleanNetworkClassifier,
    conj_out: Tensor | None = None,
) -> dict[str, Tensor]:
    loss_dict = {
        "base_loss": criterion(y_hat, y),
    }

    # Weight regularisation
    p_t = torch.cat(
        [
            parameter.view(-1)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    )
    reg_loss = torch.abs(p_t * (6 - torch.abs(p_t))).mean()
    loss_dict["weight_reg_loss"] = reg_loss.mean()

    if conj_out is not None:
        # Conjunction regularisation loss (push to ±1)
        loss_dict["conj_reg_loss"] = (1 - conj_out.abs()).mean()

    return loss_dict


def _train(
    training_cfg: DictConfig,
    model: BooleanNetworkClassifier,
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
    if isinstance(model, BooleanNetworkMLP):
        criterion = nn.BCEWithLogitsLoss()
    elif loss_func_key == "bce":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Delta delay scheduler if using NeuralDNF
    if isinstance(model, BooleanNetworkNeuralDNF):
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

    # Other training settings
    gen_weight_hist = training_cfg.get("gen_weight_hist", False)
    log_interval = training_cfg.get("log_interval", 100)
    log_lr_in_wandb = training_cfg.get("log_lr_in_wandb", False)

    # Meters
    train_loss_meters = {
        "overall_loss": MetricValueMeter("overall_loss_meter"),
        "base_loss": MetricValueMeter("base_loss_meter"),
        "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
    }
    if isinstance(model, BooleanNetworkNeuralDNF):
        train_loss_meters["conj_reg_loss"] = MetricValueMeter(
            "conj_reg_loss_meter"
        )
        acc_meter_conversion_fn = lambda y_hat: y_hat > 0.5
    else:
        acc_meter_conversion_fn = lambda y_hat: torch.sigmoid(y_hat) > 0.5
    train_acc_meter = AccuracyMeter(acc_meter_conversion_fn)
    train_jacc_meter = JaccardScoreMeter()
    epoch_val_loss_meter = MetricValueMeter("val_loss_meter")
    epoch_val_acc_meter = AccuracyMeter(acc_meter_conversion_fn)
    epoch_val_jacc_meter = JaccardScoreMeter()

    for epoch in range(training_cfg["epochs"]):
        # -------------------------------------------------------------------- #
        #  1. Training
        # -------------------------------------------------------------------- #
        for m in train_loss_meters.values():
            m.reset()
        train_acc_meter.reset()
        train_jacc_meter.reset()
        model.train()

        for data in train_loader:
            optimiser.zero_grad()

            x, y = get_x_and_y_boolean_network(
                data,
                device,
                use_ndnf=isinstance(model, BooleanNetworkNeuralDNF),
            )  # y \in {0, 1}

            y_hat = model(x)
            conj_out = None
            if isinstance(model, BooleanNetworkNeuralDNF):
                # For NeuralDNF, we need to take the tanh of the logit and
                # then scale it to (0, 1)
                y_hat = (torch.tanh(y_hat) + 1) / 2
                conj_out = model.get_conjunction(x)

            loss_dict = loss_calculation(criterion, y_hat, y, model, conj_out)

            loss = (
                loss_dict["base_loss"]
                + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                * loss_dict["weight_reg_loss"]
            )
            if isinstance(model, BooleanNetworkNeuralDNF):
                loss += (
                    training_cfg["aux_loss"]["tanh_conj_lambda"]
                    * loss_dict["conj_reg_loss"]
                )

            loss.backward()
            optimiser.step()

            # Update meters
            for key, loss_val in loss_dict.items():
                train_loss_meters[key].update(loss_val.item())
            train_loss_meters["overall_loss"].update(loss.item())
            train_acc_meter.update(y_hat, y)

            # Update jacc meter
            if isinstance(model, BooleanNetworkNeuralDNF):
                with torch.no_grad():
                    # y_hat \in (0, 1), convert to binary
                    y_hat_prime = (y_hat > 0.5).long()
            else:
                # MLP
                # y_hat \in R, convert to binary
                y_hat_prime = (torch.sigmoid(y_hat) > 0.5).long()
            train_jacc_meter.update(y_hat_prime, y)

        # Log average performance for train
        avg_loss = train_loss_meters["overall_loss"].get_average()
        avg_acc = train_acc_meter.get_average()
        avg_sample_jacc = train_jacc_meter.get_average()
        avg_macro_jacc = train_jacc_meter.get_average("macro")
        assert isinstance(avg_sample_jacc, float)
        assert isinstance(avg_macro_jacc, float)

        if epoch % log_interval == 0:
            if isinstance(model, BooleanNetworkNeuralDNF):
                log_info_str = (
                    f"  [{epoch + 1:3d}] Train  "
                    f"Delta: {model.ndnf.get_delta_val()[0]:.3f}  "
                    f"avg loss: {avg_loss:.3f}  avg acc: {avg_acc:.3f}  "
                    f"avg sample jacc: {avg_sample_jacc:.3f}  "
                    f"avg macro jacc: {avg_macro_jacc:.3f}"
                )
            else:
                log_info_str = (
                    f"  [{epoch + 1:3d}] Train                "
                    f"avg loss: {avg_loss:.3f}  avg acc: {avg_acc:.3f}  "
                    f"avg sample jacc: {avg_sample_jacc:.3f}  "
                    f"avg macro jacc: {avg_macro_jacc:.3f}"
                )
            log.info(log_info_str)

        # -------------------------------------------------------------------- #
        # 2. Evaluate performance on val
        # -------------------------------------------------------------------- #
        epoch_val_loss_meter.reset()
        epoch_val_acc_meter.reset()
        epoch_val_jacc_meter.reset()
        model.eval()

        for data in val_loader:
            with torch.no_grad():
                # Get model output and compute loss
                x, y = get_x_and_y_boolean_network(
                    data,
                    device,
                    use_ndnf=isinstance(model, BooleanNetworkNeuralDNF),
                )

                y_hat = model(x)
                conj_out = None
                if isinstance(model, BooleanNetworkNeuralDNF):
                    y_hat = (torch.tanh(y_hat) + 1) / 2
                    conj_out = model.get_conjunction(x)

                loss_dict = loss_calculation(
                    criterion, y_hat, y, model, conj_out
                )

                loss = (
                    loss_dict["base_loss"]
                    + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                    * loss_dict["weight_reg_loss"]
                )
                if isinstance(model, BooleanNetworkNeuralDNF):
                    loss += (
                        training_cfg["aux_loss"]["tanh_conj_lambda"]
                        * loss_dict["conj_reg_loss"]
                    )

                # Update meters
                epoch_val_loss_meter.update(loss.item())
                epoch_val_acc_meter.update(y_hat, y)
                if isinstance(model, BooleanNetworkNeuralDNF):
                    y_hat_prime = (y_hat > 0.5).long()
                else:
                    y_hat_prime = (torch.sigmoid(y_hat) > 0.5).long()
                epoch_val_jacc_meter.update(y_hat_prime, y)

        val_avg_loss = epoch_val_loss_meter.get_average()
        val_avg_acc = epoch_val_acc_meter.get_average()
        val_sample_jaccard = epoch_val_jacc_meter.get_average()
        val_macro_jaccard = epoch_val_jacc_meter.get_average("macro")
        assert isinstance(val_sample_jaccard, float)
        assert isinstance(val_macro_jaccard, float)
        if epoch % log_interval == 0:
            log.info(
                f"  [{epoch + 1:3d}] Val                  "
                f"avg loss: {val_avg_loss:.3f}  avg acc: {val_avg_acc:.3f}  "
                f"sample jacc: {val_sample_jaccard:.3f}  "
                f"macro jacc: {val_macro_jaccard:.3f}"
            )

        # -------------------------------------------------------------------- #
        # 3. Schedulers step
        # -------------------------------------------------------------------- #
        scheduler.step()

        if isinstance(model, BooleanNetworkNeuralDNF):
            # Update delta value
            if not isinstance(dds, DeltaDelayedMonitoringDecayScheduler):
                delta_dict = dds.step(model.ndnf)
            else:
                delta_dict = dds.step(model.ndnf, val_avg_acc)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

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
                "train/sample_jaccard": avg_sample_jacc,
                "train/macro_jaccard": avg_macro_jacc,
                "val/loss": val_avg_loss,
                "val/accuracy": val_avg_acc,
                "val/sample_jaccard": val_sample_jaccard,
                "val/macro_jaccard": val_macro_jaccard,
            }

            if isinstance(model, BooleanNetworkNeuralDNF):
                wandb_log_dict["train/delta"] = old_delta
            for key, meter in train_loss_meters.items():
                if key == "overall_loss":
                    continue
                wandb_log_dict[f"train/{key}"] = meter.get_average()

            if log_lr_in_wandb:
                # Log learning rate
                wandb_log_dict["train/lr"] = optimiser.param_groups[0]["lr"]

            if gen_weight_hist and isinstance(model, BooleanNetworkNeuralDNF):
                # Generate weight histogram
                f1, f2 = generate_weight_histogram(model.ndnf)
                wandb_log_dict["conj_w_hist"] = wandb.Image(f1)
                wandb_log_dict["disj_w_hist"] = wandb.Image(f2)
            wandb.log(wandb_log_dict)

    return {
        "train_loss": avg_loss,
        "train_accuracy": avg_acc,
        "train_sample_jaccard": avg_sample_jacc,
        "train_macro_jaccard": avg_macro_jacc,
        "val_loss": val_avg_loss,
        "val_accuracy": val_avg_acc,
        "val_sample_jaccard": val_sample_jaccard,
        "val_macro_jaccard": val_macro_jaccard,
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
    assert (
        "hold_out" in cfg["dataset"]
        and cfg["dataset"]["hold_out"]["create_hold_out"]
    ), "Hold out test data must be created. If not, use k-fold train script."

    if training_cfg.get("use_full_dataset", False):
        log.info("Using full dataset for training.")
        train_data, val_data = split_boolean_network_data(
            get_boolean_network_data_np_from_path(
                cfg["dataset"], is_test=False
            ),
            test_size=training_cfg.get("val_size", 0.1),
            random_state=training_cfg.get("val_seed", 73),
        )
    else:
        train_data = get_boolean_network_full_data_np_from_path(cfg["dataset"])
        val_data = get_boolean_network_full_data_np_from_path(cfg["dataset"])

    train_dataset = BooleanNetworkDataset(
        dataset_type=cfg["dataset"]["dataset_name"],
        subtype=None,
        data=train_data,
    )
    val_dataset = BooleanNetworkDataset(
        dataset_type=cfg["dataset"]["dataset_name"],
        subtype=None,
        data=val_data,
    )
    test_dataset = BooleanNetworkDataset(
        dataset_type=cfg["dataset"]["dataset_name"],
        subtype=None,
        data=get_boolean_network_data_np_from_path(
            cfg["dataset"], is_test=True
        ),
    )
    full_dataset = BooleanNetworkDataset(
        dataset_type=cfg["dataset"]["dataset_name"],
        subtype=None,
        data=get_boolean_network_full_data_np_from_path(cfg["dataset"]),
    )

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
    full_dataset_loader = DataLoader(
        full_dataset,
        batch_size=NON_TRAIN_LOADER_BATCH_SIZE,
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
    )

    # number of genes, also the number of input and output for our model
    num_genes = train_data.shape[2]

    # Model
    model = construct_model(training_cfg, num_genes)
    model.to(device)

    _train(training_cfg, model, train_loader, val_loader, device, use_wandb)

    if isinstance(model, BooleanNetworkNeuralDNF):
        # Set delta to 1.0 for evaluation
        model.ndnf.set_delta_val(1.0)

    model_path = run_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    # Evaluate
    final_test_loader = val_loader if is_sweep else test_loader
    test_eval_result = parse_eval_return_meters_with_logging(
        boolean_network_classifier_eval(model, device, final_test_loader),
        model_name=f"Model after training ({'val' if is_sweep else 'test'})",
        do_logging=True,
    )
    full_dataset_eval_result = parse_eval_return_meters_with_logging(
        boolean_network_classifier_eval(model, device, full_dataset_loader),
        model_name="Model on full dataset",
        do_logging=True,
    )

    if use_wandb:
        wandb.log(full_dataset_eval_result)
        wandb.save(glob_str=str(model_path.absolute()))

    with open("train_result.json", "w") as f:
        json.dump(
            {
                "val" if is_sweep else "test": test_eval_result,
                "full_dataset": full_dataset_eval_result,
            },
            f,
            indent=4,
        )
    return test_eval_result


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # We expect the experiment name to be in the format of:
    # bn_{dataset}_{mlp/ndnf}_...
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
    import warnings

    warnings.filterwarnings("ignore")

    run_experiment()
