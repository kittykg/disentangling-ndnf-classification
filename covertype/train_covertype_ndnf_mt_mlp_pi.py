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
    DeltaDelayedDecayScheduler,
    DeltaDelayedExponentialDecayScheduler,
    DeltaDelayedLinearDecayScheduler,
    DeltaDelayedMonotonicFunctionScheduler,
    DeltaDelayedMonitoringDecayScheduler,
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

from analysis import (
    MetricValueMeter,
    AccuracyMeter,
    JaccardScoreMeter,
)
from utils import post_to_discord_webhook, generate_weight_histogram

from covertype.eval.eval_common import (
    covertype_classifier_eval,
    parse_eval_return_meters_with_logging,
)
from covertype.data_utils_covertype import (
    CoverTypeDataset,
    get_covertype_data_np_from_path,
    get_x_and_y_covertype,
)
from covertype.models import (
    CoverTypeMLPPINeuralDNFMT,
    construct_model,
)

NON_TRAIN_LOADER_BATCH_SIZE = 4096
log = logging.getLogger()


def loss_calculation(
    criterion: torch.nn.Module,
    y_hat: Tensor,
    y: Tensor,
    model: CoverTypeMLPPINeuralDNFMT,
    conj_out: Tensor,
    invented_predicates: Tensor,
    all_forms_dict: dict[str, dict[str, Tensor]],
) -> dict[str, Tensor]:
    loss_dict = {
        "base_loss": criterion(y_hat, y),
        "weight_reg_loss": model.get_weight_reg_loss(),
        "conj_reg_loss": (1 - conj_out.abs()).mean(),
        "invented_predicates_reg_loss": (1 - invented_predicates.abs()).mean(),
    }

    # MT regularisation loss: push the mutex-tanh activation and the tanh
    # activation to be the same
    disj_mt = all_forms_dict["disjunction"]["mutex_tanh"]
    disj_tanh = all_forms_dict["disjunction"]["tanh"]

    p_k = (disj_mt + 1) / 2
    p_k_hat = (disj_tanh + 1) / 2
    loss_dict["mt_reg_loss"] = -torch.sum(
        p_k * torch.log(p_k_hat + 1e-8)
        + (1 - p_k) * torch.log(1 - p_k_hat + 1e-8)
    )

    return loss_dict


def _train(
    training_cfg: DictConfig,
    model: CoverTypeMLPPINeuralDNFMT,
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
    criterion = nn.CrossEntropyLoss()

    # Delta delay scheduler for NeuralDNF based model
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

    # Meters
    train_loss_meters = {
        "overall_loss": MetricValueMeter("overall_loss_meter"),
        "base_loss": MetricValueMeter("base_loss_meter"),
        "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
        "conj_reg_loss": MetricValueMeter("conj_reg_loss_meter"),
        "invented_predicates_reg_loss": MetricValueMeter(
            "invented_predicates_reg_loss_meter"
        ),
        "mt_reg_loss": MetricValueMeter("mt_reg_loss_meter"),
    }

    train_acc_meter = AccuracyMeter()
    train_jacc_meter = JaccardScoreMeter()
    epoch_val_loss_meter = MetricValueMeter("val_loss_meter")
    epoch_val_acc_meter = AccuracyMeter()
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

            x, y = get_x_and_y_covertype(data, device, use_ndnf=True)
            y_hat = model(x)
            conj_out = model.get_conjunction(x)
            invented_predicates = model.get_invented_predicates(x)
            all_forms_dict = model.get_all_forms(x)

            loss_dict = loss_calculation(
                criterion,
                y_hat,
                y,
                model,
                conj_out,
                invented_predicates,
                all_forms_dict,
            )

            loss = (
                loss_dict["base_loss"]
                + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                * loss_dict["weight_reg_loss"]
                + training_cfg["aux_loss"]["tanh_conj_lambda"]
                * loss_dict["conj_reg_loss"]
                + training_cfg["aux_loss"]["pi_lambda"]
                * loss_dict["invented_predicates_reg_loss"]
                + training_cfg["aux_loss"]["mt_reg_lambda"]
                * loss_dict["mt_reg_loss"]
            )

            loss.backward()

            # If using NDNF model, check if some weights are manually set to be
            # sparse. If so, after backward, we need to set the gradients of
            # those weights to 0, so that they are not updated.
            if (
                model.manually_sparse_conj_layer_k is not None
                and model.manually_sparse_conj_layer_k > 0
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

            # Update jacc meter
            with torch.no_grad():
                y_hat_prime = (all_forms_dict["disjunction"]["tanh"] > 0).long()
            train_jacc_meter.update(y_hat_prime, y)

        # Log average performance for train
        avg_loss = train_loss_meters["overall_loss"].get_average()
        avg_acc = train_acc_meter.get_average()
        avg_sample_jacc = train_jacc_meter.get_average()
        avg_macro_jacc = train_jacc_meter.get_average("macro")
        assert isinstance(avg_sample_jacc, float)
        assert isinstance(avg_macro_jacc, float)

        if epoch % log_interval == 0:
            log_info_str = (
                f"  [{epoch + 1:3d}] "
                f"Train  Delta: {model.ndnf.get_delta_val()[0]:.3f}  "
                f"avg loss: {avg_loss:.3f}  "
                f"avg acc: {avg_acc:.3f}  "
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
                x, y = get_x_and_y_covertype(data, device, use_ndnf=True)

                y_hat = model(x)
                conj_out = model.get_conjunction(x)
                invented_predicates = model.get_invented_predicates(x)
                all_forms_dict = model.get_all_forms(x)

                loss_dict = loss_calculation(
                    criterion,
                    y_hat,
                    y,
                    model,
                    conj_out,
                    invented_predicates,
                    all_forms_dict,
                )

                loss = (
                    loss_dict["base_loss"]
                    + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                    * loss_dict["weight_reg_loss"]
                    + training_cfg["aux_loss"]["tanh_conj_lambda"]
                    * loss_dict["conj_reg_loss"]
                    + training_cfg["aux_loss"]["pi_lambda"]
                    * loss_dict["invented_predicates_reg_loss"]
                    + training_cfg["aux_loss"]["mt_reg_lambda"]
                    * loss_dict["mt_reg_loss"]
                )

                # Update meters
                epoch_val_loss_meter.update(loss.item())
                epoch_val_acc_meter.update(y_hat, y)
                y_hat_prime = (all_forms_dict["disjunction"]["tanh"] > 0).long()
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

        # Update delta value
        if not isinstance(dds, DeltaDelayedMonitoringDecayScheduler):
            delta_dict = dds.step(model.ndnf)
        else:
            delta_dict = dds.step(model.ndnf, val_avg_acc)
        new_delta = delta_dict["new_delta_vals"][0]
        old_delta = delta_dict["old_delta_vals"][0]

        if new_delta == 1.0:
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
                "train/delta": old_delta,
            }
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
    assert cfg["dataset"].get(
        "convert_categorical_to_binary_encoding", False
    ) == training_cfg.get("convert_categorical_to_binary_encoding", False)
    X, y = get_covertype_data_np_from_path(cfg["dataset"], is_test=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=training_cfg.get("val_size", 0.2),
        random_state=training_cfg.get("val_seed", 73),
    )
    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    hold_out_test_X, hold_out_test_y = get_covertype_data_np_from_path(
        cfg["dataset"], is_test=True
    )
    test_dataset = CoverTypeDataset(hold_out_test_X, hold_out_test_y)

    # Data loaders
    if training_cfg.get("use_weighted_sampler", False):
        # Compute class counts
        class_counts = np.bincount(y_train)
        class_weights = 1 / class_counts
        sampler = torch.utils.data.WeightedRandomSampler(
            class_weights[y_train], len(y_train), replacement=True  # type: ignore
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
        sampler=sampler,
        shuffle=True if sampler is None else False,
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
    assert isinstance(model, CoverTypeMLPPINeuralDNFMT)
    model.to(device)
    log.info(f"Model: {model}")

    _train(training_cfg, model, train_loader, val_loader, device, use_wandb)

    model.ndnf.set_delta_val(1.0)

    model_path = run_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    eval_model = model.to_ndnf_model()
    eval_model.to(device)
    eval_model.eval()

    final_test_loader = val_loader if is_sweep else test_loader
    test_eval_raw_dict = covertype_classifier_eval(
        eval_model, device, final_test_loader
    )
    test_eval_result = parse_eval_return_meters_with_logging(
        test_eval_raw_dict,
        model_name="Model after training",
        do_logging=True,
    )

    with open("train_result.json", "w") as f:
        json.dump(test_eval_result, f, indent=4)

    if use_wandb:
        wandb.save(glob_str=str(model_path.absolute()))
        wandb.log(
            parse_eval_return_meters_with_logging(
                test_eval_raw_dict,
                "Model after training",
                do_logging=False,
                filter_out_list=True,
            )
        )

    return test_eval_result


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # We expect the experiment name to be in the format of:
    # covertype_mlp_pi_...
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
                if isinstance(v, float):
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
