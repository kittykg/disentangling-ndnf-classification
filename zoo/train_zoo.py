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
from sklearn.model_selection import StratifiedKFold
import torch
from torch import Tensor, nn
import wandb

from neural_dnf.neural_dnf import BaseNeuralDNF, NeuralDNF, NeuralDNFEO
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

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
    collate,
    synthesize,
)
from zoo.data_utils_zoo import *
from utils import post_to_discord_webhook, generate_weight_histogram


ZOO_PROCESSED_NUM_FEATURES = 21
ZOO_NUM_CLASSES = 7

log = logging.getLogger()


class ZooMLP(nn.Module):
    def __init__(self, num_latent: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(ZOO_PROCESSED_NUM_FEATURES, num_latent),
            nn.Tanh(),
            nn.Linear(num_latent, num_latent),
            nn.Tanh(),
            nn.Linear(num_latent, ZOO_NUM_CLASSES),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def get_weight_reg_loss(self) -> Tensor:
        # L1 regularisation
        p_t = torch.cat(
            [
                parameter.view(-1)
                for parameter in self.parameters()
                if parameter.requires_grad
            ]
        )
        return p_t.abs().mean()


def construct_model(cfg: DictConfig) -> BaseNeuralDNF | ZooMLP:
    model_arch_cfg = cfg["model_architecture"]
    model_type = cfg["model_type"]
    if model_type == "mlp":
        return ZooMLP(model_arch_cfg["num_latent"])

    model_class = NeuralDNFEO if model_type == "eo" else NeuralDNF
    return model_class(
        n_in=model_arch_cfg["n_in"],
        n_conjunctions=model_arch_cfg["n_conjunctions"],
        n_out=ZOO_NUM_CLASSES,
        delta=cfg["dds"]["initial_delta"],
        weight_init_type=model_arch_cfg["weight_init_type"],
    )


def loss_calculation(
    model: BaseNeuralDNF | ZooMLP,
    criterion: torch.nn.Module,
    y_hat: Tensor,
    y: Tensor,
    conj_out: Tensor | None,
) -> dict[str, Tensor]:
    loss_dict = {"base_loss": criterion(y_hat, y)}

    # Weight regularisation loss
    if isinstance(model, BaseNeuralDNF):
        p_t = torch.cat(
            [
                parameter.view(-1)
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        )
        weight_reg_loss = torch.abs(p_t * (6 - torch.abs(p_t))).mean()
    else:
        weight_reg_loss = model.get_weight_reg_loss()
    loss_dict["weight_reg_loss"] = weight_reg_loss

    if isinstance(model, BaseNeuralDNF):
        # Conjunction regularisation loss (push to ±1)
        assert conj_out is not None
        conj_reg_loss = (1 - conj_out.abs()).mean()
        loss_dict["conj_reg_loss"] = conj_reg_loss

    return loss_dict


def train_fold(
    fold_id: int,
    train_index: npt.NDArray[np.int64],
    test_index: npt.NDArray[np.int64],
    training_cfg: DictConfig,
    device: torch.device,
    dataset: ZooDataset,
    use_wandb: bool,
) -> tuple[BaseNeuralDNF | ZooMLP, dict[str, float]]:
    # Model
    model = construct_model(training_cfg)
    model.to(device)
    is_ndnf = isinstance(model, BaseNeuralDNF)

    # Data loaders
    train_loader, val_loader = get_zoo_dataloaders(
        dataset=dataset,
        train_index=train_index,
        test_index=test_index,
        batch_size=training_cfg["batch_size"],
        loader_num_workers=training_cfg.get("loader_num_workers", 0),
        pin_memory=device == torch.device("cuda"),
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
    # loss_func_key = training_cfg["loss_func"]
    criterion = nn.CrossEntropyLoss()

    # Delta delay scheduler
    if is_ndnf:
        dds = DeltaDelayedExponentialDecayScheduler(
            initial_delta=training_cfg["dds"]["initial_delta"],
            delta_decay_delay=training_cfg["dds"]["delta_decay_delay"],
            delta_decay_steps=training_cfg["dds"]["delta_decay_steps"],
            delta_decay_rate=training_cfg["dds"]["delta_decay_rate"],
            target_module_type=model.__class__.__name__,
        )
        delta_one_counter = 0

    # Other training settings
    gen_weight_hist = training_cfg.get("gen_weight_hist", False)
    log_interval = training_cfg.get("log_interval", 100)

    # Meters
    train_loss_meters = {
        "overall_loss": MetricValueMeter("overall_loss_meter"),
        "base_loss": MetricValueMeter("base_loss_meter"),
        "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
    }
    if is_ndnf:
        train_loss_meters["conj_reg_loss"] = MetricValueMeter(
            "conj_reg_loss_meter"
        )
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

            x, y = get_x_and_y_zoo(data, device, use_ndnf=is_ndnf)
            y_hat = model(x)
            conj_out = None
            if is_ndnf:
                model.get_conjunction(x)

            loss_dict = loss_calculation(model, criterion, y_hat, y, conj_out)

            loss = (
                loss_dict["base_loss"]
                + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                * loss_dict["weight_reg_loss"]
            )
            if is_ndnf:
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

        if is_ndnf:
            # NeuralDNF / NeuralDNFEO
            # Update meters
            with torch.no_grad():
                y_hat_prime = (
                    torch.tanh(
                        model.get_plain_output(x)
                        if isinstance(model, NeuralDNFEO)
                        else y_hat
                    )
                    > 0
                ).long()
            train_jacc_meter.update(y_hat_prime, y)

            # Update delta value
            delta_dict = dds.step(model)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

            if new_delta == 1.0:
                # The first time where new_delta_val becomes 1, the network
                # isn't train with delta being 1 for that epoch. So
                # delta_one_counter starts with -1, and when new_delta_val is
                # first time being 1, the delta_one_counter becomes 0. We do not
                # use the delta_one_counter for now, but it can be used to
                # customise when to add auxiliary loss
                delta_one_counter += 1
        else:
            # MLP
            # Update jacc meters
            # convert the max value to the class
            y_hat_prime = torch.zeros(len(y), ZOO_NUM_CLASSES).long()
            y_hat_prime[range(len(y)), torch.argmax(y_hat, dim=1).long()] = 1
            train_jacc_meter.update(y_hat_prime, y)

        # Log average performance for train
        avg_loss = train_loss_meters["overall_loss"].get_average()
        avg_acc = train_acc_meter.get_average()
        avg_sample_jacc = train_jacc_meter.get_average()
        avg_macro_jacc = train_jacc_meter.get_average("macro")
        assert isinstance(avg_sample_jacc, float)
        assert isinstance(avg_macro_jacc, float)

        if epoch % log_interval == 0:
            if is_ndnf:
                log_info_str = (
                    f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] Train  "
                    f"Delta: {old_delta:.3f}  avg loss: {avg_loss:.3f}  "
                    f"avg acc: {avg_acc:.3f}  "
                    f"avg sample jacc: {avg_sample_jacc:.3f}  "
                    f"avg macro jacc: {avg_macro_jacc:.3f}"
                )
            else:
                log_info_str = (
                    f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] "
                    f"Train                avg loss: {avg_loss:.3f}  "
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
                x, y = get_x_and_y_zoo(data, device, use_ndnf=is_ndnf)
                if isinstance(model, NeuralDNFEO):
                    y_hat = model.get_plain_output(x)
                else:
                    y_hat = model(x)

                conj_out = None
                if is_ndnf:
                    conj_out = model.get_conjunction(x)

                loss_dict = loss_calculation(
                    model, criterion, y_hat, y, conj_out
                )

                loss = (
                    loss_dict["base_loss"]
                    + training_cfg["aux_loss"]["weight_l1_mod_lambda"]
                    * loss_dict["weight_reg_loss"]
                )
                if is_ndnf:
                    loss += (
                        training_cfg["aux_loss"]["tanh_conj_lambda"]
                        * loss_dict["conj_reg_loss"]
                    )

                # Update meters
                epoch_val_loss_meter.update(loss.item())
                epoch_val_acc_meter.update(y_hat, y)
                if is_ndnf:
                    # To get the jaccard score, we need to threshold the tanh
                    # activation to get the binary prediction of each class
                    y_hat = (torch.tanh(y_hat) > 0).long()
                    epoch_val_jacc_meter.update(y_hat, y)
                else:
                    y_hat_prime = torch.zeros(len(y), ZOO_NUM_CLASSES).long()
                    y_hat_prime[
                        range(len(y)), torch.argmax(y_hat, dim=1).long()
                    ] = 1
                    epoch_val_jacc_meter.update(y_hat_prime, y)

        val_avg_loss = epoch_val_loss_meter.get_average()
        val_avg_acc = epoch_val_acc_meter.get_average()
        val_sample_jaccard = epoch_val_jacc_meter.get_average()
        val_macro_jaccard = epoch_val_jacc_meter.get_average("macro")
        assert isinstance(val_sample_jaccard, float)
        assert isinstance(val_macro_jaccard, float)
        if epoch % log_interval == 0:
            log.info(
                f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] "
                f"Val                  avg loss: {val_avg_loss:.3f}  "
                f"avg acc: {val_avg_acc:.3f}  "
                f"sample jacc: {val_sample_jaccard:.3f}  "
                f"macro jacc: {val_macro_jaccard:.3f}"
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
                f"fold_{fold_id}/train/epoch": epoch,
                f"fold_{fold_id}/train/loss": avg_loss,
                f"fold_{fold_id}/train/accuracy": avg_acc,
                f"fold_{fold_id}/train/sample_jaccard": avg_sample_jacc,
                f"fold_{fold_id}/train/macro_jaccard": avg_macro_jacc,
                f"fold_{fold_id}/val/loss": val_avg_loss,
                f"fold_{fold_id}/val/accuracy": val_avg_acc,
                f"fold_{fold_id}/val/sample_jaccard": val_sample_jaccard,
                f"fold_{fold_id}/val/macro_jaccard": val_macro_jaccard,
            }
            if is_ndnf:
                wandb_log_dict[f"fold_{fold_id}/delta"] = old_delta
            for key, meter in train_loss_meters.items():
                if key == "overall_loss":
                    continue
                wandb_log_dict[f"fold_{fold_id}/train/{key}"] = (
                    meter.get_average()
                )
            if gen_weight_hist and is_ndnf:
                # Generate weight histogram
                f1, f2 = generate_weight_histogram(model)
                wandb_log_dict[f"fold_{fold_id}/conj_w_hist"] = wandb.Image(f1)
                wandb_log_dict[f"fold_{fold_id}/disj_w_hist"] = wandb.Image(f2)
            wandb.log(wandb_log_dict)

    return model, {
        "train_loss": avg_loss,
        "train_accuracy": avg_acc,
        "train_sample_jaccard": avg_sample_jacc,
        "train_macro_jaccard": avg_macro_jacc,
        "val_loss": val_avg_loss,
        "val_accuracy": val_avg_acc,
        "val_sample_jaccard": val_sample_jaccard,
        "val_macro_jaccard": val_macro_jaccard,
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

    # Load data
    X, y, _ = get_zoo_data_np_from_path(
        data_dir_path=Path(cfg["dataset"]["save_dir"])
    )
    dataset = ZooDataset(X, y)

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
            training_cfg,
            device,
            dataset,
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
    # zoo_{mlp/ndnf/ndnf_eo}_...
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
