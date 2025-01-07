import json
import logging
from pathlib import Path
import random
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
import torch
from torch import Tensor, nn
from ucimlrepo import fetch_ucirepo
import wandb

from neural_dnf.neural_dnf import NeuralDNF
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from analysis import (
    MetricValueMeter,
    BinaryAccuracyMeter,
    collate,
    synthesize,
)
from data_utils_generic import GenericUCIDataset
from utils import post_to_discord_webhook, generate_weight_histogram


log = logging.getLogger()


class BCCClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class BCCMLP(BCCClassifier):
    def __init__(self, num_features: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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


class BCCNeuralDNF(BCCClassifier):
    def __init__(
        self,
        num_features: int,
        invented_predicate_per_input: int,
        num_conjunctions: int,
        uniform_init_boundary: float = 5,
    ):
        super().__init__()

        self.predicate_inventor = nn.Parameter(
            torch.empty(num_features, invented_predicate_per_input)
        )  # P x Q
        nn.init.uniform_(self.predicate_inventor, a=0, b=uniform_init_boundary)

        self.ndnf = NeuralDNF(
            n_in=num_features * invented_predicate_per_input,
            n_conjunctions=num_conjunctions,
            n_out=1,
            delta=1.0,
        )

    def get_invented_predicates(self, x: Tensor) -> Tensor:
        # x: B x P
        x = torch.tanh(x.unsqueeze(-1) - self.predicate_inventor)
        # x: B x P x Q, x \in (-1, 1)
        x = x.flatten(start_dim=1)
        # x: B x (P * Q)
        return x

    def get_conjunction(self, x: Tensor) -> Tensor:
        # x: B x P
        x = self.get_invented_predicates(x)
        # x: B x (P * Q)
        return self.ndnf.get_conjunction(x)

    def forward(self, x: Tensor) -> Tensor:
        # x: B x P
        x = self.get_invented_predicates(x)
        # x: B x (P * Q)
        return self.ndnf(x)

    def get_weight_reg_loss(self) -> Tensor:
        p_t = torch.cat(
            [
                parameter.view(-1)
                for parameter in self.ndnf.parameters()
                if parameter.requires_grad
            ]
        )
        return torch.abs(p_t * (6 - torch.abs(p_t))).mean()


breast_cancer_coimbra = fetch_ucirepo(id=451)
data = breast_cancer_coimbra.data  # data is pandas DataFrame
X = data.features.to_numpy().astype(np.float32)  # type: ignore
y = data.targets.to_numpy().astype(np.float32).flatten() - 1  # type: ignore
bcc_dataset = GenericUCIDataset(X, y)


def loss_calculation(
    criterion: torch.nn.Module,
    y_hat: Tensor,
    y: Tensor,
    model: BCCClassifier,
    conj_out: Tensor | None = None,
    invented_predicates: Tensor | None = None,
) -> dict[str, Tensor]:
    loss_dict = {
        "base_loss": criterion(y_hat, y),
        "weight_reg_loss": model.get_weight_reg_loss(),
    }

    if conj_out is not None:
        # Conjunction regularisation loss (push to ±1)
        loss_dict["conj_reg_loss"] = (1 - conj_out.abs()).mean()

    if invented_predicates is not None:
        # Invented predicate regularisation loss (push to ±1)
        loss_dict["invented_predicates_reg_loss"] = (
            1 - invented_predicates.abs()
        ).mean()

    return loss_dict


def train_fold(
    fold_id: int,
    train_index: npt.NDArray[np.int64],
    test_index: npt.NDArray[np.int64],
    training_cfg: DictConfig,
    device: torch.device,
    use_wandb: bool,
) -> tuple[BCCClassifier, dict[str, float]]:

    # Model
    if training_cfg["model_type"] == "ndnf":
        model = BCCNeuralDNF(
            num_features=bcc_dataset.X.shape[1],
            invented_predicate_per_input=training_cfg["model_architecture"][
                "invented_predicate_per_input"
            ],
            num_conjunctions=training_cfg["model_architecture"][
                "n_conjunctions"
            ],
        )
    else:
        model = BCCMLP(num_features=bcc_dataset.X.shape[1])
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
    loss_func_key = training_cfg["loss_func"]
    # MLP -> BCEWithLogitsLoss, NDNF -> BCELoss / MSELoss
    if isinstance(model, BCCMLP):
        criterion = nn.BCEWithLogitsLoss()
    elif loss_func_key == "bce":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Delta delay scheduler if using BCCNeuralDNF
    if isinstance(model, BCCNeuralDNF):
        dds = DeltaDelayedExponentialDecayScheduler(
            initial_delta=training_cfg["dds"]["initial_delta"],
            delta_decay_delay=training_cfg["dds"]["delta_decay_delay"],
            delta_decay_steps=training_cfg["dds"]["delta_decay_steps"],
            delta_decay_rate=training_cfg["dds"]["delta_decay_rate"],
            target_module_type=model.ndnf.__class__.__name__,
        )
        model.ndnf.set_delta_val(0.1)
        delta_one_counter = 0

    # Other training settings
    gen_weight_hist = training_cfg.get("gen_weight_hist", False)
    log_interval = training_cfg.get("log_interval", 100)

    for epoch in range(training_cfg["epochs"]):
        # -------------------------------------------------------------------- #
        #  1. Training
        # -------------------------------------------------------------------- #
        train_loss_meters = {
            "overall_loss": MetricValueMeter("overall_loss_meter"),
            "base_loss": MetricValueMeter("base_loss_meter"),
            "weight_reg_loss": MetricValueMeter("weight_reg_loss_meter"),
        }
        if isinstance(model, BCCNeuralDNF):
            train_loss_meters["conj_reg_loss"] = MetricValueMeter(
                "conj_reg_loss_meter"
            )
            train_loss_meters["invented_predicates_reg_loss"] = (
                MetricValueMeter("invented_predicates_reg_loss_meter")
            )
        train_acc_meter = BinaryAccuracyMeter()

        model.train()

        for data in train_loader:
            optimiser.zero_grad()

            x = data[0].to(device)
            y = data[1].to(device)  # y \in {0, 1}

            y_hat = model(x).squeeze()
            conj_out, invented_predicates = None, None
            if isinstance(model, BCCNeuralDNF):
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
            if isinstance(model, BCCNeuralDNF):
                loss += (
                    training_cfg["aux_loss"]["tanh_conj_lambda"]
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

            if isinstance(model, BCCNeuralDNF):
                train_acc_meter.update(y_hat > 0.5, y)
            else:
                train_acc_meter.update(torch.sigmoid(y_hat) > 0.5, y)

        if isinstance(model, BCCNeuralDNF):
            # Update delta value
            delta_dict = dds.step(model.ndnf)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

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
            if isinstance(model, BCCNeuralDNF):
                log_info_str = (
                    f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] Train  "
                    f"Delta: {old_delta:.3f}  avg loss: {avg_loss:.3f}  "
                    f"avg perf: {avg_acc:.3f}"
                )
            else:
                log_info_str = (
                    f"  Fold [{fold_id:2d}] [{epoch + 1:3d}] "
                    f"Train                avg loss: {avg_loss:.3f}  "
                    f"avg acc: {avg_acc:.3f}"
                )
            log.info(log_info_str)

        # -------------------------------------------------------------------- #
        # 2. Evaluate performance on val
        # -------------------------------------------------------------------- #
        epoch_val_loss_meter = MetricValueMeter("val_loss_meter")
        epoch_val_acc_meter = BinaryAccuracyMeter()

        model.eval()

        for data in val_loader:
            with torch.no_grad():
                # Get model output and compute loss
                x = data[0].to(device)
                y = data[1].to(device)

                y_hat = model(x).squeeze()
                conj_out, invented_predicates = None, None
                if isinstance(model, BCCNeuralDNF):
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
                if isinstance(model, BCCNeuralDNF):
                    loss += (
                        training_cfg["aux_loss"]["tanh_conj_lambda"]
                        * loss_dict["conj_reg_loss"]
                        + training_cfg["aux_loss"]["pi_lambda"]
                        * loss_dict["invented_predicates_reg_loss"]
                    )

                # Update meters
                epoch_val_loss_meter.update(loss.item())
                if isinstance(model, BCCNeuralDNF):
                    epoch_val_acc_meter.update(y_hat > 0.5, y)
                else:
                    epoch_val_acc_meter.update(torch.sigmoid(y_hat) > 0.5, y)

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
                f"fold_{fold_id}/val/loss": val_avg_loss,
                f"fold_{fold_id}/val/accuracy": val_avg_acc,
            }
            if isinstance(model, BCCNeuralDNF):
                wandb_log_dict[f"fold_{fold_id}/delta"] = old_delta
            for key, meter in train_loss_meters.items():
                if key == "overall_loss":
                    continue
                wandb_log_dict[f"fold_{fold_id}/train/{key}"] = (
                    meter.get_average()
                )
            if gen_weight_hist:
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
            fold_id, train_index, test_index, training_cfg, device, use_wandb
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # We expect the experiment name to be in the format of:
    # cub_{no. classes}_ndnf_{eo/plain}_...
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
