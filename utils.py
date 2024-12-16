from datetime import datetime
import os
import requests

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from omegaconf import DictConfig
import torch
from torch import nn

from neural_dnf import NeuralDNF, NeuralDNFEO
from neural_dnf.neural_dnf import BaseNeuralDNF

################################################################################
#                                   Plotting                                   #
################################################################################


def generate_weight_histogram(
    model: BaseNeuralDNF, num_bins: int = 20
) -> tuple[Figure, Figure]:
    conj_w = model.conjunctions.weights.data.flatten().detach().cpu().numpy()
    disj_w = model.disjunctions.weights.data.flatten().detach().cpu().numpy()

    f1 = plt.figure(figsize=(20, 15))
    plt.title("Conjunction weight distribution")
    arr = plt.hist(conj_w, bins=num_bins)
    for i in range(num_bins):
        plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))  # type: ignore

    f2 = plt.figure(figsize=(20, 15))
    plt.title("Disjunction weight distribution")
    arr = plt.hist(disj_w, bins=num_bins)
    for i in range(num_bins):
        plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))  # type: ignore

    return f1, f2


################################################################################
#                                 Model management                             #
################################################################################

model_type_str_to_class: dict[str, type[BaseNeuralDNF]] = {
    "plain": NeuralDNF,
    "eo": NeuralDNFEO,
}


def construct_mlp(cfg: DictConfig, delta: float = 1.0) -> nn.Sequential:
    model_arch_cfg = cfg["model_architecture"]
    layers = []
    n_in = model_arch_cfg["n_in"]
    for n_out in model_arch_cfg["hidden_layers"]:
        layers.append(nn.Linear(n_in, n_out))
        layers.append(nn.ReLU())
        n_in = n_out
    layers.append(nn.Linear(n_in, model_arch_cfg["num_classes"]))
    return nn.Sequential(*layers)


def construct_ndnf_based_model(
    cfg: DictConfig, delta: float = 1.0
) -> BaseNeuralDNF:
    model_class = model_type_str_to_class[cfg["model_type"]]
    model_arch_cfg = cfg["model_architecture"]
    return model_class(
        n_in=model_arch_cfg["n_in"],
        n_conjunctions=model_arch_cfg["n_conjunctions"],
        n_out=model_arch_cfg["num_classes"],
        delta=delta,
        weight_init_type=model_arch_cfg["weight_init_type"],
    )


def load_pretrained_model_state_dict(
    model: nn.Module, model_pth: str, device: torch.device
) -> None:
    pretrain_dict = torch.load(
        model_pth, map_location=device, weights_only=True
    )
    model.load_state_dict(pretrain_dict)
    model.to(device)


def freeze_model(model: nn.Module):
    for _, param in model.named_parameters():
        param.requires_grad = False


################################################################################
#                              Webhook utils                                   #
################################################################################


def post_to_discord_webhook(
    webhook_url: str,
    experiment_name: str,
    message_body: str,
    errored: bool,
    keyboard_interrupt: bool | None = None,
) -> None:
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    nodename = os.uname().nodename
    if keyboard_interrupt:
        message_head = (
            f"[{dt}]\n"
            f"Experiment {experiment_name} on hine {nodename} "
            f"INTERRUPTED!!\n"
        )
    else:
        message_head = (
            f"[{dt}]\n"
            f"Experiment {experiment_name} on hine {nodename} "
            f"{'ERRORED' if errored else 'FINISHED'}!!\n"
        )

    requests.post(webhook_url, json={"content": message_head + message_body})
