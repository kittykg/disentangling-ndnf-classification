from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf import NeuralDNF


class BooleanNetworkClassifier(nn.Module):
    num_input: int
    num_output: int

    def __init__(self, num_input: int, num_output: int):
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class BooleanNetworkMLP(BooleanNetworkClassifier):
    num_latent: int

    def __init__(self, num_input: int, num_output: int, num_latent: int):
        super().__init__(num_input, num_output)
        self.num_latent = num_latent

        self.mlp = nn.Sequential(
            nn.Linear(num_input, num_latent),
            nn.Tanh(),
            nn.Linear(num_latent, num_output),
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


class BooleanNetworkNeuralDNF(BooleanNetworkClassifier):
    num_conjunctions: int

    def __init__(self, num_input: int, num_output: int, num_conjunctions: int):
        super().__init__(num_input, num_output)
        self.num_conjunctions = num_conjunctions

        self.ndnf = NeuralDNF(
            n_in=num_input,
            n_conjunctions=num_conjunctions,
            n_out=num_output,
            delta=1.0,
        )

    def get_conjunction(self, x: Tensor) -> Tensor:
        # x: B x P
        return self.ndnf.get_conjunction(x)

    def forward(self, x: Tensor) -> Tensor:
        # x: B x P
        return self.ndnf(x)

    def get_weight_reg_loss(self, take_mean: bool = True) -> Tensor:
        p_t = torch.cat(
            [
                parameter.view(-1)
                for parameter in self.ndnf.parameters()
                if parameter.requires_grad
            ]
        )
        reg_loss = torch.abs(p_t * (6 - torch.abs(p_t)))
        if take_mean:
            return reg_loss.mean()
        return reg_loss

    def change_ndnf(self, new_ndnf: NeuralDNF):
        self.ndnf = new_ndnf


def construct_model(
    cfg: DictConfig, num_genes: int
) -> BooleanNetworkClassifier:
    if cfg["model_type"] == "ndnf":
        return BooleanNetworkNeuralDNF(
            num_input=num_genes,
            num_output=num_genes,
            num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
        )

    return BooleanNetworkMLP(
        num_input=num_genes,
        num_output=num_genes,
        num_latent=cfg["model_architecture"].get("num_latent", 64),
    )
