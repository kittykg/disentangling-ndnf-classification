from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf import NeuralDNF


class MushroomClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class MushroomMLP(MushroomClassifier):
    def __init__(self, num_features: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Tanh(),
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


class MushroomNeuralDNF(MushroomClassifier):
    def __init__(self, num_features: int, num_conjunctions: int):
        super().__init__()

        self.ndnf = NeuralDNF(
            n_in=num_features,
            n_conjunctions=num_conjunctions,
            n_out=1,
            delta=1.0,
        )

    def get_conjunction(self, x: Tensor) -> Tensor:
        # x: B x P
        return self.ndnf.get_conjunction(x)

    def forward(self, x: Tensor) -> Tensor:
        # x: B x P
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

    def change_ndnf(self, new_ndnf: NeuralDNF):
        self.ndnf = new_ndnf


def construct_model(cfg: DictConfig, num_features: int) -> MushroomClassifier:
    if cfg["model_type"] == "ndnf":
        return MushroomNeuralDNF(
            num_features=num_features,
            num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
        )

    return MushroomMLP(num_features=num_features)
