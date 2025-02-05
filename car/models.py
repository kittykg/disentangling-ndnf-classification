from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf.neural_dnf import BaseNeuralDNF, NeuralDNF, NeuralDNFEO

CAR_NUM_CLASSES: int = 4


class CarClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class CarMLP(CarClassifier):
    def __init__(self, num_features: int, num_latent: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_latent),
            nn.ReLU(),
            nn.Linear(num_latent, num_latent),
            nn.ReLU(),
            nn.Linear(num_latent, CAR_NUM_CLASSES),
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


class CarBaseNeuralDNF(CarClassifier):
    ndnf: BaseNeuralDNF
    num_features: int
    num_conjunctions: int

    def __init__(self, num_features: int, num_conjunctions: int):
        super().__init__()
        self.num_features = num_features
        self.num_conjunctions = num_conjunctions
        self.ndnf = self._create_ndnf_model()

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

    def _create_ndnf_model(self) -> BaseNeuralDNF:
        raise NotImplementedError


class CarNeuralDNF(CarBaseNeuralDNF):
    """
    This class is not expected to be trained directely
    """

    ndnf: NeuralDNF

    def _create_ndnf_model(self):
        return NeuralDNF(
            n_in=self.num_features,
            n_conjunctions=self.num_conjunctions,
            n_out=CAR_NUM_CLASSES,
            delta=1.0,
        )


class CarNeuralDNFEO(CarBaseNeuralDNF):
    ndnf: NeuralDNFEO

    def _create_ndnf_model(self):
        return NeuralDNFEO(
            n_in=self.num_features,
            n_conjunctions=self.num_conjunctions,
            n_out=CAR_NUM_CLASSES,
            delta=1.0,
        )

    def get_pre_eo_output(self, x: Tensor) -> Tensor:
        # x: B x P
        return self.ndnf.get_plain_output(x)

    def to_ndnf_model(self) -> CarNeuralDNF:
        ndnf_model = CarNeuralDNF(
            num_features=self.num_features,
            num_conjunctions=self.num_conjunctions,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        return ndnf_model


def construct_model(cfg: DictConfig, num_features: int) -> CarClassifier:
    if cfg["model_type"] == "eo":
        return CarNeuralDNFEO(
            num_features=num_features,
            num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
        )

    return CarMLP(
        num_features=num_features,
        num_latent=cfg["model_architecture"]["num_latent"],
    )
