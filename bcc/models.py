from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf import NeuralDNF

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from predicate_invention import NeuralDNFPredicateInventor


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
    predicate_inventor: NeuralDNFPredicateInventor
    ndnf: NeuralDNF

    def __init__(
        self,
        num_features: int,
        invented_predicate_per_input: int,
        num_conjunctions: int,
        predicate_inventor_tau: float = 1.0,
    ):
        super().__init__()

        self.predicate_inventor = NeuralDNFPredicateInventor(
            num_features=num_features,
            invented_predicate_per_input=invented_predicate_per_input,
            tau=predicate_inventor_tau,
        )

        self.ndnf = NeuralDNF(
            n_in=num_features * invented_predicate_per_input,
            n_conjunctions=num_conjunctions,
            n_out=1,
            delta=1.0,
        )

    def get_invented_predicates(
        self, x: Tensor, discretised: bool = False
    ) -> Tensor:
        return self.predicate_inventor(x, discretised)

    def get_conjunction(self, x: Tensor) -> Tensor:
        # x: B x P
        x = self.predicate_inventor(x)
        # x: B x (P * Q)
        return self.ndnf.get_conjunction(x)

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x P
        x = self.predicate_inventor(x, discretise_invented_predicate)
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

    def change_ndnf(self, new_ndnf: NeuralDNF):
        self.ndnf = new_ndnf
