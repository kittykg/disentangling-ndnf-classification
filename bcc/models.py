import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf import NeuralDNF


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
    ):
        super().__init__()

        self.predicate_inventor = nn.Parameter(
            torch.empty(num_features, invented_predicate_per_input)
        )  # P x Q
        nn.init.normal_(self.predicate_inventor)

        self.ndnf = NeuralDNF(
            n_in=num_features * invented_predicate_per_input,
            n_conjunctions=num_conjunctions,
            n_out=1,
            delta=1.0,
        )

    def get_invented_predicates(
        self, x: Tensor, discretised: bool = False
    ) -> Tensor:
        # x: B x P
        x = torch.tanh(x.unsqueeze(-1) - self.predicate_inventor)
        # x: B x P x Q, x \in (-1, 1)
        x = x.flatten(start_dim=1)
        # x: B x (P * Q)
        if discretised:
            x = torch.sign(x)
        return x

    def get_conjunction(self, x: Tensor) -> Tensor:
        # x: B x P
        x = self.get_invented_predicates(x)
        # x: B x (P * Q)
        return self.ndnf.get_conjunction(x)

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x P
        x = self.get_invented_predicates(x, discretise_invented_predicate)
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
