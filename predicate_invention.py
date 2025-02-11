"""
This file provides a pytorch implementation of the predicate inventor module and
its tau scheduler.

The predicate inventor module can convert real-valued input features into a set
of invented predicates in (-1, 1), by learning a set of thresholds:
for each input feature f_i, the predicate inventor learns t_{i, 0}, t_{i, 1},
..., and each invented predicate p_{i, j} is defined as p_{i, j} = tanh((f_i -
t_{i, j}) / tau). gbl(p_{i, j}) = \top if f_i > t_{i, j}.

The tau parameter controls the smoothness of the invented predicates. The tau
scheduler can be used to decay the tau scheduler.
"""

import torch
from torch import nn, Tensor


class NeuralDNFPredicateInventor(nn.Module):
    predicate_inventor: nn.Parameter
    tau: float

    def __init__(
        self,
        num_features: int,
        invented_predicate_per_input: int,
        tau: float = 1.0,
    ):
        super().__init__()

        self.predicate_inventor = nn.Parameter(
            torch.empty(num_features, invented_predicate_per_input)
        )
        nn.init.normal_(self.predicate_inventor)

        self.tau = tau

    def forward(self, x: Tensor, discretised: bool = False) -> Tensor:
        # x: B x P
        x = torch.tanh((x.unsqueeze(-1) - self.predicate_inventor) / self.tau)
        # x: B x P x Q, x \in (-1, 1)
        x = x.flatten(start_dim=1)
        # x: B x (P * Q)
        if discretised:
            x = torch.sign(x)
        return x


class PredicateInventorTauScheduler:
    initial_tau: float
    tau_decay_delay: int
    tau_decay_steps: int
    tau_decay_rate: float
    min_tau: float

    internal_step_counter: int

    def __init__(
        self,
        initial_tau: float,
        tau_decay_delay: int,
        tau_decay_steps: int,
        tau_decay_rate: float,
        min_tau: float,
    ):
        self.initial_tau = initial_tau
        self.tau_decay_delay = tau_decay_delay
        self.tau_decay_steps = tau_decay_steps
        self.tau_decay_rate = tau_decay_rate
        self.min_tau = min_tau

        self.internal_step_counter = 0

    def step(self, predicate_inventor: NeuralDNFPredicateInventor):
        old_tau = predicate_inventor.tau
        if old_tau == self.min_tau:
            new_tau = self.min_tau
        else:
            new_tau = self._calculate_new_tau(self.internal_step_counter)
            new_tau = max(new_tau, self.min_tau)
        predicate_inventor.tau = new_tau
        self.internal_step_counter += 1
        return {
            "old_tau": old_tau,
            "new_tau": new_tau,
        }

    def _calculate_new_tau(self, step: int) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        self.internal_step_counter = 0


class DelayedExpontentialTauDecayScheduler(PredicateInventorTauScheduler):
    def _calculate_new_tau(self, step: int) -> float:
        if step < self.tau_decay_delay:
            return self.initial_tau

        return self.initial_tau * self.tau_decay_rate ** (
            (step - self.tau_decay_delay) // self.tau_decay_steps + 1
        )


class DelayedLinearTauDecayScheduler(PredicateInventorTauScheduler):
    def _calculate_new_tau(self, step: int) -> float:
        if step < self.tau_decay_delay:
            return self.initial_tau

        step_diff = step - self.tau_decay_delay
        return (
            self.initial_tau
            - (step_diff // self.tau_decay_steps) * self.tau_decay_rate
        )
