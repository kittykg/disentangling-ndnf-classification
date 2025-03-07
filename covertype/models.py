from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf.neural_dnf import (
    BaseNeuralDNF,
    NeuralDNF,
    NeuralDNFEO,
    NeuralDNFMutexTanh,
)
from predicate_invention import NeuralDNFPredicateInventor

COVERTYPE_NUM_CLASSES: int = 7
COVERTYPE_NUM_REAL_VALUED_FEATURES: int = 4
COVERTYPE_NUM_BINARY_FEATURES: int = 40
COVERTYPE_TOTAL_NUM_FEATURES: int = (
    COVERTYPE_NUM_REAL_VALUED_FEATURES + COVERTYPE_NUM_BINARY_FEATURES
)


class CoverTypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class CoverTypeMLP(CoverTypeClassifier):
    def __init__(self, num_latent: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(COVERTYPE_TOTAL_NUM_FEATURES, num_latent),
            nn.ReLU(),
            nn.Linear(num_latent, num_latent),
            nn.ReLU(),
            nn.Linear(num_latent, COVERTYPE_NUM_CLASSES),
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


class CoverTypeBaseNeuralDNF(CoverTypeClassifier):
    invented_predicate_per_input: int
    num_conjunctions: int

    predicate_inventor: NeuralDNFPredicateInventor
    ndnf: BaseNeuralDNF

    def __init__(
        self,
        invented_predicate_per_input: int,
        num_conjunctions: int,
        predicate_inventor_tau: float = 1.0,
    ):
        super().__init__()

        self.invented_predicate_per_input = invented_predicate_per_input
        self.num_conjunctions = num_conjunctions

        self.predicate_inventor = NeuralDNFPredicateInventor(
            num_features=COVERTYPE_NUM_REAL_VALUED_FEATURES,
            invented_predicate_per_input=invented_predicate_per_input,
            tau=predicate_inventor_tau,
        )

        self.ndnf = self._create_ndnf_model()

    def get_invented_predicates(
        self, x: Tensor, discretised: bool = False
    ) -> Tensor:
        """
        This function compute the invented predicates from the real valued
        features of the input data, and concat them with the binary features.
        """
        # x: B x 44
        # We only take the real valued features
        real_val_features = x[:, :COVERTYPE_NUM_REAL_VALUED_FEATURES]
        # real_val_features: B x 4
        binary_features = x[:, COVERTYPE_NUM_REAL_VALUED_FEATURES:]
        # binary_features: B x 40
        invented_predicates = self.predicate_inventor(
            real_val_features, discretised
        )
        # invented_predicates: B x (4 * IP)
        final_tensor = torch.cat([invented_predicates, binary_features], dim=1)
        # final_tensor: B x (4 * IP + 40)
        return final_tensor

    def get_conjunction(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (4 * IP + 40)
        return self.ndnf.get_conjunction(x)

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (4 * IP + 40)
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


class CoverTypeNeuralDNF(CoverTypeBaseNeuralDNF):
    """
    This class is not expected to be trained directely
    """

    ndnf: NeuralDNF

    def _create_ndnf_model(self) -> NeuralDNF:
        return NeuralDNF(
            n_in=self.invented_predicate_per_input
            * COVERTYPE_NUM_REAL_VALUED_FEATURES
            + COVERTYPE_NUM_BINARY_FEATURES,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def change_ndnf(self, new_ndnf: NeuralDNF) -> None:
        self.ndnf = new_ndnf


class CoverTypeNeuralDNFEO(CoverTypeBaseNeuralDNF):
    ndnf: NeuralDNFEO

    def _create_ndnf_model(self) -> NeuralDNFEO:
        return NeuralDNFEO(
            n_in=self.invented_predicate_per_input
            * COVERTYPE_NUM_REAL_VALUED_FEATURES
            + COVERTYPE_NUM_BINARY_FEATURES,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def get_pre_eo_output(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (4 * IP + 40)
        return self.ndnf.get_plain_output(x)

    def to_ndnf_model(self) -> CoverTypeNeuralDNF:
        ndnf_model = CoverTypeNeuralDNF(
            invented_predicate_per_input=self.invented_predicate_per_input,
            num_conjunctions=self.num_conjunctions,
            predicate_inventor_tau=self.predicate_inventor.tau,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        ndnf_model.predicate_inventor.predicate_inventor.data = (
            self.predicate_inventor.predicate_inventor.data.clone()
        )
        ndnf_model.predicate_inventor.tau = self.predicate_inventor.tau

        return ndnf_model


class CoverTypeNeuralDNFMT(CoverTypeBaseNeuralDNF):
    """
    Car classifier with NeuralDNFMutexTanh as the underlying model.
    This model is expected to be trained with NLLLoss, since it outputs log
    probabilities.
    """

    ndnf: NeuralDNFMutexTanh

    def _create_ndnf_model(self):
        return NeuralDNFMutexTanh(
            n_in=self.invented_predicate_per_input
            * COVERTYPE_NUM_REAL_VALUED_FEATURES
            + COVERTYPE_NUM_BINARY_FEATURES,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns the raw logits of the model. This is useful for training with
        CrossEntropyLoss.
        """
        return self.ndnf.get_raw_output(x)

    def get_all_forms(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> dict[str, dict[str, Tensor]]:
        # x: B x 44
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (4 * IP + 40)
        return self.ndnf.get_all_forms(x)

    def to_ndnf_model(self) -> CoverTypeNeuralDNF:
        ndnf_model = CoverTypeNeuralDNF(
            invented_predicate_per_input=self.invented_predicate_per_input,
            num_conjunctions=self.num_conjunctions,
            predicate_inventor_tau=self.predicate_inventor.tau,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        ndnf_model.predicate_inventor.predicate_inventor.data = (
            self.predicate_inventor.predicate_inventor.data.clone()
        )
        ndnf_model.predicate_inventor.tau = self.predicate_inventor.tau

        return ndnf_model


def construct_model(cfg: DictConfig) -> CoverTypeClassifier:
    if cfg["model_type"] in ["eo", "mt"]:
        ndnf_class = (
            CoverTypeNeuralDNFEO
            if cfg["model_type"] == "eo"
            else CoverTypeNeuralDNFMT
        )
        return ndnf_class(
            invented_predicate_per_input=cfg["model_architecture"][
                "invented_predicate_per_input"
            ],
            num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
            predicate_inventor_tau=cfg["model_architecture"].get(
                "predicate_inventor_tau", 1.0
            ),
        )

    return CoverTypeMLP(
        num_latent=cfg["model_architecture"]["num_latent"],
    )
