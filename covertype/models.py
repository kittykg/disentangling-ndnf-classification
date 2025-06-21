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
COVERTYPE_NUM_REAL_VALUED_FEATURES: int = 9
COVERTYPE_NUM_BINARY_FEATURES: int = 35
COVERTYPE_TOTAL_NUM_FEATURES: int = (
    COVERTYPE_NUM_REAL_VALUED_FEATURES + COVERTYPE_NUM_BINARY_FEATURES
)  # 44
COVERTYPE_C2B_NUM_BINARY_FEATURES: int = 7
COVERTYPE_C2B_TOTAL_NUM_FEATURES: int = (
    COVERTYPE_NUM_REAL_VALUED_FEATURES + COVERTYPE_C2B_NUM_BINARY_FEATURES
)  # 16


class CoverTypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class CoverTypeMLP(CoverTypeClassifier):
    def __init__(self, num_latents: list[int] = [64]):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(COVERTYPE_TOTAL_NUM_FEATURES, num_latents[0]),
            nn.Tanh(),
            *[
                layer
                for d_in, d_out in zip(num_latents[:-1], num_latents[1:])
                for layer in [nn.Linear(d_in, d_out), nn.Tanh()]
            ],
            nn.Linear(num_latents[-1], COVERTYPE_NUM_CLASSES),
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
    num_conjunctions: int
    c2b: bool
    manually_sparse_conj_layer_k: int | None = None

    predicate_inventor: nn.Module
    ndnf_num_input_features: int
    ndnf: BaseNeuralDNF

    def __init__(self):
        super().__init__()

    def manually_sparse_conjunctive_layer(self) -> None:
        """
        This function is used to randomly set the k connections between input
        and a conjunctive node to zero. Once set to zero, the connections will
        not be updated during training. This is useful to create a sparse model.
        """
        # Set the weights to zero
        for i in range(self.num_conjunctions):
            indices_to_zero = torch.randperm(self.ndnf_num_input_features)[
                : self.manually_sparse_conj_layer_k
            ]
            self.ndnf.conjunctions.weights.data[i, indices_to_zero] = 0.0
            # disable the gradient for these weights via masking
            self.ndnf.conj_weight_mask[i, indices_to_zero] = 0.0

    def get_invented_predicates(
        self, x: Tensor, discretised: bool = False
    ) -> Tensor:
        raise NotImplementedError

    def get_conjunction(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_conjunction(x)

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7) or B x IP
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


class CovertTypeThresholdPIBaseNeuralDNF(CoverTypeBaseNeuralDNF):
    invented_predicate_per_input: int
    predicate_inventor: NeuralDNFPredicateInventor

    def __init__(
        self,
        invented_predicate_per_input: int,
        num_conjunctions: int,
        predicate_inventor_tau: float = 1.0,
        c2b: bool = False,
        manually_sparse_conj_layer_k: int | None = None,
    ):
        super().__init__()

        self.invented_predicate_per_input = invented_predicate_per_input
        self.num_conjunctions = num_conjunctions
        self.c2b = c2b

        self.predicate_inventor = NeuralDNFPredicateInventor(
            num_features=COVERTYPE_NUM_REAL_VALUED_FEATURES,
            invented_predicate_per_input=invented_predicate_per_input,
            tau=predicate_inventor_tau,
        )

        self.ndnf_num_input_features = (
            invented_predicate_per_input * COVERTYPE_NUM_REAL_VALUED_FEATURES
        )
        if c2b:
            self.ndnf_num_input_features += COVERTYPE_C2B_NUM_BINARY_FEATURES
        else:
            self.ndnf_num_input_features += COVERTYPE_NUM_BINARY_FEATURES

        self.ndnf = self._create_ndnf_model()

        self.manually_sparse_conj_layer_k = manually_sparse_conj_layer_k
        if (
            manually_sparse_conj_layer_k is not None
            and manually_sparse_conj_layer_k > 0
        ):
            # Manually set some
            self.manually_sparse_conjunctive_layer()

    def get_invented_predicates(
        self, x: Tensor, discretised: bool = False
    ) -> Tensor:
        """
        This function compute the invented predicates from the real valued
        features of the input data, and concat them with the binary features.
        """
        # x: B x 44 or B x 16
        # We only take the real valued features
        real_val_features = x[:, :COVERTYPE_NUM_REAL_VALUED_FEATURES]
        # real_val_features: B x 9
        binary_features = x[:, COVERTYPE_NUM_REAL_VALUED_FEATURES:]
        # binary_features: B x 35 or B x 7
        invented_predicates = self.predicate_inventor(
            real_val_features, discretised
        )
        # invented_predicates: B x (9 * IP)
        final_tensor = torch.cat([invented_predicates, binary_features], dim=1)
        # final_tensor: B x (9 * IP + 35) or B x (9 * IP + 7)
        return final_tensor


class CoverTypeThresholdPINeuralDNF(CovertTypeThresholdPIBaseNeuralDNF):
    """
    This class is not expected to be trained directly
    """

    ndnf: NeuralDNF

    def _create_ndnf_model(self) -> NeuralDNF:
        return NeuralDNF(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def change_ndnf(self, new_ndnf: NeuralDNF) -> None:
        self.ndnf = new_ndnf


class CoverTypeThresholdPINeuralDNFEO(CovertTypeThresholdPIBaseNeuralDNF):
    ndnf: NeuralDNFEO

    def _create_ndnf_model(self) -> NeuralDNFEO:
        return NeuralDNFEO(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def get_pre_eo_output(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_plain_output(x)

    def to_ndnf_model(self) -> CoverTypeThresholdPINeuralDNF:
        ndnf_model = CoverTypeThresholdPINeuralDNF(
            invented_predicate_per_input=self.invented_predicate_per_input,
            num_conjunctions=self.num_conjunctions,
            predicate_inventor_tau=self.predicate_inventor.tau,
            c2b=self.c2b,
            manually_sparse_conj_layer_k=self.manually_sparse_conj_layer_k,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        ndnf_model.predicate_inventor.predicate_inventor.data = (
            self.predicate_inventor.predicate_inventor.data.clone()
        )
        ndnf_model.predicate_inventor.tau = self.predicate_inventor.tau

        return ndnf_model


class CoverTypeThresholdPINeuralDNFMT(CovertTypeThresholdPIBaseNeuralDNF):
    """
    Car classifier with NeuralDNFMutexTanh as the underlying model.
    This model is expected to be trained with NLLLoss, since it outputs log
    probabilities.
    """

    ndnf: NeuralDNFMutexTanh

    def _create_ndnf_model(self):
        return NeuralDNFMutexTanh(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        """
        Returns the raw logits of the model. This is useful for training with
        CrossEntropyLoss.
        """
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_raw_output(x)

    def get_all_forms(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> dict[str, dict[str, Tensor]]:
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_all_forms(x)

    def to_ndnf_model(self) -> CoverTypeThresholdPINeuralDNF:
        ndnf_model = CoverTypeThresholdPINeuralDNF(
            invented_predicate_per_input=self.invented_predicate_per_input,
            num_conjunctions=self.num_conjunctions,
            predicate_inventor_tau=self.predicate_inventor.tau,
            c2b=self.c2b,
            manually_sparse_conj_layer_k=self.manually_sparse_conj_layer_k,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        ndnf_model.predicate_inventor.predicate_inventor.data = (
            self.predicate_inventor.predicate_inventor.data.clone()
        )
        ndnf_model.predicate_inventor.tau = self.predicate_inventor.tau

        return ndnf_model


class CoverTypeMLPPIBaseNeuralDNF(CoverTypeBaseNeuralDNF):
    """
    A NeuralDNF model that uses an MLP for predicate invention instead of
    NeuralDNFPredicateInventor. The MLP takes all features as input and outputs
    invented predicates.
    """

    predicate_inventor: nn.Sequential

    def __init__(
        self,
        predicate_inventor_dims: list[int],
        num_conjunctions: int,
        c2b: bool = False,
        manually_sparse_conj_layer_k: int | None = None,
    ):
        super().__init__()

        self.predicate_inventor_dims = predicate_inventor_dims
        self.num_conjunctions = num_conjunctions
        self.c2b = c2b
        self.manually_sparse_conj_layer_k = manually_sparse_conj_layer_k

        # Replace the predicate inventor with an MLP
        self.predicate_inventor = nn.Sequential(
            nn.Linear(
                (
                    COVERTYPE_C2B_TOTAL_NUM_FEATURES
                    if c2b
                    else COVERTYPE_TOTAL_NUM_FEATURES
                ),
                predicate_inventor_dims[0],
            ),
            nn.Tanh(),
            *[
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(
                    predicate_inventor_dims[:-1], predicate_inventor_dims[1:]
                )
            ],
        )

        self.ndnf_num_input_features = predicate_inventor_dims[-1]

        self.ndnf = self._create_ndnf_model()

        self.manually_sparse_conj_layer_k = manually_sparse_conj_layer_k
        if (
            manually_sparse_conj_layer_k is not None
            and manually_sparse_conj_layer_k > 0
        ):
            # Manually set some
            self.manually_sparse_conjunctive_layer()

    def get_invented_predicates(
        self, x: Tensor, discretised: bool = False
    ) -> Tensor:
        """
        This function computes the invented predicates from the all features
        of the input data using an MLP, and concatenates them with the binary
        features.
        """
        # x: B x 44 or B x 16
        # Get invented predicates from MLP
        invented_predicates = self.predicate_inventor(x)
        # invented_predicates: B x IP
        invented_predicates = torch.tanh(invented_predicates)

        if discretised:
            # Discretise the invented predicates
            invented_predicates = torch.sign(invented_predicates)

        return invented_predicates


class CoverTypeMLPPINeuralDNF(CoverTypeMLPPIBaseNeuralDNF):
    """
    This class is not expected to be trained directly.
    """

    ndnf: NeuralDNF

    def _create_ndnf_model(self) -> NeuralDNF:
        return NeuralDNF(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def change_ndnf(self, new_ndnf: NeuralDNF) -> None:
        self.ndnf = new_ndnf


class CoverTypeMLPPINeuralDNFEO(CoverTypeMLPPIBaseNeuralDNF):
    ndnf: NeuralDNFEO

    def _create_ndnf_model(self) -> NeuralDNFEO:
        return NeuralDNFEO(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def get_pre_eo_output(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_plain_output(x)

    def to_ndnf_model(self) -> CoverTypeMLPPINeuralDNF:
        ndnf_model = CoverTypeMLPPINeuralDNF(
            predicate_inventor_dims=self.predicate_inventor_dims,
            num_conjunctions=self.num_conjunctions,
            c2b=self.c2b,
            manually_sparse_conj_layer_k=self.manually_sparse_conj_layer_k,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        # copy the sequential model
        ndnf_model.predicate_inventor.load_state_dict(
            self.predicate_inventor.state_dict()
        )

        return ndnf_model


class CoverTypeMLPPINeuralDNFMT(CoverTypeMLPPIBaseNeuralDNF):
    ndnf: NeuralDNFMutexTanh

    def _create_ndnf_model(self) -> NeuralDNFMutexTanh:
        return NeuralDNFMutexTanh(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=self.num_conjunctions,
            n_out=COVERTYPE_NUM_CLASSES,
            delta=1.0,
        )

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        """
        Returns the raw logits of the model. This is useful for training with
        CrossEntropyLoss.
        """
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_raw_output(x)

    def get_all_forms(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> dict[str, dict[str, Tensor]]:
        # x: B x 44 or B x 16
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (9 * IP + 35) or B x (9 * IP + 7)
        return self.ndnf.get_all_forms(x)

    def to_ndnf_model(self) -> CoverTypeMLPPINeuralDNF:
        ndnf_model = CoverTypeMLPPINeuralDNF(
            predicate_inventor_dims=self.predicate_inventor_dims,
            num_conjunctions=self.num_conjunctions,
            c2b=self.c2b,
            manually_sparse_conj_layer_k=self.manually_sparse_conj_layer_k,
        )
        ndnf_model.ndnf = self.ndnf.to_ndnf()
        # copy the sequential model
        ndnf_model.predicate_inventor.load_state_dict(
            self.predicate_inventor.state_dict()
        )

        return ndnf_model


def construct_model(cfg: DictConfig) -> CoverTypeClassifier:
    if cfg["model_type"] in ["eo", "mt"]:
        if (
            cfg["model_architecture"].get("predicate_inventor_mlp_dims", None)
            is not None
        ):
            ndnf_class = (
                CoverTypeMLPPINeuralDNFEO
                if cfg["model_type"] == "eo"
                else CoverTypeMLPPINeuralDNFMT
            )
            return ndnf_class(
                predicate_inventor_dims=cfg["model_architecture"][
                    "predicate_inventor_mlp_dims"
                ],
                num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
                c2b=cfg.get("convert_categorical_to_binary_encoding", False),
                manually_sparse_conj_layer_k=cfg["model_architecture"].get(
                    "manually_sparse_conj_layer_k", None
                ),
            )
        else:
            ndnf_class = (
                CoverTypeThresholdPINeuralDNFEO
                if cfg["model_type"] == "eo"
                else CoverTypeThresholdPINeuralDNFMT
            )
            return ndnf_class(
                invented_predicate_per_input=cfg["model_architecture"][
                    "invented_predicate_per_input"
                ],
                num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
                predicate_inventor_tau=cfg["model_architecture"].get(
                    "predicate_inventor_tau", 1.0
                ),
                c2b=cfg.get("convert_categorical_to_binary_encoding", False),
                manually_sparse_conj_layer_k=cfg["model_architecture"].get(
                    "manually_sparse_conj_layer_k", None
                ),
            )

    return CoverTypeMLP(
        num_latents=cfg["model_architecture"]["num_latents"],
    )
