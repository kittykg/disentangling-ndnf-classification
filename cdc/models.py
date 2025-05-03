from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from neural_dnf import NeuralDNF

from predicate_invention import NeuralDNFPredicateInventor


# The following numbers of features assume that all volatile features are
# included as numeric
CDC_NUM_REAL_VALUED_FEATURES: int = 1
CDC_NUM_VOLATILE_FEATURES: int = 3
CDC_TOTAL_NUM_REAL_VALUED_FEATURES: int = (
    CDC_NUM_REAL_VALUED_FEATURES + CDC_NUM_VOLATILE_FEATURES
)
CDC_NUM_BINARY_FEATURES: int = 28
CDC_TOTAL_NUM_FEATURES: int = (
    CDC_TOTAL_NUM_REAL_VALUED_FEATURES + CDC_NUM_BINARY_FEATURES
)


class CDCClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_weight_reg_loss(self) -> Tensor:
        raise NotImplementedError


class CDCMLP(CDCClassifier):
    def __init__(self, num_latent: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(CDC_TOTAL_NUM_FEATURES, num_latent),
            nn.Tanh(),
            nn.Linear(num_latent, num_latent),
            nn.Tanh(),
            nn.Linear(num_latent, 1),
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


class CDCNeuralDNF(CDCClassifier):
    invented_predicate_per_input: int
    num_conjunctions: int
    manually_sparse_conj_layer_k: int | None = None

    predicate_inventor: NeuralDNFPredicateInventor
    ndnf_num_input_features: int
    ndnf: NeuralDNF

    def __init__(
        self,
        invented_predicate_per_input: int,
        num_conjunctions: int,
        predicate_inventor_tau: float = 1.0,
        manually_sparse_conj_layer_k: int | None = None,
    ):
        super().__init__()

        self.invented_predicate_per_input = invented_predicate_per_input
        self.num_conjunctions = num_conjunctions

        self.predicate_inventor = NeuralDNFPredicateInventor(
            num_features=CDC_NUM_REAL_VALUED_FEATURES
            + CDC_NUM_VOLATILE_FEATURES,
            invented_predicate_per_input=invented_predicate_per_input,
            tau=predicate_inventor_tau,
        )

        self.ndnf_num_input_features = (
            invented_predicate_per_input * CDC_TOTAL_NUM_REAL_VALUED_FEATURES
            + CDC_NUM_BINARY_FEATURES
        )
        self.ndnf = NeuralDNF(
            n_in=self.ndnf_num_input_features,
            n_conjunctions=num_conjunctions,
            n_out=1,
            delta=1.0,
        )

        self.manually_sparse_conj_layer_k = manually_sparse_conj_layer_k
        if (
            manually_sparse_conj_layer_k is not None
            and manually_sparse_conj_layer_k > 0
        ):
            # Manually set some
            self.manually_sparse_conjunctive_layer()

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
        """
        This function compute the invented predicates from the real valued
        features of the input data, and concat them with the binary features.
        """
        # x: B x 32
        # We only take the real valued features
        real_val_features = x[:, :CDC_TOTAL_NUM_REAL_VALUED_FEATURES]
        # real_val_features: B x 4
        binary_features = x[:, CDC_TOTAL_NUM_REAL_VALUED_FEATURES:]
        # binary_features: B x 28
        invented_predicates = self.predicate_inventor(
            real_val_features, discretised
        )
        # invented_predicates: B x (4 * IP)
        final_tensor = torch.cat([invented_predicates, binary_features], dim=1)
        # final_tensor: B x (4 * IP + 28)
        return final_tensor

    def get_conjunction(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 32
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (4 * IP + 28)
        return self.ndnf.get_conjunction(x)

    def forward(
        self, x: Tensor, discretise_invented_predicate: bool = False
    ) -> Tensor:
        # x: B x 32
        x = self.get_invented_predicates(x, discretise_invented_predicate)
        # x: B x (4 * IP + 28)
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


def construct_model(cfg: DictConfig) -> CDCClassifier:
    if cfg["model_type"] == "ndnf":
        return CDCNeuralDNF(
            invented_predicate_per_input=cfg["model_architecture"][
                "invented_predicate_per_input"
            ],
            num_conjunctions=cfg["model_architecture"]["n_conjunctions"],
            predicate_inventor_tau=cfg["model_architecture"].get(
                "predicate_inventor_tau", 1.0
            ),
            manually_sparse_conj_layer_k=cfg["model_architecture"].get(
                "manually_sparse_conj_layer_k", None
            ),
        )

    return CDCMLP(num_latent=cfg["model_architecture"]["num_latent"])
