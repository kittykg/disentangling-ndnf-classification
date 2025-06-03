from collections import OrderedDict
from itertools import chain, combinations

import torch
from torch import Tensor

from neural_dnf.neural_dnf import NeuralDNF

def split_positively_used_conjunction(
    w: Tensor, j_minus_explore_limit: int = -1
) -> list[Tensor]:
    # Pre-condition: there are more than one non-zero weights, and this
    # conjunction `w` is used positively in a disjunction
    # Return the split weights if there are more than one non-zero weights

    # J = {j | w_j \neq 0}
    # For a possible input x, we can split \mathcal{J} into two sets
    # J+ = {j ∈ J | w_j x_j > 0} -- the sign match set
    # J- = {j ∈ J | w_j x_j < 0} -- the sign mismatch set
    # g(x) = max_{j ∈ J} |w_j| - 2 \sum_{j ∈ J-} |w_j|
    # For g(x) > 0, any weights |w_i| >= 0.5 * max_{j ∈ J} |w_j| has to be
    # in J+
    # The rest of the weights, as long as their sum is less than
    # 0.5 * max_{j ∈ J} |w_j|, they can be in J-

    abs_w = torch.abs(w)
    max_abs_w = torch.max(abs_w)
    half_max_abs_w = max_abs_w / 2
    non_zero_idx = torch.where(w != 0)[0]

    non_zero_abs_w = abs_w[non_zero_idx]
    half_max_abs_w = max_abs_w / 2
    less_than_half_indices = torch.where(non_zero_abs_w < half_max_abs_w)[0]

    if less_than_half_indices.numel() == 0:
        # Return itself since no split will be valid
        return [torch.sign(w) * 6]

    j_minus_candidates = non_zero_idx[less_than_half_indices].tolist()

    # We search for the candidates in a breadth-first manner. We start with only
    # one element in the J- set, and we try to add more elements to the J- set.
    # With each tuple of indices, we try to add more elements to the J- set. If
    # we can add more elements, we keep adding elements to it (so that the J+
    # rule is shorter and thus more general). If we can't add any more, this
    # tuple of indices itself or its last parent is a valid J- set. We then add
    # this tuple to the overall valid split set.
    overall_valid_split_set = set()
    split_candidate_queue: OrderedDict[tuple[int], None | tuple] = OrderedDict()
    for j in j_minus_candidates:
        split_candidate_queue[tuple([j])] = None

    while len(split_candidate_queue) > 0:
        head = split_candidate_queue.popitem(last=False)
        removal_idx = list(head[0])
        parent = head[1]

        # bias offset is the sum of the absolute weights of the elements in the
        # current J- set candidate
        bias_offset = torch.sum(abs_w[removal_idx], dtype=torch.float64)
        new_half_max_abs_w = max_abs_w / 2 - bias_offset
        new_less_than_half_indices = torch.where(
            non_zero_abs_w < new_half_max_abs_w
        )[0]

        if new_half_max_abs_w <= 0 or len(new_less_than_half_indices) == 0:
            # new_half_max_abs_w <= 0 when max_abs_w / 2 <= bias_offset, the
            # current J- candidate (`removal_idx`) is a not valid J- set;
            # len(new_less_than_half_indices) == 0 when there are no more
            # elements in the J- set that can be added to the current J- set
            if new_half_max_abs_w > 0:
                overall_valid_split_set.add(tuple(removal_idx))
            elif parent is not None:
                # if the bias offset <= 0, we can't add removal_idx itself, but
                # we can add its parent to the overall valid split set
                overall_valid_split_set.add(parent)
            continue

        # Look for the new candidates that are not in the current J- set
        new_j_minus_candidates = non_zero_idx[
            new_less_than_half_indices
        ].tolist()
        new_j_minus_candidates = list(
            set(new_j_minus_candidates) - set(removal_idx)
        )

        if len(new_j_minus_candidates) == 0:
            # There are no more elements that are not in the current J- set
            # that can be added
            overall_valid_split_set.add(tuple(removal_idx))
            continue

        # There are candidates to explore
        for j in new_j_minus_candidates:
            new_removal_idx: tuple[int] = tuple(sorted(removal_idx + [j]))  # type: ignore
            if (
                j_minus_explore_limit != -1
                and len(new_removal_idx) > j_minus_explore_limit
            ):
                # We have reached the limit of the number of elements in the J-
                # set, we will not add any more elements to the J- set but will
                # add the current removal_idx to the overall valid split set
                overall_valid_split_set.add(tuple(removal_idx))
                continue

            if new_removal_idx not in split_candidate_queue:
                # we add the new candidate to the queue, with its parent as the
                # value
                split_candidate_queue[new_removal_idx] = tuple(removal_idx)

    split_tensors = []
    non_zero_idx = torch.where(w != 0)[0]
    for t in overall_valid_split_set:
        c = torch.zeros_like(w, device=w.device)
        for i in non_zero_idx:
            c[i] = w[i].sign() * (0 if i in t else 6)
        split_tensors.append(c)

    return split_tensors


def split_negatively_used_conjunction(
    w: Tensor, j_minus_explore_limit: int = -1
) -> list[Tensor]:
    # there are more than one non-zero weights, and this conjunction `w` is
    # used negatively in a disjunction
    # Return the split weights if there are more than one non-zero weights

    # Compute the bias
    abs_w = torch.abs(w)
    max_abs_w = torch.max(abs_w)
    non_zero_idx = torch.where(w != 0)[0]

    # J = {j | w_j \neq 0}
    # For a possible input x, we can split \mathcal{J} into two sets
    # J+ = {j ∈ J | w_j x_j > 0} -- the sign match set
    # J- = {j ∈ J | w_j x_j < 0} -- the sign mismatch set
    # g(x) = max_{j ∈ J} |w_j| - 2 \sum_{j ∈ J-} |w_j|
    # For g(x) < 0, any weights |w_i| >= 0.5 * max_{j ∈ J} |w_j| can be the only
    # element in J-
    # The rest of the weights, as long as their sum is more than
    # 0.5 * max_{j ∈ J} |w_j|, they can be in J-

    # Check which weight can be candidates for J-
    non_zero_abs_w = abs_w[non_zero_idx]
    half_max_abs_w = max_abs_w / 2
    more_than_half_indices = torch.where(non_zero_abs_w > half_max_abs_w)[0]

    if more_than_half_indices.numel() == 0:
        # Return itself since no split will be valid
        return [torch.sign(w) * 6]

    less_than_half_indices = torch.where(non_zero_abs_w <= half_max_abs_w)[0]
    j_minus_candidates = non_zero_idx[less_than_half_indices].tolist()

    overall_valid_split_set = set()
    for i in non_zero_idx[more_than_half_indices]:
        overall_valid_split_set.add(tuple([i.item()]))

    # We search for the candidates in a breadth-first manner. We start with only
    # one element in the J- set, and we try to add more elements to the J- set.
    # With each tuple of indices, if it's the first time that it goes above the
    # threshold, we add it to the overall valid split set and stop the search.

    accumulators: OrderedDict[tuple[int], None] = OrderedDict()
    for i in j_minus_candidates:
        accumulators[tuple([i])] = None

    while len(accumulators) > 0:

        head = accumulators.popitem(last=False)
        keep_idx_attempt = list(head[0])

        if w.abs()[keep_idx_attempt].sum() > half_max_abs_w:
            # This is a minimal split
            not_subsumed = True
            for t in overall_valid_split_set:
                if set(t).issubset(set(keep_idx_attempt)):
                    # This removal set is already covered by a previous split
                    not_subsumed = False
                    break

            if not_subsumed:
                overall_valid_split_set.add(tuple(sorted(keep_idx_attempt)))

        else:
            for i in j_minus_candidates:
                if i not in keep_idx_attempt:
                    new_index = sorted(keep_idx_attempt + [i])
                    if (
                        j_minus_explore_limit != -1
                        and len(new_index) > j_minus_explore_limit
                    ):
                        # We have reached the limit of the number of elements in
                        # the J- set. We will stop the search here
                        continue
                    accumulators[tuple(new_index)] = None  # type: ignore

    # Convert the indices to actual weight tensor. Each indices set, we generate
    # a new weight tensor, where if the index is in the set the weight is
    # negated and otherwise removed
    valid_split = []

    for t in overall_valid_split_set:

        c = torch.zeros_like(w, device=w.device)
        for i in non_zero_idx:
            c[i] = w[i].sign() * (-6 if i in t else 0)
        valid_split.append(c)

    return valid_split


def split_entangled_conjunction(
    w: Tensor,
    sign: int = 1,
    positive_disentangle_j_minus_limit: int = -1,
    negative_disentangle_j_minus_limit: int = -1,
) -> None | list[Tensor]:
    assert sign in [-1, 1], "Sign should be either -1 or 1"

    # Return None if all weights are zero
    if torch.all(w == 0):
        return None

    # Return itself if there is only one non-zero weight
    if torch.sum(w != 0) == 1:
        return [torch.sign(w) * 6 * sign]

    if sign > 0:
        return split_positively_used_conjunction(
            w, positive_disentangle_j_minus_limit
        )
    return split_negatively_used_conjunction(
        w, negative_disentangle_j_minus_limit
    )


def split_entangled_disjunction(w: Tensor) -> None | list[tuple[Tensor, bool]]:
    """
    Return a list of split

    Each split is a tuple of a Tensor and a boolean. The boolean indicates
    whether the split should be a disjunction or not (i.e. if FALSE, then the
    tensor should be treated a conjunction)
    """
    # Return None if all weights are zero
    if torch.all(w == 0):
        return None

    # Return itself if there is only one non-zero weight
    if torch.sum(w != 0) == 1:
        return [(torch.sign(w) * 6, True)]

    # ------------------------------- Otherwise --------------------------------
    # There are more than one non-zero weights
    # By default, disjunction `w` the last layer of the neural DNF model, so
    # we don't need to consider whether it's being used positively or negatively

    # Compute the bias
    abs_w = torch.abs(w)
    max_abs_w = torch.max(abs_w)
    sum_abs_w = torch.sum(abs_w)
    bias = sum_abs_w - max_abs_w

    non_zero_idx = torch.where(w != 0)[0]

    # J = {j | w_j \neq 0}
    # For a possible input x, we can split \mathcal{J} into two sets
    # J+ = {j ∈ J | w_j x_j > 0} -- the sign match set
    # J- = {j ∈ J | w_j x_j < 0} -- the sign mismatch set
    # g(x) =  2 \sum_{j ∈ J+} |w_j| - max_{j ∈ J} |w_j|
    # For g(x) > 0, any weights |w_i| > 0.5 * max_{j ∈ J} |w_j| can be the only
    # element in J+
    # The rest of the weights, we compute the combination. For a combination, as
    # long as the sum of those abs weights is greater than 0.5 * max_{j ∈ J}
    # |w_j|, that combination can be a valid J+ set without including other
    # weights (if include a > 0.5 *m max weight, it will be subsumed)

    # Check which weights construct singleton J+
    non_zero_abs_w = abs_w[non_zero_idx]
    half_max_abs_w = max_abs_w / 2
    gt_half_indices = torch.where(non_zero_abs_w > half_max_abs_w)[0]
    le_half_indices = torch.where(non_zero_abs_w <= half_max_abs_w)[0]

    if le_half_indices.numel() == 0:
        # Return itself since no need to split
        return [(torch.sign(w) * 6, True)]

    valid_split: list[tuple[Tensor, bool]] = []

    # For any abs weights > 0.5 * max_abs_w, they can be the only element in J+
    for i in non_zero_idx[gt_half_indices]:
        c = torch.zeros_like(w, device=w.device)
        c[i] = torch.sign(w[i]) * 6
        valid_split.append((c, False))

    j_plus_candidates = non_zero_idx[le_half_indices].tolist()
    pws_list = list(list(map(list, power_set(j_plus_candidates))))

    # Each set of `pws_list` represent a possible mismatch J- set
    for pws in pws_list:
        # input_entry \in {-1, 0, 1}, and:
        # - for all i not in `pws` w_i * x_i < 0;
        # - for all j in `pws` w_j * x_j > 0
        input_entry = torch.sign(w).to(w.device) * -1
        for i in pws:
            input_entry[i] *= -1

        disj_out = torch.sum(w * input_entry) + bias

        if disj_out > 0:
            # This combination can activate the disjunction
            c = torch.zeros_like(w, device=w.device)
            sign_match_indices = torch.where(input_entry == torch.sign(w))[0]
            c[sign_match_indices] = torch.sign(w[sign_match_indices]) * 6
            valid_split.append((c, False))

    return valid_split


def condense_neural_dnf_model(model: NeuralDNF) -> NeuralDNF:
    """
    Remove any unused conjunctions from the model and reduce the number of
    conjunctions used. Prerequisite: the model's weights are strictly in the set
    {-6, 0, 6}.
    This function will create a new NeuralDNF model with the minimum number of
    conjunctions, without changing the model's behaviour.
    """
    conj_w = model.conjunctions.weights.data.clone()
    disj_w = model.disjunctions.weights.data.clone()

    # Check weights are in the set {-6, 0, 6}
    conj_w_abs_unique = torch.abs(conj_w).unique()
    disj_w_abs_unique = torch.abs(disj_w).unique()
    range_tensor = torch.tensor([0, 6])
    assert conj_w_abs_unique.shape == range_tensor.shape and torch.all(
        conj_w_abs_unique == range_tensor
    ), "Conjunction weights are not in the set {-6, 0, 6}"
    assert disj_w_abs_unique.shape == range_tensor.shape and torch.all(
        disj_w_abs_unique == range_tensor
    ), "Disjunction weights are not in the set {-6, 0, 6}"

    # Get the unique conjunctions' indices
    unique_conjunctions = set()
    for w in disj_w:
        for i in torch.where(w != 0)[0]:
            unique_conjunctions.add(i.item())

    assert len(unique_conjunctions) > 0, "No conjunctions are used in the model"

    condensed_model = NeuralDNF(
        conj_w.shape[1], len(unique_conjunctions), disj_w.shape[0], 1.0
    )

    # Create the new conjunctions weight matrix
    new_conj_list = []
    old_conj_id_to_new = dict()

    for conj_id in sorted(unique_conjunctions):
        new_id = len(new_conj_list)
        old_conj_id_to_new[conj_id] = new_id
        new_conj_list.append(conj_w[conj_id])

    condensed_model.conjunctions.weights.data = torch.stack(new_conj_list)

    condensed_model.disjunctions.weights.data = torch.zeros(
        len(disj_w), len(unique_conjunctions), dtype=torch.float32
    )
    for disj_id, w in enumerate(disj_w):
        for old_conj_ids in torch.where(w != 0)[0]:
            new_conj_id = old_conj_id_to_new[old_conj_ids.item()]
            condensed_model.disjunctions.weights.data[disj_id, new_conj_id] = w[
                old_conj_ids
            ]

    return condensed_model


# ============================================================================ #
#                               Helper functions                               #
# ============================================================================ #


def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
