import torch
from torch import nn


def create_fisher_mask(
    fisher_diag: torch.Tensor, model: nn.Module, sparsity: float = 0.9
) -> Mask:
    """
    Create a dictionary of binary gradient masks based on Fisher importance scores.

    Keeps sparsity% of the most important parameters masking them to 0 and sets
    the rest to 1 (sets their gradients to zero during training).

    Args:
        fisher_diag (torch.Tensor): Flattened tensor of Fisher Information scores (1D).
        model (nn.Module): The model whose parameters will be masked.
        sparsity (float): Fraction of total parameters that are frozen (set mask=0).

    Returns:
        Mask:  Dictionary mapping parameter names to binary masks
                                 with the same shape as the parameter tensors.
    """
    assert 0 < sparsity < 1, "sparsity needs to be between 0 and 1"

    k = round(len(fisher_diag) * sparsity)

    # Default: all parameters allowed to update
    flat_mask = torch.ones_like(fisher_diag)

    # Find top-k important indices to freeze
    # 1 for unimportant paramters, 0 for the important ones so that gradient does not update in SparseSGD

    if k > 0:
        important_indices = torch.topk(
            fisher_diag, k=k, largest=True
        ).indices  # top k scores
        flat_mask[important_indices] = 0.0

    param_sizes = [p.numel() for _, p in model.named_parameters()]
    param_shapes = [p.shape for _, p in model.named_parameters()]
    param_names = [name for name, _ in model.named_parameters()]
    split_masks = torch.split(flat_mask, param_sizes)

    return {
        name: mask.view(shape)
        for name, mask, shape in zip(param_names, split_masks, param_shapes)
    }


def generate_global_mask1(
    fisher_info, top_k: float = 0.2, strategy: str = "fisher_least"
):
    if strategy.startswith("fisher"):
        all_scores = torch.cat([f.view(-1) for f in fisher_info.values()])

        # Use kthvalue for better memory efficiency with large tensors
        total_elements = all_scores.numel()

        if strategy == "fisher_least":
            k = max(1, int(top_k * total_elements))
            threshold = torch.kthvalue(all_scores, k).values
            compare = lambda x: x <= threshold
        elif strategy == "fisher_most":
            k = max(1, int((1 - top_k) * total_elements))
            threshold = torch.kthvalue(all_scores, k).values
            compare = lambda x: x >= threshold
        elif strategy == "fisher_left_only":
            # New strategy: only parameters on the left side of distribution (least important)
            # This sets mask to 1 ONLY for the leftmost top_k fraction of Fisher values
            k = max(1, int(top_k * total_elements))
            threshold = torch.kthvalue(all_scores, k).values
            compare = lambda x: x <= threshold
        else:
            raise ValueError(f"Unknown Fisher strategy: {strategy}")

        mask = {name: compare(tensor).float() for name, tensor in fisher_info.items()}

    elif strategy in {"magnitude_lowest", "magnitude_highest"}:
        all_params = torch.cat([p.view(-1).abs() for p in fisher_info.values()])
        total_elements = all_params.numel()

        if strategy == "magnitude_lowest":
            k = max(1, int(top_k * total_elements))
            threshold = torch.kthvalue(all_params, k).values
            compare = lambda x: x.abs() <= threshold
        else:
            k = max(1, int((1 - top_k) * total_elements))
            threshold = torch.kthvalue(all_params, k).values
            compare = lambda x: x.abs() >= threshold
        mask = {name: compare(p).float() for name, p in fisher_info.items()}

    elif strategy == "random":
        mask = {
            name: (torch.rand_like(p) < top_k).float()
            for name, p in fisher_info.items()
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return mask
