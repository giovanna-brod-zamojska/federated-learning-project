from __future__ import annotations

import enum

import attr
import logging

import numpy as np
import torch
from werkzeug.utils import cached_property
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union, Iterable, Tuple, List
from utils.model_utils import get_device

__all__ = [
    "PruningType",
    "Mask",
    "compute_fisher_diagonal",
    "create_fisher_mask",
    "progressive_mask_calibration",
]


class PruningType(enum.Enum):
    FISHER = enum.auto()
    HESSIAN_PARAM_SQUARED = enum.auto()


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_batches: Optional[int] = None,
    mask: Optional[Mask] = None,
    pruning_type: PruningType = PruningType.FISHER,
) -> torch.Tensor:
    """
    Approximate the diagonal of the Fisher Information Matrix (FIM) using squared gradients computed
    over mini-batches. If `mask_dict` is provided, gradients of pruned parameters will be zeroed out
    before accumulation, preserving previously frozen weights.

    The Fisher Information is given by the expected value of the squared gradient of the loss function:

        Fisher(θ) = E_{(x, y) ~ D} [ (∂L(f_θ(x), y) / ∂θ)^2 ]

    The sample approximation of the above expectation should be:

        Fisher(θ) ≈ (1 / N) * ∑_{i=1}^{N} (∇_θ L^{(i)}(θ))²

    Where:
        - L^{(i)}(θ) is the loss on the i-th data point
        - ∇_θ L^{(i)}(θ) is the gradient of the loss with respect to parameters θ
        - N is the number of samples (or mini-batches)
        - The square is element-wise and gives the diagonal approximation

    However, this function only computes an approximation of the Fisher diagonal and returns:

        F_diag ≈ (1/N) ∑_b (∇_θ L_b)²

    where:
        - b indexes batches from the dataset
        - N is the number of batches
        - L_b = (1 / |B|) ∑_{i∈B} ℓ(f_θ(x_i), y_i) is the **average loss over a mini-batch**
        - ℓ(·,·) is the per-sample loss (e.g., negative log-likelihood)
        - ∇_θ L_b is the gradient of the average batch loss with respect to the model parameters

    In other words, for computational simplicity, we square the gradient of the average loss over the mini batch,
    while in fact we should square the gradient of the loss over the individual samples.

    Notes:
    - This method does **not** compute per-sample gradients.
    - It squares the gradient of the averaged batch loss, which introduces bias.
    - It underestimates the true Fisher diagonal, as E[g]² < E[g²].
    - It's computationally efficient and works with any model

    Args:
        model (nn.Module): The model whose parameters are being analyzed.
        dataloader (DataLoader): DataLoader providing input–target pairs.
        loss_fn (nn.Module): Loss function used to compute gradients.
        num_batches (Optional[int]): If set, only the first `num_batches` are used
            to estimate the Fisher information. Useful for faster computation.
        mask (Optional[Mask]): If provided, compute gradients only for parameters that are not frozen.
        pruning_type (Optional[PruningType]): Pruning type to use, defaults to PruningType.FISHER.
            if PruningType.HESSIAN_PARAM_SQUARED returns (theta_i ** 2) * fisher_diag[i], which is an estimator
            for expected loss increase after pruning, as in Delta L ≈ 0.5 * theta_i^2 * H_ii.
    Returns:
        fisher_diag (torch.Tensor): A flattened tensor containing the Fisher diagonal estimate,
        one element per parameter.
    """
    if mask is not None:
        assert isinstance(mask, Mask)

    model.eval()
    device = get_device()
    model.to(device)
    if mask is not None:
        mask.to(device)
        mask.validate_against(model.named_parameters())

    fisher_diag = [torch.zeros_like(p, device=device) for p in model.parameters()]
    total_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)  # L_b = average over the batch

        # Approximation: we compute ∇_θ L_b, then square it,
        # which is not equal to the average of per-sample (∇_θ ℓ_i)^2
        loss.backward()

        for i, (name, p) in enumerate(model.named_parameters()):
            if p.grad is not None:
                grad = p.grad.detach()
                if mask is not None and name in mask:
                    grad = grad * mask[name]
                fisher_diag[i] += grad**2

        total_batches += 1

    if total_batches == 0:
        raise ValueError("No batches processed for Fisher approximation.")

    # Normalize by the number of batches (approximate Fisher expectation over data)
    fisher_diag = [f / total_batches for f in fisher_diag]

    # Flatten and concatenate all parameter diagonals into one tensor
    fisher_diag = torch.cat([f.flatten() for f in fisher_diag])
    if pruning_type == PruningType.FISHER:
        return fisher_diag
    elif pruning_type == PruningType.HESSIAN_PARAM_SQUARED:
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        # Elementwise square and multiply by fisher_diag
        param_squared_fisher = (params**2) * fisher_diag
        return param_squared_fisher
    else:
        raise ValueError(f"Unknown pruning type: {pruning_type}")


def compute_fisher_information(model, dataloader, device, loss_fn, num_samples=100):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    count = 0
    for inputs, targets in dataloader:
        if count >= num_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

        count += 1

    for name in fisher:
        fisher[name] /= count

    return fisher