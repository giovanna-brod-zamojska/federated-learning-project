import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union


def _num_total_params(mask: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> int:
    if isinstance(mask, torch.Tensor):
        return mask.numel()

    return sum(t.numel() for t in mask.values())


def _num_zero_params(mask: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> int:
    if isinstance(mask, torch.Tensor):
        return (mask == 0.0).sum().item()

    return sum((t == 0).sum().item() for t in mask.values())


def _num_one_params(mask: Union[torch.Tensor, Dict[str, torch.Tensor]]):
    if isinstance(mask, torch.Tensor):
        return (mask == 1.0).sum().item()

    return sum((t == 1.0).sum().item() for t in mask.values())


def _compute_sparsity(mask: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> float:
    return _num_zero_params(mask) / _num_total_params(mask)


def _compute_num_trainable_params(mask) -> float:
    return _num_one_params(mask) / _num_total_params(mask)


def _compute_approximated_fisher_scores(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_batches: Optional[int] = None,
    mask: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
):

    scores = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total_batches = len(dataloader) if num_batches is None else num_batches

    for batch_idx, (inputs, targets) in enumerate(
        tqdm(dataloader, total=total_batches, desc="Computing Fisher")
    ):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        # outputs = model(inputs)
        features = model(inputs)  # only backbone if using torch.load(dino_model..)
        outputs = model.head(features)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                scores[name] += param.grad.detach() ** 2

    for name in scores:
        scores[name] /= total_batches

    return scores


# too much time
def _compute_fisher_scores(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module = None,
    num_batches: Optional[int] = None,
    mask: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    model.eval()
    model.to(device)

    scores = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total_batches_to_process = len(dataloader) if num_batches is None else num_batches

    for batch_idx, (inputs, targets) in enumerate(
        tqdm(dataloader, total=total_batches_to_process, desc="Computing Fisher")
    ):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # Based on https://github.com/iurada/talos-task-arithmetic/blob/ae102473de0e57ebf625eca22e10781c371149ec/vision/pruners.py
        logits = model(inputs)
        outdx = (
            torch.distributions.Categorical(logits=logits)
            .sample()
            .unsqueeze(1)
            .detach()
        )
        sampled_y = logits.gather(1, outdx)

        for i in range(inputs.size(0)):  # process each sample individually
            # Compute per-sample gradients.
            model.zero_grad()
            torch.autograd.backward(sampled_y[i], retain_graph=True)

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    scores[name] += param.grad.detach() ** 2

    return scores


def calibrate_mask(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sparsity: float = 0.9,
    rounds: int = 5,
    num_batches: Optional[int] = None,
    strategy: str = "train_least_important",
    approximate_fisher: bool = True,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
) -> Dict[str, torch.Tensor]:
    print("*" * 50)
    print(f"Progressive Mask Calibration - Strategy: {strategy}")
    print("*" * 50)

    model.to(device)

    mask = {
        n: torch.ones_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    inv_sparsity = 1.0 - sparsity

    for r in range(1, rounds + 1):
        print(f"[Round {r}]")
        print(
            f"Current Sparsity: {_compute_sparsity(mask):.4f} "
            f"({_num_zero_params(mask)}/{_num_total_params(mask)} parameters zeroed out)\n"
        )

        # Recompute Fisher scores at each round
        _func = (
            _compute_approximated_fisher_scores
            if approximate_fisher
            else _compute_fisher_scores
        )
        scores = _func(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            mask=mask,
            num_batches=num_batches,
            device=device,
        )

        all_scores = torch.cat([v.flatten() for v in scores.values()])

        # Select scores of currently unpruned params, scores[mask == 1.0],  # only unpruned params / trainable params
        active_scores = torch.cat(
            [score[mask[name] != 0].flatten() for name, score in scores.items()]
        )

        total_params = all_scores.numel()
        total_active_params = active_scores.numel() - 1

        keep_fraction = (inv_sparsity) ** (r / rounds)
        # this decreases at each round (it indicates the number of params with mask=1), we want to increase the number of zeroed params at each round and decrease the number of one params
        k = int(keep_fraction * total_params)
        print(f"Current keep fraction: {keep_fraction:.4f} | Keeping only top k: {k}")

        if strategy == "train_least_important":

            threshold, _ = torch.kthvalue(
                active_scores,
                min(
                    k,
                    total_active_params,
                ),
            )
            print("Threshold (below which params are kept):", threshold)

            # set as trainable (mask=1) all parameters with scores below this threshold
            for name, score in scores.items():

                new_mask = (score <= threshold).float()
                mask[name] = mask[name] * new_mask  # retain previously zeroed params

        elif strategy == "train_most_important":
            raise NotImplementedError()
            # select the k-th largest score as the threshold
            # ....

            # set as trainable (mask=1) all parameters with scores above this threshold
            for name, score in scores.items():
                new_mask = (score >= threshold).float()
                mask[name] = mask[name] * new_mask  # retain previously zeroed params

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(
            f"After round {r} | Adjusted mask (Old mask * New mask): Fraction trainable (1) {_compute_num_trainable_params(mask)} - sparsity (0): {_compute_sparsity(mask)}"
        )
        print()

    print("Progressive Mask Calibration completed.")

    return mask
