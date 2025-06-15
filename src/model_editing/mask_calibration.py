import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional


def _num_total_params(mask) -> int:
    return sum(t.numel() for t in mask.values())


def _num_zero_params(mask) -> int:
    return sum((t == 0).sum().item() for t in mask.values())


def _compute_sparsity(mask) -> float:
    return _num_zero_params(mask) / _num_total_params(mask)


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
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                scores[name] += param.grad.detach() ** 2

    # TODO: not sure this is correct or necessary
    # for name in scores:
    #     scores[name] /= total_batches

    return scores


def _compute_fisher_scores(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
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
    strategy: str = "least",
    approximate_fisher: bool = False,
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

    for r in range(1, rounds + 1):
        print(f"[Round {r}]")
        print(
            f"Current Sparsity: {_compute_sparsity(mask):.4f} "
            f"({_num_zero_params(mask)}/{_num_total_params(mask)} parameters zeroed out)\n"
        )

        # Recompute Fisher scores at each round.
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

        # accumulated scores
        all_scores = torch.cat([v.flatten() for v in scores.values()])

        print(f"Max score found: {all_scores.max().item()}")
        target_sparsity = sparsity ** (r / rounds)
        total_params = all_scores.numel()

        # if sparsity = 0.9 , we want the 90% values to be 0 (frozen -not trainable)
        k = int(
            (1 - target_sparsity) * total_params
        )  # k elements to keep /  be trainable /  mask = 1
        print(f"Target Sparsity: {target_sparsity:.4f} | k: {k}")

        if strategy == "train_least_important":
            # select the k-th smallest score as the threshold
            threshold, _ = torch.kthvalue(input=all_scores, k=k, largest=False)
            print("Threshold:", threshold)

            # set as trainable (mask=1) all parameters with scores below this threshold
            for name, score in scores.items():
                new_mask = (score <= threshold).float()

        elif strategy == "train_most_important":
            # select the k-th largest score as the threshold
            threshold, _ = torch.kthvalue(input=all_scores, k=k, largest=True)
            print("Threshold:", threshold)

            # set as trainable (mask=1) all parameters with scores above this threshold
            for name, score in scores.items():
                new_mask = (score >= threshold).float()
                # we will set as trainable (mask=1) all parameters with scores above this threshold

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        mask[name] = mask[name] * new_mask  # Keep previous frozen params frozen

        print(
            f"Round {r}. Reached Sparsity: {_compute_sparsity(mask):.4f} "
            f"({_num_zero_params(mask)}/{_num_total_params(mask)} parameters zeroed out)"
        )

    print("Progressive Mask Calibration completed.")

    return mask
