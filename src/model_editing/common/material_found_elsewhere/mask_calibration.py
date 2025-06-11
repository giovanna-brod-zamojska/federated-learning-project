
def progressive_mask_calibration(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    target_sparsity: float = 0.9,
    rounds: int = 5,
    warn_tolerance: float = 0.02,
    pruning_type: PruningType = PruningType.FISHER,
) -> Mask:
    """
    Progressively create a gradient mask using Fisher info, applying pruning at each round.

    This method is based on:
        "The Lottery Ticket Hypothesis: finding sparse, trainable neural networks" by Frankle and Carbin.
         https://arxiv.org/abs/1803.03635.

    Note: Raises a RuntimeError if the final sparsity deviates significantly from the target.

    Args:
        model (nn.Module): Model to prune.
        dataloader (DataLoader): DataLoader (not used in dummy logic).
        loss_fn (nn.Module): Loss function (not used in dummy logic).
        target_sparsity (float): Target sparsity at the end of pruning.
        rounds (int): Number of pruning rounds.
        warn_tolerance (float): Relative deviation tolerance to trigger a warning.
        pruning_type (PruningType): Type of pruning, defaults to PruningType.FISHER.
    """
    assert isinstance(rounds, int)
    assert rounds > 1, "rounds needs to be greater than 1"
    device = get_device()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    grad_mask = Mask(
        mask_dict={
            name: torch.ones_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    )

    sparsity_targets = np.geomspace(0.1, target_sparsity, rounds)

    for r, sparsity_target in enumerate(sparsity_targets):
        logging.info(f"[Round {r}] Target sparsity: {sparsity_target}")

        # Recompute Fisher based on the masked model
        fisher_diag = compute_fisher_diagonal(
            model, dataloader, loss_fn, mask=grad_mask, pruning_type=pruning_type
        )

        # Count how many parameters are already frozen and how many parameters are still to mask
        already_masked = grad_mask.num_zeroed_parameters
        parameters_to_mask = round(total_params * sparsity_target)
        adjusted_sparsity = (parameters_to_mask - already_masked) / total_params
        logging.debug(f"[Round {r}]: Target Adjusted sparsity: {adjusted_sparsity}")
        # Create new mask (0 = freeze, 1 = allow update)
        new_mask = create_fisher_mask(fisher_diag, model, sparsity=adjusted_sparsity)
        new_sparsity = new_mask.num_zeroed_parameters
        logging.debug(
            f"[Round {r}] Actual Adjusted Sparsity: {new_sparsity/total_params}."
        )

        # Progressive pruning.
        # Update cumulative mask (once frozen, always frozen) i.e. once a parameter has been set to zero because it's
        # important it's going to stay set to zero. If it is unimportant in the old mask (grad_mask=1) and important in the
        # new_mask (new_mask=0), update it so that grad_mask=0, thus gradually increasing sparsity.
        grad_mask = grad_mask.update(new_mask)
        masked = grad_mask.num_zeroed_parameters
        logging.info(
            f"[Round {r}] Actual Sparsity: {masked/total_params}. Masked: {masked} / {total_params}. "
        )

    # Final sparsity check and warning
    actual_sparsity = grad_mask.sparsity
    rel_error = abs(actual_sparsity - target_sparsity) / target_sparsity

    if rel_error > warn_tolerance:
        raise RuntimeError(
            f"Final sparsity {actual_sparsity:.4f} deviates from target {target_sparsity:.4f} "
            f"by {rel_error:.2%} (exceeds allowed tolerance of {warn_tolerance:.2%})."
        )
    return grad_mask