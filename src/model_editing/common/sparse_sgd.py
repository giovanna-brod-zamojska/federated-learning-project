import torch
from torch.optim import Optimizer
from typing import Dict, Iterable

# TODO: i dont know if this is correct

class SparseSGDM(Optimizer):
    """
    SGD with support for sparse (masked) gradient updates.
    Gradients are multiplied by a mask before applying the update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        *,
        grad_mask: Dict[str, torch.Tensor],
        named_params: Dict[str, torch.nn.Parameter],
    ):
        # Initialize base SGD optimizer
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )
        super(SparseSGDM, self).__init__(params, defaults)

        # Validate and store named parameters and masks
        # grad_mask.validate_against(named_params)

        self.named_params = named_params
        self.grad_mask = grad_mask
        self.param_id_to_name = {id(p): n for n, p in named_params.items()}


    def step(self, closure=None):
        """
        Performs a single optimization step with gradient masking.
        This is essentially the same as PyTorch’s SGD but adds gradient masking just before any updates:
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            lr = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply gradient mask if the parameter is in the mask
                name = self.param_id_to_name.get(id(param))
                if name in self.grad_mask:
                    grad = grad * self.grad_mask[name] 

                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                state = self.state[param]

                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                param.data.add_(buf, alpha=-lr)

        return loss
