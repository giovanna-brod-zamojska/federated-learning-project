import torch
from torch.optim import SGD
from typing import Dict, Iterable


class SparseSGD(SGD):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        named_params: Dict[str, torch.nn.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        mask=None,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.mask = mask  # Dict {param_name: mask_tensor}
        self.named_params = named_params
        self.param_id_to_name = {id(p): n for n, p in named_params.items()}

    def step(self, closure=None):

        if closure is not None:
            closure()

        # The mask is applied to the gradients.
        for group in self.param_groups:
            for p in group["params"]:
                # Applying the gradient mask only if that name is in the mask
                name = self.param_id_to_name.get(id(p))
                if p.grad is not None and name in self.mask:
                    p.grad.data *= self.grad_mask[name]

        return super().step(closure)
