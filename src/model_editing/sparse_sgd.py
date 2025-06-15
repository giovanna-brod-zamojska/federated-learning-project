from torch.optim import SGD


class SparseSGD(SGD):

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        weight_decay=0,
        mask=None,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.mask = mask  # Dict {param_name: mask_tensor}

    def step(self, closure=None):

        if closure is not None:
            closure()

        # The mask is applied to the gradients.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    pname = p._name if hasattr(p, "_name") else None
                    if not pname:  # if pname is None, computation will be wrong
                        raise ValueError("Parameter does not have a name attribute.")

                    # Apply gradient mask if the parameter is in the mask
                    if self.mask is not None and pname in self.mask:
                        p.grad *= self.mask[
                            pname
                        ]  # TODO: or maybe we should do: p.grad.data *= self.mask[pname] ?

        return super().step(closure)
