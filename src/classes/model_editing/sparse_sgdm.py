import torch
from torch.optim import SGD, Optimizer


class SparseSGDM_v1(SGD):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        gradient_masks=None,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.gradient_masks = gradient_masks or {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Mask the gradient
                if p in self.gradient_masks:
                    p.grad *= self.gradient_masks[p]

        # Now call the original SGD step
        super().step()
        return loss


class SparseSGDM_v2(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        gradient_masks=None,
    ):
        """
        gradient_masks: a dict {param: mask_tensor} where mask_tensor is 0/1 of same shape as param
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super(SparseSGDM_v2, self).__init__(params, defaults)
        self.gradient_masks = gradient_masks or {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply gradient mask if available
                if param in self.gradient_masks:
                    grad = grad * self.gradient_masks[param]

                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                state = self.state[param]

                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                param.data.add_(update, alpha=-group["lr"])

        return loss
