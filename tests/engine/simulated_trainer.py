"""A stand-in trainer: AdamW on an FP32 master copy, exported in the parameter's dtype each step.

Gradients are N(0, grad_scale^2), seeded by (seed, tensor, step), so one tensor's whole
trajectory is computed on its own and no model-wide optimizer state is held. grad_scale sets how
far Adam moves each element: far above eps every element moves a full lr per step, near eps (as
real, norm-clipped gradients are) most move less, so it sets the density of changed bytes. The
update matches torch.optim.AdamW (single-tensor path).
"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SimulatedTrainer:
    lr: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    weight_decay: float = 0.1
    seed: int = 0
    grad_scale: float = 1.0

    def _gradient(self, name: str, shape, step: int, device) -> torch.Tensor:
        key = hashlib.sha256(f"{self.seed}/{name}/{step}".encode()).digest()
        generator = torch.Generator(device=device).manual_seed(
            int.from_bytes(key[:8], "little") >> 1
        )
        gradient = torch.randn(
            shape, generator=generator, device=device, dtype=torch.float32
        )
        return gradient.mul_(self.grad_scale)

    def trajectory(
        self, name: str, initial: torch.Tensor, no_gradient: list[torch.Tensor]
    ) -> Iterator[torch.Tensor]:
        """The parameter after steps 1, 2, ... in its own dtype. `no_gradient`: masks, broadcast to
        the parameter and taken in turn by the steps, of the elements a real trainer gives no gradient."""
        master = initial.detach().to(torch.float32, copy=True)
        exp_avg, exp_avg_sq = torch.zeros_like(master), torch.zeros_like(master)
        step = 0
        while True:
            step += 1
            grad = self._gradient(name, master.shape, step, master.device)
            if no_gradient:
                grad.masked_fill_(no_gradient[(step - 1) % len(no_gradient)], 0)
            master.mul_(1 - self.lr * self.weight_decay)
            exp_avg.lerp_(grad, 1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            bias_correction1 = 1 - self.beta1**step
            bias_correction2_sqrt = (1 - self.beta2**step) ** 0.5
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
            master.addcdiv_(exp_avg, denom, value=-self.lr / bias_correction1)
            # A copy even for FP32 parameters: the next step updates the master in place.
            yield master.to(initial.dtype, copy=True)
