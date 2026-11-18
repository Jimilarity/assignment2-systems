from __future__ import annotations

import torch

from collections.abc import Iterable

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)

def cross_entropy(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted = x - x.amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return
    total_norm = torch.stack([grad.norm(2) for grad in gradients]).norm(2)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in gradients:
            grad.mul_(scale)