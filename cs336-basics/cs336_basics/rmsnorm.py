from __future__ import annotations

import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #转换输入类型防止溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(in_dtype)