from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError("d_k must be a positive even number")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inverse_frequencies = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((cos * even - sin * odd, sin * even + cos * odd), dim=-1)
        # (..., seq_len, d_k // 2, 2) -> (..., seq_len, d_k)
        return rotated.flatten(-2)