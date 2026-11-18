from __future__ import annotations

import math
import torch

from torch import nn

from .nn_utils import softmax
from .linear import Linear
from .rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores.masked_fill_(~mask, float('-inf'))
    return softmax(scores, dim=-1) @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by a positive num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        #Llama2 和 Qwen2 这里名字叫做 o_proj，这里为了迎合测试代码命名 output_proj
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None =None) -> torch.Tensor:
        seq_len = x.shape[-2]
        #(..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
        head_shape = (*x.shape[:-1], self.num_heads, self.d_k)
        q = self.q_proj(x).reshape(head_shape).transpose(-3, -2)
        k = self.k_proj(x).reshape(head_shape).transpose(-3, -2)
        v = self.v_proj(x).reshape(head_shape).transpose(-3, -2)
        
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # (..., seq_len) -> (..., 1, seq_len), shared across heads.
            '''如果默认token_positions是一位张量(seq_len,) ，则可以正常广播，无需unsqueeze，但是token_positions可能传入batch
            假设 Q/K 的形状为 (B, H, S, D)：
            - token_positions 为 (S,) 时，查出的 cos/sin 是 (S, D/2)，可以直接广播到所有 batch 和 head。
            - token_positions 为 (B, S) 时，查出的 cos/sin 是 (B, S, D/2)，就需要补上 head 轴。
            positions = token_positions.unsqueeze(-2)
            因为广播是从右向左对齐的，不加这一行会变成：
            even/odd:  (B, H, S, D/2)
            cos/sin:   (1, B, S, D/2)'''
            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        attend = scaled_dot_product_attention(q, k, v, mask)
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        combined = attend.transpose(-3, -2).reshape(*x.shape[:-1], self.d_model)
        return self.output_proj(combined)