from __future__ import annotations

import torch
from torch import nn

from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .rope import RotaryPositionalEmbedding
from .attention import MultiHeadSelfAttention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by a positive num_heads")
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        #残差连接
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) 
                for _ in range(num_layers)
            ]  
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("Input sequence length exceeds context_length")
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device)
        
        #(..., seq_len) -> (..., seq_len, d_model)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
         # (..., seq_len, d_model) -> (..., seq_len, vocab_size)
        return self.lm_head(self.ln_final(x))
            