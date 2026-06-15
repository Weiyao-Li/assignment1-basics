import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        d_ff = int(8 / 3 * d_model)
        d_ff = (d_ff + 63) // 64 * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(torch.sigmoid(gate) * gate * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    '''
    对输入向量的每两个维度组成的二维子空间，根据 token 的位置旋转不同角度，
    从而把位置信息编码进 query/key 向量中，同时保持向量维度不变。
    '''
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        dims = torch.arange(0, d_k, 2, device=device)
        freqs = 1.0 / (theta ** (dims / d_k))
        angles = positions * freqs
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    # Compute attention scores and scale by sqrt(d_k) to prevent vanishing gradients
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # Fill masked positions with -inf so they become 0 after softmax
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: "RotaryPositionalEmbedding | None" = None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch, seq_len, d_model = x.shape
        # Project and split into heads: (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
        Q = self.q_proj(x).unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)
        K = self.k_proj(x).unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)
        V = self.v_proj(x).unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).expand(*batch, seq_len)
            # Expand token_positions for heads dimension: (..., 1, seq_len)
            pos = token_positions.unsqueeze(-2)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # Causal mask: only attend to current and previous positions
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads back: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        out = out.transpose(-2, -3).flatten(-2)
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: "RotaryPositionalEmbedding | None" = None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU.__new__(SwiGLU)
        nn.Module.__init__(self.ffn)
        self.ffn.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ffn.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.ffn.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)