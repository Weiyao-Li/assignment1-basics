import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int

# Linear transformation: y = x @ W^T + b
# weight stored as (out, in) following PyTorch convention
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., in_features) -> (..., out_features)
        return x @ self.weight.T + self.bias

# Token lookup table: maps integer token IDs to dense embedding vectors
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (...) -> (..., embedding_dim)
        return self.weight[token_ids]

# Root Mean Square Layer Normalization: normalizes by RMS instead of mean+std
# Upcasts to float32 internally to avoid overflow when squaring activations
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter g_i, initialized to 1 (identity at init)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., d_model) -> (..., d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

# SwiGLU feed-forward network: SiLU(W1·x) * W3·x, then project back with W2
# d_ff is set to (8/3)*d_model rounded up to multiple of 64 for hardware efficiency
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        d_ff = int(8 / 3 * d_model)
        d_ff = (d_ff + 63) // 64 * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., d_model) -> (..., d_ff) -> (..., d_model)
        gate = self.w1(x)
        # SiLU(gate) = sigmoid(gate) * gate; element-wise multiply with w3(x) as content path
        return self.w2(torch.sigmoid(gate) * gate * self.w3(x))

# Rotary Positional Embedding: rotates pairs of query/key dimensions by position-dependent angles
# Pre-computes cos/sin tables at init; shared across all layers to save memory
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        dims = torch.arange(0, d_k, 2, device=device)
        # freq for pair k: 1 / theta^(2k/d_k)
        freqs = 1.0 / (theta ** (dims / d_k))
        angles = positions * freqs
        # Register as buffer (not a parameter): moves with the model but not trained
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # (..., seq_len, d_k) -> (..., seq_len, d_k)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # Split into even/odd pairs and apply 2D rotation to each pair
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

# Numerically stable softmax: subtract max before exp to prevent overflow
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
# Supports optional causal or padding mask (True = keep, False = mask out)
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # (..., queries, d_k) x (..., keys, d_k) -> (..., queries, keys)
    d_k = Q.shape[-1]
    # Compute attention scores and scale by sqrt(d_k) to prevent vanishing gradients
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # Fill masked positions with -inf so they become 0 after softmax
        scores = scores.masked_fill(~mask, float("-inf"))
    # (..., queries, keys) -> (..., queries, d_v)
    attn = softmax(scores, dim=-1)
    return attn @ V

# Causal multi-head self-attention with optional RoPE
# Each head independently attends with d_k = d_model / num_heads
# num_heads dimension acts as a batch dimension inside scaled_dot_product_attention
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
        # (..., seq_len, d_model) -> (..., seq_len, d_model)
        *batch, seq_len, d_model = x.shape
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
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
        # (..., num_heads, seq_len, d_k) -> (..., num_heads, seq_len, d_k)
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        out = out.transpose(-2, -3).flatten(-2)
        return self.output_proj(out)

# Pre-norm Transformer block: RMSNorm before each sub-layer, residual connection after
# Sub-layer 1: x = x + Attn(RMSNorm(x))
# Sub-layer 2: x = x + FFN(RMSNorm(x))
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: "RotaryPositionalEmbedding | None" = None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        # Use __new__ to bypass SwiGLU.__init__ so we can set d_ff externally
        self.ffn = SwiGLU.__new__(SwiGLU)
        nn.Module.__init__(self.ffn)
        self.ffn.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ffn.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.ffn.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # (..., seq_len, d_model) -> (..., seq_len, d_model)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

# Full Transformer language model: token embeddings -> N blocks -> final norm -> lm_head
# lm_head projects to vocab_size logits for next-token prediction at each position
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # Single shared RoPE instance across all layers
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embeddings(in_indices)
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model) x num_layers
        for layer in self.layers:
            x = layer(x)
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.ln_final(x)
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.lm_head(x)

# Cross-entropy loss: -log p(target) = log_sum_exp(logits) - logits[target]
# Uses log-sum-exp trick (subtract max) for numerical stability
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Subtract max for numerical stability before log/exp
    logits = inputs - inputs.max(dim=-1, keepdim=True).values
    # log(sum(exp(logits))) - logits[target]  (log-sum-exp trick cancels log and exp)
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))
    correct_logits = logits[torch.arange(len(targets)), targets]
    return (log_sum_exp - correct_logits).mean()

# AdamW optimizer: Adam with decoupled weight decay
# m_t = beta1*m_{t-1} + (1-beta1)*g  (first moment / gradient EMA)
# v_t = beta2*v_{t-1} + (1-beta2)*g^2  (second moment / squared gradient EMA)
# theta = theta * (1 - lr*lambda) - alpha_t * m_t / (sqrt(v_t) + eps)
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # Initialize state on first step
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t = state["t"]
                m, v = state["m"], state["v"]
                # Update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Bias correction
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # Decoupled weight decay (applied to param, not gradient)
                p.data.mul_(1 - lr * lam)
                # Parameter update
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
                state["t"] = t + 1
        return loss

# Gradient clipping: scales all gradients down if their global L2 norm exceeds max_l2_norm
# Preserves gradient direction, only reduces magnitude
def gradient_clipping(parameters, max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

# Data loader: samples random batches of (input, target) pairs from a flat token array
# Input:  x[i : i+context_length], Target: x[i+1 : i+context_length+1]
def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample batch_size random start indices; valid range is [0, len-context_length)
    indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    inputs = torch.stack([torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in indices])
    targets = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)) for i in indices])
    return inputs.to(device), targets.to(device)

# Save model + optimizer state and current iteration to a file/path
def save_checkpoint(model, optimizer, iteration: int, out) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)

# Load model + optimizer state from a checkpoint; returns the saved iteration number
def load_checkpoint(src, model, optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

# Autoregressive text generation with temperature scaling and top-p (nucleus) sampling
# Generates tokens one at a time, appending each to the context for the next step
def generate(
    model: "TransformerLM",
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> list[int]:
    model.eval()
    tokens = list(prompt)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(context)
            # Take logits at the last position: (vocab_size,)
            next_logits = logits[0, -1, :]

            # Temperature scaling: low tau -> sharper distribution (more greedy)
            next_logits = next_logits / temperature
            probs = softmax(next_logits, dim=0)

            # Top-p (nucleus) sampling: keep only the smallest set of tokens whose cumulative prob >= p
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                # Remove tokens once cumulative prob exceeds top_p (shift by 1 to keep the token that crosses p)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                # Renormalize so probabilities sum to 1
                sorted_probs = sorted_probs / sorted_probs.sum()
                # Sample from the truncated distribution and map back to original indices
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[sampled].item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens[len(prompt):]
