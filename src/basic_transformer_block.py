"""
A from-scratch implementation of a single Transformer encoder block
(multi-head self-attention + position-wise feed-forward network),
built with PyTorch primitives to demonstrate understanding of the
core Transformer architecture described in "Attention Is All You Need".

This module is self-contained and has no dependency on Hugging Face
Transformers -- it is meant to prove foundational knowledge of the
mechanism, independent of the library used later for fine-tuning.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Given an input of shape (batch, seq_len, embed_dim), this layer:
      1. Projects the input into Query, Key, and Value tensors.
      2. Splits each projection into `num_heads` independent heads.
      3. Computes scaled dot-product attention per head.
      4. Concatenates the heads and applies a final output projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined linear projections for Q, K, V (more efficient than 3 separate layers)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
            attn_mask: Optional boolean/float mask broadcastable to
                       (batch, num_heads, seq_len, seq_len). Positions with
                       True / -inf are masked out (e.g. for causal LM training).

        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected last dim {self.embed_dim}, got {embed_dim}"
            )

        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)

        out = self.out_proj(context)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """
    A single Transformer block combining:
      - Multi-head self-attention with a residual connection + LayerNorm
      - A feed-forward network with a residual connection + LayerNorm

    Uses the "pre-norm" convention (LayerNorm before the sub-layer), which
    is the same convention used by GPT-2 and most modern LLMs, and tends
    to give more stable training than the original post-norm design.

    Shape contract:
        Input:  (batch_size, seq_len, embed_dim)
        Output: (batch_size, seq_len, embed_dim)  -- shape-preserving
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # --- Input shape verification ---
        if x.dim() != 3:
            raise ValueError(
                f"Expected input of shape (batch, seq_len, embed_dim), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim} on last axis, got {x.shape[-1]}"
            )

        # --- Self-attention sub-layer (pre-norm + residual) ---
        residual = x
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, attn_mask=attn_mask)
        x = residual + attn_out

        # --- Feed-forward sub-layer (pre-norm + residual) ---
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out

        # --- Output shape verification ---
        assert x.shape[-1] == self.embed_dim, "Block must be shape-preserving on embed_dim"
        return x


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Builds a boolean causal (look-ahead) mask of shape (1, 1, seq_len, seq_len)
    where True marks positions that must be masked out (future tokens).
    Useful if this block is reused for autoregressive/decoder-style training.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _verify_forward_pass() -> None:
    """
    Sanity-check routine: constructs a BasicTransformerBlock, runs a forward
    pass on a random batch, and verifies input/output shape equality plus
    a basic numerical health check (no NaNs/Infs).
    """
    torch.manual_seed(42)

    batch_size, seq_len, embed_dim, num_heads, ff_dim = 4, 16, 64, 8, 256

    block = BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 1) Bidirectional (encoder-style) forward pass
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: in={x.shape}, out={out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf values"
    print(f"[OK] Encoder-style forward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}")

    # 2) Causal (decoder-style) forward pass with masking
    mask = build_causal_mask(seq_len)
    out_causal = block(x, attn_mask=mask)
    assert out_causal.shape == x.shape
    assert torch.isfinite(out_causal).all()
    print(f"[OK] Causal-masked forward pass: input {tuple(x.shape)} -> output {tuple(out_causal.shape)}")

    # 3) Gradient flow check (ensures the block is actually trainable)
    loss = out.sum()
    loss.backward()
    total_grad_norm = sum(
        p.grad.norm().item() for p in block.parameters() if p.grad is not None
    )
    assert total_grad_norm > 0, "No gradients flowed through the block"
    print(f"[OK] Backward pass succeeded, total grad norm = {total_grad_norm:.4f}")

    num_params = sum(p.numel() for p in block.parameters())
    print(f"[INFO] BasicTransformerBlock parameter count: {num_params:,}")
    print("\nAll BasicTransformerBlock checks passed.")


if __name__ == "__main__":
    _verify_forward_pass()