import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from art.config import ArtConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ArtConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Flash Attention: fused causal softmax, faster on MPS/CUDA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(
        self, x: Tensor, kv_cache: dict | None
    ) -> tuple[Tensor, dict]:
        """Single-step forward with KV-cache for autoregressive generation."""
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v}

        # Manual attention (T=1 query against full KV sequence, not causal-masked needed)
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, config: ArtConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ArtConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def forward_cached(self, x: Tensor, kv_cache: dict | None) -> tuple[Tensor, dict]:
        normed = self.ln1(x)
        attn_out, new_cache = self.attn.forward_cached(normed, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache


class PixelGPT(nn.Module):
    def __init__(self, config: ArtConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    @torch.no_grad()
    def forward_with_activations(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass that also returns intermediate layer activations.

        Each activation has shape (B, T, d_model) — the output of each
        transformer block before residual connections sum up.
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)
        layer_activations = []
        for block in self.blocks:
            h = block(h)
            layer_activations.append(h.detach())
        h = self.ln_f(h)
        return self.head(h), layer_activations

    def _init_kv_cache(self) -> list[dict | None]:
        return [None] * len(self.blocks)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: int = 0,
        device: str = "cpu",
    ) -> Tensor:
        tokens, _ = self.generate_with_confidence(
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        return tokens

    @torch.no_grad()
    def generate_with_confidence(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: int = 0,
        device: str = "cpu",
        on_token: Callable[[int, Tensor, Tensor], None] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate sequences using KV-cache for O(n) instead of O(n²) compute.

        Returns (tokens, confidences) where confidences[b, t] is the probability
        the model assigned to the chosen token at position t.
        """
        config = self.config
        # Generate on CPU — MPS dispatch overhead dominates for short sequences
        gen_device = torch.device("cpu")
        model_device = next(self.parameters()).device
        if model_device != gen_device:
            self.to(gen_device)

        seq = torch.full(
            (batch_size, config.seq_length), config.PAD, dtype=torch.long, device=gen_device
        )
        seq[:, 0] = config.BOS
        confidences = torch.zeros(batch_size, config.seq_length, device=gen_device)

        # KV-cache: list of per-layer dicts, each holding (B, H, T, d_head) k and v
        kv_caches = self._init_kv_cache()

        # Prime the cache with BOS token
        bos = seq[:, :1]  # (B, 1)
        positions = torch.zeros(1, 1, dtype=torch.long, device=gen_device)
        h = self.tok_emb(bos) + self.pos_emb(positions)
        new_caches = []
        for block, cache in zip(self.blocks, kv_caches):
            h, updated = block.forward_cached(h, cache)
            new_caches.append(updated)
        kv_caches = new_caches

        for t in range(1, config.seq_length):
            # Only process the single new token (T=1 each step)
            cur_tok = seq[:, t - 1 : t]  # (B, 1)
            pos = torch.tensor([[t - 1]], dtype=torch.long, device=gen_device)
            h = self.tok_emb(cur_tok) + self.pos_emb(pos)

            new_caches = []
            for block, cache in zip(self.blocks, kv_caches):
                h, updated = block.forward_cached(h, cache)
                new_caches.append(updated)
            kv_caches = new_caches

            logits = self.head(self.ln_f(h))[:, 0, :]  # (B, vocab)
            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq[:, t] = next_token.squeeze(-1)
            confidences[:, t] = probs.gather(1, next_token).squeeze(-1)

            if on_token is not None:
                on_token(t, seq, confidences)

        if model_device != gen_device:
            self.to(model_device)

        return seq, confidences

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
