"""MicroAdder: minimal split-subspace transformer for 10-digit addition.

Architecture (default 242p configuration):
  d_model = 6 = tok_dim(3) + pos_dim(3)
  1 layer, 2 heads, head_dim=3
  Split attention: Q,K from pos dims; V from tok dims (SEPARATE Q and K)
  FFN dim=2 with bias
  RMSNorm (3 norms, weight only)
  Tied output head: head_proj(6->3) @ tok_emb.T
  LSB-first digit ordering, shared XYZ positional encoding
"""

import copy
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    VOCAB_SIZE, SEQ_LEN, MAX_DIGITS, ANSWER_LEN,
    POS_SOURCES, POS_INDICES,
)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model: int = 6
    tok_dim: int = 3
    pos_dim: int = 3
    n_heads: int = 2
    head_dim: int = 3
    n_layers: int = 1
    ffn_dim: int = 2
    ffn_bias: bool = True

    pos_mode: str = "learned"       # "learned" | "spiral_correct"
    pos_correction_mode: str = "full"  # "full" (10 params) | "linear" (2 params)
    freeze_special: str = "none"   # "none" | "eos" | "plus_eos"
    tie_qk: bool = False            # True = share Q,K projection
    attn_out_rank: int = 0          # 0 = full rank
    share_layers: bool = False      # Universal transformer style

    token_init: str = "spiral"      # "spiral" | "normal"
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = SEQ_LEN

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Modules ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LowRankLinear(nn.Module):
    """y = x @ A @ B, params = in*rank + rank*out."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


# ── Attention ──────────────────────────────────────────────────────────────

class SplitAttention(nn.Module):
    """Split-subspace attention: Q,K from pos_dim; V from tok_dim."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.n_heads * cfg.head_dim
        self.tok_dim = cfg.tok_dim
        self.pos_dim = cfg.pos_dim
        self.tie_qk = cfg.tie_qk

        self.q_proj = nn.Linear(cfg.pos_dim, self.inner_dim, bias=False)
        if not cfg.tie_qk:
            self.k_proj = nn.Linear(cfg.pos_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(cfg.tok_dim, self.inner_dim, bias=False)

        if cfg.attn_out_rank > 0:
            self.out_proj = LowRankLinear(self.inner_dim, cfg.d_model, cfg.attn_out_rank)
        else:
            self.out_proj = nn.Linear(self.inner_dim, cfg.d_model, bias=False)

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_tok = x[:, :, :self.tok_dim]
        x_pos = x[:, :, self.tok_dim:]

        q = self.q_proj(x_pos)
        k = self.q_proj(x_pos) if self.tie_qk else self.k_proj(x_pos)
        v = self.v_proj(x_tok)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.out_proj(out)


# ── FFN ────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ── Transformer block ─────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = SplitAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.ffn_dim, bias=cfg.ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────

class MicroAdder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model == cfg.tok_dim + cfg.pos_dim
        self.cfg = cfg

        # Token embedding (learnable, spiral-initialized)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.tok_dim)

        # Positional encoding tables
        self._init_positions(cfg)

        # Position index buffers (shared XYZ mapping)
        self.register_buffer(
            "_pos_sources", torch.tensor(POS_SOURCES, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_pos_indices", torch.tensor(POS_INDICES, dtype=torch.long), persistent=False
        )

        # Transformer blocks
        if cfg.share_layers:
            self._shared_block = TransformerBlock(cfg)
            self.blocks = nn.ModuleList([self._shared_block])
            self._n_passes = cfg.n_layers
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
            )
            self._n_passes = cfg.n_layers

        # Output head
        self.ln_f = RMSNorm(cfg.d_model)
        self.head_proj = nn.Linear(cfg.d_model, cfg.tok_dim, bias=False)
        # Output logits = head_proj(x) @ tok_emb.weight.T  (tied)

        self._init_weights()

    # ── Position encoding ──────────────────────────────────────────────

    def _init_positions(self, cfg: ModelConfig) -> None:
        if cfg.pos_mode == "learned":
            # Full learned: 10 x pos_dim
            self.digit_pos = nn.Parameter(torch.zeros(MAX_DIGITS, cfg.pos_dim))
        elif cfg.pos_mode == "spiral_correct":
            # Parametric spiral (4 params)
            self.spiral_amp = nn.Parameter(torch.tensor(1.0))
            self.spiral_phase = nn.Parameter(torch.tensor(0.0))
            self.spiral_slope = nn.Parameter(torch.tensor(1.0 / max(1, MAX_DIGITS - 1)))
            self.spiral_offset = nn.Parameter(torch.tensor(0.0))
            # Per-position scale correction
            if cfg.pos_correction_mode == "full":
                self.pos_correction = nn.Parameter(torch.zeros(MAX_DIGITS))  # 10 params
            elif cfg.pos_correction_mode == "linear":
                self.pos_corr_slope = nn.Parameter(torch.tensor(0.0))        # 2 params
                self.pos_corr_intercept = nn.Parameter(torch.tensor(0.0))
            else:
                raise ValueError(f"Unknown pos_correction_mode: {cfg.pos_correction_mode}")
        else:
            raise ValueError(f"Unknown pos_mode: {cfg.pos_mode}")

        # Carry position
        self.z_hi_pos = nn.Parameter(torch.zeros(1, cfg.pos_dim))

        # Special token positions (PLUS=0, EQUALS=1, EOS=2)
        if cfg.freeze_special == "none":
            self.special_pos = nn.Parameter(torch.zeros(3, cfg.pos_dim))
        elif cfg.freeze_special == "eos":
            # PLUS and EQUALS are learnable, EOS is fixed to zero
            self.special_pos_learned = nn.Parameter(torch.zeros(2, cfg.pos_dim))
            self.register_buffer("_eos_pos", torch.zeros(1, cfg.pos_dim))
        elif cfg.freeze_special == "plus_eos":
            # Only EQUALS is learnable, PLUS and EOS are fixed to zero
            self.special_pos_equals = nn.Parameter(torch.zeros(1, cfg.pos_dim))
            self.register_buffer("_plus_pos", torch.zeros(1, cfg.pos_dim))
            self.register_buffer("_eos_pos", torch.zeros(1, cfg.pos_dim))
        else:
            raise ValueError(f"Unknown freeze_special: {cfg.freeze_special}")

    def _get_special_pos(self) -> torch.Tensor:
        """Build the (3, pos_dim) special position table."""
        cfg = self.cfg
        if cfg.freeze_special == "none":
            return self.special_pos
        elif cfg.freeze_special == "eos":
            return torch.cat([self.special_pos_learned, self._eos_pos], dim=0)
        else:  # plus_eos
            return torch.cat([self._plus_pos, self.special_pos_equals, self._eos_pos], dim=0)

    def _get_digit_positions(self) -> torch.Tensor:
        """Compute the (MAX_DIGITS, pos_dim) position table."""
        cfg = self.cfg
        if cfg.pos_mode == "learned":
            return self.digit_pos
        # spiral_correct: base spiral * (1 + correction)
        idx = torch.arange(MAX_DIGITS, device=self.spiral_amp.device, dtype=self.spiral_amp.dtype)
        angle = 2.0 * math.pi * idx / float(MAX_DIGITS) + self.spiral_phase
        base = torch.zeros(MAX_DIGITS, cfg.pos_dim, device=idx.device, dtype=idx.dtype)
        if cfg.pos_dim > 0:
            base[:, 0] = self.spiral_amp * torch.cos(angle)
        if cfg.pos_dim > 1:
            base[:, 1] = self.spiral_amp * torch.sin(angle)
        if cfg.pos_dim > 2:
            base[:, 2] = self.spiral_slope * idx + self.spiral_offset
        # Apply correction scaling
        if cfg.pos_correction_mode == "full":
            correction = self.pos_correction
        else:  # linear
            correction = self.pos_corr_intercept + self.pos_corr_slope * idx
        scale = (1.0 + correction).unsqueeze(1)  # (10, 1)
        return base * scale

    def _get_positions(self, T: int) -> torch.Tensor:
        """Build the full (T, pos_dim) position tensor for sequence length T."""
        cfg = self.cfg
        device = self.z_hi_pos.device
        dtype = self.z_hi_pos.dtype

        digit_pos = self._get_digit_positions()  # (10, pos_dim)
        special_pos = self._get_special_pos()    # (3, pos_dim)
        tables = [digit_pos, self.z_hi_pos, special_pos]  # source 0, 1, 2

        src = self._pos_sources[:T]
        idx = self._pos_indices[:T]
        out = torch.zeros(T, cfg.pos_dim, device=device, dtype=dtype)
        for source_id, table in enumerate(tables):
            mask = src == source_id
            if mask.any():
                out[mask] = table[idx[mask]]
        return out

    # ── Weight initialization ──────────────────────────────────────────

    def _init_weights(self) -> None:
        cfg = self.cfg
        # Spiral token init
        if cfg.token_init == "spiral":
            with torch.no_grad():
                self.tok_emb.weight.zero_()
                for d in range(min(10, cfg.vocab_size)):
                    c = math.cos(2 * math.pi * d / 10)
                    s = math.sin(2 * math.pi * d / 10)
                    lin = d / 9.0
                    if cfg.tok_dim > 0: self.tok_emb.weight[d, 0] = c
                    if cfg.tok_dim > 1: self.tok_emb.weight[d, 1] = s
                    if cfg.tok_dim > 2: self.tok_emb.weight[d, 2] = lin
                # Special tokens: distinct corners
                specials = {
                    10: (2.0, 0.0, -1.0),   # PLUS
                    11: (0.0, 2.0, -1.0),   # EQUALS
                    12: (-2.0, 0.0, -1.0),  # EOS
                    13: (0.0, -2.0, -1.0),  # PAD
                }
                for tid, vals in specials.items():
                    if tid < cfg.vocab_size:
                        for j in range(min(cfg.tok_dim, 3)):
                            self.tok_emb.weight[tid, j] = vals[j]

        # Learned position init
        with torch.no_grad():
            if cfg.pos_mode == "learned":
                nn.init.normal_(self.digit_pos, std=0.02)
            nn.init.normal_(self.z_hi_pos, std=0.02)
            if cfg.freeze_special == "none":
                nn.init.normal_(self.special_pos, std=0.02)
            elif cfg.freeze_special == "eos":
                nn.init.normal_(self.special_pos_learned, std=0.02)
            else:  # plus_eos
                nn.init.normal_(self.special_pos_equals, std=0.02)

        # Xavier for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Forward / generate ─────────────────────────────────────────────

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        tok = self.tok_emb(idx)                                     # (B, T, tok_dim)
        pos = self._get_positions(T).unsqueeze(0).expand(B, -1, -1) # (B, T, pos_dim)
        x = torch.cat([tok, pos], dim=-1)                           # (B, T, d_model)

        if self.cfg.share_layers:
            for _ in range(self._n_passes):
                x = self.blocks[0](x)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head_proj(x) @ self.tok_emb.weight.T         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive greedy decoding."""
        idx = prompt
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx[:, prompt.shape[1]:]


# ── Utilities ──────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count unique learnable parameters (respects weight tying)."""
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def parameter_breakdown(model: nn.Module) -> dict:
    """Return {name: numel} for every unique parameter."""
    seen = set()
    breakdown = {}
    for name, p in model.named_parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            breakdown[name] = p.numel()
    return breakdown
