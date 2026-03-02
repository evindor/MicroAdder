"""MicroAdder: 74-parameter transformer for 10-digit addition.

Architecture:
    d_model = 5 = tok_dim(2) + pos_dim(3)
    1 layer, 1 head, head_dim = 5 (= d_model)
    Split attention: Q,K from pos_dim (3->5), V from tok_dim via tied head_proj
    Rank-1 attention output projection
    FFN dim=2, GELU activation, no bias
    Shared RMSNorm (single weight vector for all 3 norm sites)
    Parametric circular token embeddings (3 arc params)
    Spiral positional encoding (4 params)
    vocab_size=10 (digits 0-9 only)

Parameter count: exactly 74.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    VOCAB_SIZE, SEQ_LEN, MAX_DIGITS, ANSWER_LEN,
    POS_SOURCES, POS_INDICES,
)


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """All architectural hyperparameters for the 74p MicroAdder."""
    d_model: int = 5
    tok_dim: int = 2
    pos_dim: int = 3
    n_heads: int = 1
    head_dim: int = 5
    ffn_dim: int = 2
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = SEQ_LEN

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── RMSNorm ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable per-dim weights."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ── Rank-1 Linear ────────────────────────────────────────────────────────

class Rank1Linear(nn.Module):
    """Low-rank projection: y = (x @ A) @ B, where A is (in, 1) and B is (1, out).

    Total params: in_features + out_features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, 1))
        self.B = nn.Parameter(torch.empty(1, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A) @ self.B


# ── Main Model ───────────────────────────────────────────────────────────

class MicroAdder(nn.Module):
    """74-parameter autoregressive decoder for 10-digit addition.

    Parameter breakdown (74 total):
        tok_arc (A, start, stride)      3
        spiral (amp, phase, slope, off) 4
        z_hi_pos                        3
        special_pos_equals              3
        q_phase_angle                   1
        q_proj (3 -> 5, no bias)       15
        out_proj (5+5 rank-1)          10
        fc1 (5 -> 2, no bias)          10
        fc2 (2 -> 5, no bias)          10
        head_proj (5 -> 2, no bias)    10
        norm_weight (shared, dim 5)     5
        ────────────────────────────────
        TOTAL                          74
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        assert cfg.d_model == cfg.tok_dim + cfg.pos_dim
        self.cfg = cfg

        # ── Parametric circular token embeddings (3 params) ──────────
        # emb[d] = [A*cos(start + d*stride), A*sin(start + d*stride)]
        # A=B tied (circular arc)
        self.tok_arc_A = nn.Parameter(torch.tensor(2.5))
        self.tok_arc_start = nn.Parameter(torch.tensor(-1.2))
        self.tok_arc_stride = nn.Parameter(torch.tensor(0.29))

        # ── Spiral positional encoding (4 params) ────────────────────
        # pos[i] = [amp*cos(2*pi*i/10 + phase),
        #           amp*sin(2*pi*i/10 + phase),
        #           slope*i + offset]
        self.spiral_amp = nn.Parameter(torch.tensor(1.0))
        self.spiral_phase = nn.Parameter(torch.tensor(0.0))
        self.spiral_slope = nn.Parameter(torch.tensor(1.0 / 9.0))
        self.spiral_offset = nn.Parameter(torch.tensor(0.0))

        # ── Special positions ────────────────────────────────────────
        # PLUS (frozen zero), EQUALS (learned, 3 params), EOS (frozen zero)
        self.register_buffer("_plus_pos", torch.zeros(1, cfg.pos_dim))
        self.special_pos_equals = nn.Parameter(torch.zeros(1, cfg.pos_dim))  # 3 params
        self.register_buffer("_eos_pos", torch.zeros(1, cfg.pos_dim))

        # Carry position (learned, 3 params)
        self.z_hi_pos = nn.Parameter(torch.zeros(1, cfg.pos_dim))

        # ── Position index buffers ───────────────────────────────────
        self.register_buffer(
            "_pos_sources", torch.tensor(POS_SOURCES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_pos_indices", torch.tensor(POS_INDICES, dtype=torch.long),
            persistent=False,
        )

        # ── Attention (tied Q/K with phase rotation) ─────────────────
        # Q and K share the same projection: pos_dim -> head_dim (3->5)
        self.q_proj = nn.Linear(cfg.pos_dim, cfg.head_dim, bias=False)  # 15 params

        # Learnable phase angle for Q (1 param) -- breaks Q/K symmetry
        self.q_phase_angle = nn.Parameter(torch.zeros(cfg.n_heads))  # 1 param

        # Rank-1 output projection (5+5 = 10 params)
        self.out_proj = Rank1Linear(cfg.head_dim, cfg.d_model)

        # Causal mask
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        # ── FFN (no bias, 10+10 = 20 params) ────────────────────────
        self.fc1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)  # 10 params
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)  # 10 params

        # ── Output head (10 params, also serves as V projection) ─────
        self.head_proj = nn.Linear(cfg.d_model, cfg.tok_dim, bias=False)  # 10 params

        # ── Shared RMSNorm (5 params) ────────────────────────────────
        # Single weight vector shared across all 3 norm sites:
        # pre-attention, pre-FFN, pre-output
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.norm_f = RMSNorm(cfg.d_model)
        # Tie all norm weights to the same parameter
        shared_weight = self.norm1.weight
        self.norm2.weight = shared_weight
        self.norm_f.weight = shared_weight

        self._init_weights()

    # ── Parametric token embeddings ──────────────────────────────────

    def _compute_tok_emb(self) -> torch.Tensor:
        """Compute token embedding table from arc parameters.

        Returns (vocab_size, tok_dim) tensor.
        Each digit d maps to [A*cos(start + d*stride), A*sin(start + d*stride)].
        """
        d = torch.arange(self.cfg.vocab_size, device=self.tok_arc_A.device,
                         dtype=self.tok_arc_A.dtype)
        angles = self.tok_arc_start + d * self.tok_arc_stride
        return torch.stack([
            self.tok_arc_A * torch.cos(angles),
            self.tok_arc_A * torch.sin(angles),
        ], dim=1)  # (vocab_size, 2)

    # ── Spiral positions ─────────────────────────────────────────────

    def _get_digit_positions(self) -> torch.Tensor:
        """Compute the (MAX_DIGITS, pos_dim) digit position table from spiral params."""
        idx = torch.arange(MAX_DIGITS, device=self.spiral_amp.device,
                           dtype=self.spiral_amp.dtype)
        angle = 2.0 * math.pi * idx / float(MAX_DIGITS) + self.spiral_phase
        pos = torch.zeros(MAX_DIGITS, self.cfg.pos_dim,
                          device=idx.device, dtype=idx.dtype)
        pos[:, 0] = self.spiral_amp * torch.cos(angle)
        pos[:, 1] = self.spiral_amp * torch.sin(angle)
        pos[:, 2] = self.spiral_slope * idx + self.spiral_offset
        return pos

    def _get_positions(self, T: int) -> torch.Tensor:
        """Build the full (T, pos_dim) position tensor for sequence length T.

        Assembles positions from three tables using the position source mapping:
        - Source 0: digit positions (shared across X, Y, Z sections)
        - Source 1: z_hi carry position
        - Source 2: special positions (PLUS=zero, EQUALS=learned, EOS=zero)
        """
        digit_pos = self._get_digit_positions()                         # (10, 3)
        special_pos = torch.cat([                                       # (3, 3)
            self._plus_pos, self.special_pos_equals, self._eos_pos
        ], dim=0)
        tables = [digit_pos, self.z_hi_pos, special_pos]

        src = self._pos_sources[:T]
        idx = self._pos_indices[:T]
        out = torch.zeros(T, self.cfg.pos_dim,
                          device=self.z_hi_pos.device, dtype=self.z_hi_pos.dtype)
        for source_id, table in enumerate(tables):
            mask = src == source_id
            if mask.any():
                out[mask] = table[idx[mask]]
        return out

    # ── Phase rotation (Q/K asymmetry) ───────────────────────────────

    def _apply_q_phase(self, q: torch.Tensor) -> torch.Tensor:
        """Apply learnable 2D rotation to Q vectors.

        Rotates pairs of dimensions: (0,1), (2,3). Dim 4 is untouched.
        q shape: (B, n_heads, T, head_dim)
        """
        cos_a = self.q_phase_angle.cos()  # (n_heads,)
        sin_a = self.q_phase_angle.sin()
        q_rot = q.clone()
        # Rotate dim pairs (0,1) and (2,3); dim 4 left as-is
        for p in range(self.cfg.head_dim // 2):
            d0, d1 = 2 * p, 2 * p + 1
            c = cos_a[None, :, None]  # (1, n_heads, 1)
            s = sin_a[None, :, None]
            q0 = q[:, :, :, d0]
            q1 = q[:, :, :, d1]
            q_rot[:, :, :, d0] = q0 * c - q1 * s
            q_rot[:, :, :, d1] = q0 * s + q1 * c
        return q_rot

    # ── Weight initialization ────────────────────────────────────────

    def _init_weights(self) -> None:
        """Initialize learnable parameters."""
        with torch.no_grad():
            nn.init.normal_(self.z_hi_pos, std=0.02)
            nn.init.normal_(self.special_pos_equals, std=0.02)
        # Xavier for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)

    # ── Forward pass ─────────────────────────────────────────────────

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional loss computation.

        Args:
            idx: (B, T) input token ids
            targets: (B, T) target ids with -100 for ignored positions

        Returns:
            (logits, loss) where loss is None if targets not provided
        """
        B, T = idx.shape
        tok_emb_table = self._compute_tok_emb()                        # (10, 2)

        # 1. Token + position embeddings -> (B, T, 5)
        tok = tok_emb_table[idx]                                        # (B, T, 2)
        pos = self._get_positions(T).unsqueeze(0).expand(B, -1, -1)     # (B, T, 3)
        x = torch.cat([tok, pos], dim=-1)                               # (B, T, 5)

        # 2. Pre-attention norm
        h = self.norm1(x)

        # 3. Attention: Q,K from pos subspace, V from tok subspace via tied head_proj
        pos_h = h[:, :, self.cfg.tok_dim:]                              # (B, T, 3)
        tok_h = h[:, :, :self.cfg.tok_dim]                              # (B, T, 2)

        Q = self.q_proj(pos_h)                                          # (B, T, 5)
        K = self.q_proj(pos_h)                                          # (B, T, 5) tied
        # Tied V/O: V projection uses head_proj.weight as the (tok_dim->d_model) map.
        # head_proj.weight shape is (tok_dim, d_model) = (2, 5), so
        # V = tok_h @ weight = (B,T,2) @ (2,5) = (B,T,5).
        V = tok_h @ self.head_proj.weight                               # (B, T, 5)

        # Reshape for multi-head (trivial with 1 head)
        Q = Q.view(B, T, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        K = K.view(B, T, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        V = V.view(B, T, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)

        # 4. Phase rotation on Q (asymmetry for tied Q/K)
        Q = self._apply_q_phase(Q)

        # 5. Scaled dot-product attention with causal mask
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # 6. Attention output -> rank-1 projection -> residual
        out = (att @ V).transpose(1, 2).contiguous().view(B, T, self.cfg.head_dim)
        x = x + self.out_proj(out)

        # 7. FFN with pre-norm (shared norm weights)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))

        # 8. Output logits: head_proj(norm(x)) @ tok_emb.T
        logits = self.head_proj(self.norm_f(x)) @ tok_emb_table.T      # (B, T, 10)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ── Autoregressive generation ────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Args:
            prompt: (B, T) token ids for the prompt
            max_new_tokens: number of tokens to generate

        Returns:
            (B, max_new_tokens) tensor of generated token ids
        """
        idx = prompt
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx[:, prompt.shape[1]:]


# ── Utilities ────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
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
