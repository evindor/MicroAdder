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
    X_START, PLUS_POS, Y_START, EQ_POS, Z_START, EOS_POS,
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

    pos_mode: str = "learned"       # "learned" | "spiral_correct" | "zero"
    pos_correction_mode: str = "full"  # "full" (10 params) | "linear" (2 params) | "none" (0 params)
    freeze_special: str = "none"   # "none" | "eos" | "plus_eos" | "all"
    alibi: bool = False             # Add ALiBi attention bias (learned slopes)
    qk_source: str = "pos"         # "pos" = Q,K read pos_dim; "tok" = Q,K read tok_dim
    tie_qk: bool = False            # True = share Q,K projection
    attn_out_rank: int = 0          # 0 = full rank
    num_kv_heads: int = 0           # 0 = same as n_heads (MHA); <n_heads = GQA
    q_phase: bool = False           # Add learnable per-head phase rotation to Q (for tied Q/K asymmetry)
    share_layers: bool = False      # Universal transformer style
    norm_mode: str = "full"         # "full" (18p) | "shared" (6p) | "scalar" (1p) | "fixed" (0p) | "no_ln2" (12p)
    tie_vo: bool = False            # Tie v_proj.weight = head_proj.weight.T (saves tok_dim*head_dim params)
    freeze_pad: bool = False        # Freeze PAD token embedding to zero (saves tok_dim params)
    freeze_z_hi: bool = False       # Freeze z_hi carry position to zero (saves pos_dim params)
    freeze_spiral: str = ""         # Comma-separated spiral params to freeze: "slope,offset" etc
    freeze_tok_arc: str = ""        # Comma-separated arc params to freeze: "start,stride" etc
    tie_tok_arc_ab: bool = False    # Tie tok_arc_A = tok_arc_B (circular, saves 1p)
    q_proj_rank: int = 0            # 0 = full rank q_proj; >0 = low-rank factorization (saves params)
    q_proj_mode: str = "full"       # "full" | "toeplitz" — toeplitz uses (in+out-1) params instead of in*out
    softmax1: bool = False          # Use softmax1 (add 1 to denominator, allows attn sum < 1)
    attn_mode: str = "split"        # "split" | "offset" | "hard_offset"

    token_init: str = "spiral"      # "spiral" | "normal"
    tok_emb_mode: str = "learned"   # "learned" | "parametric"
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


class ScalarRMSNorm(nn.Module):
    """RMSNorm with a single shared scalar weight (1 param instead of d_model)."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


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


class ToeplitzLinear(nn.Module):
    """Linear layer with Toeplitz-constrained weight matrix.

    A (out_features x in_features) Toeplitz matrix has constant diagonals,
    requiring only (in_features + out_features - 1) params instead of in*out.
    Equivalent to a 1D convolution / cross-correlation.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        n_params = in_features + out_features - 1
        self.vals = nn.Parameter(torch.empty(n_params))
        # Kaiming-like init scaled to effective fan_in
        nn.init.normal_(self.vals, std=1.0 / math.sqrt(in_features))

    def _build_weight(self) -> torch.Tensor:
        """Construct (out_features, in_features) Toeplitz matrix from vals."""
        # vals layout: [row_{out-1}, ..., row_1, diag, col_1, ..., col_{in-1}]
        # Row i, col j uses vals[out_features - 1 - i + j]
        idx = torch.arange(self.in_features, device=self.vals.device)
        row_offsets = torch.arange(self.out_features - 1, -1, -1, device=self.vals.device)
        indices = row_offsets.unsqueeze(1) + idx.unsqueeze(0)  # (out, in)
        return self.vals[indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._build_weight()
        return x @ W.T


# ── Attention ──────────────────────────────────────────────────────────────

class SplitAttention(nn.Module):
    """Split-subspace attention: Q,K from pos_dim; V from tok_dim.

    Supports GQA (num_kv_heads < n_heads), ALiBi, tied Q/K, and q-phase.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.n_heads * cfg.head_dim
        self.tok_dim = cfg.tok_dim
        self.pos_dim = cfg.pos_dim
        self.qk_source = cfg.qk_source
        self.tie_qk = cfg.tie_qk
        self.use_alibi = cfg.alibi
        self.use_q_phase = cfg.q_phase
        self.use_softmax1 = cfg.softmax1

        # GQA: num_kv_heads <= n_heads; 0 means same as n_heads (standard MHA)
        self.num_kv_heads = cfg.num_kv_heads if cfg.num_kv_heads > 0 else cfg.n_heads
        assert cfg.n_heads % self.num_kv_heads == 0, \
            f"n_heads ({cfg.n_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.kv_inner_dim = self.num_kv_heads * cfg.head_dim
        self.kv_repeat = cfg.n_heads // self.num_kv_heads

        qk_in_dim = cfg.tok_dim if cfg.qk_source == "tok" else cfg.pos_dim
        if cfg.q_proj_rank > 0:
            self.q_proj = LowRankLinear(qk_in_dim, self.inner_dim, cfg.q_proj_rank)
        elif cfg.q_proj_mode == "toeplitz":
            self.q_proj = ToeplitzLinear(qk_in_dim, self.inner_dim)
        else:
            self.q_proj = nn.Linear(qk_in_dim, self.inner_dim, bias=False)
        if not cfg.tie_qk:
            self.k_proj = nn.Linear(qk_in_dim, self.kv_inner_dim, bias=False)
        else:
            # When tied, K uses q_proj. If GQA with tie_qk, K is sliced from q_proj output.
            pass
        self.v_proj = nn.Linear(cfg.tok_dim, self.kv_inner_dim, bias=False)

        if cfg.attn_out_rank > 0:
            self.out_proj = LowRankLinear(self.inner_dim, cfg.d_model, cfg.attn_out_rank)
        else:
            self.out_proj = nn.Linear(self.inner_dim, cfg.d_model, bias=False)

        # Q-phase: learnable per-head angle applied as 2D rotation to Q
        # Applies to pairs of dimensions: (0,1), (2,3), etc. Last dim untouched if odd.
        if self.use_q_phase:
            self.q_phase_angle = nn.Parameter(torch.zeros(cfg.n_heads))

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        # ALiBi: learned slope per head, initialized to log(10) (base-10 structure)
        if self.use_alibi:
            self.alibi_log_slopes = nn.Parameter(
                torch.full((cfg.n_heads,), math.log(math.log(10)))
            )
            positions = torch.arange(cfg.max_seq_len)
            rel_pos = -(positions.unsqueeze(0) - positions.unsqueeze(1)).float()
            self.register_buffer("_alibi_rel_pos", rel_pos.unsqueeze(0).unsqueeze(0))

    def _apply_q_phase(self, q: torch.Tensor) -> torch.Tensor:
        """Apply per-head 2D rotation to Q. q: (B, n_heads, T, head_dim)."""
        angles = self.q_phase_angle  # (n_heads,)
        cos_a = angles.cos()  # (n_heads,)
        sin_a = angles.sin()  # (n_heads,)
        # Rotate pairs of dims: (d0,d1), (d2,d3), ...
        n_pairs = self.head_dim // 2
        if n_pairs == 0:
            return q
        q_rot = q.clone()
        for p in range(n_pairs):
            d0, d1 = 2 * p, 2 * p + 1
            c = cos_a[None, :, None]  # (1, n_heads, 1)
            s = sin_a[None, :, None]  # (1, n_heads, 1)
            q0 = q[:, :, :, d0]  # (B, n_heads, T)
            q1 = q[:, :, :, d1]
            q_rot[:, :, :, d0] = q0 * c - q1 * s
            q_rot[:, :, :, d1] = q0 * s + q1 * c
        return q_rot

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads. x: (B, num_kv_heads, T, head_dim)."""
        if self.kv_repeat == 1:
            return x
        B, H, T, D = x.shape
        return x.unsqueeze(2).expand(B, H, self.kv_repeat, T, D).reshape(B, H * self.kv_repeat, T, D)

    def forward(self, x: torch.Tensor, v_weight: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        x_tok = x[:, :, :self.tok_dim]
        x_pos = x[:, :, self.tok_dim:]
        x_qk = x_tok if self.qk_source == "tok" else x_pos

        q = self.q_proj(x_qk)
        if self.tie_qk:
            # For GQA + tied: take first kv_inner_dim outputs as K
            k = self.q_proj(x_qk)[:, :, :self.kv_inner_dim]
        else:
            k = self.k_proj(x_qk)
        if v_weight is not None:
            # tie_vo: use external weight (head_proj.weight) as v_proj
            # v_weight shape: (tok_dim, d_model), we want x_tok @ v_weight = (B,T,tok_dim) @ (tok_dim, d_model) = (B,T,d_model)
            v = x_tok @ v_weight
        else:
            v = self.v_proj(x_tok)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply phase rotation to Q (asymmetry for tied Q/K)
        if self.use_q_phase:
            q = self._apply_q_phase(q)

        # Repeat KV heads for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add ALiBi bias: slope_h * (j - i) for each head h
        if self.use_alibi:
            slopes = self.alibi_log_slopes.exp()
            alibi_bias = slopes.view(1, self.n_heads, 1, 1) * self._alibi_rel_pos[:, :, :T, :T]
            att = att + alibi_bias

        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        if self.use_softmax1:
            # Softmax1: exp(x) / (1 + sum(exp(x))), allows attention sum < 1
            att_exp = torch.exp(att - att.max(dim=-1, keepdim=True).values)  # numerical stability
            att = att_exp / (1.0 + att_exp.sum(dim=-1, keepdim=True))
        else:
            att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.out_proj(out)


class OffsetAttention(nn.Module):
    """Fixed-offset attention: replaces Q/K projections with learned positional biases.

    Exploits the discovery that attention in split-subspace addition is purely
    positional routing with fixed offsets:
      Head 0: attends to X_{i+2} and Y_{i+1} (carry lookahead)
      Head 1: attends to X_{i+1}, Y_i, and self (current context)

    Instead of learning a full q_proj (24p) + q_phase (2p) = 26p, we learn:
      - Per-head X-offset and Y-offset (what digit offset to attend to): 4p
      - Per-head sharpness (how peaked the attention is): 2p
      - Per-head self-attention weight: 2p
      - Per-head carry/special weight: 2p
    Total: ~10p instead of 26p, saving ~16p.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.n_heads * cfg.head_dim
        self.tok_dim = cfg.tok_dim
        self.pos_dim = cfg.pos_dim

        # V projection (same as SplitAttention — reads token subspace)
        self.v_proj = nn.Linear(cfg.tok_dim, self.inner_dim, bias=False)

        # Output projection
        if cfg.attn_out_rank > 0:
            self.out_proj = LowRankLinear(self.inner_dim, cfg.d_model, cfg.attn_out_rank)
        else:
            self.out_proj = nn.Linear(self.inner_dim, cfg.d_model, bias=False)

        # Offset attention parameters
        # Per-head preferred offset for X section and Y section
        # Corrected init: Head0=(+1, 0), Head1=(0, -1) based on re-analysis
        self.x_offset = nn.Parameter(torch.tensor([1.0, 0.0]))   # (n_heads,)
        self.y_offset = nn.Parameter(torch.tensor([0.0, -1.0]))  # (n_heads,)
        # Per-head sharpness (higher = more peaked attention)
        self.sharpness = nn.Parameter(torch.tensor([3.0, 3.0]))  # (n_heads,)
        # Per-head self-attention logit
        self.self_weight = nn.Parameter(torch.tensor([-2.0, 0.0]))  # (n_heads,)
        # Per-head carry/special position weight
        self.special_weight = nn.Parameter(torch.tensor([0.0, -2.0]))  # (n_heads,)

        # Build section membership buffers (which sequence positions are X, Y, Z, special)
        T = cfg.max_seq_len
        # digit_index[j] = which digit position (0-9) is at sequence position j, or -1
        digit_index = torch.full((T,), -1, dtype=torch.float32)
        # section[j] = 0 (X), 1 (Y), 2 (Z/carry), 3 (special)
        section = torch.full((T,), 3, dtype=torch.long)

        for j in range(MAX_DIGITS):
            digit_index[X_START + j] = j
            section[X_START + j] = 0
        for j in range(MAX_DIGITS):
            digit_index[Y_START + j] = j
            section[Y_START + j] = 1
        for j in range(MAX_DIGITS):
            digit_index[Z_START + j] = j
            section[Z_START + j] = 2
        digit_index[Z_START + MAX_DIGITS] = MAX_DIGITS  # carry position
        section[Z_START + MAX_DIGITS] = 2

        self.register_buffer("_digit_index", digit_index)
        self.register_buffer("_section", section)

        # Causal mask
        mask = torch.tril(torch.ones(T, T))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def _compute_attn_bias(self, T: int) -> torch.Tensor:
        """Compute attention bias matrix. Returns (1, n_heads, T, T)."""
        digit_idx = self._digit_index[:T]  # (T,)
        section = self._section[:T]         # (T,)

        # For each query position i and key position j:
        # If j is in X section: score = -sharpness * (digit_index[j] - digit_index[i] - x_offset)^2
        # If j is in Y section: score = -sharpness * (digit_index[j] - digit_index[i] - y_offset)^2
        # If j == i (self): score = self_weight
        # If j is special/carry: score = special_weight

        # Compute digit offset matrix: digit_index[j] - digit_index[i]
        # Shape: (T, T)
        offset = digit_idx.unsqueeze(0) - digit_idx.unsqueeze(1)  # (T, T) — offset[i,j] = digit[j] - digit[i]

        # Per-head attention bias: (n_heads, T, T)
        bias = torch.zeros(self.n_heads, T, T, device=digit_idx.device)

        x_mask = (section == 0).float()  # (T,) — which positions are X
        y_mask = (section == 1).float()  # (T,) — which positions are Y

        for h in range(self.n_heads):
            sharp = self.sharpness[h].abs() + 0.1  # ensure positive sharpness
            # X section contribution
            x_score = -sharp * (offset - self.x_offset[h]).pow(2)  # (T, T)
            # Y section contribution
            y_score = -sharp * (offset - self.y_offset[h]).pow(2)  # (T, T)

            # Combine: multiply by section masks
            score = x_score * x_mask.unsqueeze(0) + y_score * y_mask.unsqueeze(0)

            # Self-attention
            eye = torch.eye(T, device=digit_idx.device)
            score = score + self.self_weight[h] * eye

            # Special/carry positions
            special_mask = ((section != 0) & (section != 1)).float()
            score = score + self.special_weight[h] * special_mask.unsqueeze(0)

            bias[h] = score

        return bias.unsqueeze(0)  # (1, n_heads, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_tok = x[:, :, :self.tok_dim]

        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention from positional bias only (no Q/K projections!)
        att = self._compute_attn_bias(T)  # (1, n_heads, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Expand attention for batch
        att = att.expand(B, -1, -1, -1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.out_proj(out)


class HardOffsetAttention(nn.Module):
    """Hardcoded attention patterns — 0 learnable params for Q/K.

    Based on the structural analysis discovery:
      Head 0: A_i → 50% X_{i+2} + 50% Y_{i+1} (carry lookahead)
      Head 1: A_i → 33% X_{i+1} + 33% Y_i + 33% self (current context)

    Special cases:
      A_8 (Head 0): → z_hi carry position (97%)
      A_9 (both): → self (100%)
      Prompt positions: uniform causal attention

    The only learnable parts are V projection and output projection.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.n_heads * cfg.head_dim
        self.tok_dim = cfg.tok_dim

        # V projection (reads token subspace)
        self.v_proj = nn.Linear(cfg.tok_dim, self.inner_dim, bias=False)

        # Output projection
        if cfg.attn_out_rank > 0:
            self.out_proj = LowRankLinear(self.inner_dim, cfg.d_model, cfg.attn_out_rank)
        else:
            self.out_proj = nn.Linear(self.inner_dim, cfg.d_model, bias=False)

        # Build hardcoded attention weights
        T = cfg.max_seq_len
        attn = torch.zeros(cfg.n_heads, T, T)  # (n_heads, T, T)

        # For prompt positions (0-21): uniform causal
        for i in range(min(22, T)):
            attn[0, i, :i+1] = 1.0 / (i + 1)
            attn[1, i, :i+1] = 1.0 / (i + 1)

        # For answer positions Z_i (positions 22+i):
        # Corrected patterns based on re-analysis of structural_analysis.md:
        # (The diagnostics had an off-by-one: their "A0" was actually Z_1)
        #
        # Head 0: Z_i → 50% X_{i+1} + 50% Y_i (lookahead by +1 in X)
        # Head 1: Z_i → 33% X_i + 33% Y_{i-1} + 33% self (current context)
        #
        # Special cases:
        # Z_9 (Head 0): → PLUS position (97%) — uses PLUS as summary signal
        # Z_10 (carry digit): → self
        for i in range(min(ANSWER_LEN + 1, T - Z_START)):
            pos = Z_START + i  # sequence position
            if i <= 8:
                # Head 0: 50% X_{i+1} + 50% Y_i
                if i + 1 < MAX_DIGITS:
                    x_pos = X_START + i + 1
                    y_pos = Y_START + i
                    attn[0, pos, x_pos] = 0.5
                    attn[0, pos, y_pos] = 0.5
                else:
                    # i=9: X_10 doesn't exist, attend to PLUS as fallback
                    attn[0, pos, PLUS_POS] = 1.0

                # Head 1: 33% X_i + 33% Y_{max(i-1,0)} + 33% self
                x_pos1 = X_START + min(i, MAX_DIGITS - 1)
                y_pos1 = Y_START + max(i - 1, 0)
                attn[1, pos, x_pos1] = 1.0 / 3
                attn[1, pos, y_pos1] = 1.0 / 3
                attn[1, pos, pos] = 1.0 / 3
            elif i == 9:
                # Z_9 (Head 0): PLUS position
                attn[0, pos, PLUS_POS] = 1.0
                # Head 1: X_9 + Y_8 + self
                attn[1, pos, X_START + 9] = 1.0 / 3
                attn[1, pos, Y_START + 8] = 1.0 / 3
                attn[1, pos, pos] = 1.0 / 3
            else:
                # Z_10 (carry) and EOS: self-attention
                attn[0, pos, pos] = 1.0
                attn[1, pos, pos] = 1.0

        self.register_buffer("_hard_attn", attn.unsqueeze(0))  # (1, n_heads, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_tok = x[:, :, :self.tok_dim]

        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Use precomputed attention (no Q/K computation!)
        att = self._hard_attn[:, :, :T, :T].expand(B, -1, -1, -1)
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

class FixedRMSNorm(nn.Module):
    """RMSNorm with fixed all-ones weight (0 parameters)."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms


def _make_norm(cfg: ModelConfig) -> nn.Module:
    """Create a norm layer based on norm_mode. For 'shared'/'scalar' mode, the caller
    must handle weight sharing — this returns a normal RMSNorm/ScalarRMSNorm."""
    if cfg.norm_mode == "fixed":
        return FixedRMSNorm(cfg.d_model)
    if cfg.norm_mode == "scalar":
        return ScalarRMSNorm(cfg.d_model)
    return RMSNorm(cfg.d_model)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = _make_norm(cfg)
        if cfg.attn_mode == "offset":
            self.attn = OffsetAttention(cfg)
        elif cfg.attn_mode == "hard_offset":
            self.attn = HardOffsetAttention(cfg)
        else:
            self.attn = SplitAttention(cfg)
        self.has_ln2 = cfg.norm_mode != "no_ln2"
        if self.has_ln2:
            self.ln2 = _make_norm(cfg)
        self.ffn = FFN(cfg.d_model, cfg.ffn_dim, bias=cfg.ffn_bias)

    def forward(self, x: torch.Tensor, v_weight: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), v_weight=v_weight)
        if self.has_ln2:
            x = x + self.ffn(self.ln2(x))
        else:
            x = x + self.ffn(x)
        return x


# ── Main model ─────────────────────────────────────────────────────────────

class MicroAdder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model == cfg.tok_dim + cfg.pos_dim
        self.cfg = cfg

        # Token embedding
        if cfg.tok_emb_mode == "parametric":
            # Parametric: digits placed on a learnable arc in 2D
            # emb[d] = [A * cos(start + d * stride), B * sin(start + d * stride)]
            # 4 params for 10+ digits, instead of 20+ learned
            assert cfg.tok_dim in (1, 2), "Parametric tok_emb supports tok_dim=1 or 2"
            frozen_arc = set(cfg.freeze_tok_arc.split(",")) if cfg.freeze_tok_arc else set()
            for name, init_val in [("A", 2.5), ("B", 2.5), ("start", -1.2), ("stride", 0.29)]:
                if name in frozen_arc:
                    self.register_buffer(f"tok_arc_{name}", torch.tensor(init_val))
                else:
                    setattr(self, f"tok_arc_{name}", nn.Parameter(torch.tensor(init_val)))
            if cfg.tie_tok_arc_ab and "B" not in frozen_arc and "A" not in frozen_arc:
                # Share A and B parameters (circular arc)
                self.tok_arc_B = self.tok_arc_A
            # No nn.Embedding — embeddings computed on the fly from arc params
        elif cfg.freeze_pad:
            # Learnable embeddings for tokens 0..vocab_size-2; PAD (last) is frozen zero
            self.tok_emb = nn.Embedding(cfg.vocab_size - 1, cfg.tok_dim)
            self.register_buffer("_pad_emb", torch.zeros(1, cfg.tok_dim))
        else:
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
        self.ln_f = _make_norm(cfg)

        # Shared norm weights: point all norm weight attrs to a single parameter
        if cfg.norm_mode == "shared":
            shared_w = self.blocks[0].ln1.weight
            for block in self.blocks:
                block.ln1.weight = shared_w
                if block.has_ln2:
                    block.ln2.weight = shared_w
            self.ln_f.weight = shared_w
        elif cfg.norm_mode == "scalar":
            shared_s = self.blocks[0].ln1.scale
            for block in self.blocks:
                block.ln1.scale = shared_s
                if block.has_ln2:
                    block.ln2.scale = shared_s
            self.ln_f.scale = shared_s
        self.head_proj = nn.Linear(cfg.d_model, cfg.tok_dim, bias=False)
        # Output logits = head_proj(x) @ tok_emb.weight.T  (tied)

        # Tie V projection weight to head_proj: v_proj uses head_proj.weight.T
        # v_proj needs (tok_dim→inner_dim), head_proj is (d_model→tok_dim)
        # When inner_dim == d_model, v_proj.weight (inner_dim, tok_dim) = head_proj.weight.T (d_model, tok_dim)
        if cfg.tie_vo:
            assert cfg.n_heads * cfg.head_dim == cfg.d_model, \
                "tie_vo requires inner_dim (n_heads*head_dim) == d_model"
            for block in self.blocks:
                attn = block.attn
                # Remove v_proj params — forward will use head_proj.weight instead
                del attn.v_proj
            self._tie_vo = True
        else:
            self._tie_vo = False

        self._init_weights()

    # ── Position encoding ──────────────────────────────────────────────

    def _init_positions(self, cfg: ModelConfig) -> None:
        if cfg.pos_mode == "zero":
            # No learnable positions — pos_dim filled with zeros.
            # Used with ALiBi, which handles position via attention bias.
            pass
        elif cfg.pos_mode == "learned":
            # Full learned: 10 x pos_dim
            self.digit_pos = nn.Parameter(torch.zeros(MAX_DIGITS, cfg.pos_dim))
        elif cfg.pos_mode == "spiral_correct":
            # Parametric spiral (4 params)
            # pos_dim>=3: amp*cos, amp*sin, slope*i+offset (circle + linear ramp)
            # pos_dim==2: amp*cos(+phase), slope*sin(+offset) (ellipse with independent phases)
            frozen_spiral = set(cfg.freeze_spiral.split(",")) if cfg.freeze_spiral else set()
            slope_init = 1.0 if cfg.pos_dim == 2 else 1.0 / max(1, MAX_DIGITS - 1)
            # When freezing slope, use 0.0 (learned value converges near 0 anyway)
            frozen_overrides = {"slope": 0.0, "offset": 0.0}
            for name, init_val in [("amp", 1.0), ("phase", 0.0), ("slope", slope_init), ("offset", 0.0)]:
                if name in frozen_spiral:
                    freeze_val = frozen_overrides.get(name, init_val)
                    self.register_buffer(f"spiral_{name}", torch.tensor(freeze_val))
                else:
                    setattr(self, f"spiral_{name}", nn.Parameter(torch.tensor(init_val)))
            # Per-position scale correction
            if cfg.pos_correction_mode == "full":
                self.pos_correction = nn.Parameter(torch.zeros(MAX_DIGITS))  # 10 params
            elif cfg.pos_correction_mode == "linear":
                self.pos_corr_slope = nn.Parameter(torch.tensor(0.0))        # 2 params
                self.pos_corr_intercept = nn.Parameter(torch.tensor(0.0))
            elif cfg.pos_correction_mode == "none":
                pass  # No correction parameters — spiral positions used as-is
            else:
                raise ValueError(f"Unknown pos_correction_mode: {cfg.pos_correction_mode}")
        else:
            raise ValueError(f"Unknown pos_mode: {cfg.pos_mode}")

        if cfg.pos_mode == "zero":
            # All positions are zero buffers — no learnable params
            self.register_buffer("_zero_z_hi", torch.zeros(1, cfg.pos_dim))
            self.register_buffer("_zero_special", torch.zeros(3, cfg.pos_dim))
        else:
            # Carry position
            if cfg.freeze_z_hi:
                self.register_buffer("z_hi_pos", torch.zeros(1, cfg.pos_dim))
            else:
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
            elif cfg.freeze_special == "all":
                # All special positions fixed to zero (saves all special_pos params)
                self.register_buffer("_frozen_special", torch.zeros(3, cfg.pos_dim))
            elif cfg.freeze_special == "plus_eos_equals":
                # All frozen as buffers (keeps key names for warm-start compatibility)
                self.register_buffer("special_pos_equals", torch.zeros(1, cfg.pos_dim))
                self.register_buffer("_plus_pos", torch.zeros(1, cfg.pos_dim))
                self.register_buffer("_eos_pos", torch.zeros(1, cfg.pos_dim))
            else:
                raise ValueError(f"Unknown freeze_special: {cfg.freeze_special}")

    def _get_special_pos(self) -> torch.Tensor:
        """Build the (3, pos_dim) special position table."""
        cfg = self.cfg
        if cfg.pos_mode == "zero":
            return self._zero_special
        if cfg.freeze_special == "none":
            return self.special_pos
        elif cfg.freeze_special == "eos":
            return torch.cat([self.special_pos_learned, self._eos_pos], dim=0)
        elif cfg.freeze_special in ("plus_eos", "plus_eos_equals"):
            return torch.cat([self._plus_pos, self.special_pos_equals, self._eos_pos], dim=0)
        else:  # all
            return self._frozen_special

    def _get_digit_positions(self) -> torch.Tensor:
        """Compute the (MAX_DIGITS, pos_dim) position table."""
        cfg = self.cfg
        if cfg.pos_mode == "zero":
            return torch.zeros(MAX_DIGITS, cfg.pos_dim, device=self._zero_z_hi.device)
        if cfg.pos_mode == "learned":
            return self.digit_pos
        # spiral_correct: base spiral * (1 + correction)
        idx = torch.arange(MAX_DIGITS, device=self.spiral_amp.device, dtype=self.spiral_amp.dtype)
        angle = 2.0 * math.pi * idx / float(MAX_DIGITS) + self.spiral_phase
        base = torch.zeros(MAX_DIGITS, cfg.pos_dim, device=idx.device, dtype=idx.dtype)
        if cfg.pos_dim > 0:
            base[:, 0] = self.spiral_amp * torch.cos(angle)
        if cfg.pos_dim > 1:
            if cfg.pos_dim == 2:
                # Ellipse mode: independent amp/phase per axis (all 4 spiral params used)
                angle2 = 2.0 * math.pi * idx / float(MAX_DIGITS) + self.spiral_offset
                base[:, 1] = self.spiral_slope * torch.sin(angle2)
            else:
                base[:, 1] = self.spiral_amp * torch.sin(angle)
        if cfg.pos_dim > 2:
            base[:, 2] = self.spiral_slope * idx + self.spiral_offset
        # Apply correction scaling
        if cfg.pos_correction_mode == "full":
            correction = self.pos_correction
        elif cfg.pos_correction_mode == "linear":
            correction = self.pos_corr_intercept + self.pos_corr_slope * idx
        else:  # none
            return base
        scale = (1.0 + correction).unsqueeze(1)  # (10, 1)
        return base * scale

    def _get_positions(self, T: int) -> torch.Tensor:
        """Build the full (T, pos_dim) position tensor for sequence length T."""
        cfg = self.cfg
        if cfg.pos_mode == "zero":
            device = self._zero_z_hi.device
            dtype = self._zero_z_hi.dtype
        else:
            device = self.z_hi_pos.device
            dtype = self.z_hi_pos.dtype

        digit_pos = self._get_digit_positions()  # (10, pos_dim)
        special_pos = self._get_special_pos()    # (3, pos_dim)
        z_hi = self._zero_z_hi if cfg.pos_mode == "zero" else self.z_hi_pos
        tables = [digit_pos, z_hi, special_pos]  # source 0, 1, 2

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
        # Spiral token init (skip for parametric mode — arc params handle init)
        if cfg.tok_emb_mode != "parametric" and cfg.token_init == "spiral":
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
                n_emb = self.tok_emb.weight.shape[0]
                for tid, vals in specials.items():
                    if tid < n_emb:
                        for j in range(min(cfg.tok_dim, 3)):
                            self.tok_emb.weight[tid, j] = vals[j]

        # Learned position init (skip for zero mode — no learnable positions)
        if cfg.pos_mode != "zero":
            with torch.no_grad():
                if cfg.pos_mode == "learned":
                    nn.init.normal_(self.digit_pos, std=0.02)
                if not cfg.freeze_z_hi:
                    nn.init.normal_(self.z_hi_pos, std=0.02)
                if cfg.freeze_special == "none":
                    nn.init.normal_(self.special_pos, std=0.02)
                elif cfg.freeze_special == "eos":
                    nn.init.normal_(self.special_pos_learned, std=0.02)
                elif cfg.freeze_special == "plus_eos":
                    nn.init.normal_(self.special_pos_equals, std=0.02)
                # "all" → no learnable special positions to initialize

        # Xavier for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Forward / generate ─────────────────────────────────────────────

    def _compute_parametric_emb(self) -> torch.Tensor:
        """Compute parametric token embeddings from arc parameters.

        emb[d] = [A * cos(start + d * stride), B * sin(start + d * stride)]
        Returns (vocab_size, tok_dim).
        """
        d = torch.arange(self.cfg.vocab_size, device=self.tok_arc_A.device,
                         dtype=self.tok_arc_A.dtype)
        angles = self.tok_arc_start + d * self.tok_arc_stride
        if self.cfg.tok_dim == 1:
            # 1D: digits on a cosine curve
            return (self.tok_arc_A * torch.cos(angles)).unsqueeze(1)
        emb = torch.stack([
            self.tok_arc_A * torch.cos(angles),
            self.tok_arc_B * torch.sin(angles),
        ], dim=1)
        return emb

    def _full_tok_weight(self) -> torch.Tensor:
        """Return (vocab_size, tok_dim) embedding table, with frozen PAD row if needed."""
        if self.cfg.tok_emb_mode == "parametric":
            return self._compute_parametric_emb()
        if self.cfg.freeze_pad:
            return torch.cat([self.tok_emb.weight, self._pad_emb], dim=0)
        return self.tok_emb.weight

    def _embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings, routing PAD through the frozen buffer."""
        if self.cfg.tok_emb_mode == "parametric":
            # Compute embeddings on the fly and index into them
            emb_table = self._compute_parametric_emb()  # (vocab_size, tok_dim)
            return emb_table[idx]
        if self.cfg.freeze_pad:
            # Clamp PAD id to valid range for nn.Embedding, then zero it out
            clamped = idx.clamp(max=self.cfg.vocab_size - 2)
            emb = self.tok_emb(clamped)
            pad_mask = (idx == self.cfg.vocab_size - 1).unsqueeze(-1)
            return emb.masked_fill(pad_mask, 0.0)
        return self.tok_emb(idx)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        tok = self._embed_tokens(idx)                               # (B, T, tok_dim)
        pos = self._get_positions(T).unsqueeze(0).expand(B, -1, -1) # (B, T, pos_dim)
        x = torch.cat([tok, pos], dim=-1)                           # (B, T, d_model)

        v_weight = self.head_proj.weight if self._tie_vo else None
        if self.cfg.share_layers:
            for _ in range(self._n_passes):
                x = self.blocks[0](x, v_weight=v_weight)
        else:
            for block in self.blocks:
                x = block(x, v_weight=v_weight)

        x = self.ln_f(x)
        logits = self.head_proj(x) @ self._full_tok_weight().T      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ── Scaffold weights ────────────────────────────────────────────────

    def get_scaffold_l1(self, target_cfg: 'ModelConfig') -> torch.Tensor:
        """Compute L1 penalty on scaffold (extra) dimensions.

        Penalizes:
          - out_proj: extra rank columns/rows beyond target_cfg.attn_out_rank
          - FFN: extra hidden dims beyond target_cfg.ffn_dim
          - norms: difference between ln2/ln_f weights and ln1 weight (drives toward shared)
        """
        device = next(self.parameters()).device
        l1 = torch.tensor(0.0, device=device)

        for block in self.blocks:
            # out_proj scaffold: extra rank indices
            if (target_cfg.attn_out_rank > 0
                    and self.cfg.attn_out_rank > target_cfg.attn_out_rank
                    and isinstance(block.attn.out_proj, LowRankLinear)):
                target_rank = target_cfg.attn_out_rank
                l1 = l1 + block.attn.out_proj.A[:, target_rank:].abs().sum()
                l1 = l1 + block.attn.out_proj.B[target_rank:, :].abs().sum()

            # FFN scaffold: extra hidden dims
            if self.cfg.ffn_dim > target_cfg.ffn_dim:
                target_dim = target_cfg.ffn_dim
                l1 = l1 + block.ffn.fc1.weight[target_dim:, :].abs().sum()
                l1 = l1 + block.ffn.fc2.weight[:, target_dim:].abs().sum()
                if block.ffn.fc1.bias is not None:
                    l1 = l1 + block.ffn.fc1.bias[target_dim:].abs().sum()
                if block.ffn.fc2.bias is not None:
                    l1 = l1 + block.ffn.fc2.bias.abs().sum()  # full bias since output dim unchanged

            # Norm scaffold: difference-based (drives ln2, ln_f toward ln1)
            if (self.cfg.norm_mode != "shared"
                    and target_cfg.norm_mode == "shared"
                    and hasattr(block, 'ln2') and block.has_ln2
                    and hasattr(block.ln1, 'weight')):
                l1 = l1 + (block.ln2.weight - block.ln1.weight).abs().sum()

        # ln_f vs ln1 of first block
        if (self.cfg.norm_mode != "shared"
                and target_cfg.norm_mode == "shared"
                and hasattr(self.ln_f, 'weight')
                and hasattr(self.blocks[0].ln1, 'weight')):
            l1 = l1 + (self.ln_f.weight - self.blocks[0].ln1.weight).abs().sum()

        return l1

    def prune_scaffold(self, target_cfg: 'ModelConfig') -> None:
        """Hard-prune scaffold dimensions, restructuring layers in-place.

        After pruning, the model has exactly the target architecture.
        """
        for block in self.blocks:
            # Prune out_proj rank
            if (target_cfg.attn_out_rank > 0
                    and self.cfg.attn_out_rank > target_cfg.attn_out_rank
                    and isinstance(block.attn.out_proj, LowRankLinear)):
                old = block.attn.out_proj
                target_rank = target_cfg.attn_out_rank
                new_proj = LowRankLinear(
                    old.A.shape[0], old.B.shape[1], target_rank
                ).to(old.A.device)
                with torch.no_grad():
                    new_proj.A.copy_(old.A[:, :target_rank])
                    new_proj.B.copy_(old.B[:target_rank, :])
                block.attn.out_proj = new_proj

            # Prune FFN hidden dim
            if self.cfg.ffn_dim > target_cfg.ffn_dim:
                old_fc1 = block.ffn.fc1
                old_fc2 = block.ffn.fc2
                target_dim = target_cfg.ffn_dim
                has_bias = old_fc1.bias is not None

                new_fc1 = nn.Linear(old_fc1.in_features, target_dim, bias=has_bias).to(old_fc1.weight.device)
                new_fc2 = nn.Linear(target_dim, old_fc2.out_features, bias=has_bias).to(old_fc2.weight.device)
                with torch.no_grad():
                    new_fc1.weight.copy_(old_fc1.weight[:target_dim, :])
                    new_fc2.weight.copy_(old_fc2.weight[:, :target_dim])
                    if has_bias:
                        new_fc1.bias.copy_(old_fc1.bias[:target_dim])
                        new_fc2.bias.copy_(old_fc2.bias)
                block.ffn.fc1 = new_fc1
                block.ffn.fc2 = new_fc2

            # Prune norms to shared mode
            if (self.cfg.norm_mode != "shared"
                    and target_cfg.norm_mode == "shared"
                    and hasattr(block, 'ln2') and block.has_ln2
                    and hasattr(block.ln1, 'weight')):
                block.ln2.weight = block.ln1.weight

        # Share ln_f weight with ln1
        if (self.cfg.norm_mode != "shared"
                and target_cfg.norm_mode == "shared"
                and hasattr(self.ln_f, 'weight')
                and hasattr(self.blocks[0].ln1, 'weight')):
            self.ln_f.weight = self.blocks[0].ln1.weight

        # Update config
        self.cfg = copy.deepcopy(target_cfg)

    def freeze_params(self, param_names: list) -> int:
        """Convert named parameters to frozen buffers in-place, preserving their values.

        Returns the number of params frozen (for logging).
        """
        frozen = 0
        for name in param_names:
            parts = name.split(".")
            # Navigate to parent module
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr = parts[-1]
            param = getattr(obj, attr)
            if isinstance(param, nn.Parameter):
                data = param.data.clone()
                delattr(obj, attr)
                obj.register_buffer(attr, data)
                frozen += data.numel()
        return frozen

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
