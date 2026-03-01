"""
141-parameter trained transformer for 10-digit addition.

Based on JackCai's 242p split-subspace architecture with:
  - d_model=5 (tok_dim=2, pos_dim=3) — saves 29p over d_model=6
  - 1 head, head_dim=5 (full d_model) — decoupled inner_dim
  - Spiral+correction positional encoding with linear correction (2 params)
  - Rank-2 attention output projection
  - Frozen EOS special position
  - Tied Q/K with per-head phase rotation
  - Grokked from resumed checkpoint using aggressive weight decay scheduling

Architecture: 1-layer decoder, d_model=5 (2 tok + 3 pos), 1 head, hd=5, ff=2
Training: AdamW (lr=0.02, wd=0.01 adaptive with ultra-low thresholds),
          cosine decay, carry-focused curriculum.
          Originally trained ~200K steps, then resumed with aggressive WD drop
          (Stage 2 at val_exact=0.05). Grokked at ~243K steps from resume.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Constants --

VOCAB_SIZE = 14
D_MODEL = 5
TOK_DIM = 2
POS_DIM = 3
N_HEADS = 1
HEAD_DIM = 5
FFN_DIM = 2
MAX_SEQ_LEN = 34
MAX_DIGITS = 10
ANSWER_LEN = 11

PLUS_TOKEN = 10
EQUALS_TOKEN = 11
EOS_TOKEN = 12
PAD_TOKEN = 13


# -- Position index map (shared XYZ: X[i]=Y[i]=Z[i]) --


def _build_pos_map():
    sources, indices = [], []
    for i in range(10):  # X_0..X_9
        sources.append(0)
        indices.append(i)
    sources.append(2)
    indices.append(0)  # PLUS
    for i in range(10):  # Y_0..Y_9
        sources.append(0)
        indices.append(i)
    sources.append(2)
    indices.append(1)  # EQUALS
    for i in range(10):  # Z_0..Z_9
        sources.append(0)
        indices.append(i)
    sources.append(1)
    indices.append(0)  # Z_10 (carry)
    sources.append(2)
    indices.append(2)  # EOS
    return torch.tensor(sources), torch.tensor(indices)


# -- Modules --


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LowRankLinear(nn.Module):
    """y = x @ A @ B (rank-2 factorization)."""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))

    def forward(self, x):
        return x @ self.A @ self.B


class MicroAdder(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embedding (spiral-initialized, tied with output head)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)

        # Spiral+correction positional encoding (pos_dim=3)
        self.spiral_amp = nn.Parameter(torch.tensor(1.0))
        self.spiral_phase = nn.Parameter(torch.tensor(0.0))
        # For pos_dim==2, these become ellipse params; for pos_dim>=3, linear ramp
        self.spiral_slope = nn.Parameter(torch.tensor(1.0 / 9.0))
        self.spiral_offset = nn.Parameter(torch.tensor(0.0))
        # Linear correction (2 params instead of 10)
        self.pos_corr_slope = nn.Parameter(torch.tensor(0.0))
        self.pos_corr_intercept = nn.Parameter(torch.tensor(0.0))

        # Carry position + special token positions (PLUS, EQUALS learned; EOS fixed to 0)
        self.z_hi_pos = nn.Parameter(torch.zeros(1, POS_DIM))
        self.special_pos_learned = nn.Parameter(torch.zeros(2, POS_DIM))  # PLUS, EQUALS
        self.register_buffer("_eos_pos", torch.zeros(1, POS_DIM))

        # Position index buffers
        sources, indices = _build_pos_map()
        self.register_buffer("_pos_sources", sources)
        self.register_buffer("_pos_indices", indices)

        # Split attention: Q,K from pos dims; V from tok dims
        # Tied Q/K: single q_proj used for both Q and K
        # 1 head with head_dim=5 (= d_model), inner_dim=5
        self.q_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.out_proj = LowRankLinear(N_HEADS * HEAD_DIM, D_MODEL, rank=2)

        # Q-phase: per-head rotation angle that breaks Q/K symmetry
        self.q_phase_angle = nn.Parameter(torch.zeros(N_HEADS))

        # Norms
        self.ln1 = RMSNorm(D_MODEL)
        self.ln2 = RMSNorm(D_MODEL)
        self.ln_f = RMSNorm(D_MODEL)

        # FFN
        self.ffn_fc1 = nn.Linear(D_MODEL, FFN_DIM)
        self.ffn_fc2 = nn.Linear(FFN_DIM, D_MODEL)

        # Tied output head
        self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0),
        )

    def _get_digit_positions(self):
        idx = torch.arange(
            MAX_DIGITS, device=self.spiral_amp.device, dtype=self.spiral_amp.dtype
        )
        angle = 2.0 * math.pi * idx / float(MAX_DIGITS) + self.spiral_phase
        base = torch.zeros(MAX_DIGITS, POS_DIM, device=idx.device)
        base[:, 0] = self.spiral_amp * torch.cos(angle)
        base[:, 1] = self.spiral_amp * torch.sin(angle)
        base[:, 2] = self.spiral_slope * idx + self.spiral_offset
        correction = self.pos_corr_intercept + self.pos_corr_slope * idx
        return base * (1.0 + correction).unsqueeze(1)

    def _get_positions(self, T):
        digit_pos = self._get_digit_positions()
        special_pos = torch.cat([self.special_pos_learned, self._eos_pos], dim=0)
        tables = [digit_pos, self.z_hi_pos, special_pos]
        src = self._pos_sources[:T]
        idx = self._pos_indices[:T]
        out = torch.zeros(T, POS_DIM, device=digit_pos.device)
        for sid, table in enumerate(tables):
            mask = src == sid
            if mask.any():
                out[mask] = table[idx[mask]]
        return out

    def _apply_q_phase(self, q):
        """Apply per-head 2D rotation to Q. q: (B, n_heads, T, head_dim)."""
        cos_a = self.q_phase_angle.cos()
        sin_a = self.q_phase_angle.sin()
        # Rotate pairs (d0,d1), (d2,d3); d4 untouched (head_dim=5, 2 pairs)
        q_rot = q.clone()
        c = cos_a[None, :, None]  # (1, n_heads, 1)
        s = sin_a[None, :, None]
        for p in range(HEAD_DIM // 2):
            d0, d1 = 2 * p, 2 * p + 1
            q0 = q[:, :, :, d0]
            q1 = q[:, :, :, d1]
            q_rot[:, :, :, d0] = q0 * c - q1 * s
            q_rot[:, :, :, d1] = q0 * s + q1 * c
        return q_rot

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self._get_positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        # Attention
        h = self.ln1(x)
        q = self.q_proj(h[:, :, TOK_DIM:]).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.q_proj(h[:, :, TOK_DIM:]).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h[:, :, :TOK_DIM]).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        # Apply phase rotation to Q only (breaks Q=K symmetry)
        q = self._apply_q_phase(q)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, N_HEADS * HEAD_DIM)
        x = x + self.out_proj(out)

        # FFN
        x = x + self.ffn_fc2(F.gelu(self.ffn_fc1(self.ln2(x))))

        # Output
        x = self.ln_f(x)
        return self.head_proj(x) @ self.tok_emb.weight.T

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, T_prompt = prompt.shape
        full_seq = torch.zeros(
            B, T_prompt + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device
        )
        full_seq[:, :T_prompt] = prompt
        for step in range(ANSWER_LEN + 1):
            T = T_prompt + step
            logits = self.forward(full_seq[:, :T])
            full_seq[:, T] = logits[:, -1].argmax(dim=-1)
        return full_seq[:, T_prompt:]


# -- Submission interface --


def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroAdder()

    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoint_141p.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Remap keys from training checkpoint format to submission model format
    KEY_MAP = {
        "blocks.0.attn.q_proj.weight": "q_proj.weight",
        "blocks.0.attn.v_proj.weight": "v_proj.weight",
        "blocks.0.attn.out_proj.A": "out_proj.A",
        "blocks.0.attn.out_proj.B": "out_proj.B",
        "blocks.0.attn.q_phase_angle": "q_phase_angle",
        "blocks.0.ln1.weight": "ln1.weight",
        "blocks.0.ln2.weight": "ln2.weight",
        "blocks.0.ffn.fc1.weight": "ffn_fc1.weight",
        "blocks.0.ffn.fc1.bias": "ffn_fc1.bias",
        "blocks.0.ffn.fc2.weight": "ffn_fc2.weight",
        "blocks.0.ffn.fc2.bias": "ffn_fc2.bias",
    }

    raw = ckpt["model_state_dict"]
    state = {}
    skip_keys = {"blocks.0.attn.causal_mask", "_pos_sources", "_pos_indices"}
    for k, v in raw.items():
        if k in skip_keys:
            continue
        state[KEY_MAP.get(k, k)] = v

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    seen = set()
    n_params = sum(
        p.numel()
        for p in model.parameters()
        if id(p) not in seen and not seen.add(id(p))
    )

    metadata = {
        "name": "MicroAdder 141p",
        "author": "Arseniy Zarechnev",
        "params": n_params,
        "architecture": "1L decoder, d=5 (2 tok + 3 pos), 1h, hd=5, ff=2, rank-2 out_proj, tied Q/K + q-phase, RMSNorm, tied output",
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device

    x_digits = [(a // 10**i) % 10 for i in range(MAX_DIGITS)]
    y_digits = [(b // 10**i) % 10 for i in range(MAX_DIGITS)]
    prompt = x_digits + [PLUS_TOKEN] + y_digits + [EQUALS_TOKEN]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(prompt_tensor)

    result = 0
    for i, tok in enumerate(generated[0].tolist()):
        if tok >= 10:
            break
        result += tok * (10**i)
    return result
