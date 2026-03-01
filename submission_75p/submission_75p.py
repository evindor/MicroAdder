"""
75-parameter trained transformer for 10-digit addition.

Architecture: 1-layer decoder, d_model=5 (2 tok + 3 pos), 1 head, hd=5, ff=2
Key innovations:
  - d_model=5 with 1 head (head_dim=5 = d_model, full-rank attention)
  - vocab=10: all special tokens map to digit-0, distinguished by position
  - Parametric token embeddings: 4 arc params instead of 20 learned params
  - Rank-1 attention output projection: A(5x1) @ B(1x5) = 10p instead of 20p
  - No FFN bias: saves 7p (fc1_bias=2, fc2_bias=5)
  - Spiral positions (no correction), frozen PLUS/EOS, tied Q/K + q-phase
  - Shared norms: single RMSNorm weight shared across ln1/ln2/ln_f (5p not 15p)
  - Tied V/output: head_proj doubles as v_proj (no separate v_proj)

Training: AdamW (lr=0.02, wd=0.01 adaptive), freeze_special=plus_eos,
          from-scratch seed 80085, grokked at ~36K steps.
          10010/10010 on AdderBoard evaluation.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Constants --

VOCAB_SIZE = 10  # digits 0-9 only; special tokens map to digit 0
D_MODEL = 5
TOK_DIM = 2
POS_DIM = 3
N_HEADS = 1
HEAD_DIM = 5
FFN_DIM = 2
MAX_SEQ_LEN = 34
MAX_DIGITS = 10
ANSWER_LEN = 11

PLUS_TOKEN = 0
EQUALS_TOKEN = 0


# -- Position index map --


def _build_pos_map():
    sources, indices = [], []
    for i in range(10):
        sources.append(0); indices.append(i)
    sources.append(2); indices.append(0)  # PLUS
    for i in range(10):
        sources.append(0); indices.append(i)
    sources.append(2); indices.append(1)  # EQUALS
    for i in range(10):
        sources.append(0); indices.append(i)
    sources.append(1); indices.append(0)  # Z_10
    sources.append(2); indices.append(2)  # EOS
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
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))

    def forward(self, x):
        return x @ self.A @ self.B


class MicroAdder(nn.Module):
    def __init__(self):
        super().__init__()

        # Parametric token embedding: 4 arc params for all 10 digits
        self.tok_arc_A = nn.Parameter(torch.tensor(2.5))
        self.tok_arc_B = nn.Parameter(torch.tensor(2.5))
        self.tok_arc_start = nn.Parameter(torch.tensor(-1.2))
        self.tok_arc_stride = nn.Parameter(torch.tensor(0.29))

        # Spiral positional encoding (no correction)
        self.spiral_amp = nn.Parameter(torch.tensor(1.0))
        self.spiral_phase = nn.Parameter(torch.tensor(0.0))
        self.spiral_slope = nn.Parameter(torch.tensor(1.0 / 9.0))
        self.spiral_offset = nn.Parameter(torch.tensor(0.0))

        # Carry position (learnable)
        self.z_hi_pos = nn.Parameter(torch.zeros(1, POS_DIM))

        # Special positions: PLUS and EOS are frozen zero buffers, EQUALS is learnable
        self.register_buffer("_plus_pos", torch.zeros(1, POS_DIM))
        self.special_pos_equals = nn.Parameter(torch.zeros(1, POS_DIM))
        self.register_buffer("_eos_pos", torch.zeros(1, POS_DIM))

        sources, indices = _build_pos_map()
        self.register_buffer("_pos_sources", sources)
        self.register_buffer("_pos_indices", indices)

        # Attention (rank-1 output, tied Q/K, no separate v_proj -- tie_vo)
        self.q_proj = nn.Linear(POS_DIM, HEAD_DIM, bias=False)
        self.out_proj = LowRankLinear(HEAD_DIM, D_MODEL, rank=1)
        self.q_phase_angle = nn.Parameter(torch.zeros(1))

        # Shared norm: single RMSNorm weight used for ln1, ln2, ln_f
        self.ln1 = RMSNorm(D_MODEL)

        # FFN (no bias!)
        self.ffn_fc1 = nn.Linear(D_MODEL, FFN_DIM, bias=False)
        self.ffn_fc2 = nn.Linear(FFN_DIM, D_MODEL, bias=False)

        # Tied output head (also serves as v_proj via tie_vo)
        self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0),
        )

    def _tok_emb_table(self):
        d = torch.arange(VOCAB_SIZE, device=self.tok_arc_A.device, dtype=self.tok_arc_A.dtype)
        angles = self.tok_arc_start + d * self.tok_arc_stride
        return torch.stack([self.tok_arc_A * torch.cos(angles),
                           self.tok_arc_B * torch.sin(angles)], dim=1)

    def _get_digit_positions(self):
        idx = torch.arange(MAX_DIGITS, device=self.spiral_amp.device, dtype=self.spiral_amp.dtype)
        angle = 2.0 * math.pi * idx / float(MAX_DIGITS) + self.spiral_phase
        base = torch.zeros(MAX_DIGITS, POS_DIM, device=idx.device)
        base[:, 0] = self.spiral_amp * torch.cos(angle)
        base[:, 1] = self.spiral_amp * torch.sin(angle)
        base[:, 2] = self.spiral_slope * idx + self.spiral_offset
        # No position correction -- just return base spiral
        return base

    def _get_positions(self, T):
        digit_pos = self._get_digit_positions()
        special_pos = torch.cat([self._plus_pos, self.special_pos_equals, self._eos_pos], dim=0)
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
        cos_a = self.q_phase_angle.cos()
        sin_a = self.q_phase_angle.sin()
        q_rot = q.clone()
        c = cos_a[None, :, None]
        s = sin_a[None, :, None]
        for p in range(HEAD_DIM // 2):
            d0, d1 = 2 * p, 2 * p + 1
            q_rot[:, :, :, d0] = q[:, :, :, d0] * c - q[:, :, :, d1] * s
            q_rot[:, :, :, d1] = q[:, :, :, d0] * s + q[:, :, :, d1] * c
        return q_rot

    def forward(self, idx):
        B, T = idx.shape
        tok_table = self._tok_emb_table()
        tok = tok_table[idx]
        pos = self._get_positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        # Attention (tie_vo: V uses head_proj.weight transposed)
        h = self.ln1(x)
        q = self.q_proj(h[:, :, TOK_DIM:]).view(B, T, 1, HEAD_DIM).transpose(1, 2)
        k = self.q_proj(h[:, :, TOK_DIM:]).view(B, T, 1, HEAD_DIM).transpose(1, 2)
        # V = h_tok @ head_proj.weight (tie_vo: v_proj.weight = head_proj.weight.T)
        # F.linear(h_tok, head_proj.weight.T) = h_tok @ head_proj.weight
        v = F.linear(h[:, :, :TOK_DIM], self.head_proj.weight.T).view(B, T, 1, HEAD_DIM).transpose(1, 2)
        q = self._apply_q_phase(q)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, HEAD_DIM)
        x = x + self.out_proj(out)

        # FFN (shared norm: reuse ln1)
        x = x + self.ffn_fc2(F.gelu(self.ffn_fc1(self.ln1(x))))

        # Output (shared norm: reuse ln1)
        x = self.ln1(x)
        return self.head_proj(x) @ tok_table.T

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, T_prompt = prompt.shape
        full_seq = torch.zeros(B, T_prompt + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
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

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_75p.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    KEY_MAP = {
        "blocks.0.attn.q_proj.weight": "q_proj.weight",
        "blocks.0.attn.out_proj.A": "out_proj.A",
        "blocks.0.attn.out_proj.B": "out_proj.B",
        "blocks.0.attn.q_phase_angle": "q_phase_angle",
        "blocks.0.ln1.weight": "ln1.weight",
        "blocks.0.ffn.fc1.weight": "ffn_fc1.weight",
        "blocks.0.ffn.fc2.weight": "ffn_fc2.weight",
    }

    # Keys to skip entirely (buffers regenerated by model, or shared norm duplicates)
    SKIP_KEYS = {
        "blocks.0.attn.causal_mask",
        "_pos_sources",
        "_pos_indices",
        "blocks.0.ln2.weight",  # shared with ln1
        "ln_f.weight",          # shared with ln1
    }

    raw = ckpt["model_state_dict"]
    state = {}
    for k, v in raw.items():
        if k in SKIP_KEYS:
            continue
        state[KEY_MAP.get(k, k)] = v

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    seen = set()
    n_params = sum(p.numel() for p in model.parameters() if id(p) not in seen and not seen.add(id(p)))

    metadata = {
        "name": "MicroAdder 75p",
        "author": "Arseniy Zarechnev",
        "params": n_params,
        "architecture": "1L decoder, d=5, 1h, hd=5, ff=2, rank-1 out, no FFN bias, parametric tok_emb, vocab=10, shared norms, tie_vo, no pos correction",
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
        if i >= ANSWER_LEN:
            break
        result += tok * (10**i)
    return result
