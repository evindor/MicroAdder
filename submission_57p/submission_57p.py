"""
57-parameter trained transformer for 10-digit addition.
100% accuracy (10010/10010) on AdderBoard. Trained from scratch, seed 777.

Architecture: 1-layer decoder, d_model=5 (tok_dim=2 + pos_dim=3), 1 head,
qk_dim=4, head_dim=5, ffn_dim=2, rank-1 attention output, no FFN bias,
shared RMSNorm, tied V/output, tied Q/K with phase rotation,
parametric circular token embeddings, frozen sinusoidal positions,
tied fc2=head_proj (triple-duty head_proj).

Parameter budget (57):
  tok_arc (A, start, stride)    3   circular embedding (A=B tied)
  z_hi_pos                      3   carry position
  special_pos_equals            3   EQUALS position (PLUS/EOS frozen zero)
  q_phase_angle                 1   Q/K asymmetry rotation
  q_proj                       12   3->4 (shared Q/K, qk_dim=4)
  out_proj                     10   rank-1 (5x1 + 1x5)
  FFN fc1                      10   5->2 (no bias)
  head_proj                    10   5->2 (triple-duty: V proj, output head, fc2)
  RMSNorm (shared)              5   one weight for all 3 norms
  TOTAL                        57

Free (sinusoidal, frozen at init):
  spiral (amp, phase, slope, off)  4   fixed: amp=3.5, phase=0, slope=0.15, offset=0
  PLUS/EOS positions               6   frozen at zero
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Constants --

VOCAB_SIZE = 10
D_MODEL = 5
TOK_DIM = 2
POS_DIM = 3
HEAD_DIM = 5
QK_DIM = 4
FFN_DIM = 2
MAX_DIGITS = 10
ANSWER_LEN = 11
SEQ_LEN = 34

# Frozen sinusoidal position parameters
SPIRAL_AMP = 3.5
SPIRAL_PHASE = 0.0
SPIRAL_SLOPE = 0.15
SPIRAL_OFFSET = 0.0


# -- Position map --

def _build_pos_map():
    sources, indices = [], []
    for i in range(10):           # X digits
        sources.append(0); indices.append(i)
    sources.append(2); indices.append(0)  # PLUS
    for i in range(10):           # Y digits
        sources.append(0); indices.append(i)
    sources.append(2); indices.append(1)  # EQUALS
    for i in range(10):           # Z digits (answer)
        sources.append(0); indices.append(i)
    sources.append(1); indices.append(0)  # Z_10 (carry-out)
    sources.append(2); indices.append(2)  # EOS
    return torch.tensor(sources), torch.tensor(indices)


# -- Model --

class MicroAdder(nn.Module):
    def __init__(self):
        super().__init__()

        # Parametric token embedding: circular arc (A=B tied, 3 params)
        self.tok_arc_A = nn.Parameter(torch.tensor(2.5))
        self.tok_arc_start = nn.Parameter(torch.tensor(-1.2))
        self.tok_arc_stride = nn.Parameter(torch.tensor(0.29))

        # Frozen sinusoidal positional encoding (0 params)
        self.register_buffer("spiral_amp", torch.tensor(SPIRAL_AMP))
        self.register_buffer("spiral_phase", torch.tensor(SPIRAL_PHASE))
        self.register_buffer("spiral_slope", torch.tensor(SPIRAL_SLOPE))
        self.register_buffer("spiral_offset", torch.tensor(SPIRAL_OFFSET))

        # Carry position (3 params)
        self.z_hi_pos = nn.Parameter(torch.zeros(1, POS_DIM))

        # Special positions: PLUS/EOS frozen at zero, EQUALS learned (3 params)
        self.register_buffer("_plus_pos", torch.zeros(1, POS_DIM))
        self.special_pos_equals = nn.Parameter(torch.zeros(1, POS_DIM))
        self.register_buffer("_eos_pos", torch.zeros(1, POS_DIM))

        # Position lookup buffers
        sources, indices = _build_pos_map()
        self.register_buffer("_pos_sources", sources)
        self.register_buffer("_pos_indices", indices)

        # Q/K projection (shared, 12 params: 3->4) + phase rotation (1 param)
        self.q_proj = nn.Linear(POS_DIM, QK_DIM, bias=False)
        self.q_phase_angle = nn.Parameter(torch.zeros(1))

        # Rank-1 attention output (10 params)
        self.out_proj_A = nn.Parameter(torch.empty(HEAD_DIM, 1))
        self.out_proj_B = nn.Parameter(torch.empty(1, D_MODEL))
        nn.init.kaiming_uniform_(self.out_proj_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj_B, a=math.sqrt(5))

        # Shared RMSNorm (5 params, used for ln1/ln2/ln_f)
        self.norm_weight = nn.Parameter(torch.ones(D_MODEL))

        # FFN fc1 (10 params, no bias) -- fc2 uses head_proj.weight (tied)
        self.fc1 = nn.Linear(D_MODEL, FFN_DIM, bias=False)

        # Output head (10 params, triple-duty: V proj, output head, fc2 expansion)
        self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)

        # Causal mask
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).unsqueeze(0).unsqueeze(0))

    def _tok_emb_table(self):
        d = torch.arange(VOCAB_SIZE, device=self.tok_arc_A.device, dtype=self.tok_arc_A.dtype)
        angles = self.tok_arc_start + d * self.tok_arc_stride
        A = self.tok_arc_A
        return torch.stack([A * torch.cos(angles), A * torch.sin(angles)], dim=1)

    def _digit_positions(self):
        idx = torch.arange(MAX_DIGITS, device=self.spiral_amp.device, dtype=self.spiral_amp.dtype)
        angle = 2.0 * math.pi * idx / MAX_DIGITS + self.spiral_phase
        pos = torch.zeros(MAX_DIGITS, POS_DIM, device=idx.device)
        pos[:, 0] = self.spiral_amp * torch.cos(angle)
        pos[:, 1] = self.spiral_amp * torch.sin(angle)
        pos[:, 2] = self.spiral_slope * idx + self.spiral_offset
        return pos

    def _positions(self, T):
        digit_pos = self._digit_positions()
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

    def _rmsnorm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.norm_weight

    def _q_phase(self, q):
        cos_a = self.q_phase_angle.cos()
        sin_a = self.q_phase_angle.sin()
        q_rot = q.clone()
        for p in range(QK_DIM // 2):
            d0, d1 = 2 * p, 2 * p + 1
            q_rot[..., d0] = q[..., d0] * cos_a - q[..., d1] * sin_a
            q_rot[..., d1] = q[..., d0] * sin_a + q[..., d1] * cos_a
        return q_rot

    def forward(self, idx):
        B, T = idx.shape
        tok_table = self._tok_emb_table()

        # Embed: [tok(2) | pos(3)] = 5D residual stream
        tok = tok_table[idx]
        pos = self._positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        # === Attention ===
        h = self._rmsnorm(x)
        pos_h = h[:, :, TOK_DIM:]
        tok_h = h[:, :, :TOK_DIM]

        q = self.q_proj(pos_h).unsqueeze(1)                # (B, 1, T, 4)
        k = self.q_proj(pos_h).unsqueeze(1)                # tied Q/K
        v = F.linear(tok_h, self.head_proj.weight.T).unsqueeze(1)  # tied V/output -> (B, 1, T, 5)

        q = self._q_phase(q)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(QK_DIM)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).squeeze(1)                         # (B, T, 5)
        x = x + (out @ self.out_proj_A) @ self.out_proj_B # rank-1 projection

        # === FFN (fc2 = head_proj.weight, triple-duty) ===
        ffn_hidden = F.gelu(self.fc1(self._rmsnorm(x)))
        x = x + ffn_hidden @ self.head_proj.weight         # tied fc2

        # === Output ===
        return self.head_proj(self._rmsnorm(x)) @ tok_table.T

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, T_prompt = prompt.shape
        seq = torch.zeros(B, T_prompt + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
        seq[:, :T_prompt] = prompt
        for step in range(ANSWER_LEN + 1):
            T = T_prompt + step
            logits = self.forward(seq[:, :T])
            seq[:, T] = logits[:, -1].argmax(dim=-1)
        return seq[:, T_prompt:]


# -- Submission interface --

def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroAdder()

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_57p.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    KEY_MAP = {
        "out_proj.A": "out_proj_A",
        "out_proj.B": "out_proj_B",
        "norm1.weight": "norm_weight",
    }
    SKIP_KEYS = {
        "_pos_sources", "_pos_indices",
        "norm2.weight", "norm_f.weight",  # shared norm duplicates
        "spiral_amp", "spiral_phase", "spiral_slope", "spiral_offset",  # frozen sinusoidal
        "_plus_pos", "_eos_pos",  # frozen zero
        "causal_mask",
    }

    raw = ckpt["model_state_dict"]
    state = {}
    for k, v in raw.items():
        if k in SKIP_KEYS:
            continue
        state[KEY_MAP.get(k, k)] = v

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    seen = set()
    n_params = sum(p.numel() for p in model.parameters() if id(p) not in seen and not seen.add(id(p)))

    return model, {
        "name": "MicroAdder 57p",
        "author": "Arseniy Zarechnev",
        "params": n_params,
        "architecture": "1L decoder, d=5(2+3), 1h, qk=4, hd=5, ff=2, rank-1 out, "
                        "no bias, parametric circular emb, sinusoidal pos, "
                        "shared norm, tied V/O, tied Q/K+phase, tied fc2=head_proj",
    }


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    x_digits = [(a // 10**i) % 10 for i in range(MAX_DIGITS)]
    y_digits = [(b // 10**i) % 10 for i in range(MAX_DIGITS)]
    prompt = x_digits + [0] + y_digits + [0]
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(prompt_t)

    result = 0
    for i, tok in enumerate(generated[0].tolist()):
        if i >= ANSWER_LEN:
            break
        result += tok * (10 ** i)
    return result
