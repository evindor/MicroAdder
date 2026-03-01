# MicroAdder 75p Submission

**75 learnable parameters. 100% accuracy. Trained from scratch.**

A 1-layer autoregressive decoder that performs 10-digit addition with 100% accuracy (10010/10010 on [AdderBoard](https://github.com/anadim/AdderBoard)). Every parameter is learned from random initialization — no warm-starting, no frozen pretrained values.

## Architecture

```
Input: X_0..X_9 + Y_0..Y_9 = Z_0..Z_10 EOS     (34 tokens, LSB-first)

                    ┌─────────────────────────────────────┐
   tok_emb(4p)      │  Parametric arc: 10 digits in 2D    │
   + spiral_pos(4p) │  Spiral: 10 positions in 3D         │
                    └──────────┬──────────────────────────┘
                               │  x ∈ R^5 = [tok(2) ‖ pos(3)]
                               ▼
                    ┌──────────────────────┐
                    │  RMSNorm (5p shared) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │           Attention              │
              │  Q = q_proj(pos) (15p)           │
              │  K = q_proj(pos) (tied with Q)   │
              │  Q = rotate(Q, phase) (1p)       │
              │  V = tok @ head_proj.W (tied)    │
              │  out = rank-1 proj (10p)         │
              └────────────────┬────────────────┘
                               │  + residual
                    ┌──────────────────────┐
                    │  RMSNorm (shared)    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │       FFN (no bias)              │
              │  fc1: 5 → 2 (10p)  GELU         │
              │  fc2: 2 → 5 (10p)               │
              └────────────────┬────────────────┘
                               │  + residual
                    ┌──────────────────────┐
                    │  RMSNorm (shared)    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │  head_proj: 5 → 2 (10p)         │
              │  logits = head_proj(x) @ tok.T   │
              └─────────────────────────────────┘
```

1-layer decoder. d_model=5, split into tok_dim=2 + pos_dim=3, concatenated. Single attention head with head_dim=5 (full d_model rank). FFN hidden dim=2 with GELU activation. All three RMSNorm layers share a single 5D weight vector.

## Parameter Budget

| Component | Params | Description |
|---|---|---|
| `tok_arc` (A, B, start, stride) | 4 | All 10 digit embeddings from `(A cos(s+i*d), B sin(s+i*d))` |
| `spiral` (amp, phase, slope, offset) | 4 | Positional encoding: `(amp*cos(2pi*i/10+ph), amp*sin(...), slope*i+off)` |
| `z_hi_pos` | 3 | Carry-out position (Z_10), learned 3D vector |
| `special_pos_equals` | 3 | EQUALS delimiter position (PLUS/EOS fixed at zero) |
| `q_phase_angle` | 1 | Rotation angle for Q/K asymmetry |
| `q_proj` | 15 | 3 -> 5 linear, shared by Q and K |
| `out_proj` | 10 | Rank-1 factored: A(5x1) + B(1x5) |
| `ffn_fc1` | 10 | 5 -> 2, no bias |
| `ffn_fc2` | 10 | 2 -> 5, no bias |
| `head_proj` | 10 | 5 -> 2, doubles as V projection (tied V/O) |
| `RMSNorm` (shared) | 5 | One 5D weight vector, used 3 times |
| **Total** | **75** | |

## Key Design Choices

### Split-Subspace Attention (from JackCai's 242p)

The residual stream is split: the first 2 dimensions carry **token** (digit identity) information, the last 3 carry **positional** (where am I?) information. Q and K attend using only the positional subspace. V reads only the token subspace. This hard split means attention routing is purely positional — the model learns fixed offset patterns (carry-lookahead), not content-dependent attention.

### Vocab = 10

PLUS, EQUALS, and EOS are all token ID 0 (same as digit zero). The model distinguishes them entirely by position. This eliminates 4 special token embeddings and simplifies the output head to discriminate only 10 classes in 2D.

### Parametric Token Embeddings (4p instead of 20p)

Instead of a 10x2 learned table, all 10 digit embeddings lie on a 2D arc: `(A*cos(start + i*stride), B*sin(start + i*stride))`. The model learns the arc shape (A, B, start, stride) during training. Since the same embedding table is used for both input and output (via `head_proj(x) @ tok_table.T`), the arc must simultaneously encode digits and provide good classification boundaries.

### Tied Q/K with Phase Rotation (16p saved)

Q and K share the same 3->5 projection matrix. A single learnable angle (1 param) rotates pairs of Q dimensions: `Q_rot = Q*cos(theta) - Q_swap*sin(theta)`. This gives the asymmetry the carry circuit needs — K says "I am at position i", Q says "I want position i+offset" — without a separate K projection.

### Tied V/Output (10p saved)

The value projection and output head share weights: `v_proj.weight = head_proj.weight.T`. Both map between tok_dim (2) and d_model (5). Despite the trained matrices not being naturally similar in untied models (cosine sim = -0.30), tying them acts as beneficial regularization at d_model=5.

### Shared RMSNorm (10p saved)

All three normalization layers (pre-attention, pre-FFN, final) share a single 5D weight vector. At d_model=5, one normalization scale works in all three positions.

### Rank-1 Attention Output (10p saved)

The attention output matrix is factored as `A(5x1) @ B(1x5)` — 10 params instead of a full 5x5 = 25. The trained solution at d_model=5 is effectively rank-1.

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=0.02, cosine decay to 0.002) |
| Weight decay | 0.01, adaptive (drops at val_exact 1% and 5%) |
| Batch size | 256 |
| Curriculum | 1-3 digits for 2K steps, 1-6 for 5K, then full 1-10 |
| Carry mix | 30% structured carry-heavy examples, faded to 0% |
| Seed | 80085 (from scratch) |
| Grokking | ~36K steps |
| Total steps | 500K budget |

**Adaptive weight decay** is critical. The carry circuit needs large weights to approximate hard step functions; constant WD fights this sharpening. WD drops from 0.01 to 0.001 when val_exact first exceeds 1%, then to 0.0001 at 5%.

**Carry-mix curriculum** oversamples carry-heavy problems (cascading 9s, boundary crossings) early in training, fading to uniform as token accuracy rises from 70% to 90%. This addresses exponentially rare long carry chains without distorting the final distribution.

**Seed sensitivity**: Only ~10-20% of random seeds show grokking signals at 75p. Seed 80085 is the only confirmed seed that stably groks to 100%.

## Verification

```bash
# Via AdderBoard verifier
uv run python ../AdderBoard/verify.py submission_75p/submission_75p.py

# Standalone test
python -c "
from submission_75p.submission_75p import build_model, add
model, meta = build_model()
print(f'{meta[\"params\"]} params')
print(add(model, 9999999999, 1))          # 10000000000
print(add(model, 5678901234, 4321098765)) # 9999999999
"
```

## Reproduce From Scratch

```bash
uv run python -m src.train --run-name repro_75p \
    --d-model 5 --tok-dim 2 --pos-dim 3 --n-heads 1 --head-dim 5 \
    --ffn-dim 2 --no-ffn-bias --tie-qk --q-phase --attn-out-rank 1 \
    --vocab-size 10 --tok-emb-mode parametric \
    --pos-mode spiral_correct --pos-correction-mode none \
    --freeze-special plus_eos --norm-mode shared --tie-vo \
    --lr 0.02 --carry-mix 0.3 --wd-adaptive \
    --seed 80085 --steps 500000
```
