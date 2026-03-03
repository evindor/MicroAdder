# MicroAdder 67p Submission

**67 learnable parameters. 100% accuracy. Trained from scratch.**

[Interactive model](https://evindor.github.io/MicroAdder/) | [Paper](https://evindor.github.io/MicroAdder/PAPER) (not a real scientific paper, it's a weekend project)

A 1-layer autoregressive decoder that performs 10-digit addition with 100% accuracy (10010/10010 on [AdderBoard](https://github.com/anadim/AdderBoard)). Every parameter is learned from random initialization — no warm-starting, no frozen pretrained values. Positional encodings use a fixed sinusoidal scheme (0 learned parameters).

## Parameter Budget

| Component | Params | Description |
|---|---|---|
| `tok_arc` (A, start, stride) | 3 | Circular arc: `A·cos(s+d·stride), A·sin(s+d·stride)` (A=B tied) |
| `z_hi_pos` | 3 | Carry-out position (Z_10), learned 3D vector |
| `special_pos_equals` | 3 | EQUALS delimiter position |
| `q_phase_angle` | 1 | Rotation angle for Q/K asymmetry |
| `q_proj` | 12 | 3→4 linear, shared by Q and K |
| `out_proj` | 10 | Rank-1 factored: A(5×1) + B(1×5) |
| `ffn_fc1` | 10 | 5→2, no bias |
| `ffn_fc2` | 10 | 2→5, no bias |
| `head_proj` | 10 | 5→2, doubles as V projection (tied V/O) |
| `RMSNorm` (shared) | 5 | One 5D weight vector, used 3 times |
| **Total** | **67** | |

Free (frozen at initialization):

| Component | Params | Value |
|---|---|---|
| Sinusoidal positions | 4 | amp=3.5, phase=0, slope=0.15, offset=0 |
| PLUS/EOS positions | 6 | Frozen at zero |

## What Changed from 74p → 67p

Two independent compressions, both motivated by analysis of 74p trained weights:

1. **Sinusoidal positions (−4p):** The 74p model's learned spiral converged near sinusoidal defaults (amp=3.56→3.5, phase=−25.3°→0°, slope=0.17→0.15). Freezing all spiral params to fixed sinusoidal values works — the Q/K projection compensates for the difference. Fixed sinusoidal encodings are free by competition rules.

2. **Reduced Q/K dimension (−3p):** The 74p model's Q/K projection had a dead 5th row (all near-zero weights). Removing it: `q_proj` shrinks from 3→5 (15p) to 3→4 (12p). Four dimensions suffice for position-based attention routing.

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=0.02, cosine decay) |
| Weight decay | 0.01 (constant — no adaptive WD needed) |
| Batch size | 256 |
| Steps | 60K |
| Curriculum | 1-3 digits for 2K steps, 1-6 for 5K, then full 1-10 |
| Carry mix | 80% carry-heavy, step-based fade 15K→45K |
| Seed | 71046 |
| Grokking | ~46K steps |

The training breakthrough that made this reproducible: **80% carry-heavy examples** with step-based linear fade. Long carry chains (9999999999 + 1) are exponentially rare in uniform sampling but are the hardest test case. Step-based fade avoids the oscillation feedback loop that metric-based fade creates at high carry-mix.

## Compression History

Each step validated at 100% accuracy (10010/10010), trained from scratch:

```
242p  JackCai's split-subspace (starting point)
  ↓   spiral positions, rank-2 out_proj, linear correction, frozen EOS
226p
  ↓   tied Q/K + phase rotation
187p
  ↓   tok_dim 3→2
170p
  ↓   d_model 6→5, vocab 14→10, parametric embeddings, 1 head
133p
  ↓   rank-1 out_proj, no FFN bias
100p
  ↓   shared norms, tied V/output, no position correction
 78p
  ↓   freeze PLUS/EOS positions
 75p
  ↓   tie A=B (circular embedding) + high carry-mix training
 74p  previous from-scratch frontier
  ↓   sinusoidal positions (freeze spiral) + qk_dim 5→4
 67p  ← current from-scratch frontier
```

## Verification

```bash
# Via AdderBoard verifier
uv run python ../AdderBoard/verify.py submission_67p/submission_67p.py

# Standalone test
python -c "
from submission_67p.submission_67p import build_model, add
model, meta = build_model()
print(f'{meta[\"params\"]} params')
print(add(model, 9999999999, 1))          # 10000000000
print(add(model, 5678901234, 4321098765)) # 9999999999
"
```

## Reproduce From Scratch

```bash
uv run python -m microadder.train --run-name sub100_67p_repro --seed 71046
```

## Credits

Built on the work of:
- **[JackCai1206](https://github.com/JackCai1206/smallest-addition-transformer)** — the 242p split-subspace architecture
- **[Wonderfall](https://github.com/Wonderfall)** (param_40) — tied Q/K with phase rotation
- **[Alex Litzenberger](https://alexlitzenberger.com)** (TinyAdder) — 36p hand-coded model, the representational floor
- **[Dimitris Papailiopoulos](https://github.com/anadim)** — the [AdderBoard](https://github.com/anadim/AdderBoard) challenge

Made by Arseniy Zarechnev with help from Claude.
