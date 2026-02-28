# MicroAdder

A **170-parameter** trained transformer that performs 10-digit addition with 100% accuracy. Built for the [AdderBoard](https://github.com/anadim/AdderBoard) challenge.

## Current Best Result

**170 parameters, 100% accuracy** (10010/10010 confirmed on AdderBoard verification).

| | Params | Accuracy (10K) | Seed |
|---|---|---|---|
| **Best (uncompressed)** | **170** | **100%** | 80085 |
| Previous best | 187 | 100% | 777, 1995 |
| Earlier | 203 | 100% | 777, 69420, 1995 |

### Model Architecture

1-layer autoregressive decoder with split-subspace attention and tied Q/K.

```
d_model = 6 = tok_dim(2) + pos_dim(4)
1 layer, 2 heads, head_dim = 3
FFN dim = 2 (with bias, GELU activation)
RMSNorm (weight only, no bias)
Tied Q/K with per-head phase rotation (2 params)
Tied output head: head_proj(6->2) @ tok_emb.T
```

### Parameter Budget (170p)

```
tok_emb           28  (14 x 2, tied with output)
spiral params      4  (amp, phase, slope, offset)
linear pos corr    2  (slope + intercept)
z_hi_pos           4  (1 x 4, carry position)
special_pos        8  (2 x 4, PLUS + EQUALS; EOS frozen to zero)
q_proj            24  (4 -> 6, shared with K)
q_phase            2  (per-head rotation angle)
v_proj            12  (2 -> 6)
out_proj          24  (6x2 + 2x6, rank-2 factorized)
ln1 + ln2 + ln_f  18  (3 x 6)
fc1 (w+b)         14  (6 -> 2)
fc2 (w+b)         18  (2 -> 6)
head_proj         12  (6 -> 2)
─────────────────────
TOTAL            170
```

## Our Contributions

### tok_dim=2, pos_dim=4 Reshape

The most recent compression: **-17 parameters**. The token embedding subspace is shrunk from 3D to 2D (SVD of trained embeddings shows 96% energy in 2 dimensions). This reduces tok_emb (14x3 -> 14x2), v_proj (3->6 to 2->6), and head_proj (6->3 to 6->2), while slightly expanding q_proj (3->6 to 4->6). The tied output head must discriminate 14 classes from 2D embeddings — this is tight, making grokking seed-sensitive (1/3 seeds succeed).

### Tied Q/K with Per-Head Phase Rotation

The single biggest compression: **-16 parameters**. Q and K share the same projection matrix. A learnable per-head angle (2 params) rotates pairs of Q dimensions — `Q_rot = Q*cos(θ) - Q_swap*sin(θ)` — giving each head a unique "viewing angle" on the shared key space. This provides the asymmetry that carry routing requires, replacing an 18-parameter K projection with just 2 parameters. This overturns the earlier finding that tied Q/K caps at 39% accuracy.

### Adaptive Weight Decay

Weight decay drops from 0.01 → 0.001 → 0.0001 as grokking is detected (val_exact crosses 20% then 50%), enabling quicker convergence. The carry circuit needs large weights to approximate hard step functions; constant WD fights this sharpening process.

### Carry-Mix Training Curriculum

A training-time data sampling strategy that oversamples carry-heavy examples early, then fades them out so the model finishes on a natural distribution. Under uniform random sampling, long carry chains (e.g., `9999999999 + 1`) are exponentially rare — carry-mix guarantees the model sees them regularly.

**How it works:** 30% of each batch is replaced with structured carry-heavy problems from four patterns — isolated single carries, cascading all-9s chains, single-digit place additions, and power-of-10 boundary crossings. This fraction fades linearly to 0% as token accuracy rises from 0.7 to 0.9, ensuring the model's final training is on the natural distribution.

Combined with a digit curriculum (1-3 digits for 2K steps, 1-6 for 5K, then full 1-10), carry-mix accelerates grokking dramatically by addressing the long tail of hard addition problems without distorting what the model converges to.

See **[carry-mix.md](carry-mix.md)** for the full design, the four carry patterns, the adaptive fading schedule, and results.

### Spiral+Correction Positional Encoding

Replaced JackCai's 30 learned position parameters with a parametric spiral (4 params) plus a linear correction (2 params), saving **24 parameters**. The spiral `(amp*cos(angle), amp*sin(angle), slope*i + offset)` captures the base-10 periodic structure of digit positions, while the linear correction `(1 + intercept + slope*i)` gives per-position magnitude control.

### Rank-2 Attention Output Projection

The attention output matrix is factored as `A(6x2) @ B(2x6)` = 24 params instead of a full `6x6` = 36 params, saving **12 parameters**. We verified via SVD that trained checkpoints concentrate >85% of energy in the top 2 singular values, then confirmed this can be trained natively from scratch.

### Frozen EOS Special Position

The EOS token's positional encoding is fixed to zero (a buffer, not a parameter), saving **3 parameters**. Post-training analysis showed EOS position is exactly zero in every grokked checkpoint — it carries no positional information the model needs.


## Path to 170p

```
242p  JackCai's split-subspace architecture (SOTA at the time)
  |   - d_model=6, split Q/K/V, shared XYZ positions, tied output head
  |
226p  Spiral+correction positions (-16p)
  |   - Parametric spiral replaces 30 learned position params with 14
  |   - Carry-focused curriculum helps grokking
  |
214p  + Rank-2 attention output (-12p)
  |   - out_proj factored as A(6x2) @ B(2x6), trained natively
  |   - 4/10 seeds grok at this size
  |
203p  + Linear pos correction + frozen EOS (-11p)
  |   - 10 per-position corrections -> 2 linear params (-8p)
  |   - EOS special position frozen to zero (-3p)
  |   - 3/4 seeds grok at this size
  |
187p  + Tied Q/K with q-phase rotation (-16p)
  |   - k_proj (18p) eliminated, replaced by q_phase_angle (2p)
  |   - Per-head angle: Q_rot = Q·cos(θ) - Q_swap·sin(θ)
  |   - Adaptive weight decay: 18x faster grokking (~33K steps)
  |   - 2/3 seeds grok at this size
  |
170p  + tok_dim=2, pos_dim=4 reshape (-17p)
      - Token embedding subspace shrunk from 3D to 2D
      - Seed-sensitive: 1/3 seeds grok (80085 at 15K steps)
```

## Quick Start

```bash
cd microadder

# Verify submission
uv run python ../AdderBoard/verify.py submission_170p/submission_170p.py

# Train from scratch (170p)
uv run python -m src.train --run-name my_170p_run \
    --tie-qk --q-phase --tok-dim 2 --pos-dim 4 \
    --pos-mode spiral_correct --attn-out-rank 2 \
    --pos-correction-mode linear --freeze-special eos \
    --wd-adaptive --wd-drop-exact 0.2 --wd-drop-exact-final 0.5 \
    --seed 80085 --steps 500000 --lr 0.02 --carry-mix 0.3

# Evaluate a checkpoint
uv run python -m src.eval --checkpoint results/runs/<run>/checkpoints/best.pt --autoregressive
```

## Credits

This work builds directly on:

- **[JackCai1206](https://github.com/JackCai1206/smallest-addition-transformer)** — the 242p split-subspace architecture. The core design (split Q/K/V attention, shared XYZ positions, tied output head, RMSNorm, spiral token init) is from their work.
- **[rezabyt](https://github.com/rezabyt)** — the 311p model and rank-3 factorization approach that informed our earlier attempts.
- **[Dimitris Papailiopoulos](https://github.com/anadim)** — for creating the [AdderBoard](https://github.com/anadim/AdderBoard) challenge.

Made by Arseniy Zarechnev with help from Claude Code and Codex.

## Citation
```bibtex
@misc{zarechnev2026microadder,
  author       = {Arseniy Zarechnev},
  title        = {micro-ten-digit-addition-transformer},
  year         = {2026},
  url          = {https://github.com/evindor/MicroAdder},
}
```
