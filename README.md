# MicroAdder

A **203-parameter** trained transformer that performs 10-digit addition with 100% accuracy. Built for the [AdderBoard](https://github.com/anadim/AdderBoard) challenge.

## Current Best Result

**203 parameters, 100% accuracy** (10010/10010 confirmed on AdderBoard verification).

| | Params | Accuracy (1K) | Seed |
|---|---|---|---|
| **Best (uncompressed)** | **203** | **100%** | 777 |
| Other qualified seeds | 203 | 100% | 69420, 1995 |

### Model Architecture

1-layer autoregressive decoder with split-subspace attention.

```
d_model = 6 = tok_dim(3) + pos_dim(3)
1 layer, 2 heads, head_dim = 3
FFN dim = 2 (with bias, GELU activation)
RMSNorm (weight only, no bias)
Tied output head: head_proj(6->3) @ tok_emb.T
```

### Parameter Budget (203p)

```
tok_emb           42  (14 x 3, tied with output)
spiral params      4  (amp, phase, slope, offset)
linear pos corr    2  (slope + intercept)
z_hi_pos           3  (1 x 3, carry position)
special_pos        6  (2 x 3, PLUS + EQUALS; EOS frozen to zero)
q_proj            18  (3 -> 6)
k_proj            18  (3 -> 6)
v_proj            18  (3 -> 6)
out_proj          24  (6x2 + 2x6, rank-2 factorized)
ln1 + ln2 + ln_f  18  (3 x 6)
fc1 (w+b)         14  (6 -> 2)
fc2 (w+b)         18  (2 -> 6)
head_proj         18  (6 -> 3)
─────────────────────
TOTAL            203
```

## Our Contributions

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


## Path to 203p

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
      - 10 per-position corrections -> 2 linear params (-8p)
      - EOS special position frozen to zero (-3p)
      - 3/4 seeds grok at this size
```

## Quick Start

```bash
cd microadder

# Verify submission
uv run python ../AdderBoard/verify.py submission_203p/submission_203p.py

# Train from scratch (203p)
uv run python -m src.train --run-name my_203p_run \
    --pos-mode spiral_correct --attn-out-rank 2 \
    --pos-correction-mode linear --freeze-special eos \
    --seed 777 --steps 500000 --lr 0.02 --carry-mix 0.3

# Evaluate a checkpoint
uv run python -m src.eval --checkpoint results/runs/<run>/checkpoints/best.pt --autoregressive
```

## Credits

This work builds directly on:

- **[JackCai1206](https://github.com/JackCai1206/smallest-addition-transformer)** — the 242p split-subspace architecture. The core design (split Q/K/V attention, shared XYZ positions, tied output head, RMSNorm, spiral token init) is from their work.
- **[rezabyt](https://github.com/rezabyt)** — the 311p model and rank-3 factorization approach that informed our earlier attempts.
- **[Dimitris Papailiopoulos](https://github.com/anadim)** — for creating the [AdderBoard](https://github.com/anadim/AdderBoard) challenge.

Made by Arseniy Zarechnev with help from Claude Code and Codex.
