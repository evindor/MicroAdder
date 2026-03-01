# MicroAdder

A **75-parameter** trained transformer that performs 10-digit addition with 100% accuracy. Built for the [AdderBoard](https://github.com/anadim/AdderBoard) challenge.

## Current Best Result

**75 parameters, 100% accuracy** (10010/10010). Trained from scratch — no warm-starting, no frozen pretrained values.

| | Params | Accuracy (10K) | Seed | From scratch? |
|---|---|---|---|---|
| **Current best** | **75** | **100%** | 80085 | Yes |
| Previous best | 170 | 100% | 80085 | Yes |
| Earlier | 203 | 100% | 777, 69420, 1995 | Yes |

### Model Architecture

1-layer autoregressive decoder with split-subspace attention, tied Q/K, tied V/output, shared norms.

```
d_model = 5 = tok_dim(2) + pos_dim(3)
1 layer, 1 head, head_dim = 5 (full d_model rank)
FFN dim = 2 (no bias, GELU activation)
Shared RMSNorm (one weight vector for all 3 norms)
Tied Q/K with per-head phase rotation (1 param)
Tied V/output: v_proj = head_proj.T (no separate V projection)
Rank-1 attention output: A(5x1) @ B(1x5)
Parametric token embedding: 4 arc params for 10 digits
Spiral positional encoding: 4 params, no correction
Vocab = 10 (special tokens → digit-0, distinguished by position)
```

### Parameter Budget (75p)

```
tok_arc (A, B, start, stride)   4  parametric token embeddings
spiral (amp, phase, slope, off) 4  positional encoding
z_hi_pos                        3  carry position (1 x 3)
special_pos_equals              3  EQUALS position (PLUS/EOS frozen to zero)
q_phase_angle                   1  Q rotation for tied Q/K asymmetry
q_proj                         15  (3 -> 5, shared with K)
out_proj                       10  (5x1 + 1x5, rank-1 factorized)
FFN fc1                        10  (5 -> 2, no bias)
FFN fc2                        10  (2 -> 5, no bias)
head_proj                      10  (5 -> 2, also used as v_proj)
RMSNorm (shared)                5  (one weight, 3 norms)
───────────────────────────────
TOTAL                          75
```

## Our Contributions

### d_model=5, Single Head (170p → 75p)

The largest structural change: shrinking from d_model=6 with 2 heads (head_dim=3) to d_model=5 with 1 head (head_dim=5). One head with full d_model rank is more expressive than two heads with head_dim=3 each, and the reduced d_model cascades savings across every layer. Combined with vocab=10, this unlocked the entire sub-100p compression path.

### Parametric Token Embeddings

Instead of a 10×2 learned embedding table (20p), we parameterize all 10 digit embeddings as points on a 2D arc: `(A·cos(start + i·stride), B·sin(start + i·stride))` — just 4 parameters. The model learns the optimal arc shape during training.

### Tied V/Output

The value projection and the output head share weights: `v_proj.weight = head_proj.weight.T`. Both map between tok_dim (2) and d_model (5). Despite the trained matrices not being naturally similar (cosine sim = -0.30 in the untied model), tying them acts as beneficial regularization.

### Shared RMSNorm

All three RMSNorm layers (pre-attention, pre-FFN, final) share a single 5D weight vector — 5 params instead of 15. The model finds a single normalization scale that works in all three positions.

### Tied Q/K with Per-Head Phase Rotation

Q and K share the same projection matrix. A learnable rotation angle (1 param) rotates pairs of Q dimensions — `Q_rot = Q·cos(θ) - Q_swap·sin(θ)` — providing the asymmetry the carry circuit needs. This replaces a separate K projection (15p) with just 1 parameter.

### Adaptive Weight Decay

Weight decay drops from 0.01 → 0.001 → 0.0001 as grokking is detected (val_exact crosses thresholds), enabling quicker convergence. The carry circuit needs large weights to approximate hard step functions; constant WD fights this sharpening process.

### Carry-Mix Training Curriculum

30% of each batch is replaced with structured carry-heavy problems (cascading carries, all-9s chains, etc.), fading to 0% as token accuracy rises. Combined with a digit curriculum (1-3 digits → 1-6 → 1-10), this addresses the exponentially rare long carry chains without distorting the final training distribution.

See **[carry-mix.md](carry-mix.md)** for the full design.

### Spiral Positional Encoding

Parametric spiral `(amp·cos(2πi/10 + phase), amp·sin(2πi/10 + phase), slope·i + offset)` captures the base-10 periodic structure of digit positions with just 4 parameters, replacing 30 learned position parameters.

### Rank-1 Attention Output

The attention output matrix is factored as `A(5×1) @ B(1×5)` = 10 params instead of a full 5×5 = 25 params.

## Compression Path

```
242p  JackCai's split-subspace architecture (SOTA at the time)
  |   d_model=6, split Q/K/V, shared XYZ positions, tied output, RMSNorm
  |
226p  Spiral+correction positions (-16p)
  |   Parametric spiral replaces 30 learned position params
  |
214p  Rank-2 attention output (-12p)
  |   out_proj factored as A(6x2) @ B(2x6)
  |
203p  Linear pos correction + frozen EOS (-11p)
  |   10 corrections → 2 linear params, EOS position → zero
  |
187p  Tied Q/K with q-phase rotation (-16p)
  |   k_proj eliminated; Q_rot = Q·cos(θ) - Q_swap·sin(θ)
  |   Adaptive weight decay (18x faster grokking)
  |
170p  tok_dim=2, pos_dim=4 reshape (-17p)
  |   Token subspace 3D→2D (96% SVD energy)
  |
133p  vocab=10, d_model=5, 1 head, parametric tok_emb (-37p)
  |   All special tokens → digit-0; 4 arc params vs 20 learned
  |   Rank-1 out_proj, no FFN bias, no pos correction
  |
100p  Same + all the above stacked
  |
 80p  Shared norms + tied V/output (-20p)
  |   Single RMSNorm weight, v_proj = head_proj.T
  |
 78p  Remove position correction (-2p)
  |   Spiral amp absorbs constant scaling
  |
 75p  Freeze PLUS+EOS positions (-3p)
      Delimiter positions fixed at zero
      ** FROM-SCRATCH FRONTIER **
```

## Quick Start

```bash
cd microadder

# Verify submission (75p)
uv run python ../AdderBoard/verify.py submission_75p/submission_75p.py

# Train from scratch (75p, ~276K steps to grok)
uv run python -m src.train --run-name my_75p_run \
    --d-model 5 --tok-dim 2 --pos-dim 3 --n-heads 1 --head-dim 5 \
    --ffn-dim 2 --no-ffn-bias --tie-qk --q-phase --attn-out-rank 1 \
    --vocab-size 10 --tok-emb-mode parametric \
    --pos-mode spiral_correct --pos-correction-mode none \
    --freeze-special plus_eos --norm-mode shared --tie-vo \
    --lr 0.02 --carry-mix 0.3 --wd-adaptive \
    --seed 80085 --steps 400000

# Evaluate a checkpoint
uv run python -m src.eval --checkpoint results/runs/<run>/checkpoints/best.pt --autoregressive
```

## Project Structure

```
src/
  model.py    - MicroAdder model definition
  train.py    - Training loop with curriculum, adaptive WD
  data.py     - Data generation, tokenization, positional encoding maps
  eval.py     - Evaluation scripts

submission_75p/   - Current best: 75p from scratch
submission_100p/  - 100p milestone
submission_170p/  - 170p (original sub-200p breakthrough)

RESEARCH.md       - Detailed research notes and architecture analysis
JOURNEY.md        - Chronological research narrative
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
