# MicroAdder

A **67-parameter** trained transformer that performs 10-digit addition with 100% accuracy. Built for the [AdderBoard](https://github.com/anadim/AdderBoard) challenge.

## Result

**67 parameters, 100% accuracy** (10,010/10,010). Trained from scratch — no warm-starting, no frozen pretrained values. Sinusoidal positional encoding (0 learned position parameters).

| Params | Accuracy | Seed | Key innovations | Training |
|--------|----------|------|-----------------|----------|
| **67** | **100%** | 71046 | sinusoidal pos, qk_dim=4 | carry_mix=0.8, step-fade, 60K steps |
| 74 | 100% | 45214, 71046, 78988 | learned spiral, qk_dim=5 | carry_mix=0.8, step-fade, 120K steps |
| 242 | 100% | (baseline) | — | JackCai's architecture |

## Architecture

Single-layer autoregressive decoder. 5-dimensional residual stream (2D token + 3D position). One attention head with 4D Q/K subspace and 5D value pathway.

```
d_model = 5 = tok_dim(2) + pos_dim(3)
1 layer, 1 head, qk_dim = 4, head_dim = 5
FFN dim = 2 (no bias, GELU)
```

Every component is compressed:

| Component | Params | Innovation |
|-----------|--------|------------|
| Token embeddings | 3 | Circular arc: `A·cos(s+d·stride), A·sin(s+d·stride)` |
| Positions | 0 | Frozen sinusoidal (amp=3.5, phase=0, slope=0.15) |
| Carry position | 3 | Learned (high norm, far from digits) |
| EQUALS position | 3 | Learned (PLUS/EOS frozen at zero) |
| Q/K projection | 12+1 | Tied Q/K with phase rotation (3→4) |
| Attention output | 10 | Rank-1 factorization: A(5×1) @ B(1×5) |
| FFN | 20 | 5→2→5, no bias, GELU |
| Output head | 10 | Also serves as V projection (tied V/O) |
| Normalization | 5 | Single RMSNorm weight shared across all 3 norms |
| **Total** | **67** | |

## Compression Path

```
242p  JackCai's split-subspace (starting point)
  ↓  spiral positions, rank-2 out_proj, linear correction, frozen EOS
203p  tied Q/K + phase rotation, adaptive weight decay
  ↓  tok_dim 3→2, d_model 6→5, vocab 14→10, 1 head
133p  parametric embeddings, rank-1 out_proj, no FFN bias
  ↓  shared norms, tied V/output, no position correction
 75p  frozen PLUS/EOS positions
  ↓  tied A=B (circular embedding) + high carry-mix training
 74p  previous from-scratch frontier
  ↓  sinusoidal positions (freeze spiral) + qk_dim 5→4
 67p  ← current from-scratch frontier
```

## Quick Start

```bash
# Verify (100% accuracy, ~80s)
uv run python ../AdderBoard/verify.py submission_67p/submission_67p.py

# Train from scratch (~60K steps, ~5 min)
uv run python -m microadder.train --run-name sub100_my_67p --seed 71046

# Evaluate a checkpoint
uv run python -m microadder.eval --checkpoint results/runs/sub100_my_67p/checkpoints/best.pt
```

## Key Training Innovation: High Carry-Mix

The training breakthrough that made sub-100p reproducible: 80% carry-heavy examples with step-based linear fade.

Long carry chains (9999999999 + 1) are exponentially rare in uniform sampling but are the hardest test case. Flooding early training with carry-heavy examples forces the carry circuit to form. Step-based fade (linear ramp from 80% to 0% over steps 15K-45K) avoids the oscillation feedback loop that metric-based fade creates at high carry-mix.

## Project Structure

```
microadder/           Clean reimplementation (model + training)
  model.py            67p/74p model definition
  train.py            Training loop
  data.py             Data generation
  eval.py             Evaluation

submission_67p/       Current best submission (67p)
submission_74p/       Previous best (74p)

src/                  Original research codebase (full feature set)

PAPER.md              Full write-up of the research
ARCHITECTURE.md       How the model performs addition (weight analysis)
RESEARCH.md           Detailed research notes
JOURNEY.md            Chronological research narrative
REFLECTION.md         Retrospective analysis
```

## Credits

Built on the work of:
- **[JackCai1206](https://github.com/JackCai1206/smallest-addition-transformer)** — the 242p split-subspace architecture
- **[Wonderfall](https://github.com/Wonderfall)** (param_40) — tied Q/K with phase rotation
- **[Alex Litzenberger](https://alexlitzenberger.com)** (TinyAdder) — 36p hand-coded model, the representational floor
- **[Dimitris Papailiopoulos](https://github.com/anadim)** — the [AdderBoard](https://github.com/anadim/AdderBoard) challenge

Made by Arseniy Zarechnev with help from Claude.

## Citation

```bibtex
@misc{zarechnev2026microadder,
  author       = {Arseniy Zarechnev and Claude},
  title        = {67 Parameters Is All You Need: Training a Transformer for Perfect 10-Digit Addition},
  year         = {2026},
  url          = {https://github.com/evindor/MicroAdder},
}
```
