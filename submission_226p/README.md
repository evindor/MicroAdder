# MicroAdder 226p Submission

226-parameter trained transformer that does 10-digit addition with 99.98% accuracy (10,008/10,010, seed=2025), failing on a single edge case (max + max).

```
Results: 10008/10010 correct (99.98%)
Time: 64.6s (155 additions/sec)
Status: QUALIFIED (threshold: 99%)

Failures (2):
  9999999999 + 9999999999 = 19999999998, got 919999999998
  9999999999 + 9999999999 = 19999999998, got 919999999998
```

## Architecture

1-layer autoregressive decoder. The split-subspace design is from [JackCai's 242p model](https://github.com/JackCai1206/smallest-addition-transformer) — we replaced learned positional encodings with a spiral+correction scheme, saving 16 parameters.

| Component | Params |
|-----------|--------|
| Token embedding (14x3, tied with output head) | 42 |
| Spiral position params (amp, phase, slope, offset) | 4 |
| Per-digit position corrections (10) | 10 |
| Carry position (1x3) | 3 |
| Special token positions (3x3) | 9 |
| Q projection (3->6, no bias) | 18 |
| K projection (3->6, no bias) | 18 |
| V projection (3->6, no bias) | 18 |
| Attention output projection (6->6, no bias) | 36 |
| RMSNorm x3 (weight only, 6 each) | 18 |
| FFN fc1 (6->2, with bias) | 14 |
| FFN fc2 (2->6, with bias) | 18 |
| Output head projection (6->3, no bias) | 18 |
| **Total** | **226** |

Key design choices (all from JackCai's architecture):
- **d_model=6** split into tok_dim=3 + pos_dim=3, concatenated
- **Split attention**: Q, K attend from positional dims; V carries token dims
- **Shared XYZ positions**: same encoding for X[i], Y[i], Z[i]
- **Tied output**: `head_proj(x) @ tok_emb.weight.T` instead of a full linear layer
- **LSB-first** digit ordering for natural carry propagation

Our contributions are the **spiral+correction positions** (14 params instead of 30 learned) and the **[carry-mix curriculum](../carry-mix.md)** — a training-time data sampling strategy that oversamples carry-heavy examples early, then fades them out so the model finishes on the natural distribution.

## Training

- **Optimizer**: AdamW, lr=0.02, weight_decay=0.01, cosine decay
- **[Carry-mix curriculum](../carry-mix.md)**: 30% of training batches are replaced with structured carry-heavy problems (cascading all-9s chains, isolated single carries, power-of-10 boundary crossings, single-digit place additions). This fraction fades linearly to 0% as token accuracy rises from 0.7 to 0.9, so the model finishes on the natural distribution. Addresses the exponential rarity of long carry chains under uniform sampling and accelerates grokking.
- **Grokking**: happened at ~78K steps with seed 420. Exact match jumped from 48% to 100% in about 6K steps.
- Trained on a single GPU in ~7 minutes.

## Verification

```bash
# Using AdderBoard's verify.py
python verify.py submission_226p/submission_226p.py

# Or standalone
python -c "
from submission_226p.submission_226p import build_model, add
model, meta = build_model()
print(f'{meta[\"params\"]} params')
print(add(model, 9999999999, 1))  # 10000000000
"
```

## Credits

This builds directly on prior work:

- **[JackCai1206](https://github.com/JackCai1206/smallest-addition-transformer)** — the 242p split-subspace architecture that this is based on. The core design (split Q/K/V, shared XYZ positions, tied output head, RMSNorm, spiral token init) is all from their work.
- **[rezabyt](https://github.com/rezabyt/digit-addition-311p)** — the 311p model and rank-3 factorization approach that informed our earlier attempts and understanding of grokking dynamics.
- **[Dimitris Papailiopoulos](https://github.com/anadim/AdderBoard)** — for creating the AdderBoard challenge and the original exploration.

Made by Arseniy Zarechnev with help from Claude Code and Codex.
