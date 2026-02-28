# MicroAdder 170p Submission

170-parameter trained transformer that does 10-digit addition with 100% accuracy.

## Architecture

1-layer autoregressive decoder. The split-subspace design is from [JackCai's 242p model](https://github.com/JackCai1206/smallest-addition-transformer). We applied six compression techniques to reduce from 242p to 170p while maintaining full accuracy.

| Component | Params |
|-----------|--------|
| Token embedding (14x2, tied with output head) | 28 |
| Spiral position params (amp, phase, slope, offset) | 4 |
| Linear position correction (slope, intercept) | 2 |
| Carry position (1x4) | 4 |
| Special token positions (2x4, EOS frozen to zero) | 8 |
| Q projection (4->6, no bias, shared with K) | 24 |
| Q-phase angle (per-head rotation for Q/K asymmetry) | 2 |
| V projection (2->6, no bias) | 12 |
| Attention output projection, rank-2 (6x2 + 2x6) | 24 |
| RMSNorm x3 (weight only, 6 each) | 18 |
| FFN fc1 (6->2, with bias) | 14 |
| FFN fc2 (2->6, with bias) | 18 |
| Output head projection (6->2, no bias) | 12 |
| **Total** | **170** |

Key design choices (from JackCai's architecture):
- **d_model=6** split into tok_dim=2 + pos_dim=4, concatenated
- **Split attention**: Q, K attend from positional dims; V carries token dims
- **Shared XYZ positions**: same encoding for X[i], Y[i], Z[i]
- **Tied output**: `head_proj(x) @ tok_emb.weight.T` instead of a full linear layer
- **LSB-first** digit ordering for natural carry propagation

Our contributions:
- **tok_dim=2, pos_dim=4 reshape** (saves 17p vs tok_dim=3, pos_dim=3): digit embeddings have 96% energy in 2D. Shrinking tok_dim from 3 to 2 reduces tok_emb, v_proj, and head_proj while expanding q_proj slightly. Net saving of 17 parameters.
- **Tied Q/K with per-head phase rotation** (2 params instead of 18 for separate K projection): Q and K share the same projection. A learnable per-head angle rotates pairs of Q dimensions — `Q_rot = Q*cos(θ) - Q_swap*sin(θ)` — giving each head a unique "viewing angle" on the shared key space. This provides the asymmetry carry routing requires with just 2 parameters.
- **Spiral+correction positions** (6 params instead of 30 learned): parametric spiral captures base-10 periodicity, linear correction gives per-position magnitude control
- **Rank-2 attention output** (24 params instead of 36): the attention output matrix is effectively rank-2, so we train it factored from the start
- **Frozen EOS position** (0 params instead of 4): EOS special position is always zero after grokking
- **Adaptive weight decay**: WD drops from 0.01 to 0.001 to 0.0001 at grokking onset
- **[Carry-mix curriculum](../carry-mix.md)**: 30% of batches replaced with structured carry-heavy problems, faded to 0% as accuracy rises

## Training

- **Optimizer**: AdamW, lr=0.02, cosine decay
- **Adaptive weight decay**: starts at 0.01, drops at val_exact thresholds (0.2, 0.5)
- **[Carry-mix curriculum](../carry-mix.md)**: 30% of batches are structured carry-heavy problems, faded to 0% as token accuracy rises from 0.7 to 0.9
- **Digit curriculum**: 1-3 digit pairs for 2K steps, 1-6 for 5K, then full 1-10
- **Grokking**: seed 80085 grokked at ~15K steps. Seed-sensitive (1/3 seeds grok).
- Trained on a single GPU.

## Path from 187p to 170p

```
187p  Previous best (tied Q/K + q-phase, tok_dim=3, pos_dim=3)
  |
170p  + tok_dim=2, pos_dim=4 reshape (-17p)
      - tok_emb: 14x3 -> 14x2 (-14p)
      - v_proj: 3->6 -> 2->6 (-6p)
      - head_proj: 6->3 -> 6->2 (-6p)
      - q_proj: 3->6 -> 4->6 (+6p)
      - z_hi_pos: 1x3 -> 1x4 (+1p)
      - special_pos: 2x3 -> 2x4 (+2p)
      - Seed-sensitive: 1/3 seeds grok (80085 at 15K steps)
```

## Verification

```bash
python verify.py submission_170p/submission_170p.py

# Or standalone
python -c "
from submission_170p.submission_170p import build_model, add
model, meta = build_model()
print(f'{meta[\"params\"]} params')
print(add(model, 9999999999, 1))  # 10000000000
"
```

## Credits

- **[JackCai](https://github.com/jackcai)** — the 242p split-subspace architecture that this builds on
- **[rezabyt](https://github.com/rezabyt)** — the 311p model and rank-3 factorization approach
- **[Dimitris Papailiopoulos](https://github.com/anadim)** — for creating the AdderBoard challenge

Made by Arseniy Zarechnev with help from Claude Code and Codex.
