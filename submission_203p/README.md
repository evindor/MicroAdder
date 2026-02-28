# MicroAdder 203p Submission

203-parameter trained transformer that does 10-digit addition with 100% accuracy (1000/1000, seed=2025).

## Architecture

1-layer autoregressive decoder. The split-subspace design is from [JackCai's 242p model](https://github.com/JackCai1206/smallest-addition-transformer). We applied four compression techniques to reduce from 242p to 203p while maintaining full accuracy.

| Component | Params |
|-----------|--------|
| Token embedding (14x3, tied with output head) | 42 |
| Spiral position params (amp, phase, slope, offset) | 4 |
| Linear position correction (slope, intercept) | 2 |
| Carry position (1x3) | 3 |
| Special token positions (2x3, EOS frozen to zero) | 6 |
| Q projection (3->6, no bias) | 18 |
| K projection (3->6, no bias) | 18 |
| V projection (3->6, no bias) | 18 |
| Attention output projection, rank-2 (6x2 + 2x6) | 24 |
| RMSNorm x3 (weight only, 6 each) | 18 |
| FFN fc1 (6->2, with bias) | 14 |
| FFN fc2 (2->6, with bias) | 18 |
| Output head projection (6->3, no bias) | 18 |
| **Total** | **203** |

Key design choices (from JackCai's architecture):
- **d_model=6** split into tok_dim=3 + pos_dim=3, concatenated
- **Split attention**: Q, K attend from positional dims; V carries token dims
- **Shared XYZ positions**: same encoding for X[i], Y[i], Z[i]
- **Tied output**: `head_proj(x) @ tok_emb.weight.T` instead of a full linear layer
- **LSB-first** digit ordering for natural carry propagation

Our contributions:
- **Spiral+correction positions** (6 params instead of 30 learned): parametric spiral captures base-10 periodicity, linear correction gives per-position magnitude control
- **Rank-2 attention output** (24 params instead of 36): the attention output matrix is effectively rank-2, so we train it factored from the start
- **Frozen EOS position** (0 params instead of 3): EOS special position is always zero after grokking
- **[Carry-mix curriculum](../carry-mix.md)**: 30% of batches replaced with structured carry-heavy problems (all-9s chains, single carries, boundary crossings, place additions), faded to 0% as accuracy rises — accelerates grokking by addressing the long tail of hard cases

## Training

- **Optimizer**: AdamW, lr=0.02, weight_decay=0.01, cosine decay
- **[Carry-mix curriculum](../carry-mix.md)**: 30% of batches are structured carry-heavy problems, faded to 0% as token accuracy rises from 0.7 to 0.9
- **Digit curriculum**: 1-3 digit pairs for 2K steps, 1-6 for 5K, then full 1-10
- **Grokking**: seed 777 grokked. 3/4 seeds grokked overall at 203p.
- Trained on a single GPU.

## Verification

```bash
python verify.py submission_203p/submission_203p.py

# Or standalone
python -c "
from submission_203p.submission_203p import build_model, add
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
