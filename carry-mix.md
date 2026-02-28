# Carry-Mix Curriculum

The carry-mix curriculum is a **training-time data sampling strategy**. It affects only *which examples the model sees during training* — it has zero impact on the model architecture, the forward pass, or the inference loop.

### Core Idea

Addition is easy for most random number pairs but *hard* when carries cascade. Under uniform random sampling, long carry chains (e.g., `9999999999 + 1`) are exponentially rare. The model must grok carry propagation to reach 99%+ accuracy, but it rarely sees the hardest cases. Carry-mix solves this by **oversampling carry-heavy examples early in training**, then fading them out so the model finishes on a natural distribution.

### Two-Level Design

The curriculum operates on two orthogonal axes:

**Axis 1 — Digit Range (outer curriculum):**
| Phase | Steps | Digit Range | Purpose |
|-------|-------|-------------|---------|
| 1 | 0–2,000 | 1–3 digits | Learn basic single-digit addition and simple carries |
| 2 | 2,000–7,000 | 1–6 digits | Extend to medium-length carry chains |
| 3 | 7,000+ | 1–10 digits | Full difficulty, up to 11-digit sums |

Configured via `--curriculum "1-3:2000,1-6:5000,1-10:rest"`.

**Axis 2 — Carry-Mix (inner overlay):**

At each step, a fraction (`carry_mix`, default 0.3 = 30%) of batch examples are replaced with structured carry-heavy problems. The remaining 70% are standard uniform random samples from the current digit range.

### The Four Carry Patterns

Each carry-focused example is randomly assigned one of four patterns (`src/data.py:120-152`):

1. **"single"** — One digit position forces a carry (both digits ≥ 5 at that position, ≤ 4 elsewhere)
   - Example: `00050 + 00070 = 00120` — carry at position 1 only
   - Teaches: isolated carry detection

2. **"chain"** — All-9s number plus a small addend, creating a cascading carry
   - Example: `9999999999 + 7 = 10000000006` — carry propagates through all 10 digits
   - Teaches: long-range carry propagation

3. **"place"** — A full random number plus a single non-zero digit at one position
   - Example: `34567 + 10000 = 44567`
   - Teaches: position-independent digit arithmetic

4. **"boundary"** — A number just below a power of 10, plus a small addend
   - Example: `9999993 + 12 = 10000005` — crosses the 10^7 boundary
   - Teaches: overflow at digit-length boundaries

### Adaptive Fading Schedule

The carry-mix fraction is not constant — it fades based on the model's token accuracy (`src/train.py:43-60`):

```
tok_acc ≤ 0.7  →  full carry_mix (30%)
0.7 < tok_acc < 0.9  →  linear fade from 30% → 0%
tok_acc ≥ 0.9  →  0% (pure uniform sampling)
step ≥ 100,000  →  0% (hard cutoff)
```

**Rationale:** Early on, the model needs exposure to carry mechanics. Once it achieves ~70% token accuracy, it has learned the basics and should transition toward the natural distribution. By 90% accuracy, carry-mix is fully off — the model grokks the remaining cases from uniform data alone.

### Example Training Timeline

```
Step     0: curriculum=1-3 digits, carry_mix=0.30, tok_acc=0.05
Step  1000: curriculum=1-3 digits, carry_mix=0.30, tok_acc=0.45
Step  3000: curriculum=1-6 digits, carry_mix=0.30, tok_acc=0.65
Step  5000: curriculum=1-6 digits, carry_mix=0.30, tok_acc=0.68
Step  8000: curriculum=1-10 digits, carry_mix=0.28, tok_acc=0.72  ← fade begins
Step 15000: curriculum=1-10 digits, carry_mix=0.10, tok_acc=0.82
Step 30000: curriculum=1-10 digits, carry_mix=0.00, tok_acc=0.91  ← fade complete
Step 78000: curriculum=1-10 digits, carry_mix=0.00, tok_acc=1.00  ← grokked
```

### Why It Works

1. **Addresses the long tail** — Uniform sampling under-represents the hardest ~1% of addition problems. Carry-mix guarantees the model sees them regularly.
2. **Doesn't distort the final distribution** — The fade ensures the model's final training is on uniform data, so it doesn't overfit to carry-heavy patterns.
3. **Composes cleanly with digit curriculum** — The outer digit progression builds complexity gradually; carry-mix ensures each digit range is learned *thoroughly* before moving on.
4. **Accelerates grokking** — The submissions (203p and 226p) both achieved 100% accuracy with this method, demonstrating it helps small models find generalizing solutions faster.

### Results

Both submitted models used carry-mix curriculum:
- **203 params** — 100% accuracy, grokked at ~78K steps (seed 777)
- **226 params** — 100% accuracy, grokked at ~78K steps (seed 420)

### Credit
Original idea by Arseniy Zarechnev. Implementation with Claude Code and Codex.
