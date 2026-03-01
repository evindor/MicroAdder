# Structural Analysis of 170p Checkpoint

Diagnostics run on `submission_170p/checkpoint_170p.pt` (100% accuracy, 10010/10010 on AdderBoard).
Script: `diagnostics_170p.py`

---

## 1. Token Embedding Geometry

14 tokens embedded in 2D (tok_dim=2). Tied with output head — embeddings double as classification centroids.

```
   Token |     dim0     dim1 |   norm |   angle
--------------------------------------------------
       0 |   0.8237  -2.4393 |  2.575 |   -71.3°
       1 |   1.6545  -1.7787 |  2.429 |   -47.1°
       2 |   2.1099  -1.2656 |  2.460 |   -31.0°
       3 |   2.4182  -0.7553 |  2.533 |   -17.3°
       4 |   2.5908  -0.2124 |  2.600 |    -4.7°
       5 |   2.6024   0.3652 |  2.628 |     8.0°
       6 |   2.4360   0.9194 |  2.604 |    20.7°
       7 |   2.1178   1.4225 |  2.551 |    33.9°
       8 |   1.5829   1.9504 |  2.512 |    50.9°
       9 |   0.5531   2.5930 |  2.651 |    78.0°
    PLUS |   0.9489  -1.0862 |  1.442 |   -48.9°
  EQUALS |  -2.6499   1.8773 |  3.248 |   144.7°
     EOS |  -1.6019  -2.9403 |  3.348 |  -118.6°
     PAD |  -1.1785  -0.2972 |  1.215 |  -165.8°
```

**Key findings:**
- Digits 0-9 form a ~149° arc (from -71° to +78°), not a full circle
- Angular spacing: ~13° in the middle (digits 3-6), ~24-27° at extremes (0-1, 8-9)
- Tightest classification margins: digits 3-6 (margin ~0.28-0.29)
- PLUS sits dangerously close to digit 1: angle -48.9° vs -47.1°, separated only by norm (1.44 vs 2.43)
- EQUALS, EOS, PAD well-separated (margins >1.1) in opposite quadrants from digits

**Pairwise distances (digits):**
- Closest: 3↔4 (0.570), 4↔5 (0.578), 5↔6 (0.579)
- Farthest: 0↔9 (5.040)

---

## 2. Attention Patterns

Q-phase angles: [-35.3°, -10.3°] (head 0, head 1)

### Head 0: "Look-ahead" Head
For answer position A_i, attends ~49%/49% to **X_{i+2}** and **Y_{i+1}**:

```
A0 → X2(49%) + Y1(49%)
A1 → X3(49%) + Y2(49%)
A2 → X4(49%) + Y3(49%)
A3 → X5(49%) + Y4(49%)
A4 → X6(49%) + Y5(49%)
A5 → X7(49%) + Y6(49%)
A6 → X8(49%) + Y7(49%)
A7 → X9(49%) + Y8(49%)
A8 → Zhi(97%)             ← carry position
A9 → self(100%)           ← pure self-attention
```

This is a **carry-lookahead** pattern. The model reads digits 1-2 positions ahead to predict incoming carries, rather than chaining through previous answer positions.

### Head 1: "Current Triple" Head
For A_i, splits ~32%/32%/32% across **X_{i+1}**, **Y_i**, and **self**:

```
A0 → X1(33%) + Y0(32%) + self(33%)
A1 → X2(32%) + Y1(32%) + self(32%)
...
A7 → X8(33%) + Y7(33%) + self(33%)
A8 → X9(32%) + Y8(32%) + self(32%)
A9 → self(100%)
```

Reads the current Y digit, next X digit, and own residual.

### Key insight: Neither head does carry-chaining
No A_{i-1} → A_i attention. Carry propagation is NOT sequential through answer tokens. The model computes carries from lookahead on input digits, implemented as fixed positional offsets.

### Attention is fixed positional routing
The patterns are **perfectly regular shifts**, not content-dependent. Head 0 always does 50/50 on two specific offset positions. Head 1 always does 33/33/33 on three. The 26 params in q_proj + q_phase are learning a simple offset function.

### Specialization summary
```
Head 0: digit-pair=0.007, carry-chain=0.000, self=0.107
Head 1: digit-pair=0.298, carry-chain=0.008, self=0.391
```

### Delimiter attention
```
Head 0: + → X1(97%)    = → X1(49%) + Y0(49%)
Head 1: + → +(47%) + X0(47%)    = → +(32%) + =(32%) + X0(32%)
```

---

## 3. FFN Activation Analysis

FFN: 6→2 (fc1+bias, GELU) → 2→6 (fc2+bias). Bottleneck dim=2.

### Unit 0: Position→Token cross-subspace bridge (carry detector)
```
fc1 reads:  tok=[-0.811, -1.986], pos=[0.020, 0.060, -0.036, -2.637], bias=0.782
            ||w_tok||=2.145, ||w_pos||=2.638 → reads mostly from pos
fc2 writes: tok=[0.336, 1.819], pos=[0.040, -0.061, -0.184, 1.419]
            → writes mostly to tok subspace
```
Reads positional information (where am I?), writes token correction (carry adjustment).

### Unit 1: Token→Token direct transformation
```
fc1 reads:  tok=[0.604, 1.691], pos=[-0.018, 0.004, 0.376, 0.323], bias=0.769
            ||w_tok||=1.795, ||w_pos||=0.496 → reads mostly from tok
fc2 writes: tok=[-0.364, -2.178], pos=[0.093, -0.042, 0.071, 1.359]
            → writes mostly to tok subspace
```
Direct digit transformation.

### Sharpness
- Max |weight| in fc1: 2.637
- Hand-coded models use ~60000 for hard step functions
- Our GELU transitions are soft — effective slope ~2.6x at threshold
- fc2 bias (constant additive to residual): [-0.636, 0.041, -0.475, 0.073, 0.707, -1.139]

---

## 4. Position Encoding Analysis

### Spiral parameters (trained vs init)
```
amp:            +2.9518  (init: +1.000, drift: +195%)
phase:          -1.0631  (init:  0.000, drift: -61°)
slope:          +0.1276  (init: +0.111, drift: +15%)
offset:         -0.7446  (init:  0.000, drift: -0.74)
corr_slope:     +0.0041  (init:  0.000, drift: +0.004)
corr_intercept: +2.3398  (init:  0.000, drift: +2.34)
```

**Not freezeable.** Total drift: 6.12 across 6 params. Amp tripled, phase rotated -61°, correction intercept grew to +2.34 (scaling all positions ~3.3x).

### Digit positions [10x4]
```
pos |     dim0     dim1     dim2     dim3 |   corr |   norm
-----------------------------------------------------------------
  0 |   4.7932  -8.6147  -2.4868   0.0000 |  2.340 | 10.167
  1 |   8.9524  -4.1572  -2.0630   0.0000 |  2.344 | 10.084
  2 |   9.6981   1.9011  -1.6382   0.0000 |  2.348 | 10.018
  3 |   6.7367   7.2473  -1.2124   0.0000 |  2.352 |  9.969
  4 |   1.1917   9.8349  -0.7855   0.0000 |  2.356 |  9.938
  5 |  -4.8226   8.6677  -0.3575   0.0000 |  2.360 |  9.925
  6 |  -9.0073   4.1827   0.0715   0.0000 |  2.364 |  9.931
  7 |  -9.7575  -1.9128   0.5015   0.0000 |  2.369 |  9.956
  8 |  -6.7779  -7.2917   0.9326   0.0000 |  2.373 |  9.999
  9 |  -1.1990  -9.8951   1.3647   0.0000 |  2.377 | 10.060
```

**Critical: dim 3 is ZERO for all digit positions.** The spiral only fills dims 0-2. Dim 3 is dead for digits but alive for special positions:

### Special positions
```
z_hi (carry): [-0.2282,  0.1359, -0.5375, -1.2559]  norm=1.392
PLUS:         [ 2.0737, -3.6982,  1.1445,  0.1785]  norm=4.395
EQUALS:       [-1.2123, -4.6892, -5.3494,  0.7297]  norm=7.253
EOS (frozen): [ 0.0000,  0.0000,  0.0000,  0.0000]
```

z_hi's dim 3 (-1.2559) is its **largest component** — the carry position uses the "dead" dimension as its primary identifier. This means pos_dim=4→3 (d_model=5) drops the carry position's most important dimension.

---

## 5. Effective Rank of All Projections

```
Matrix                    Shape     Singular Values           Eff.Rank  Cond
q_proj                    [6, 4]    6.24  4.25  4.00  3.01   3.86      2.1
v_proj                    [6, 2]    3.33  2.16                1.95      1.5
out_proj.A                [6, 2]    3.48  1.57                1.86      2.2
out_proj.B                [2, 6]    3.45  2.83                1.99      1.2
out_proj (A@B)            [6, 6]    12.01 4.45 ~0 ~0 ~0 ~0   1.79      huge
fc1                       [2, 6]    3.71  1.14                1.73      3.2
fc2                       [6, 2]    2.89  1.96                1.96      1.5
head_proj                 [2, 6]    12.08 11.11               2.00      1.1
tok_emb (digits 0-9)      [10, 2]   6.36  4.98                1.99      1.3
```

**Key findings:**
- **q_proj is genuinely rank-4** (eff_rank=3.86). All 4 singular values contribute. No compression here.
- **fc1 is nearly rank-1** (91.3% energy in SV1, eff_rank=1.73). But rank-1 fails during training (same as out_proj).
- **head_proj is maximally rank-2** (condition=1.1). Both dims fully utilized.
- **out_proj is effectively rank-2** as designed (SV: 12.0, 4.5, then ~0).

### RMSNorm weights
```
ln1: [0.375, 1.613, 1.002, 0.923, 0.961, 2.278]   std=0.60
ln2: [0.743, 1.847, 0.354, -0.295, -0.312, 1.570]  std=0.84
ln_f: [0.380, 11.823, 0.768, 0.021, 0.471, 10.061] std=5.00
```

**ln_f is doing radical feature selection**, not normalization:
- Dims 1, 5 amplified 10-12x
- Dims 3, 4 suppressed to near-zero (~0.02, ~0.47)
- This means the 6D residual is really a 2D output space + 4D scratch space
- All 3 norms are highly specialized (confirming they can't be shared or removed)

---

## Architectural Implications

### 1. Fixed-offset attention → cheaper Q/K possible
Attention is fixed positional routing with regular offsets (Head 0: +2/+1, Head 1: +1/0/self). The 26p Q/K machinery (q_proj=24, q_phase=2) learns what could be expressed as per-head positional offsets (~6-8p). However, ALiBi (which tried this) failed — likely because it uses distance-based slopes, not group-aware offsets (the model needs different shifts for X-group vs Y-group positions).

### 2. Carry-lookahead, not carry-chaining
The model implements hardware-style carry lookahead: position i reads digits i+1 and i+2 to predict carries. No A_{i-1}→A_i attention. This is optimal for 1-layer autoregressive.

### 3. pos_dim=3 risks the carry circuit
Dim 3 is the carry position's primary identifier. Shrinking pos_dim from 4→3 forces the carry position to share dimensions with digit positions, potentially degrading carry detection. This may explain d_model=5's plateau at 8-12% exact.

### 4. PLUS/digit-1 collision → vocab=10 motivation
PLUS and digit 1 are 1.8° apart in embedding space. Removing PLUS from the vocabulary eliminates this near-collision and frees angular space for digit separation.

### 5. ln_f as output selector
The final norm's extreme weights (0.02 to 11.8) show the model uses only 2 of 6 residual dims for output. The other 4 are internal scratch. This is relevant for d_model shrink — at d_model=4, the model has only 2 dims of scratch after reserving 2 for output.

### 6. Cross-subspace FFN is the carry bridge
FFN unit 0 reads position, writes token — this is how carry information (positional) becomes a digit correction (token). This pos→tok bridge is essential and must be preserved in any compression.
