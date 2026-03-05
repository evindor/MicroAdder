# 57p Model: Complete Structural Analysis

Weight-by-weight analysis of the 57-parameter trained transformer that achieves 100% accuracy on 10-digit addition. Based on the seed 777 checkpoint from `sub100_57p_tiefc2`.

## Method

All analysis was performed by loading the trained checkpoint and tracing the forward pass manually, step by step. Key techniques:

- **Weight inspection**: Extracting raw parameter values, computing norms, cosine similarities, SVDs, and subspace energy ratios (e.g. what fraction of a weight's energy lies in the tok vs pos subspace).
- **Attention pattern analysis**: Computing Q/K scores and softmax weights for specific inputs to determine which positions attend to which.
- **End-to-end tracing**: Running the full forward pass on specific examples (7+8=15, 999+1=1000, 5555555555+5555555555) with instrumented intermediates at every layer.
- **GELU gate mapping**: Cataloguing fc1 hidden values across many (digit_sum, prev_z) combinations to identify the carry detection mechanism.
- **Comparison with 67p**: Contrasting weight structure to understand how the triple-duty constraint changed the algorithm.

## The Algorithm

The model performs addition through **column-parallel digit routing + autoregressive carry propagation**:

1. Attention routes each output position Z_i to the same-column inputs X_i and Y_i
2. The value pathway extracts digit identity from those inputs
3. out_proj injects a "digit sum signal" into the residual
4. The FFN's GELU gate acts as a carry detector, reading both the current digit pair and the previous Z output
5. The output head reads the corrected residual to predict the answer digit

This is fundamentally different from the 67p model's carry-lookahead algorithm. The triple-duty constraint on head_proj forced fc1 to read from the tok subspace (99.6%) rather than bridging pos→tok, resulting in autoregressive carry propagation through Z_{i-1}'s output token.

## Block-by-Block Breakdown

### 1. Token Embeddings (3p)

```
Parameters: A=9.4662, start=0.1524, stride=0.1393
Formula: tok(d) = A * [cos(start + d*stride), sin(start + d*stride)]
```

All 10 digits live on a **circle of radius 9.47** in 2D tok space. Angular stride = 7.98°/digit, total arc span = 71.85°. The circle parameterization encodes all 10 digits with just 3 parameters (A=B tied).

```
digit 0:   +8.7°  emb=[+9.36, +1.44]
digit 1:  +16.7°  emb=[+9.07, +2.72]
digit 2:  +24.7°  emb=[+8.60, +3.96]
digit 3:  +32.7°  emb=[+7.97, +5.11]
digit 4:  +40.7°  emb=[+7.18, +6.17]
digit 5:  +48.6°  emb=[+6.25, +7.11]
digit 6:  +56.6°  emb=[+5.21, +7.91]
digit 7:  +64.6°  emb=[+4.06, +8.55]
digit 8:  +72.6°  emb=[+2.83, +9.03]
digit 9:  +80.6°  emb=[+1.55, +9.34]
```

### 2. Positions (6p learned + 0p frozen)

**Sinusoidal (frozen, 0p):**
```
pos(i) = [3.5·sin(0.15i), 3.5·cos(0.15i), 0.15i]
```
A smooth spiral in 3D pos space. The linear ramp (dim 2) provides monotonic position discrimination.

**Learned positions:**
- `z_hi_pos` (3p): `[31.14, -15.48, -6.41]`, norm=35.36. Massive norm acts as a **negative attractor** in attention — carry positions get scores around -800, completely suppressed. This prevents Z positions from attending to the carry token.
- `equals_pos` (3p): `[3.46, -1.04, 0.90]`, norm=3.73. Close to spiral(10.5) with cos=0.95.
- PLUS, EOS: frozen at `[0, 0, 0]`.

### 3. Attention — Column Routing (13p)

**Q/K projection** (12p): shape (4, 3), maps pos_dim → qk_dim.
```
q_proj = [[ 2.53,  0.85, -0.19],
          [-2.50,  1.20,  0.44],
          [ 0.74, -3.20,  0.41],
          [ 2.20,  1.46,  0.35]]
SVD singular values: [4.25, 3.81, 0.70]
```
Near rank-2 structure (3rd singular value 6× smaller).

**Phase rotation** (1p): 0.477 rad = 27.3°. Applied to Q only, breaking Q/K symmetry so Q(pos_i) ≠ K(pos_i).

**Routing result:** Z_i at pos (21+i) attends ~43% to X_i at pos (i) and ~43% to Y_i at pos (11+i). The sinusoidal positions create the pattern where the phase-rotated Q at output positions matches K at the corresponding input positions (same digit column).

```
Z0 (pos 21) attention:
  X0: 0.430    Y0: 0.430    EQ: 0.136    (rest ~0)

Z1 (pos 22) attention:
  X1: 0.428    Y1: 0.428    Z0: 0.048    Y0: 0.048    X0: 0.048

Z5 (pos 26) attention:
  X5: 0.434    Y5: 0.434    (neighbors ~0.04 each)
```

Input digits A_i and B_i get essentially zero attention from non-corresponding Z positions (e.g., Z1's attention to X0=7 is only 0.048).

### 4. Value Pathway (10p, shared with output head and fc2)

```
head_proj.weight (2, 5):
  [[-4.94,  1.85,  0.32,  1.11,  2.71],
   [ 2.52, -4.43, -0.25,  0.89,  2.47]]
```

**V = tok_h @ head_proj.weight** — head_proj does duty #1 as the V projection.

`tok_h = norm(x)[:2]` extracts only the 2D tok subspace. So V encodes **digit identity only** — position affects magnitude (through RMS normalization) but not direction.

The weighted attention sum `0.43·V(X_i) + 0.43·V(Y_i)` produces a 5D vector encoding the digit pair. Since V is linear in tok_h, this is effectively V(0.43·tok(A_i) + 0.43·tok(B_i)) — a point on the circle determined by the angle midpoint of A_i and B_i.

### 5. out_proj — Rank-1 Residual Update (10p)

```
A (5,1): [ 0.79, -1.11, -0.28, -0.08, -0.11]
B (1,5): [ 1.21, -1.16, -0.05,  0.05,  0.03]
```

`out_proj(attn_out) = (attn_out · A) × B`

A scalar gate: dot the 5D attention output with A (95% tok energy), multiply by direction B (99.8% tok energy). This injects the "digit sum signal" into the residual stream along a fixed tok-subspace direction.

### 6. FFN — The Carry Computer (10p fc1 + 0p fc2)

```
fc1.weight (2, 5):
  [[ 0.68, -0.39, -0.06,  0.02,  0.04],
   [-0.58,  0.75, -0.02,  0.02, -0.01]]

fc2 = head_proj.T (tied, 0p) — head_proj duty #3
```

**This is the heart of the model.**

`Δx = head_proj.T @ GELU(fc1 @ norm2(x_post_attn))`

fc1 reads **99.6% from tok subspace**. This is a key difference from 67p where fc1 was the critical pos→tok bridge. In 57p, carry info arrives through the autoregressive Z chain, not through position.

**fc1 is anti-correlated with head_proj** (cos=-0.83 for row 0, cos=-0.88 for row 1). The interaction matrix fc1 @ head_proj.T has singular values [8.27, 0.46] — near rank-1. The FFN is approximately a 1D nonlinear function applied along the head_proj direction.

**GELU as binary carry gate:**

The 2D hidden space has three regimes determined by the GELU activation:

| fc1 hidden | GELU output | Interpretation |
|---|---|---|
| `[+5.8, -6.5]` | `[5.8, 0.0]` | **Carry-generating**: digit sum ≥ 10 (e.g., 7+8=15, 9+9=18) |
| `[-5.3, +6.8]` | `[0.0, 6.8]` | **No-carry**: digit sum < 10 or carry absorption |
| `[+2.5, -1.0]` | `[2.5, -0.2]` | **Boundary**: digit sum = 10 exactly (e.g., 9+1, 5+5) |

The two channels push the residual in opposite directions through fc2=head_proj.T, implementing a carry-dependent correction.

**Carry detection examples:**

```
7+8=15 (carry from Z0):
  Z0: A=7+B=8  fc1=[+5.76,-6.46] GELU=[5.76,0.00] →5  (carry out)
  Z1: A=0+B=0  fc1=[-5.23,+6.82] GELU=[0.00,6.82] →1  (carry absorbed)

3+4=7 (no carry):
  Z0: A=3+B=4  fc1=[-4.40,+6.35] GELU=[0.00,6.35] →7  (no carry)

999+1=1000 (carry chain):
  Z0: A=9+B=1  fc1=[+2.43,-0.79] GELU=[2.41,-0.17] →0  (boundary, carry out)
  Z1: A=9+B=0  fc1=[+2.73,-1.26] GELU=[2.72,-0.13] →0  (carry propagates)
  Z2: A=9+B=0  fc1=[+2.55,-0.84] GELU=[2.53,-0.17] →0  (carry propagates)
  Z3: A=0+B=0  fc1=[-5.32,+6.87] GELU=[0.00,6.87] →1  (carry absorbed)

5555555555+5555555555 (all carries):
  Z0: A=5+B=5  fc1=[+2.92,-1.44] GELU=[2.92,-0.11] →0  (boundary)
  Z1: A=5+B=5  fc1=[+4.90,-4.47] GELU=[4.90,0.00]  →1  (carry in + sum=10 → carry out)
  Z2: A=5+B=5  fc1=[+4.78,-4.09] GELU=[4.78,0.00]  →1  (same pattern propagates)
```

### 7. Carry Propagation Mechanism

The carry is **NOT** computed through attention lookahead (as in the 67p model). Instead:

1. Z_i's **input token** is Z_{i-1}'s output (autoregressive / teacher-forced)
2. This previous digit value enters the residual at Z_i's position as a tok embedding
3. Attention mixes it with V(X_i) + V(Y_i) from the same column
4. fc1 sees the **combined signal**: current digit pair + previous output value
5. GELU gates determine carry in / carry out

The key insight: a "0" from a carry-producing position looks different in the residual than a "0" from a non-carry position. The attention correction and out_proj leave different traces depending on the digit pair sum, and fc1 reads these traces to detect carries.

For carry chain propagation (999+1): the "0" output from Z0 carries implicit information through its embedding that gets combined with 9+0 at Z1. The fc1 hidden values stay in the boundary regime `[+2.5, -1.0]` throughout the carry chain, only switching to the no-carry regime `[-5.3, +6.8]` when the chain terminates.

### 8. Normalization (5p)

```
Shared RMSNorm weight: [3.34, 3.22, 2.92, 3.28, 13.44]
```

Single weight vector shared across all 3 norm sites (norm1, norm2, norm_f). Dim 4 (position ramp) is amplified 4.2× relative to dims 0-3. This amplifies the linear ramp dimension, making position discrimination sharper for the attention routing.

Dims 0-3 vary by ~15% (range 2.92-3.34), which is why the `structured2` norm experiment with `[b,b,b,b,b+d]` failed — forcing dims 0-3 identical removed needed expressivity.

### 9. Output Head (0p, shared)

```
logits = head_proj @ norm_f(x_final) @ tok_emb.T
```

Head_proj duty #2: extracts a 2D signal from the final residual, dots with each digit's embedding on the circle. The digit with maximum dot product wins.

head_proj row directions: row 0 at 159.4°, row 1 at -60.3° in tok space.

## Parameter Budget

| Component | Params | Role |
|---|---|---|
| tok_arc (A, start, stride) | 3 | Circle embedding for 10 digits |
| z_hi_pos | 3 | Carry position (attention suppressor, norm=35) |
| special_pos_equals | 3 | EQUALS position (≈spiral(10.5)) |
| q_proj | 12 | Column routing (pos → qk_dim) |
| q_phase_angle | 1 | Q/K symmetry breaking (27.3°) |
| out_proj.A | 5 | Rank-1 gate: scalar extraction |
| out_proj.B | 5 | Rank-1 gate: write direction |
| fc1.weight | 10 | Carry detector (GELU gate, reads tok) |
| head_proj.weight | 10 | **Triple duty**: V projection, output head, fc2 |
| norm1.weight (shared) | 5 | RMSNorm, shared across all 3 sites |
| **Total** | **57** | |

## Key Structural Properties

Properties relevant for further compression attempts:

1. **out_proj.B is 99.8% tok subspace**: `[1.21, -1.16, -0.05, 0.05, 0.03]`. Constraining to `[b1, b2, 0, 0, 0]` saves 3p but proved hard to train.

2. **equals_pos ≈ spiral(10.5)**: cos=0.95. Freezing saves 3p but may be too tight.

3. **fc1 anti-correlated with head_proj**: cos ≈ -0.85. Tying fc1 = M @ head_proj saves 6p but the model failed to learn the mixing matrix M from scratch (best: 58.6% at 51p with 2×2 M).

4. **q_proj near rank-2**: singular values [4.25, 3.81, 0.70]. The 3rd dimension contributes little.

5. **Norm dims 0-3 vary by 15%**: The `[b,b,b,b,b+d]` structured norm (saving 3p) reached 93.5% but couldn't close the gap — the model needs that small variation.

6. **fc1 is the bottleneck for compression**: Every attempt to reduce fc1's freedom (tying to head_proj, reducing to scalar) broke carry detection. The GELU gate needs precise thresholds that can't be derived from head_proj's weights alone.

## Comparison with 67p Algorithm

| Aspect | 67p | 57p |
|---|---|---|
| Carry mechanism | Lookahead through attention | Autoregressive through Z_{i-1} token |
| fc1 reads | Position subspace (pos→tok bridge) | Token subspace (99.6%) |
| FFN role | Position-dependent carry detection | Token-dependent carry gating |
| fc2 | Independent (10p) | Tied to head_proj.T (0p) |
| head_proj duties | 2 (V proj, output head) | 3 (V proj, output head, fc2) |

The triple-duty constraint didn't just save 10 parameters — it forced the model to discover a fundamentally different algorithm for carry propagation.
