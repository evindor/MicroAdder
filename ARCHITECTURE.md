# Architecture of the 74-Parameter MicroAdder

A detailed analysis of how a 74-parameter, single-layer transformer achieves 100% accuracy on 10-digit addition (10,010/10,010 test cases on AdderBoard).

**Checkpoint:** `results/runs/sub100_74p_cm80_sf_120k_s45214_s45214/checkpoints/best.pt`
**Submission:** `submission_74p/submission_74p.py`

---

## 1. Overview

The MicroAdder is a 1-layer decoder-only transformer with `d_model=5`, split into a 2D **token subspace** and a 3D **position subspace**. It performs 10-digit decimal addition in LSB-first (least-significant-bit first) order using autoregressive generation.

### Sequence Format

The input sequence has 34 positions total:

```
Positions  0-9:   X[0]..X[9]   (first number, LSB first)
Position  10:     PLUS          (separator, token=0)
Positions 11-20:  Y[0]..Y[9]   (second number, LSB first)
Position  21:     EQUALS        (separator, token=0)
Positions 22-31:  Z[0]..Z[9]   (answer digits, LSB first)
Position  32:     Z[10]         (carry-out / 11th digit)
Position  33:     EOS           (end of sequence)
```

### Next-Token Prediction Convention

The model uses standard next-token prediction: the output at position `t` predicts the token at position `t+1`. During autoregressive generation:
- Position 21 (EQUALS) predicts Z[0]
- Position 22+i (Z[i]) predicts Z[i+1]
- Position 31 (Z[9]) predicts Z[10] (carry-out)
- Position 32 (Z[10]) predicts EOS

### The 5D Residual Stream

Each position's hidden state is a 5D vector:

```
x = [tok_dim0, tok_dim1 | pos_dim0, pos_dim1, pos_dim2]
     ^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     Token subspace (2D)   Position subspace (3D)
```

The token and position subspaces serve completely different roles:
- **Token dims (0, 1):** Carry digit identity information; read by the V projection and output head
- **Position dims (2, 3, 4):** Carry position information; read by Q/K projections for attention routing

### Parameter Budget (74 total)

| Component | Parameters | Description |
|---|---|---|
| tok_arc (A, start, stride) | 3 | Circular token embedding (A=B tied) |
| spiral (amp, phase, slope, offset) | 4 | Spiral positional encoding |
| z_hi_pos | 3 | Carry-out position vector |
| special_pos_equals | 3 | EQUALS token position |
| q_phase_angle | 1 | Q/K asymmetry rotation |
| q_proj | 15 | Shared Q/K projection (3 -> 5) |
| out_proj | 10 | Rank-1 attention output (5x1 + 1x5) |
| FFN fc1 | 10 | First FFN layer (5 -> 2, no bias) |
| FFN fc2 | 10 | Second FFN layer (2 -> 5, no bias) |
| head_proj | 10 | Output/V projection (5 -> 2) |
| RMSNorm (shared) | 5 | Single weight vector for all 3 norms |
| **Total** | **74** | |

### Forward Pass Summary

```
1. Embed:     x = [tok_table[token] | position_vector]        (2D + 3D = 5D)
2. Norm1:     h = RMSNorm(x)
3. Attention: Q,K from h[:, 2:5] (position); V from h[:, 0:2] (token)
4. Out_proj:  x += rank1_proj(attention_output)                (writes to dim1)
5. Norm2:     h2 = RMSNorm(x)
6. FFN:       x += fc2(GELU(fc1(h2)))                         (carry detection)
7. Norm3:     h3 = RMSNorm(x)
8. Output:    logits = head_proj(h3) @ tok_table.T             (2D -> 10 logits)
```

---

## 2. Digit Representation: The Circular Token Embedding

Digits 0-9 are embedded as points on a circular arc in 2D space, parameterized by just 3 values:

```
emb(d) = [A * cos(start + d * stride), A * sin(start + d * stride)]
```

### Learned Parameters

| Parameter | Value | Description |
|---|---|---|
| A (radius) | 12.9863 | Circle radius (A=B tied, so it is a perfect circle) |
| start | -0.5325 rad (-30.51 deg) | Starting angle for digit 0 |
| stride | 0.1213 rad (6.95 deg) | Angular spacing between consecutive digits |
| total arc | 1.0914 rad (62.53 deg) | Arc span from digit 0 to digit 9 |

### Digit Coordinates

| Digit | x | y | Angle |
|---|---|---|---|
| 0 | 11.188 | -6.594 | -30.51 deg |
| 1 | 11.903 | -5.192 | -23.56 deg |
| 2 | 12.444 | -3.714 | -16.62 deg |
| 3 | 12.802 | -2.181 | -9.67 deg |
| 4 | 12.972 | -0.616 | -2.72 deg |
| 5 | 12.951 | 0.957 | 4.23 deg |
| 6 | 12.740 | 2.517 | 11.18 deg |
| 7 | 12.342 | 4.040 | 18.12 deg |
| 8 | 11.763 | 5.503 | 25.07 deg |
| 9 | 11.010 | 6.886 | 32.02 deg |

### Key Properties

1. **Uniform angular spacing:** All consecutive digits are separated by exactly 6.95 degrees. This means the chord distance between digit `d` and digit `d+k` depends only on `k`, not on `d` -- a translation-invariant property that helps the model treat all digits equivalently.

2. **Monotonic y-coordinate:** Digit 0 has the lowest y-value (-6.594) and digit 9 has the highest (6.886). This monotonic relationship between digit value and the y-coordinate is critical for the carry detection mechanism (see Section 6).

3. **Pairwise distances are Toeplitz:** The distance between digits `i` and `j` depends only on `|i-j|`:

```
|i-j|:  0     1      2      3      4      5      6      7      8      9
dist:  0.00  1.574  3.142  4.699  6.238  7.754  9.242  10.696  12.110  13.480
```

### Decision Boundaries

The output logit for digit `d` is `<head_proj(norm(x)), tok_table[d]>`. Since all digit embeddings lie on a circle of radius A=12.986, the decision boundary between any two digits is the perpendicular bisector of the chord connecting them. The equal spacing means these bisectors are evenly spaced, giving each digit an equal-width "wedge" of the 2D token space.

---

## 3. Position Encoding: The Spiral and Special Positions

Positions are encoded in the 3D position subspace. There are four types of positions:

### 3.1 Digit Positions (Spiral)

The 10 digit positions (shared across X, Y, and Z) lie on a 3D spiral:

```
pos[i] = [amp * cos(2*pi*i/10 + phase),
          amp * sin(2*pi*i/10 + phase),
          slope * i + offset]
```

| Parameter | Value |
|---|---|
| amp | 3.5630 |
| phase | -0.4420 rad (-25.33 deg) |
| slope | 0.1694 |
| offset | -2.5476 |

The spiral is **nearly flat**: the z-range is [-2.548, -1.023] (span of 1.524), while the xy-amplitude is 3.563. The ratio `slope*9 / amp = 0.43`, meaning the helix is stretched wide relative to its height. In the xy-plane, the 10 positions are evenly distributed around a circle at 36-degree intervals.

Digit position vectors:

| Position | x | y | z | Norm |
|---|---|---|---|---|
| pos[0] | 3.221 | -1.524 | -2.548 | 4.380 |
| pos[1] | 3.501 | 0.660 | -2.378 | 4.284 |
| pos[2] | 2.445 | 2.592 | -2.209 | 4.192 |
| pos[3] | 0.454 | 3.534 | -2.040 | 4.105 |
| pos[4] | -1.710 | 3.126 | -1.870 | 4.024 |
| pos[5] | -3.221 | 1.524 | -1.701 | 3.948 |
| pos[6] | -3.501 | -0.660 | -1.531 | 3.878 |
| pos[7] | -2.445 | -2.592 | -1.362 | 3.815 |
| pos[8] | -0.454 | -3.534 | -1.193 | 3.757 |
| pos[9] | 1.710 | -3.126 | -1.023 | 3.707 |

### 3.2 Position Sharing: The Critical Design Choice

X[i], Y[i], and Z[i] all share the same position vector `pos[i]`. This means the Q/K attention mechanism cannot distinguish between X[i], Y[i], and Z[i] -- they are "the same position" to the attention routing. This is what allows position-based attention to work with only 10 position vectors instead of 30+.

### 3.3 Special Positions

| Position | Vector | Norm | Notes |
|---|---|---|---|
| PLUS (pos 10) | [0, 0, 0] | 0.000 | Frozen at zero (not learned) |
| EQUALS (pos 21) | [2.644, -2.414, -1.033] | 3.726 | Learned; near pos[0] in direction |
| EOS (pos 33) | [0, 0, 0] | 0.000 | Frozen at zero (not learned) |
| Z[10] carry (pos 32) | [36.013, -32.834, 6.452] | 49.159 | Learned; **12.3x** larger than digit positions |

### 3.4 The Enormous Carry Position

The z_hi carry position has norm 49.16, which is **12.3x** the average digit position norm of 4.01. This extreme scale serves a precise purpose: when multiplied by the Q/K projection, the carry position produces attention scores that are vastly larger than any digit position. This ensures Z[10] (the carry-out digit) attends **100%** to itself, completely ignoring all other positions. The carry position acts as an attention "black hole."

---

## 4. Attention: Positional Routing with Phase Rotation

### 4.1 The Split Attention Design

Attention in this model has a unique split-subspace design:

- **Q and K** are computed from the **position subspace** only (dims 2, 3, 4)
- **V** is computed from the **token subspace** only (dims 0, 1)

This means **attention patterns are 100% determined by positions, independent of input tokens**. The same set of attention weights is used for every possible addition problem. The positions route information; the tokens carry the payload.

### 4.2 Tied Q/K with Phase Rotation

Q and K share the same linear projection (15 parameters for a 3->5 linear map):

```
q_proj weight (5x3):
  [-3.376,  3.042, -0.650]
  [ 0.478, -3.591, -0.211]
  [-2.515, -2.498,  1.458]
  [ 3.495,  1.677, -0.043]
  [ 0.000, -0.000,  0.000]     <-- dim 4 is dead
```

Note: the 5th row is effectively zero, meaning the projection only uses 4 of the 5 head dimensions. This is equivalent to a head_dim=4 model with one wasted dimension.

Without the phase rotation, Q = K exactly, which forces `Q[i] . K[j] = K[i] . K[j]` to be symmetric. This means each position would attend most strongly to **itself** (since `K[i] . K[i]` is maximal by Cauchy-Schwarz).

### 4.3 The Phase Rotation Breaks Self-Attention

The phase angle is **0.7201 rad (41.26 degrees)**. It applies a rotation to Q in the (dim0, dim1) and (dim2, dim3) planes:

```
cos(phase) = 0.7517
sin(phase) = 0.6595
```

This rotation transforms self-attention into **lookahead attention**. The effect is dramatic:

**Without phase (Q=K):** Position Z[0] attends uniformly to {X[0], Y[0], Z[0]} (33.3% each).

**With phase:** Position Z[0] attends to {X[1], Y[1]} (49.96% each), with only 0.02% to X[0]/Y[0]/Z[0].

The phase rotation shifts each position's attention to its **+1 neighbor** on the spiral. Since digit positions are evenly spaced at 36-degree intervals on a circle, a ~41-degree phase rotation maps each position's Q vector to align with the K vector of the next position.

### 4.4 The Fixed Attention Pattern

Since attention depends only on position, we can compute the complete attention map once for all inputs. For autoregressive generation (which determines the actual context each position sees):

| Position | Predicts | Attends to (>0.1%) |
|---|---|---|
| EQUALS (21) | Z[0] | X[0]: 49.94%, Y[0]: 49.94%, EQUALS: 0.12% |
| Z[0] (22) | Z[1] | X[1]: 49.96%, Y[1]: 49.96% |
| Z[1] (23) | Z[2] | X[2]: 50.00%, Y[2]: 50.00% |
| Z[2] (24) | Z[3] | X[3]: 50.00%, Y[3]: 50.00% |
| Z[3] (25) | Z[4] | X[4]: 49.98%, Y[4]: 49.98% |
| Z[4] (26) | Z[5] | X[5]: 49.92%, Y[5]: 49.92% |
| Z[5] (27) | Z[6] | X[6]: 49.97%, Y[6]: 49.97% |
| Z[6] (28) | Z[7] | X[7]: 49.99%, Y[7]: 49.99% |
| Z[7] (29) | Z[8] | X[8]: 50.00%, Y[8]: 50.00% |
| Z[8] (30) | Z[9] | X[9]: 49.98%, Y[9]: 49.98% |
| Z[9] (31) | Z[10] | EQUALS: 99.99% |
| Z[10]/carry (32) | EOS | Z[10]/carry: 100.00% |

**Key patterns:**
1. Each Z[i] position (predicting Z[i+1]) attends ~50/50 to X[i+1] and Y[i+1]
2. Z[9] (predicting Z[10], the carry-out) attends 99.99% to EQUALS
3. Z[10] (predicting EOS) attends 100% to itself (the enormous carry position dominates)

### 4.5 The Q.K/K.K Ratio

The phase rotation uniformly scales the self-attention score:

```
Q[i] . K[i] / (K[i] . K[i]) = cos(phase) = 0.7517  (constant for ALL positions)
```

This means the phase reduces each position's self-attention score by a fixed factor, while the cross-attention scores to neighbor positions are boosted. The 41-degree rotation is optimized to make the +1 neighbor the strongest match.

---

## 5. Value Pathway: Tied V/Output

### 5.1 The head_proj Double Duty

The `head_proj` weight matrix (2x5, 10 parameters) serves two roles simultaneously:

1. **As V projection:** `V = tok_h @ head_proj.weight` (2D token -> 5D attention value)
2. **As output head:** `logits = head_proj(norm(x)) @ tok_table.T` (5D -> 2D -> 10 logits)

```
head_proj.weight (2x5):
  [-21.456,  0.280,  1.058,  2.536,  12.424]
  [ -2.099, -4.889,  0.000,  0.118,  -0.162]
```

### 5.2 V Values Encode Digit Sum

When `head_proj.weight` is applied to the 2D token embedding of digit `d`, the resulting 5D value vector has a critical property: **dimension 1 is a near-linear function of digit value**.

| Digit | V[dim1] |
|---|---|
| 0 | +35.37 |
| 1 | +28.71 |
| 2 | +21.64 |
| 3 | +14.25 |
| 4 | +6.64 |
| 5 | -1.06 |
| 6 | -8.74 |
| 7 | -16.30 |
| 8 | -23.61 |
| 9 | -30.58 |

Linear fit: `V[1](d) = -7.43 * d + 36.07` (max residual from linear: 0.71)

This near-linearity arises because the token embeddings lie on a small arc of a circle, and over a 62.5-degree arc, `cos(theta)` is nearly linear. The slope of -7.43 per digit means:

- **The average V[dim1] of two digits encodes their sum:** `avg_V[1](x, y) = -3.72 * (x + y) + 36.07`
- **The crossover from positive to negative V[1] is at digit 4.85**, right between digits 4 and 5
- **For a pair sum S:** avg_V[1] is positive when S < 10 (no carry) and negative when S >= 10 (carry)

### 5.3 The Rank-1 Output Projection

After attention computes the weighted average of V vectors, the rank-1 output projection compresses the result:

```
out_proj.A (5x1): [-0.018, -1.386,  0.003,  0.022, -0.043]
out_proj.B (1x5): [-0.018, -1.593,  0.036, -0.001,  0.072]
```

The A vector reads **almost exclusively from dimension 1** of the attention output (weight -1.386, all others < 0.043). The B vector writes **almost exclusively to dimension 1** of the residual stream (weight -1.593, all others < 0.072).

The effective gain is: `out_proj_gain = (-1.386) * (-1.593) = 2.208`

So the attention's contribution to the residual stream is:
```
delta_residual[dim1] = 2.208 * (0.5 * V[1](x_{i+1}) + 0.5 * V[1](y_{i+1}))
```

This means: **the attention mechanism computes a scalar proportional to the digit sum at the next position and writes it to dimension 1 of the residual stream.**

### 5.4 Residual Stream After Attention

For a position predicting Z[i+1], the residual stream after attention is approximately:

```
dim0 = tok_table[Z[i]][0]           (token x-coordinate, ~unchanged)
dim1 = tok_table[Z[i]][1] + 2.208 * avg_V[1](x_{i+1}, y_{i+1})
dim2 = pos[i][0]                    (position, ~unchanged)
dim3 = pos[i][1]                    (position, ~unchanged)
dim4 = pos[i][2]                    (position, ~unchanged)
```

The key: **dim1 combines two signals:**
1. The token embedding's y-coordinate (encodes the current digit Z[i])
2. The attention-injected sum information (encodes x_{i+1} + y_{i+1})

---

## 6. Carry Detection: The FFN

The FFN has 2 hidden units with GELU activation and no bias:

```
fc1 (2x5): reads from normalized residual
  Unit 0: [-0.172,  1.394, -0.512, -0.010, -0.869]
  Unit 1: [-0.813, -1.304,  0.012,  0.185,  0.366]

fc2 (5x2): writes to residual
  Unit 0: [-0.727, -1.936,  0.072,  0.226,  1.516]
  Unit 1: [-0.757,  2.108, -0.020,  0.149,  1.478]
```

### 6.1 How the Two Units Specialize

**Unit 0** primarily reads dim1 (weight +1.394) -- the combined token+sum signal. It activates (positive pre-GELU) when dim1 is positive, which happens when:
- The current token Z[i] has a high y-embedding (low digit value, 0-4)
- AND/OR the attention-injected sum is small (no carry in the next position)

When active, Unit 0 writes **negative** values to tok dims: [-0.727, -1.936]. This pushes dim1 strongly negative.

**Unit 1** primarily reads dim0 and dim1 with negative weights [-0.813, -1.304]. It activates when both token dimensions are negative, which corresponds to digits in a specific arc region. When active, Unit 1 writes **positive** to dim1 (+2.108), pushing it upward.

### 6.2 Observed FFN Behavior

**Carry case (e.g., 9999999999 + 1, Z[i]=0 for all i):**
- After attention, dim1 is approximately -4.3 (negative: emb[1](0) = -6.59, plus small positive attention delta)
- FFN: Unit 0 is zero (inactive), Unit 1 activates at ~5.3
- FFN delta to tok dims: [-4.06, +11.30]
- Result: dim1 shifts positive -- the carry signal

**No-carry case (e.g., 4444444444 + 5555555555, Z[i]=9 after Z[0]):**
- After attention, dim1 is approximately +1.7 (positive: emb[1](9) = +6.89, plus negative attention delta)
- FFN: Unit 0 activates at ~3.7, Unit 1 is zero (inactive)
- FFN delta to tok dims: [-2.67, -7.10]
- Result: dim1 shifts negative -- the no-carry signal

**Carry-out detection (Z[9] predicting Z[10], attends to EQUALS):**
- The EQUALS token is always 0, giving V[1] = +35.37 (very positive)
- After attention, dim1 jumps to ~64.4 (token 0 emb[1]=-6.59, but attention delta = +71.01)
- FFN: Unit 0 activates at ~20.7, Unit 1 is zero
- FFN delta: [-15.02, -40.01]
- This massive negative push to dim1 encodes "predict 1" for the carry-out

### 6.3 The Carry Propagation Mechanism

The model propagates carries through autoregressive generation:

1. **Position 21 (EQUALS) predicts Z[0]:** Attends to X[0]/Y[0]. Computes `(x_0 + y_0) mod 10` from the sum encoded in V[dim1]. No carry input needed for the least significant digit.

2. **Position 22+i (Z[i]) predicts Z[i+1]:** The token at this position is Z[i], the **previously predicted digit**. The attention reads X[i+1]/Y[i+1] to get the sum at the next position. The FFN combines:
   - Z[i]'s token embedding (encodes whether there was a carry from position i)
   - The attention signal (encodes x_{i+1} + y_{i+1})

   To determine: Z[i+1] = (x_{i+1} + y_{i+1} + carry_in) mod 10

3. **Carry is encoded in the token value:** If Z[i] is a "low" digit (0-4), it signals that a carry occurred at position i (because the sum was >= 10 and wrapped around). If Z[i] is a "high" digit (5-9), no carry occurred.

This is not perfectly reliable for all digit combinations (e.g., Z[i]=0 could mean the sum was exactly 0 or exactly 10), but the FFN learns to handle the ambiguity using the full 5D residual stream context, including position information.

### 6.4 FFN Hidden Activation Patterns Summary

| Scenario | Unit 0 | Unit 1 | Effect on dim1 |
|---|---|---|---|
| No carry, large sum at i+1 | ~4-10 | 0 | Negative (push toward high digit) |
| No carry, small sum at i+1 | ~15-21 | 0 | Very negative (push toward 0) |
| Carry, any sum at i+1 | 0 | ~5-19 | Positive (add carry, shift digit up) |

---

## 7. Output: Reading the Answer

### 7.1 The Final Norm + Head Projection

The output pipeline is:

```
1. x_final = residual after FFN (5D)
2. h = RMSNorm(x_final)           (normalize, then scale by norm_weight)
3. tok_2d = head_proj(h)           (5D -> 2D: project to token space)
4. logits = tok_2d @ tok_table.T   (2D -> 10: dot product with each digit embedding)
```

### 7.2 The Shared Norm Weight as Feature Gate

The same RMSNorm weight is shared across all three normalization points (pre-attention, pre-FFN, pre-output):

```
norm_weight: [0.401, 6.574, 3.180, 3.114, 4.984]
```

The key asymmetry: **dim0 is suppressed (0.40) while dim1 is amplified (6.57)** -- a 16.4x ratio. This has different effects at different stages:

**At the output head:** The effective contribution of each dimension to the 2D token logit (after norm scaling and head_proj):

| Dim | Raw head_proj col | After norm scaling | Role |
|---|---|---|---|
| dim0 | [-21.46, -2.10] | [-8.61, -0.84] | Suppressed by 0.40x norm |
| dim1 | [0.28, -4.89] | [1.84, -32.14] | **Dominant** via 6.57x norm |
| dim2 | [1.06, 0.00] | [3.37, 0.00] | Position leakage (small) |
| dim3 | [2.54, 0.12] | [7.90, 0.37] | Position leakage (moderate) |
| dim4 | [12.42, -0.16] | [61.92, -0.81] | Position-dependent offset |

**Dim1 is the primary digit discriminator.** After norm amplification, dim1 contributes [-32.14] to the second component of the 2D logit vector. Since digits are arranged on an arc where the y-coordinate increases with digit value, a large negative second-component pushes the logit point toward low digits (near digit 0), while a large positive value pushes toward high digits (near digit 9).

**Dim4 contributes a position-dependent offset** to the first component (61.92x), which shifts all logits equally in the x-direction. Since all digit embeddings have similar x-coordinates (range 11.0 to 13.0), this primarily controls the overall logit scale rather than discrimination.

### 7.3 The 2D Decision Process

The output head projects the 5D normalized residual to a 2D point in the same space as the digit embeddings. The predicted digit is whichever embedding is closest (highest dot product) to this 2D point.

Since the embeddings lie on a 62.5-degree arc of a circle, the decision boundaries are radial lines emanating from the origin. The 2D logit point's **angle** relative to the center of the arc determines the predicted digit.

---

## 8. The Full Picture: End-to-End Walkthrough

### Example: 9999999999 + 1 = 10000000000

This is the hardest test case: a full carry chain propagating through all 10 digits.

```
X (LSB first): [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
Y (LSB first): [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Z (expected):  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

#### Step 0: EQUALS (pos 21) predicts Z[0]

- **Token at pos 21:** 0 (EQUALS separator)
- **Position:** EQUALS = [2.644, -2.414, -1.033]
- **Attention:** 49.94% to X[0]=9, 49.94% to Y[0]=1
- **V signal:** avg_V[1](9, 1) = 0.5 * (-30.58) + 0.5 * (28.71) = -0.93
  - Slightly negative: sum 10 is right at the carry boundary
- **Attention writes to dim1:** 2.208 * (-0.93) = -2.06
- **After attention, dim1:** -6.59 + (-2.06) ≈ -4.34 (but exact value differs due to normalization and position interaction)
- **FFN:** Unit 0 inactive, Unit 1 activates at 5.36 -> adds [−4.06, +11.30] to tok dims
- **Prediction:** digit **0** (correct: 9+1=10, ones digit is 0, carry=1)
- **Logit gap:** 66.0 vs 42.4 in 2D tok space -> confident

#### Steps 1-9: Z[i] (pos 22+i) predicts Z[i+1]

For all steps 1-9, the pattern is identical:
- **Token at position:** Z[i] = 0 (from previous prediction)
- **Attention:** ~50/50 to X[i+1] and Y[i+1]
  - Steps 1-9: X[i+1]=9, Y[i+1]=0 (sum=9)
- **V signal:** avg_V[1](9, 0) = 0.5 * (-30.58) + 0.5 * (35.37) = 2.39
  - Positive: sum 9 is just below carry threshold
- **After attention, dim1:** emb[1](0) + delta ≈ -6.59 + 2.48*2.21 ≈ -4.3
  - Dim1 is negative because the carry-indicating token (Z[i]=0) has emb[1]=-6.59
- **FFN:** Unit 0 inactive, Unit 1 activates at ~5.3 -> adds carry adjustment
  - The carry from the previous position (encoded in Z[i]=0 being a low digit) causes Unit 1 to fire, effectively computing (9+0+1) mod 10 = 0
- **Prediction:** digit **0** at each step (correct)

The carry propagates: Z[i]=0 signals carry -> FFN adds +1 to the next position's sum -> result wraps to 0 -> carry continues.

#### Step 10: Z[9] (pos 31) predicts Z[10] (carry-out)

- **Token at pos 31:** Z[9] = 0
- **Attention:** 99.99% to EQUALS (pos 21), token=0
- **V signal:** V[1](0) = 35.37 (very positive -- EQUALS always has token 0)
- **After attention, dim1:** -6.59 + 2.208 * 35.37 ≈ -6.59 + 78.09 ≈ 64.4
  - **Massive positive dim1** -- the largest value in the entire computation
- **FFN:** Unit 0 activates at 20.67, Unit 1 inactive -> adds [-15.02, -40.01]
  - The enormous FFN correction reshapes the 2D logit point
- **2D tok logit:** [107.74, -48.99]
- **Prediction:** digit **1** (correct: carry-out = 1)
- **Logit gap:** 1522.0 vs 1516.1 (digit 1 vs digit 0) -- only 5.9 gap, the tightest margin

#### Step 11: Z[10] (pos 32) predicts EOS

- **Token:** Z[10] = 1
- **Attention:** 100% to self (the carry position's enormous norm dominates)
- **Prediction:** 0 (EOS token)

### Contrast: 4444444444 + 5555555555 = 9999999999 (No Carries)

- **Z[0]:** EQUALS attends to X[0]=4, Y[0]=5. Sum=9, no carry. Predicts 9.
- **Z[1]-Z[9]:** Each Z[i]=9, attention reads X[i+1]=4, Y[i+1]=5.
  - Token dim1: emb[1](9) = +6.89 (positive -- high digit means no carry)
  - V signal: avg_V[1](4,5) = 0.5*(6.64)+0.5*(-1.06) = 2.79 (positive, sum=9)
  - FFN: Unit 0 activates (~3.7), Unit 1 inactive -> subtracts from dim1
  - This correctly predicts 9 at each position (no carry added)
- **Z[10]:** Carry-out = 0 (correct)

### Summary of the Algorithm

The 74-parameter transformer implements addition through this pipeline:

1. **Position routing (attention):** The phase-rotated Q/K mechanism routes each position to read the X and Y digits at the **position it is predicting** (the next position in the sequence). This is a fixed, input-independent routing.

2. **Sum encoding (V pathway):** The tied V/output projection maps digits to a 5D space where dimension 1 is nearly linear in digit value. The 50/50 attention average computes `V[1] ~ -3.72 * (x + y) + 36.07`, encoding the digit pair sum.

3. **Sum + carry combination (residual):** After attention, dim1 of the residual stream contains the sum of the current token's y-embedding (encoding Z[i], the carry signal) and the attention-injected sum signal (encoding x_{i+1} + y_{i+1}).

4. **Carry detection (FFN):** Two GELU units act as a soft carry detector. Unit 1 fires when the current token indicates a carry (low digit, negative dim1), adding +1 worth of shift. Unit 0 fires in the no-carry case, applying the appropriate correction.

5. **Digit selection (output head):** The norm amplifies dim1 by 6.57x, and the output head projects back to 2D token space. The angle of the resulting point on the arc determines the output digit.

---

## Appendix A: Exact Weight Values

### Token Embedding

```
tok_arc_A = 12.986257
tok_arc_start = -0.532547
tok_arc_stride = 0.121268
```

### Spiral Position

```
spiral_amp = 3.562954
spiral_phase = -0.442029
spiral_slope = 0.169358
spiral_offset = -2.547637
```

### Special Positions

```
z_hi_pos = [36.013222, -32.833508, 6.451802]
special_pos_equals = [2.643782, -2.413533, -1.032774]
_plus_pos = [0, 0, 0]  (frozen)
_eos_pos = [0, 0, 0]   (frozen)
```

### Attention

```
q_phase_angle = 0.720147

q_proj.weight (5x3):
  [-3.375872,  3.042024, -0.649680]
  [ 0.477801, -3.590906, -0.210741]
  [-2.515178, -2.498113,  1.458350]
  [ 3.494784,  1.676627, -0.042561]
  [ 0.000000, -0.000000,  0.000000]

out_proj_A (5x1): [-0.017880, -1.386409,  0.003480,  0.022128, -0.042946]
out_proj_B (1x5): [-0.018258, -1.592626,  0.036357, -0.001274,  0.071765]
```

### FFN

```
fc1.weight (2x5):
  [-0.172500,  1.393574, -0.511556, -0.009794, -0.869375]
  [-0.812804, -1.303782,  0.011639,  0.185164,  0.366109]

fc2.weight (5x2):
  [-0.726745, -0.756547]
  [-1.935689,  2.108446]
  [ 0.072297, -0.020406]
  [ 0.225507,  0.149272]
  [ 1.515820,  1.478060]
```

### Output Head and Norm

```
head_proj.weight (2x5):
  [-21.456442,  0.279907,  1.058445,  2.535547,  12.424424]
  [ -2.098896, -4.889176,  0.000474,  0.117969,  -0.161811]

norm_weight (5): [0.401474, 6.573721, 3.179817, 3.114479, 4.983604]
```

## Appendix B: Data Flow Diagram

```
INPUT: tokens [x0..x9, +, y0..y9, =, z0..z9, carry, eos]

                    ┌─────────────────┐
  tokens ──────────>│  tok_table[tok]  │──> tok_emb (2D)  ──┐
                    └─────────────────┘                     │
                    ┌─────────────────┐                     │    ┌───────────┐
  positions ──────>│  spiral/special   │──> pos_emb (3D)  ──┼──>│ CONCAT    │──> x (5D)
                    └─────────────────┘                     │    │ [tok|pos] │
                                                            │    └───────────┘
                                                            │         │
                                                            │    ┌────▼────┐
                                                            │    │ RMSNorm │ (shared weight)
                                                            │    └────┬────┘
                                                            │         │
                                                ┌───────────┤    ┌────▼────┐
                                                │  pos dims ├───>│ q_proj  │──> K (5D)
                                                │  (2:5)    │    │ (shared)│──> Q (5D, +phase)
                                                │           │    └─────────┘
                                                │           │         │
                                                │  tok dims │    ┌────▼──────────┐
                                                │  (0:2)    ├───>│ V = tok_h @   │
                                                │           │    │ head_proj.wt   │──> V (5D)
                                                │           │    └────────────────┘
                                                │           │         │
                                                │           │    ┌────▼────┐
                                                │           │    │ softmax │ Q·K^T/√5
                                                │           │    │ (causal)│
                                                │           │    └────┬────┘
                                                │           │         │
                                                │           │    ┌────▼──────┐
                                                │           │    │ att @ V   │──> attn_out (5D)
                                                │           │    └────┬──────┘
                                                │           │         │
                                                │           │    ┌────▼──────────┐
                                                │           │    │ out_proj      │ rank-1
                                                │           │    │ A(5x1)·B(1x5) │──> delta (5D)
                                                │           │    └────┬──────────┘   writes dim1
                                                │           │         │
                                    ┌───────────┤           │    ┌────▼────┐
                              x ────┤ + residual├───────────┤    │   ADD   │──> x' (5D)
                                    └───────────┘           │    └────┬────┘
                                                            │         │
                                                            │    ┌────▼────┐
                                                            │    │ RMSNorm │ (same weight)
                                                            │    └────┬────┘
                                                            │         │
                                                            │    ┌────▼────┐
                                                            │    │ fc1(5→2)│ no bias
                                                            │    │  GELU   │ carry detect
                                                            │    │ fc2(2→5)│ no bias
                                                            │    └────┬────┘
                                                            │         │
                                    ┌───────────┐           │    ┌────▼────┐
                              x' ───┤ + residual├───────────┤    │   ADD   │──> x'' (5D)
                                    └───────────┘           │    └────┬────┘
                                                            │         │
                                                            │    ┌────▼────┐
                                                            │    │ RMSNorm │ (same weight)
                                                            │    └────┬────┘
                                                            │         │
                                                            │    ┌────▼────────┐
                                                            │    │ head_proj   │ 5D → 2D
                                                            │    │ (same as V) │
                                                            │    └────┬────────┘
                                                            │         │
                                                       ┌────┘    ┌────▼──────┐
                                          tok_table ───>│────────>│ @ tok.T   │──> logits (10)
                                                       └────┘    └───────────┘
```

## Appendix C: Position Map

The position map assigns each of the 34 sequence positions to a position vector:

```
Seq pos  Label        Source      Index    Position vector
  0      X[0]         digit       0       [3.221, -1.524, -2.548]
  1      X[1]         digit       1       [3.501, 0.660, -2.378]
  2      X[2]         digit       2       [2.445, 2.592, -2.209]
  3      X[3]         digit       3       [0.454, 3.534, -2.040]
  4      X[4]         digit       4       [-1.710, 3.126, -1.870]
  5      X[5]         digit       5       [-3.221, 1.524, -1.701]
  6      X[6]         digit       6       [-3.501, -0.660, -1.531]
  7      X[7]         digit       7       [-2.445, -2.592, -1.362]
  8      X[8]         digit       8       [-0.454, -3.534, -1.193]
  9      X[9]         digit       9       [1.710, -3.126, -1.023]
 10      PLUS         special     0       [0.000, 0.000, 0.000]
 11      Y[0]         digit       0       [3.221, -1.524, -2.548]
 12      Y[1]         digit       1       [3.501, 0.660, -2.378]
  ...    (same as X positions)
 20      Y[9]         digit       9       [1.710, -3.126, -1.023]
 21      EQUALS       special     1       [2.644, -2.414, -1.033]
 22      Z[0]         digit       0       [3.221, -1.524, -2.548]
 23      Z[1]         digit       1       [3.501, 0.660, -2.378]
  ...    (same as X positions)
 31      Z[9]         digit       9       [1.710, -3.126, -1.023]
 32      Z[10]/carry  z_hi        0       [36.013, -32.834, 6.452]
 33      EOS          special     2       [0.000, 0.000, 0.000]
```
