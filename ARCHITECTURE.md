# Architecture of the 67-Parameter MicroAdder

A detailed analysis of how a 67-parameter, single-layer transformer achieves 100% accuracy on 10-digit addition (10,010/10,010 test cases on AdderBoard).

**Checkpoint:** `results/runs/sub100_67p_qkdim4_scaledsin_2_noawd_fast_s71046/checkpoints/best.pt`
**Submission:** `submission_67p/submission_67p.py`

---

## 1. Overview

The MicroAdder is a 1-layer decoder-only transformer with `d_model=5`, split into a 2D **token subspace** and a 3D **position subspace**. It uses a 4D Q/K attention subspace (decoupled from the 5D value pathway) and frozen sinusoidal positional encoding. It performs 10-digit decimal addition in LSB-first (least-significant-bit first) order using autoregressive generation.

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

### Parameter Budget (67 total)

| Component | Parameters | Description |
|---|---|---|
| tok_arc (A, start, stride) | 3 | Circular token embedding (A=B tied) |
| z_hi_pos | 3 | Carry-out position vector |
| special_pos_equals | 3 | EQUALS token position |
| q_phase_angle | 1 | Q/K asymmetry rotation |
| q_proj | 12 | Shared Q/K projection (3 -> 4) |
| out_proj | 10 | Rank-1 attention output (5x1 + 1x5) |
| FFN fc1 | 10 | First FFN layer (5 -> 2, no bias) |
| FFN fc2 | 10 | Second FFN layer (2 -> 5, no bias) |
| head_proj | 10 | Output/V projection (5 -> 2) |
| RMSNorm (shared) | 5 | Single weight vector for all 3 norms |
| **Total** | **67** | |

Free (frozen): spiral (amp=3.5, phase=0, slope=0.15, offset=0), PLUS/EOS positions (zero).

### Forward Pass Summary

```
1. Embed:     x = [tok_table[token] | position_vector]        (2D + 3D = 5D)
2. Norm1:     h = RMSNorm(x)
3. Attention: Q,K from h[:, 2:5] via q_proj (3->4); V from h[:, 0:2] (5D via tied head_proj)
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
| A (radius) | 6.0641 | Circle radius (A=B tied, so it is a perfect circle) |
| start | -0.4902 rad (-28.09 deg) | Starting angle for digit 0 |
| stride | 0.1387 rad (7.95 deg) | Angular spacing between consecutive digits |
| total arc | 1.2480 rad (71.51 deg) | Arc span from digit 0 to digit 9 |

### Digit Coordinates

| Digit | x | y | Angle |
|---|---|---|---|
| 0 | 5.350 | -2.855 | -28.09 deg |
| 1 | 5.693 | -2.088 | -20.14 deg |
| 2 | 5.927 | -1.281 | -12.20 deg |
| 3 | 6.047 | -0.449 | -4.25 deg |
| 4 | 6.051 | 0.391 | 3.70 deg |
| 5 | 5.939 | 1.224 | 11.64 deg |
| 6 | 5.713 | 2.033 | 19.59 deg |
| 7 | 5.377 | 2.803 | 27.53 deg |
| 8 | 4.938 | 3.519 | 35.48 deg |
| 9 | 4.405 | 4.168 | 43.42 deg |

### Key Properties

1. **Uniform angular spacing:** All consecutive digits are separated by exactly 6.95 degrees. This means the chord distance between digit `d` and digit `d+k` depends only on `k`, not on `d` -- a translation-invariant property that helps the model treat all digits equivalently.

2. **Monotonic y-coordinate:** Digit 0 has the lowest y-value (-6.594) and digit 9 has the highest (6.886). This monotonic relationship between digit value and the y-coordinate is critical for the carry detection mechanism (see Section 6).

3. **Pairwise distances are Toeplitz:** The distance between digits `i` and `j` depends only on `|i-j|`, a consequence of uniform angular spacing on a circle.

### Decision Boundaries

The output logit for digit `d` is `<head_proj(norm(x)), tok_table[d]>`. Since all digit embeddings lie on a circle of radius A=6.064, the decision boundary between any two digits is the perpendicular bisector of the chord connecting them. The equal spacing means these bisectors are evenly spaced, giving each digit an equal-width "wedge" of the 2D token space.

---

## 3. Position Encoding: The Spiral and Special Positions

Positions are encoded in the 3D position subspace. There are four types of positions:

### 3.1 Digit Positions (Frozen Sinusoidal)

The 10 digit positions (shared across X, Y, and Z) lie on a fixed sinusoidal spiral with zero learned parameters:

```
pos[i] = [3.5 * cos(2*pi*i/10),
          3.5 * sin(2*pi*i/10),
          0.15 * i]
```

| Parameter | Value | Status |
|---|---|---|
| amp | 3.5000 | Frozen |
| phase | 0.0000 | Frozen |
| slope | 0.1500 | Frozen |
| offset | 0.0000 | Frozen |

The positions form a **nearly flat sinusoidal ring**: the z-range is [0, 1.35] (span of 1.35), while the xy-amplitude is 3.5. The ratio `slope*9 / amp = 0.39`, meaning the helix is stretched wide relative to its height. In the xy-plane, the 10 positions are evenly distributed around a circle at 36-degree intervals, starting at 0° (unlike the 74p model which learned phase=-25.3°).

Digit position vectors:

| Position | x | y | z | Norm |
|---|---|---|---|---|
| pos[0] | 3.500 | 0.000 | 0.000 | 3.500 |
| pos[1] | 2.832 | 2.057 | 0.150 | 3.503 |
| pos[2] | 1.082 | 3.329 | 0.300 | 3.513 |
| pos[3] | -1.082 | 3.329 | 0.450 | 3.529 |
| pos[4] | -2.832 | 2.057 | 0.600 | 3.551 |
| pos[5] | -3.500 | 0.000 | 0.750 | 3.579 |
| pos[6] | -2.832 | -2.057 | 0.900 | 3.614 |
| pos[7] | -1.082 | -3.329 | 1.050 | 3.654 |
| pos[8] | 1.082 | -3.329 | 1.200 | 3.700 |
| pos[9] | 2.832 | -2.057 | 1.350 | 3.751 |

### 3.2 Position Sharing: The Critical Design Choice

X[i], Y[i], and Z[i] all share the same position vector `pos[i]`. This means the Q/K attention mechanism cannot distinguish between X[i], Y[i], and Z[i] -- they are "the same position" to the attention routing. This is what allows position-based attention to work with only 10 position vectors instead of 30+.

### 3.3 Special Positions

| Position | Vector | Norm | Notes |
|---|---|---|---|
| PLUS (pos 10) | [0, 0, 0] | 0.000 | Frozen at zero (not learned) |
| EQUALS (pos 21) | [3.471, -1.040, 0.995] | 3.757 | Learned; near pos[0] in direction |
| EOS (pos 33) | [0, 0, 0] | 0.000 | Frozen at zero (not learned) |
| Z[10] carry (pos 32) | [3.073, -1.404, 15.874] | 16.230 | Learned; **4.6x** larger than digit positions |

### 3.4 The Large Carry Position

The z_hi carry position has norm 16.23, which is **4.6x** the average digit position norm of ~3.55. This is smaller than the 74p model's 49.2 (12.3x), but still large enough that Z[10] (the carry-out digit) attends **100%** to itself, completely ignoring all other positions. The carry position acts as an attention "black hole."

Notably, the 67p carry position achieves its dominance primarily through the z-dimension (15.87), while the xy-components (3.07, -1.40) are similar in magnitude to digit positions. The Q/K projection amplifies this z-component into large attention scores.

---

## 4. Attention: Positional Routing with Phase Rotation

### 4.1 The Split Attention Design

Attention in this model has a unique split-subspace design:

- **Q and K** are computed from the **position subspace** only (dims 2, 3, 4)
- **V** is computed from the **token subspace** only (dims 0, 1)

This means **attention patterns are 100% determined by positions, independent of input tokens**. The same set of attention weights is used for every possible addition problem. The positions route information; the tokens carry the payload.

### 4.2 Tied Q/K with Phase Rotation (qk_dim=4)

Q and K share the same linear projection (12 parameters for a 3->4 linear map):

```
q_proj weight (4x3):
  [-2.463, -0.472,  0.315]
  [-1.878,  1.487,  1.099]
  [-0.793,  2.211, -1.530]
  [ 1.482,  2.264,  0.268]
```

All 4 rows are active — unlike the 74p model which had a dead 5th row. The reduction from 5D to 4D Q/K space formalizes what the 74p model already learned: 4 dimensions suffice for position-based attention routing. The value pathway remains at 5D (full d_model), decoupling routing from content.

Without the phase rotation, Q = K exactly, which forces `Q[i] . K[j] = K[i] . K[j]` to be symmetric. This means each position would attend most strongly to **itself** (since `K[i] . K[i]` is maximal by Cauchy-Schwarz).

### 4.3 The Phase Rotation Breaks Self-Attention

The phase angle is **-0.5126 rad (-29.37 degrees)**. It applies a rotation to Q in the (dim0, dim1) and (dim2, dim3) planes:

```
cos(phase) = 0.8715
sin(phase) = -0.4904
```

This rotation transforms self-attention into **lookahead attention**. Compared to the 74p model's +41.3° rotation, the 67p model uses a **negative** angle (-29.4°). Despite rotating in the opposite direction, it achieves the same +1 lookahead effect — the Q/K projection compensates by mapping positions differently.

**Without phase (Q=K):** Position Z[0] attends uniformly to {X[0], Y[0], Z[0]} (33.3% each).

**With phase:** Position Z[0] attends to {X[1], Y[1]} (~48.6% each), with ~0.9% to self and adjacent positions.

The attention is slightly less concentrated than the 74p model (48.6% vs 49.96% per slot), with ~1.8% total leakage to self and adjacent positions.

### 4.4 The Fixed Attention Pattern

Since attention depends only on position, we can compute the complete attention map once for all inputs. For autoregressive generation (which determines the actual context each position sees):

| Position | Predicts | Attends to (>0.1%) |
|---|---|---|
| EQUALS (21) | Z[0] | X[0]: 47.46%, Y[0]: 47.46%, EQUALS: 5.04% |
| Z[0] (22) | Z[1] | X[1]: 48.62%, Y[1]: 48.62%, Z[0]: 0.92%, Y[0]: 0.92% |
| Z[1] (23) | Z[2] | X[2]: 48.60%, Y[2]: 48.60%, Y[1]: 0.93%, Z[1]: 0.93% |
| Z[2] (24) | Z[3] | X[3]: 48.79%, Y[3]: 48.79%, Z[2]: 0.81%, Y[2]: 0.81% |
| Z[3] (25) | Z[4] | X[4]: 48.89%, Y[4]: 48.89%, Y[3]: 0.74%, X[3]: 0.74% |
| Z[4] (26) | Z[5] | X[5]: 48.86%, Y[5]: 48.86%, X[4]: 0.76%, Z[4]: 0.76% |
| Z[5] (27) | Z[6] | X[6]: 48.82%, Y[6]: 48.82%, Y[5]: 0.79%, X[5]: 0.79% |
| Z[6] (28) | Z[7] | X[7]: 48.81%, Y[7]: 48.81%, X[6]: 0.79%, Y[6]: 0.79% |
| Z[7] (29) | Z[8] | X[8]: 48.86%, Y[8]: 48.86%, X[7]: 0.76%, Z[7]: 0.76% |
| Z[8] (30) | Z[9] | X[9]: 49.02%, Y[9]: 49.02%, Y[8]: 0.65%, X[8]: 0.65% |
| Z[9] (31) | Z[10] | EQUALS: 99.48% |
| Z[10]/carry (32) | EOS | Z[10]/carry: 100.00% |

**Key patterns:**
1. Each Z[i] position (predicting Z[i+1]) attends ~48.6/48.6 to X[i+1] and Y[i+1], with ~2.8% total leakage to self and adjacent positions
2. Z[9] (predicting Z[10], the carry-out) attends 99.48% to EQUALS
3. Z[10] (predicting EOS) attends 100% to itself (the carry position dominates)
4. The attention is slightly less concentrated than 74p (~97.2% vs ~99.9% on target pair), but still highly focused

### 4.5 The Q.K/K.K Ratio

The phase rotation uniformly scales the self-attention score:

```
Q[i] . K[i] / (K[i] . K[i]) = cos(phase) = 0.8715  (constant for ALL positions)
```

This means the phase reduces each position's self-attention score by a fixed factor, while the cross-attention scores to neighbor positions are boosted. The -29.4° rotation (cos=0.8715) is less aggressive than the 74p model's 41.3° rotation (cos=0.7517) — self-attention is less suppressed, resulting in the ~1-2% leakage to self observed in the attention patterns.

---

## 5. Value Pathway: Tied V/Output

### 5.1 The head_proj Double Duty

The `head_proj` weight matrix (2x5, 10 parameters) serves two roles simultaneously:

1. **As V projection:** `V = tok_h @ head_proj.weight` (2D token -> 5D attention value)
2. **As output head:** `logits = head_proj(norm(x)) @ tok_table.T` (5D -> 2D -> 10 logits)

```
head_proj.weight (2x5):
  [-10.044,  0.088, -0.537,  0.709, -0.114]
  [ -0.060, -4.016, -0.505,  0.144, -0.739]
```

### 5.2 V Values Encode Digit Sum

When `head_proj.weight` is applied to the 2D token embedding of digit `d`, the resulting 5D value vector has a critical property: **dimension 1 is a near-linear function of digit value**.

| Digit | V[dim0] | V[dim1] |
|---|---|---|
| 0 | -53.57 | +11.94 |
| 1 | -57.06 | +8.89 |
| 2 | -59.46 | +5.67 |
| 3 | -60.71 | +2.34 |
| 4 | -60.81 | -1.04 |
| 5 | -59.73 | -4.39 |
| 6 | -57.51 | -7.66 |
| 7 | -54.18 | -10.78 |
| 8 | -49.81 | -13.70 |
| 9 | -44.49 | -16.35 |

Linear fit: `V[1](d) = -3.20 * d + 11.90` (max residual from linear: 0.57)

This near-linearity arises because the token embeddings lie on a small arc of a circle, and over a 71.5-degree arc, `cos(theta)` is nearly linear. The slope of -3.20 per digit means:

- **The average V[dim1] of two digits encodes their sum:** `avg_V[1](x, y) = -1.60 * (x + y) + 11.90`
- **The crossover from positive to negative V[1] is at digit 3.72**, slightly lower than the 74p model's 4.85
- **For a pair sum S:** avg_V[1] is positive when S < ~7.4 and negative when S >= ~7.4

The 67p model uses smaller V magnitudes than 74p (slope -3.20 vs -7.43), compensating with different norm scaling and FFN thresholds.

### 5.3 The Rank-1 Output Projection

After attention computes the weighted average of V vectors, the rank-1 output projection compresses the result:

```
out_proj.A (5x1): [ 0.078, -0.810, -0.062,  0.001, -0.656]
out_proj.B (1x5): [ 0.046, -0.716, -0.010,  0.011, -0.002]
```

The A vector reads primarily from **dimension 1** (-0.810) and **dimension 4** (-0.656) of the attention output. The B vector writes **almost exclusively to dimension 1** (-0.716, all others < 0.046).

The effective dim1 gain is: `out_proj_gain = (-0.810) * (-0.716) = 0.580`

This is substantially smaller than the 74p model's gain of 2.208, meaning the 67p model injects a weaker attention signal into the residual stream. The model compensates with larger norm weights (dim1: 13.60 vs 6.57 in 74p).

The attention mechanism computes a scalar proportional to the digit sum at the next position and writes it to dimension 1 of the residual stream.

### 5.4 Residual Stream After Attention

For a position predicting Z[i+1], the residual stream after attention is approximately:

```
dim0 = tok_table[Z[i]][0]           (token x-coordinate, ~unchanged)
dim1 = tok_table[Z[i]][1] + 0.580 * avg_V[1](x_{i+1}, y_{i+1})
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
  Unit 0: [ 0.410, -0.836, -0.262,  0.024, -1.655]
  Unit 1: [ 0.449,  0.839, -0.166, -0.116, -1.465]

fc2 (5x2): writes to residual
  Unit 0: [-0.951, -0.871]
  Unit 1: [ 0.665, -0.901]
  dim2:   [-0.002, -0.056]
  dim3:   [ 0.005,  0.050]
  dim4:   [-0.027, -0.048]
```

### 6.1 How the Two Units Specialize

Both units read heavily from **dim4** (weights -1.655 and -1.465), which in the 67p model carries position-dependent information through the linear ramp of the sinusoidal positions (z = 0.15*i). They also read from dim1 (the combined token+sum signal) with opposite signs: unit 0 reads -0.836, unit 1 reads +0.839.

**Unit 0** activates when dim1 is negative AND dim4 contributes a large enough offset. It writes to dim1 via fc2: 0.665 (positive push).

**Unit 1** activates when dim1 is positive AND dim4 contributes. It writes to dim1 via fc2: -0.901 (negative push).

The two units form a push-pull pair on dim1, with their activation controlled by the sign of the combined token+sum signal — the same carry detection mechanism as the 74p model, but with different weight magnitudes and a stronger role for position-dependent signals.

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
norm_weight: [7.961, 13.596, 2.398, 2.407, 3.524]
```

The key feature: **dim1 is amplified most (13.60), dim0 second (7.96)** — a 1.7x ratio between them. This is much less extreme than the 74p model's 16.4x ratio (dim0 was suppressed to 0.40). The 67p model treats both token dimensions as important, while position dimensions are suppressed (2.4-3.5x).

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
tok_arc_A = 6.064079
tok_arc_start = -0.490179
tok_arc_stride = 0.138667
```

### Sinusoidal Positions (Frozen)

```
spiral_amp = 3.500000    (frozen)
spiral_phase = 0.000000  (frozen)
spiral_slope = 0.150000  (frozen)
spiral_offset = 0.000000 (frozen)
```

### Special Positions

```
z_hi_pos = [3.073271, -1.403966, 15.874219]
special_pos_equals = [3.470619, -1.040053, 0.994751]
_plus_pos = [0, 0, 0]  (frozen)
_eos_pos = [0, 0, 0]   (frozen)
```

### Attention

```
q_phase_angle = -0.512560

q_proj.weight (4x3):
  [-2.462873, -0.472051,  0.315342]
  [-1.877643,  1.486909,  1.099144]
  [-0.792821,  2.210805, -1.530270]
  [ 1.481666,  2.263654,  0.268334]

out_proj_A (5x1): [ 0.078231, -0.809933, -0.061855,  0.001240, -0.655522]
out_proj_B (1x5): [ 0.046148, -0.715690, -0.009938,  0.011434, -0.002321]
```

### FFN

```
fc1.weight (2x5):
  [ 0.409873, -0.836150, -0.262240,  0.024122, -1.655446]
  [ 0.449363,  0.839487, -0.165511, -0.115800, -1.464789]

fc2.weight (5x2):
  [-0.950666, -0.871497]
  [ 0.665394, -0.901238]
  [-0.002438, -0.056473]
  [ 0.005143,  0.049685]
  [-0.026858, -0.048339]
```

### Output Head and Norm

```
head_proj.weight (2x5):
  [-10.044181,  0.088397, -0.537309,  0.708696, -0.114469]
  [ -0.059638, -4.016446, -0.505348,  0.143598, -0.739328]

norm_weight (5): [7.960984, 13.595810, 2.397511, 2.407397, 3.523559]
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
  0      X[0]         digit       0       [3.500, 0.000, 0.000]
  1      X[1]         digit       1       [2.832, 2.057, 0.150]
  2      X[2]         digit       2       [1.082, 3.329, 0.300]
  3      X[3]         digit       3       [-1.082, 3.329, 0.450]
  4      X[4]         digit       4       [-2.832, 2.057, 0.600]
  5      X[5]         digit       5       [-3.500, 0.000, 0.750]
  6      X[6]         digit       6       [-2.832, -2.057, 0.900]
  7      X[7]         digit       7       [-1.082, -3.329, 1.050]
  8      X[8]         digit       8       [1.082, -3.329, 1.200]
  9      X[9]         digit       9       [2.832, -2.057, 1.350]
 10      PLUS         special     0       [0.000, 0.000, 0.000]
 11      Y[0]         digit       0       [3.500, 0.000, 0.000]
 12      Y[1]         digit       1       [2.832, 2.057, 0.150]
  ...    (same as X positions)
 20      Y[9]         digit       9       [2.832, -2.057, 1.350]
 21      EQUALS       special     1       [3.471, -1.040, 0.995]
 22      Z[0]         digit       0       [3.500, 0.000, 0.000]
 23      Z[1]         digit       1       [2.832, 2.057, 0.150]
  ...    (same as X positions)
 31      Z[9]         digit       9       [2.832, -2.057, 1.350]
 32      Z[10]/carry  z_hi        0       [3.073, -1.404, 15.874]
 33      EOS          special     2       [0.000, 0.000, 0.000]
```
