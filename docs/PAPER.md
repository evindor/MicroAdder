# Training a Transformer for Perfect 10-Digit Addition

**Arseniy Zarechnev and Claude**

March 2026

---

## Abstract

We train a 57-parameter autoregressive transformer that performs 10-digit addition with 100% accuracy (10,010/10,010 on the AdderBoard benchmark). The model is a single-layer decoder with a 5-dimensional residual stream, one attention head with a 4-dimensional Q/K subspace, and a 2-unit feedforward network. Every parameter is learned from random initialization — no warm-starting, no frozen pretrained values. Positional encodings use a fixed sinusoidal scheme (0 learned parameters).

Starting from a 242-parameter baseline, we achieve a 76% compression through twelve architectural innovations: a single full-rank attention head, parametric circular token embeddings, tied Q/K projections with phase rotation, tied V/output weights, shared normalization, rank-1 attention output, frozen delimiter positions, reduced Q/K dimensionality, sinusoidal positional encoding, and triple-duty weight tying (the output head simultaneously serves as V projection, FFN expansion matrix, and output classifier). A training breakthrough — high carry-mix curriculum with step-based fade — improved the grokking rate from ~10% to 100% of random seeds, making the result reproducible.

The model is the smallest known **trained** transformer achieving perfect 10-digit addition.

---

## 1. Introduction

The [AdderBoard](https://github.com/anadim/AdderBoard) challenge asks: what is the smallest transformer that can learn 10-digit addition from scratch with 100% accuracy? The task is a clean testbed for understanding transformer expressivity, learnability, and the gap between what neural networks can represent versus what gradient descent can discover.

Our starting point was JackCai's 242-parameter split-subspace architecture — a single-layer decoder with 6-dimensional embeddings, two attention heads, and a carry-lookahead mechanism. Over nine research sessions, we compressed this to 57 parameters while maintaining perfect accuracy, establishing the trained-from-scratch state of the art.

The compression path was not planned in advance. At each step, structural diagnostics of the trained model revealed convergence patterns — weights approaching symmetry, dimensions going unused, norms converging — that suggested the next constraint to impose. Many of these constraints had been previously tested and failed at larger scales, only to succeed at smaller ones. The recurring lesson: you cannot plan a compression roadmap. You must re-test every assumption at each new scale.

### The Task

The model receives two 10-digit numbers in LSB-first format separated by a delimiter, and must autoregressively predict the 11-digit sum (also LSB-first) plus an end token. The full sequence is 34 tokens: `X_0..X_9 + Y_0..Y_9 = Z_0..Z_10 EOS`. The vocabulary is just 10 digits (0-9); delimiters share the digit-0 token and are distinguished by position alone.

### What Makes This Hard

Addition requires carry propagation: to predict digit Z_i, the model needs not just X_i + Y_i but whether a carry arrives from position i-1, which in turn depends on whether a carry arrives from i-2, and so on. In a single-layer autoregressive transformer, the model cannot chain carries through previous outputs (those tokens haven't been generated yet when predicting earlier digits). Instead, it must predict carries by looking ahead at the *input* digits — a hardware-style carry-lookahead circuit implemented in attention and feedforward weights.

---

## 2. Architecture

The model is a single-layer autoregressive decoder with a 5-dimensional residual stream split into two subspaces: a 2D **token subspace** (carrying digit identity) and a 3D **position subspace** (carrying positional information). This split-subspace design, inherited from JackCai's architecture, allows the attention mechanism to route purely based on position while the value pathway carries token content. The Q/K attention operates in a 4-dimensional subspace (decoupled from the 5D value pathway), and positional encoding is a fixed sinusoidal scheme requiring zero learned parameters.

### 2.1 Parametric Circular Token Embeddings (3 parameters)

Instead of learning a 10×2 embedding table (20 parameters), we parameterize all 10 digit embeddings as points on a circle:

```
emb[d] = [A·cos(start + d·stride), A·sin(start + d·stride)]
```

Three parameters (A, start, stride) define the entire embedding table. The trained 57p model places digits 0-9 on a 69.3° arc of a circle with radius 11.07, uniformly spaced at ~7.7° per digit. This same embedding table serves double duty as the output classification layer — logits are computed as the dot product between the final hidden state and each embedding vector. The circular geometry provides roughly equal angular separation between all digit pairs, which is near-optimal for 10-class discrimination in 2D.

Early in the compression journey, we used 4 parameters (separate A and B for an elliptical arc), but structural diagnostics showed the trained A/B ratio was 1.005 — the model wanted a circle. Tying A=B saved a parameter from 75 to 74.

### 2.2 Sinusoidal Positional Encoding (0 parameters)

The 10 digit positions (shared across the X, Y, and Z groups) are encoded as a fixed sinusoidal spiral:

```
pos[i] = [3.5·cos(2πi/10), 3.5·sin(2πi/10), 0.15·i]
```

The first two dimensions form a circle at 36° intervals capturing the base-10 periodicity, while the third dimension provides a gentle linear ramp (0.15 per position) distinguishing different positions along the sequence. All four spiral parameters (amp=3.5, phase=0, slope=0.15, offset=0) are frozen at initialization — zero learned parameters. This is possible because the evenly-spaced sinusoidal positions provide sufficient structure for the Q/K projection to learn the correct attention routing.

The earlier 74p model learned these parameters, converging to amp=3.56, phase=-25.3°, slope=0.17, offset=-2.55. The frozen sinusoidal values are close enough that the Q/K projection compensates for the difference.

Three additional special positions are needed:
- **Carry position** (z_hi_pos, 3 learned parameters): The carry-out position at Z_10, learned with norm 35.3 (57p) / 16.2 (67p) — placed far from all digit positions (norm ~3.5) to dominate attention routing for the carry-out digit.
- **EQUALS position** (3 learned parameters): The delimiter between the Y operand and the answer.
- **PLUS and EOS positions** (frozen at zero): These delimiters carry no useful positional information and are fixed at the origin.

### 2.3 Tied Q/K with Phase Rotation and Reduced Dimension (13 parameters)

The attention mechanism operates on the position subspace only, projected to a 4-dimensional Q/K space (3D → 4D, 12 parameters). Q and K share the same projection matrix, and a single learnable angle (1 parameter) rotates Q relative to K:

```
Q_rotated[..., 2p]   = Q[..., 2p]·cos(θ) - Q[..., 2p+1]·sin(θ)
Q_rotated[..., 2p+1] = Q[..., 2p]·sin(θ) + Q[..., 2p+1]·cos(θ)
```

This phase rotation (trained to θ = +28.0° at 57p, -29.4° at 67p) provides the asymmetry the carry circuit requires — without it, Q·K^T is symmetric and the model cannot distinguish "I attend to you" from "you attend to me." The idea was borrowed from the hand-coded param_40 model and saves 11 parameters compared to a separate K projection.

The reduction from 5D to 4D Q/K space (saving 3 parameters) is possible because the 74p model's Q/K projection had a dead 5th row (all near-zero weights). Removing it formalizes what the model already learned: 4 dimensions suffice for position-based attention routing. The value pathway remains at full 5D (head_dim=5), decoupling attention routing from content aggregation.

### 2.4 Triple-Duty Head Projection (10 parameters for 3 roles)

The `head_proj` weight matrix (2×5) serves three simultaneous roles:

1. **V projection**: `V = tok_h @ head_proj.weight.T` maps 2D token content to 5D for attention aggregation
2. **Output classification**: `logits = head_proj(h) @ tok_emb.T` maps the final residual to token space
3. **FFN expansion**: `fc2_out = ffn_hidden @ head_proj.weight` maps the 2D FFN hidden state back to 5D residual

This triple-duty tying eliminates the separate fc2 layer (10 parameters), saving the full step from 67p to 57p. The insight: the FFN's second layer needs to write corrections back into the residual stream, and analysis of the 67p model showed these corrections land almost entirely in the output-relevant subspace — which is exactly where `head_proj` already knows how to write. By tying fc2 = head_proj.T, we force the FFN to write directly into output space by construction, rather than learning it independently.

Analysis of untied V/output showed these matrices were NOT naturally similar (cosine similarity = -0.30), which initially suggested tying would fail. In practice, tying acts as beneficial regularization — the model finds a different, equally valid joint solution.

### 2.5 Rank-1 Attention Output (10 parameters)

The attention output is projected back to the residual stream through a rank-1 factorization: A(5×1) @ B(1×5) = 10 parameters instead of 5×5 = 25. The 57p model's out_proj has an effective gain of 2.27, writing primarily to the token subspace (B dominated by tok dims 0,1).

### 2.6 Shared RMSNorm (5 parameters)

All three normalization points (pre-attention, pre-FFN, final) share a single 5-dimensional weight vector. At d_model=6, this sharing was impossible — the three norms had specialized different weights (pairwise similarity 0.45-0.67). At d_model=5, they converge to identical values. The 57p shared weight [3.64, 3.38, 3.18, 3.57, 14.92] massively amplifies dimension 4 (the position ramp) by 4.7× relative to other dims, creating a position-sensitive gate. This contrasts with the 67p model's [0.40, 6.57, 3.18, 3.11, 4.98] which amplified dim 1 by 16× — showing that the triple-duty constraint forces the model to find a qualitatively different solution.

### 2.7 FFN (10 parameters)

A minimal feedforward network: Linear(5→2, no bias) → GELU → expansion via head_proj.weight (tied, 0 extra params). Only fc1 (10 parameters) is independent; fc2 reuses head_proj (see §2.4). The two hidden units are the carry detection mechanism, computing threshold functions on the attention-enriched residual stream. FFN dim=1 fails (60% token accuracy) — carry detection genuinely requires two hidden dimensions. Removing FFN bias saves 7 parameters with no accuracy loss at d_model=5.

### 2.8 Parameter Budget

```
Component                   Params   Role
─────────────────────────────────────────
tok_arc (A, start, stride)     3     circular digit embedding
z_hi_pos                       3     carry position
special_pos_equals             3     EQUALS position
q_phase_angle                  1     Q/K asymmetry
q_proj                        12     position → attention (3→4)
out_proj (A + B)              10     rank-1 attention output
fc1                           10     FFN first layer (5→2)
head_proj                     10     triple-duty: V proj / output / fc2
norm_weight                    5     shared RMSNorm
─────────────────────────────────────────
TOTAL                         57

Free (frozen at initialization):
spiral (amp, phase, slope, off) 4    sinusoidal positions
PLUS/EOS positions              6    frozen at zero
fc2 (tied to head_proj)         0    triple-duty weight sharing
```

---

## 3. Training

### 3.1 The Grokking Phenomenon

The model learns addition through **grokking** — a sudden phase transition from memorization to generalization. The typical training trajectory:

1. **Memorization** (steps 0–20K): Token accuracy climbs to ~50-70% as the model learns per-digit lookup, but exact match stays near 0%.
2. **Grokking onset** (~20-40K): Exact match jumps from 0% to 90%+ in a few thousand steps as the carry circuit crystallizes.
3. **Oscillation** (40-80K): The model bounces between 50-100% exact match as the circuit stabilizes.
4. **Lock-in** (80K+): Exact match reaches 100% and stays.

### 3.2 Carry-Mix Curriculum: The Training Breakthrough

The single most impactful training innovation was aggressive carry-focused sampling with step-based fade. Long carry chains (e.g., 9999999999 + 1 = 10000000000) are exponentially rare in uniform sampling but represent the hardest test cases. Our approach:

- **80% carry-heavy sampling** during early training: each batch is 80% structured carry examples (cascading nines, boundary crossings, single carries)
- **Step-based linear fade** from 80% to 0% over steps 10K to 80K
- **No metric dependency**: the fade follows a fixed schedule regardless of model performance

This last point was critical. Our earlier approach used metric-triggered fade (remove carries when token accuracy > 0.9), which created a devastating feedback loop at high carry-mix: accuracy rises → carries removed → accuracy drops → carries restored → repeat. The oscillation never converges. Step-based fade eliminates this by smoothly ramping down regardless of performance.

**Impact**: With carry_mix=0.8 and step-based fade, 3/3 random seeds grokked at 74p. With the old carry_mix=0.3 and metric-based fade, only ~1/10 seeds grokked at 75p. This transformed a fragile, seed-dependent process into a robust one.

The 67p model uses a tighter fade window (15K-45K vs 10K-80K), matching its faster grokking dynamics.

### 3.3 Shorter Step Budget

Counterintuitively, training for fewer steps improves stability. The 57p model trains for just 60K steps (vs 120K for 74p), with grokking at step 44K. With cosine learning rate decay, the shorter budget means faster LR decay, which helps lock in the grokking basin once found.

### 3.4 No Adaptive Weight Decay (57p, 67p)

The 74p model required adaptive weight decay (dropping WD when grokking detected) to converge. The 57p and 67p models do not — constant WD=0.01 suffices. This may be because the sinusoidal positions provide a more structured starting point that reduces the search space, or because the faster training schedule (60K steps) naturally provides enough WD pressure at the right time.

### 3.5 Digit Curriculum

Training starts with small numbers and gradually increases:
- Steps 0–2K: 1-3 digit numbers
- Steps 2K–5K: 1-6 digit numbers
- Steps 5K+: 1-10 digit numbers

This helps the model learn the basic digit-addition circuit before encountering the full complexity of long carry chains.

---

## 4. The Compression Journey

### 4.1 From 242p to 57p: Twelve Architectural Steps

Each step was validated at 100% accuracy (10,010/10,010):

| Step | Params | Technique | Saving |
|------|--------|-----------|--------|
| Baseline | 242p | JackCai's split-subspace | — |
| 1 | 226p | Spiral positions replace 30 learned params | -16p |
| 2 | 214p | Rank-2 attention output factorization | -12p |
| 3 | 203p | Linear position correction + frozen EOS | -11p |
| 4 | 187p | Tied Q/K with phase rotation | -16p |
| 5 | 170p | tok_dim 3→2 (covers 96% SVD energy) | -17p |
| 6 | 133p | d_model 6→5, vocab 14→10, parametric embeddings, 1 head | -37p |
| 7 | 100p | Rank-1 attention output, no FFN bias | -33p |
| 8 | 78p | Shared norms + tied V/output, no position correction | -22p |
| 9 | 75p | Freeze PLUS/EOS positions to zero | -3p |
| 10 | 74p | Tie A=B (circular embedding) + carry-mix training | -1p |
| 11 | 67p | Sinusoidal positions (freeze spiral) + qk_dim 5→4 | -7p |
| 12 | **57p** | **Triple-duty head_proj (tie fc2 = head_proj.T)** | **-10p** |

The step to 67p combined two independent compressions: freezing all spiral parameters to fixed sinusoidal values (saving 4p) and reducing the Q/K projection dimension from 5 to 4 (saving 3p). Both were motivated by analysis of the 74p trained weights showing that the spiral parameters converged near sinusoidal defaults and the 5th Q/K dimension was dead (all near-zero).

The step to 57p tied the FFN's second layer to the output head, making `head_proj.weight` serve triple duty (V projection, output classification, FFN expansion). This was motivated by the observation that the 67p model's FFN already writes corrections primarily into the output-relevant subspace — the same subspace `head_proj` knows how to target. The tying makes this alignment a structural guarantee rather than a learned accident.

### 4.2 The Key Insight: One Head Beats Two

The single largest architectural change was step 6: shrinking from d_model=6 with 2 heads (head_dim=3 each) to d_model=5 with 1 head (head_dim=5). One head with full d_model rank is more expressive than two heads with head_dim=3, and the reduced d_model cascades parameter savings across every layer. This unlocked the entire sub-100p compression path.

### 4.3 Scale-Dependent Constraints

A recurring pattern: constraints that fail at one scale succeed at another.

| Constraint | d_model=6 (170p) | d_model=5 (74p) | d_model=5 (67p) | d_model=5 (57p) |
|-----------|-------------------|-------------------|-------------------|-------------------|
| Shared RMSNorm | Failed (61.5%) | Works | Works | Works |
| Rank-1 out_proj | Failed (0.08%) | Works | Works | Works |
| No FFN bias | Required for convergence | Unnecessary | Unnecessary | Unnecessary |
| Tied V/output | Destroys output (94% error) | Works | Works | Works |
| Frozen spiral | — | — | Works (saves 4p) | Works |
| qk_dim=4 | — | — | Works (saves 3p) | Works |
| Tied fc2=head_proj | — | — | Not tested | Works (saves 10p) |

This meant we couldn't plan a compression roadmap in advance. Each scale required re-testing every assumption.

---

## 5. What Failed

Not everything worked. The negative results were as informative as the successes.

### 5.1 Dead Ends in Architecture

- **d_model=4**: Mathematically impossible. With tok_dim=1, a single scalar times the embedding vector can only produce 2 classes (positive/negative), not 10 digits. With tok_dim=2, pos_dim=2 can't represent the carry dimension. d_model=5 is the proven minimum.
- **No q_phase**: Without the phase rotation breaking Q=K symmetry, the model dies at 21% token accuracy. The carry circuit fundamentally requires asymmetric attention.
- **Rank-1 q_proj**: 1D keys cannot distinguish positions. The position subspace needs at least the full 3→5 projection.
- **ffn_dim=1**: The carry function requires two hidden units. A single-unit FFN plateaus at 60% token accuracy.
- **ALiBi**: Pure distance-based attention slopes can't route content — stuck at 20% token accuracy.

### 5.2 Dead Ends in Training

- **Scaffold L1 training** (8 experiments): The idea of training wide then annealing extra capacity to zero via L1 penalty. Failed universally. Low L1 creates equilibrium (scaffold weights hover nonzero), medium L1 causes catastrophic collapse (carry circuit destroyed), high L1 suppresses learning entirely. L1 is adversarial to task learning in tiny models.
- **SAM (Sharpness-Aware Minimization)**: The adversarial weight perturbation disrupts delicate feature learning. Uniformly worse than vanilla AdamW at all tested scales.
- **WD scheduling variants**: Scheduled drops, cyclical oscillation, and warmup-then-adaptive all produced identical failure modes. The grokking bottleneck at small scales is seed/initialization, not weight decay timing.
- **Metric-based carry_mix fade at high carry_mix**: Creates an oscillation feedback loop that never converges. Step-based fade was the fix.

---

## 6. The Discoverability Gap

Three hand-coded addition transformers prove that 36-40 parameters suffice representationally. Our trained model needs 57 — a 1.43× overhead (down from 1.68× at 67p and 1.85× at 74p). This gap measures the price SGD pays for not knowing the solution in advance.

| Threshold | Params | What it means |
|-----------|--------|---------------|
| Representational floor | 36-40p | Hand-coded proof of concept |
| **Trained from scratch** | **57p** | **What SGD discovers (this work)** |
| Previous best | 67p | Before triple-duty head_proj |

The overhead pays for: learned normalization weights (5p) that SGD can't avoid, richer carry position encoding (3p for z_hi vs 0p in hand-coded models), and softer carry thresholds that require more capacity to approximate. The reduction from 67p to 57p came from recognizing that three separate weight matrices were converging to operate in the same subspace — the output head, V projection, and FFN expansion all target the output-relevant dimensions. Tying them eliminates 10 redundant parameters.

What SGD *can* do that surprises: learn circular embeddings from 3 parameters, discover that weight tying works even when analysis says it shouldn't, spontaneously converge toward minimal complexity, operate with fixed sinusoidal positions, and find a qualitatively different carry mechanism (autoregressive carry propagation) when forced to reuse weights.

---

## 7. Open Questions and Future Directions

### 7.1 The Sub-57p Frontier

The 57p model shows that aggressive weight tying can compress the trained transformer close to the hand-coded floor. The next compression targets are:

- **q_proj structure**: The q_proj (12p) is now the largest single parameter block. Structured constraints (Toeplitz, circulant, or low-rank factorization) could reduce it further.
- **Fewer special positions**: The EQUALS position (3p) might be replaceable with a fixed value near pos[0], since the trained EQUALS position has similar direction to pos[0].
- **Smaller norm**: The shared RMSNorm (5p) could potentially be replaced with a scalar or per-subspace scalar.
- **Out_proj compression**: The rank-1 out_proj (10p) might be further tightened via structured A/B vectors.

### 7.2 Structured Projections

The q_proj (12p) is now the largest single parameter block. At 4×3 it is more compact than the previous 5×3, and the removal of the dead 5th row suggests further structure may be exploitable. Toeplitz or circulant constraints (constant diagonals) would reduce it to 6 parameters. Could a different basis (DFT, Hadamard) with learned scales work?

### 7.3 Knowledge Distillation

A 57p teacher could guide a smaller student through soft logits, providing training signal without any weight transfer. Every parameter in the student would be learned from scratch — the teacher constrains the input→output mapping, not the internal weights. This is philosophically clean and architecturally legitimate.

### 7.4 Minimum Viable Transformer

Is there a theoretical lower bound on trained transformer parameters for N-digit addition? The representational floor is ~36p, but how much overhead does learnability fundamentally require? At 57p, the overhead factor is 1.43× — steadily shrinking from 1.68× at 67p and 1.85× at 74p. The gap is closing, but each step requires discovering new weight-sharing opportunities.

---

## 8. Reproducibility

### Training Command

```bash
# 57p (triple-duty head_proj)
uv run python -m microadder.train --run-name sub100_57p_repro --seed 777 --tie-fc2-head

# 67p
uv run python -m microadder.train --run-name sub100_67p_repro --seed 71046

# 74p (learned spiral, full qk_dim)
uv run python -m microadder.train --run-name sub100_74p_repro --seed 45214 \
    --qk-dim 0 --freeze-spiral "" --wd-adaptive --steps 120000 \
    --carry-mix-fade-start 10000 --carry-mix-fade-end 80000
```

Default hyperparameters match the 67p configuration. Add `--tie-fc2-head` for 57p. Known good seeds: 777 (57p), 71046 (67p).

### Verification

```bash
uv run python ../AdderBoard/verify.py submission_57p/submission_57p.py
```

Expected output: 10010/10010 correct (100.00%), QUALIFIED.

### Hardware

Trained on a single GPU. The 57p model trains for 60K steps (~5 minutes). Inference is essentially instant.

---

## Acknowledgments

This work builds on:
- **JackCai** — the 242p split-subspace architecture that serves as our starting point
- **Wonderfall** (param_40) — the hand-coded 40p model that inspired tied Q/K with phase rotation
- **Litzenberger** (TinyAdder) — the 36p hand-coded model that established the representational floor
- **Dimitris Papailiopoulos** — for creating the AdderBoard challenge

---

## Citation

```bibtex
@misc{zarechnev2026microadder,
  author       = {Arseniy Zarechnev and Claude},
  title        = {Training a Transformer for Perfect 10-Digit Addition},
  year         = {2026},
  url          = {https://github.com/evindor/MicroAdder},
}
```
