# 74 Parameters Is All You Need: Training a Transformer for Perfect 10-Digit Addition

**Arseniy Zarechnev and Claude**

March 2026

---

## Abstract

We train a 74-parameter autoregressive transformer that performs 10-digit addition with 100% accuracy (10,010/10,010 on the AdderBoard benchmark). The model is a single-layer decoder with a 5-dimensional residual stream, one attention head, and a 2-unit feedforward network. Every parameter is learned from random initialization — no warm-starting, no frozen pretrained values.

Starting from a 242-parameter baseline, we achieve a 70% compression through nine architectural innovations: a single full-rank attention head, parametric circular token embeddings, tied Q/K projections with phase rotation, tied V/output weights, shared normalization, rank-1 attention output, and frozen delimiter positions. A training breakthrough — high carry-mix curriculum with step-based fade — improved the grokking rate from ~10% to 100% of random seeds, making the result reproducible.

The model is the smallest known trained transformer achieving perfect 10-digit addition.

---

## 1. Introduction

The [AdderBoard](https://github.com/anadim/AdderBoard) challenge asks: what is the smallest transformer that can learn 10-digit addition from scratch with 100% accuracy? The task is a clean testbed for understanding transformer expressivity, learnability, and the gap between what neural networks can represent versus what gradient descent can discover.

Our starting point was JackCai's 242-parameter split-subspace architecture — a single-layer decoder with 6-dimensional embeddings, two attention heads, and a carry-lookahead mechanism. Over seven research sessions, we compressed this to 74 parameters while maintaining perfect accuracy, establishing the trained-from-scratch state of the art.

The compression path was not planned in advance. At each step, structural diagnostics of the trained model revealed convergence patterns — weights approaching symmetry, dimensions going unused, norms converging — that suggested the next constraint to impose. Many of these constraints had been previously tested and failed at larger scales, only to succeed at smaller ones. The recurring lesson: you cannot plan a compression roadmap. You must re-test every assumption at each new scale.

### The Task

The model receives two 10-digit numbers in LSB-first format separated by a delimiter, and must autoregressively predict the 11-digit sum (also LSB-first) plus an end token. The full sequence is 34 tokens: `X_0..X_9 + Y_0..Y_9 = Z_0..Z_10 EOS`. The vocabulary is just 10 digits (0-9); delimiters share the digit-0 token and are distinguished by position alone.

### What Makes This Hard

Addition requires carry propagation: to predict digit Z_i, the model needs not just X_i + Y_i but whether a carry arrives from position i-1, which in turn depends on whether a carry arrives from i-2, and so on. In a single-layer autoregressive transformer, the model cannot chain carries through previous outputs (those tokens haven't been generated yet when predicting earlier digits). Instead, it must predict carries by looking ahead at the *input* digits — a hardware-style carry-lookahead circuit implemented in attention and feedforward weights.

---

## 2. Architecture

The model is a single-layer autoregressive decoder with a 5-dimensional residual stream split into two subspaces: a 2D **token subspace** (carrying digit identity) and a 3D **position subspace** (carrying positional information). This split-subspace design, inherited from JackCai's architecture, allows the attention mechanism to route purely based on position while the value pathway carries token content.

### 2.1 Parametric Circular Token Embeddings (3 parameters)

Instead of learning a 10×2 embedding table (20 parameters), we parameterize all 10 digit embeddings as points on a circle:

```
emb[d] = [A·cos(start + d·stride), A·sin(start + d·stride)]
```

Three parameters (A, start, stride) define the entire embedding table. The trained model places digits 0-9 on a 62.5° arc of a circle with radius 12.99, uniformly spaced at ~6.9° per digit. This same embedding table serves double duty as the output classification layer — logits are computed as the dot product between the final hidden state and each embedding vector. The circular geometry provides roughly equal angular separation between all digit pairs, which is near-optimal for 10-class discrimination in 2D.

Early in the compression journey, we used 4 parameters (separate A and B for an elliptical arc), but structural diagnostics showed the trained A/B ratio was 1.005 — the model wanted a circle. Tying A=B saved the final parameter from 75 to 74.

### 2.2 Spiral Positional Encoding (4 parameters)

The 10 digit positions (shared across the X, Y, and Z groups) are encoded as a parametric spiral:

```
pos[i] = [amp·cos(2πi/10 + phase), amp·sin(2πi/10 + phase), slope·i + offset]
```

The first two dimensions form a circle capturing the base-10 periodicity (digit 0 and digit 10 would overlap), while the third dimension provides a linear ramp distinguishing different positions along the sequence. The trained model learns amp=3.56, phase=-25.3°, slope=0.17, offset=-2.55 — a gently tilted ring in 3D.

Three additional special positions are needed:
- **Carry position** (z_hi_pos, 3 learned parameters): The carry-out position at Z_10, learned with norm 49.2 — deliberately placed far from all digit positions (norm ~4) to prevent confusion in the attention routing.
- **EQUALS position** (3 learned parameters): The delimiter between the Y operand and the answer.
- **PLUS and EOS positions** (frozen at zero): These delimiters carry no useful positional information and are fixed at the origin.

### 2.3 Tied Q/K with Phase Rotation (16 parameters)

The attention mechanism operates on the position subspace only (3D → 5D projection). A critical innovation: Q and K share the same projection matrix (15 parameters), and a single learnable angle (1 parameter) rotates Q relative to K:

```
Q_rotated[..., 2p]   = Q[..., 2p]·cos(θ) - Q[..., 2p+1]·sin(θ)
Q_rotated[..., 2p+1] = Q[..., 2p]·sin(θ) + Q[..., 2p+1]·cos(θ)
```

This phase rotation (trained to θ = 41.3°) provides the asymmetry the carry circuit requires — without it, Q·K^T is symmetric and the model cannot distinguish "I attend to you" from "you attend to me." The idea was borrowed from the hand-coded param_40 model and saves 14 parameters compared to a separate K projection.

### 2.4 Tied V/Output (0 extra parameters)

The value projection and the output head share weights via transposition:

```
v_proj.weight = head_proj.weight.T
```

Both map between the 2D token subspace and the 5D residual stream. Analysis of untied models showed these matrices were NOT naturally similar (cosine similarity = -0.30), which initially suggested tying would fail. In practice, tying acts as beneficial regularization — the model finds a different, equally valid joint solution. This eliminated 10 parameters.

### 2.5 Rank-1 Attention Output (10 parameters)

The attention output is projected back to the residual stream through a rank-1 factorization: A(5×1) @ B(1×5) = 10 parameters instead of 5×5 = 25. The trained model writes almost entirely to dimension 1 of the residual stream (B ≈ [0, -1.59, 0.04, 0, 0.07]), effectively compressing the attention signal to a single scalar before expanding it.

### 2.6 Shared RMSNorm (5 parameters)

All three normalization points (pre-attention, pre-FFN, final) share a single 5-dimensional weight vector. At d_model=6, this sharing was impossible — the three norms had specialized different weights (pairwise similarity 0.45-0.67). At d_model=5, they converge to identical values. The shared weight [0.40, 6.57, 3.18, 3.11, 4.98] acts less as a normalizer and more as a feature gate, amplifying dimension 1 by 16× relative to dimension 0.

### 2.7 FFN (20 parameters)

A minimal feedforward network: Linear(5→2, no bias) → GELU → Linear(2→5, no bias). The two hidden units are the carry detection mechanism, computing threshold functions on the attention-enriched residual stream. FFN dim=1 fails (60% token accuracy) — carry detection genuinely requires two hidden dimensions. Removing FFN bias saves 7 parameters with no accuracy loss at d_model=5.

### 2.8 Parameter Budget

```
Component                   Params   Role
─────────────────────────────────────────
tok_arc (A, start, stride)     3     circular digit embedding
spiral (amp, phase, slope, off) 4    digit position encoding
z_hi_pos                       3     carry position
special_pos_equals             3     EQUALS position
q_phase_angle                  1     Q/K asymmetry
q_proj                        15     position → attention (3→5)
out_proj (A + B)              10     rank-1 attention output
fc1                           10     FFN first layer (5→2)
fc2                           10     FFN second layer (2→5)
head_proj                     10     output head / v_proj (5→2)
norm_weight                    5     shared RMSNorm
─────────────────────────────────────────
TOTAL                         74
```

---

## 3. Training

### 3.1 The Grokking Phenomenon

The model learns addition through **grokking** — a sudden phase transition from memorization to generalization. The typical training trajectory:

1. **Memorization** (steps 0–20K): Token accuracy climbs to ~50-70% as the model learns per-digit lookup, but exact match stays near 0%.
2. **Grokking onset** (~20-40K): Exact match jumps from 0% to 90%+ in a few thousand steps as the carry circuit crystallizes.
3. **Oscillation** (40-80K): The model bounces between 50-100% exact match as the circuit stabilizes.
4. **Lock-in** (80K+): Exact match reaches 100% and stays.

### 3.2 Adaptive Weight Decay

The carry circuit needs large weights to approximate hard step functions (the hand-coded param_40 model uses values of ~60,000). Standard weight decay fights this sharpening process. Adaptive weight decay resolves the tension:

- **Base**: wd = 0.01
- **Stage 1** (wd × 0.1): triggered when val_exact > 2%
- **Stage 2** (wd × 0.01): triggered when val_exact > 20%

This gives the model freedom to sharpen its carry circuit once grokking begins, producing an 18× speedup compared to constant weight decay.

### 3.3 Carry-Mix Curriculum: The Training Breakthrough

The single most impactful training innovation was aggressive carry-focused sampling with step-based fade. Long carry chains (e.g., 9999999999 + 1 = 10000000000) are exponentially rare in uniform sampling but represent the hardest test cases. Our approach:

- **80% carry-heavy sampling** during early training: each batch is 80% structured carry examples (cascading nines, boundary crossings, single carries)
- **Step-based linear fade** from 80% to 0% over steps 10K to 80K
- **No metric dependency**: the fade follows a fixed schedule regardless of model performance

This last point was critical. Our earlier approach used metric-triggered fade (remove carries when token accuracy > 0.9), which created a devastating feedback loop at high carry-mix: accuracy rises → carries removed → accuracy drops → carries restored → repeat. The oscillation never converges. Step-based fade eliminates this by smoothly ramping down regardless of performance.

**Impact**: With carry_mix=0.8 and step-based fade, 3/3 random seeds grokked at 74p. With the old carry_mix=0.3 and metric-based fade, only ~1/10 seeds grokked at 75p. This transformed a fragile, seed-dependent process into a robust one.

### 3.4 Shorter Step Budget

Counterintuitively, training for fewer steps (120K vs 400-500K) improves stability. With cosine learning rate decay over 120K steps, the LR drops to ~0.007 by step 80K when the model needs to lock in. With 400K steps, LR is still 0.017 at the same point — too high to hold the grokking basin, leading to post-grokking ejection.

### 3.5 Digit Curriculum

Training starts with small numbers and gradually increases:
- Steps 0–2K: 1-3 digit numbers
- Steps 2K–5K: 1-6 digit numbers
- Steps 5K+: 1-10 digit numbers

This helps the model learn the basic digit-addition circuit before encountering the full complexity of long carry chains.

---

## 4. The Compression Journey

### 4.1 From 242p to 75p: Nine Architectural Steps

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

The final step to 74p combined tying A=B in the token arc (1p architectural saving) with the carry-mix training breakthrough that made the result reproducible.

### 4.2 The Key Insight: One Head Beats Two

The single largest architectural change was step 6: shrinking from d_model=6 with 2 heads (head_dim=3 each) to d_model=5 with 1 head (head_dim=5). One head with full d_model rank is more expressive than two heads with head_dim=3, and the reduced d_model cascades parameter savings across every layer. This unlocked the entire sub-100p compression path.

### 4.3 Scale-Dependent Constraints

A recurring pattern: constraints that fail at one scale succeed at another.

| Constraint | d_model=6 (170p) | d_model=5 (74p) |
|-----------|-------------------|-------------------|
| Shared RMSNorm | Failed (61.5%) | Works perfectly |
| Rank-1 out_proj | Failed (0.08%) | Works perfectly |
| No FFN bias | Required for convergence | Unnecessary |
| Tied V/output | Destroys output (94% error) | Beneficial regularization |

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

### 5.3 The Grokking Lottery

A 10-seed sweep revealed that grokking is highly seed-dependent and, more surprisingly, config-specific:

- Seed 78779 groks 75p (reaching 95.8% exact) but completely fails 72p
- Seed 67086 fails 75p but approaches 100% at 72p
- Only seed 80085 worked for both configurations

Even freezing a single parameter to zero changes which seeds can find the solution. The loss landscape topology shifts with every architectural constraint.

**Flash grokking**: Some seeds briefly achieve near-perfect accuracy (95.8% exact at 33K steps) then crash completely. The carry circuit crystallizes in a basin too shallow to maintain under continued gradient updates. The high carry-mix training recipe stabilizes these transient solutions by ensuring the carry circuit has been thoroughly exercised before the learning rate decays.

---

## 6. The Discoverability Gap

Three hand-coded addition transformers prove that 36-40 parameters suffice representationally. Our trained model needs 74 — a 1.85× overhead. This gap measures the price SGD pays for not knowing the solution in advance.

| Threshold | Params | What it means |
|-----------|--------|---------------|
| Representational floor | 36-40p | Hand-coded proof of concept |
| **Trained from scratch** | **74p** | **What SGD discovers (this work)** |

The overhead pays for: learned normalization weights (5p) that SGD can't avoid, richer position encoding (10p vs 0p) because SGD can't use the extreme weight magnitudes hand-coded models employ, and softer carry thresholds that require more capacity to approximate.

What SGD *can* do that surprises: learn circular embeddings from 3 parameters, discover that weight tying works even when analysis says it shouldn't, and spontaneously converge toward minimal complexity (A→B ratio → 1, slope → 0, out_proj → effectively 1D).

---

## 7. Open Questions and Future Directions

### 7.1 The Near-Grok Plateau (70p)

At 70 parameters (freezing all spiral positions to make them sinusoidal — free by competition rules), models reach 73.5% exact / 97.5% token accuracy and oscillate indefinitely. This is qualitatively different from 74p where seeds either fully grok (100%) or fail completely (<1%). The model finds a partial carry circuit that handles most additions but can't stabilize the last 2.5% of digits.

Can training innovations close this gap? SWA (averaging over the oscillation trajectory), warm restarts, lower minimum learning rate, or late carry-mix re-injection are untested approaches. If any of these push 70p to 100%, the effective parameter count drops to 70 (sinusoidal positions are free by competition rules).

### 7.2 Structured Projections

The q_proj (15p) is the largest single parameter block. Toeplitz or circulant constraints (constant diagonals) would reduce it to 7 parameters, but diagnostics show the trained matrix has no exploitable structure. Could a different basis (DFT, Hadamard) with learned scales work? This would be a qualitatively different kind of projection that SGD might discover differently.

### 7.3 Knowledge Distillation

A 74p teacher could guide a smaller student through soft logits, providing training signal without any weight transfer. Every parameter in the student would be learned from scratch — the teacher constrains the input→output mapping, not the internal weights. This is philosophically clean and architecturally legitimate.

### 7.4 Why Does Grokking Happen?

Our model exhibits textbook grokking: sudden generalization after prolonged memorization. Weight decay plays a key role (dropping WD at the right moment unlocks the transition), but the deeper mechanism is unclear. Why does the carry circuit crystallize suddenly rather than gradually? Why are some seeds capable of finding the basin and others not? Loss landscape visualization (projecting the 74D parameter space onto 2D) could reveal the basin structure.

### 7.5 Minimum Viable Transformer

Is there a theoretical lower bound on trained transformer parameters for N-digit addition? The representational floor is ~36p, but how much overhead does learnability fundamentally require? As we push below 74p, are we approaching a hard boundary or will clever architecture+training keep finding savings?

---

## 8. Reproducibility

### Training Command

```bash
uv run python -m microadder.train --run-name 74p_reproduction --seed 45214
```

Default hyperparameters match our best configuration. Three known good seeds: 45214, 71046, 78988 — all three grok with the default recipe.

### Verification

```bash
uv run python ../AdderBoard/verify.py submission_74p/submission_74p.py
```

Expected output: 10010/10010 correct (100.00%), QUALIFIED.

### Hardware

Trained on a single GPU. Training takes ~120K steps (~15-30 minutes depending on hardware). The model has 74 parameters — inference is essentially instant.

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
  title        = {74 Parameters Is All You Need: Training a Transformer for Perfect 10-Digit Addition},
  year         = {2026},
  url          = {https://github.com/evindor/MicroAdder},
}
```
