# Hand-Coded Transformers: What They Teach Us About Trainable Models

A deep analysis of three hand-coded addition transformers (36p, 40p, 87p) and what their designs reveal about the fundamental gap between what transformers *can represent* and what SGD *can discover*.

## The Three Models

### TinyAdder — 36 parameters (Alex Litzenberger)

File: `external/tiny_adder_submission_autoregressive_gen.py`

```
2 layers, d_model=5, ALiBi positional encoding
Vocab=14, float64 precision
13 emb + 6 L0-attn + 12 L0-ffn + 2 L1-attn + 3 L1-ffn = 36
```

**How it works:**

The architecture is built around a key insight: addition is a two-phase algorithm.

**Layer 0 (digit pairing + partial sum):** Attention uses ALiBi with slope=log(10), which creates a natural base-10 weighting — position distance of 11 (the gap between corresponding X and Y digits) maps to an attention bias of `11 * log(10) ≈ 25.3`, creating a strong peak at the matching digit. The FFN computes a lookup table: 11 candidate values (one per possible digit 0-9, plus a carry scaling factor) are computed and stored in dimensions 5-15 of the residual. The gate selects based on the digit value.

**Layer 1 (carry resolution):** Q=K=0, producing *uniform causal attention* — every previous position contributes equally. This averages the carry signals from all preceding digits, essentially implementing a cumulative carry propagation. The FFN then applies a V-shaped absolute-value function (`relu(x) + relu(-x)`) to snap the continuous carry signal to the nearest integer.

**Key techniques:**
- `softmax1` (softmax with +1 in denominator) — allows attention weights to sum to less than 1, creating an implicit "attend to nothing" option
- ALiBi slope = log(10) — the decimal system's structure is baked into the attention bias
- Identity down-projections (0 params) — the FFN writes directly into specific residual dimensions
- Broadcast parameters — one scalar serves as a weight for an entire dimension
- float64 precision — avoids numerical errors in the carry resolution

### param_40 — 40 parameters (Wonderfall)

File: `external/param_40.py`

```
1 layer, d_model=2, 1 head, head_dim=2
Vocab=10, RoPE with period=19
Tied Q/K, tied V/O, parameterless RMSNorm
```

**How it works:**

The most elegant of the three. d_model=2 means the entire model operates in a 2D plane.

**Embedding:** `embed(d) = [1000 - 0.001*d², -d]`. Dimension 0 is a near-constant (~1000), dimension 1 encodes the digit value linearly. The quadratic perturbation in dim 0 creates a parabolic curve that serves double duty for output decoding.

**Positions:** RoPE with period=19, carefully chosen because 19 ≈ the gap between corresponding input digits (10 digits + separators). This means corresponding X[i] and Y[i] positions rotate by approximately the same angle, making their dot products large. The choice is ROPE_PERIOD = 19.0, not a power of 2 or a multiple of 10 — it's tuned to the sequence geometry.

**Attention (tied Q/K + tied V/O):**
- Q = K projection: both use the same `w_qk` matrix. Q gets an additional phase rotation (`q_phase = -PHI`), creating asymmetry. This is remarkable: Q and K are *tied* (same matrix) but the phase rotation breaks the symmetry that our experiments showed is fatal for trained models.
- V = O projection: both use the same `w_vo` matrix, with V applied as `x @ w_vo.T` and O applied as `out @ w_vo`. This saves parameters by reusing the same matrix in both directions.
- v_scale = -22.0 * DIGIT_SCALE — a large negative scalar that amplifies and inverts the digit values through attention.

**MLP:** Two 2x2 matrices with ReLU. w1 creates a "two-hinge" function that detects whether the weighted sum of digits exceeds a carry threshold (around 9.4-9.5). w2 writes the carry correction (-100/+100 in dim 1) back into the digit dimension.

**Output:** `embed.as_linear(h)` — tied embedding used for decoding. The parabolic structure (`1000 - 0.001*d²`) creates a distance metric where the output logit for digit d is maximized when `h·embed(d)` is largest. Because dim 0 is near-constant and dim 1 is `-d`, the decoding is dominated by the dim-1 inner product, effectively implementing `argmax(-h[1] * (-d)) = argmax(h[1] * d)`.

**Key techniques:**
- Tied Q/K with phase rotation to break symmetry (0 extra params)
- Tied V/O (transpose reuse of same matrix)
- RoPE period tuned to sequence geometry (period=19)
- Parabolic embedding for dual encode/decode use
- Parameterless RMSNorm (no learnable weights, just normalization)

### 87-param model (87_torch.py)

File: `external/87_torch.py`

```
2 layers, d_model=5, 2h/1kv (GQA), head_dim=2
Vocab=10, RoPE (default theta=10000)
ParameterlessEmbedding, Low-rank LM head
71 trainable + 16 frozen = 87 total
```

**How it works:**

A Qwen3-like architecture with hand-set weights. Less elegant than the other two — uses explicit numerical weight matrices found via optimization/analytical design.

**Embedding:** Parameterless — `[100, digit_value, 0, 0, 0]`. Dim 0 is a constant, dim 1 is the digit. No learned embedding at all.

**Attention:** GQA with 2 query heads, 1 KV head. Q and K projections are rank-1 (`x[..., 0:1] * u`), meaning they only read from dim 0 (the constant). This makes Q and K effectively position-only (after RoPE), with no digit-content influence on attention routing. V is parameterless — it copies dim 1 (digit value) into its output.

**Layer 0:** Attention routes digit values via position-only attention patterns. O_proj writes the attention output into dim 2 (accumulating digit sums). FFN gate uses a sparse 2x3 matrix on dims 0-2 with extremely large values (~60000), creating a hard binary gate for carry detection. The up/down projections implement carry computation.

**Layer 1:** Similar structure but operates on the accumulated values. FFN uses a full 5->3 gate projection to implement the final carry resolution. The extremely large gate values (~60000) create hard step functions, implementing exact arithmetic rather than approximate.

**LM head:** Low-rank (rank-2) factorization of a hand-designed 10x5 weight matrix. The matrix values are analytically computed to create decision boundaries at the correct digit thresholds.

**Key techniques:**
- Parameterless embedding (digit value is raw input, no learned representation)
- Rank-1 Q/K projections (attention routing depends only on position, not content)
- Extremely large gate values for hard step-function behavior
- Pre-computed LM head weight matrix with low-rank factorization

---

## The Fundamental Problems

### Problem 1: The Discoverability Gap

The most striking finding: hand-coded models need ~36-40 parameters while trained models need ~200+. This **~5x overhead** is not about representational capacity — it's about discoverability.

Hand-coded models can use:
- Exact numerical constants (e.g., slope = log(10), period = 19.0)
- Hard step functions via extreme weight magnitudes (~60000)
- Degenerate structures (rank-1 projections, identity mappings, tied Q/K with phase rotation)
- Task-specific sparsity patterns (zeros in exact positions)

SGD must discover all of these through gradient flow from random initialization. The loss landscape makes this hard because:

1. **Non-smooth optima:** The hand-coded solutions sit at sharp, narrow points in weight space. The carry detection gate in 87_torch.py uses values of ±60000 — gradient-based optimization would need to traverse enormous distances in weight space to reach these, and the gradients near the final solution are essentially zero (flat plateau → sharp cliff).

2. **Compositional dependency:** Each component must be correct simultaneously. The carry circuit only works if (a) attention routes correctly, (b) FFN detects carry threshold correctly, AND (c) the output head decodes correctly. Partial solutions receive no gradient signal because a partially-correct carry circuit produces no improvement on wrong-carry examples.

3. **Symmetry breaking:** The param_40 model ties Q=K but breaks symmetry with a phase rotation. Our experiments show tied Q=K fails at 39% accuracy when trained. SGD cannot discover the phase rotation trick because: (a) the model doesn't have a phase parameter, and (b) even if it did, the gradient landscape around the critical PHI value is flat until it's exactly right.

### Problem 2: Precision and Number Representation

All three hand-coded models use tricks that exploit numerical precision:

- TinyAdder uses **float64** throughout
- param_40 uses large constants (EMBED_CONST=1000, V_SCALE=1e4) to separate signal from noise
- 87_torch uses gate values of ~60000 to create hard step functions

Trained models at float32 face a fundamental tension: they need large weight magnitudes for sharp decision boundaries, but large weights cause gradient instability during training. This is why our model needs RMSNorm (to control magnitudes) and careful LR scheduling — overhead that hand-coded models don't pay for.

**Hypothesis:** The "grokking" phenomenon in trained addition models is partially explained by this: the model first memorizes (small weights, smooth boundaries), then gradually sharpens its decision boundaries during extended training. The sharp phase transition is the moment when carry-detection circuits become precise enough to work on *all* digit positions simultaneously.

### Problem 3: The Position Encoding Bottleneck

Each hand-coded model found a different zero-parameter position solution:

| Model | Position method | Params | Key insight |
|---|---|---|---|
| TinyAdder (36p) | ALiBi, slope=log(10) | 0 | Base-10 structure maps to log-linear attention decay |
| param_40 (40p) | RoPE, period=19 | 0 | Sequence geometry (gap between X[i] and Y[i]) determines period |
| 87_torch (87p) | RoPE, default theta | 0 | Position-only attention (rank-1 Q/K reads only the constant dim) |
| Taghadouini (228p, trained) | RoPE, theta=3 | 0 | Small theta creates separated frequencies |
| **Ours (203p, trained)** | **Spiral + correction** | **15** | Parametric spiral with linear correction |

Our model spends 15 parameters (7.4% of budget) on positional information that every hand-coded model gets for free. This is a clear compression target, but the deeper question is: **why can't SGD discover the right RoPE theta or ALiBi slope?**

Answer: In our architecture, position information enters via concatenation (`[tok; pos]`) rather than via attention bias (ALiBi) or rotation (RoPE). The concatenation approach requires *explicit* position vectors because Q and K only see the pos_dim subspace — they can't learn position from the attention mechanism's built-in structure. Switching to ALiBi or RoPE would eliminate the position parameter cost but requires abandoning the split-subspace design that gives us our other compression wins.

### Problem 4: Weight Tying Paradox

param_40 demonstrates something our experiments said was impossible: **tied Q=K projections**. It achieves 100% accuracy with Q=K, while our trained model caps at 39% with tied Q=K.

The resolution: param_40 adds a **phase rotation** to Q (`q_phase = -PHI`), which breaks the Q=K symmetry *without adding a separate K matrix*. The rotation is a single scalar parameter. After rotation, Q and K are effectively different linear functions of the input:

```
Q(x) = rotate(W_qk @ x, -PHI)
K(x) = W_qk @ x
```

This is functionally equivalent to having separate Q and K projections, but compressed into fewer parameters (one shared matrix + one rotation scalar). **Our training experiments never had the ability to discover this** because:
1. Our architecture doesn't include a Q-phase parameter
2. Even if it did, the gradient signal for the correct PHI is extremely weak — it requires knowing the exact RoPE period and sequence geometry

**Implication for trained models:** Adding a learnable Q-phase rotation (1-2 params) might unlock tied Q=K, saving 18 params minus the 1-2 for the phase. This is a much cheaper version of "untied Q/K" — instead of learning two independent 3->6 matrices (36p), learn one shared matrix plus a rotation angle (18+1 = 19p, saving 17p).

### Problem 5: The Embedding Dual-Use Problem

All three hand-coded models solve the embedding/decoding duality differently:

- **param_40:** Parabolic embedding `[1000 - 0.001d², -d]` — dim 0 is near-constant for all digits, dim 1 linearly encodes the digit. Decoding uses `embed.as_linear(h)`, which computes inner products. The parabolic term creates a quadratic correction that improves discrimination.

- **TinyAdder:** Sparse embedding (13 params for 14 tokens). Doesn't use tied output — instead uses `argmin` on a 10-dim output. The FFN outputs directly encode candidate digit scores.

- **87_torch:** Parameterless embedding `[100, digit, 0, 0, 0]` with a separate rank-2 LM head. The LM head matrix is analytically designed with exact decision boundaries.

**Our model:** tok_emb (14x3 = 42p) tied with `head_proj(6->3) @ tok_emb.T`. This is 42p for both encoding and decoding. The hand-coded models suggest that:
1. Token identity can be encoded in 1 dimension (just the digit value)
2. The remaining dimensions can be constants or positional signals
3. Decoding doesn't require the same representation as encoding — a separate low-rank head can be cheaper

---

## Theories

### Theory 1: SGD Cannot Discover Degenerate Solutions

The hand-coded models all use **degenerate** weight configurations: rank-1 projections, exact zeros, identity mappings, extreme magnitude separations. These sit at sharp cusps in the loss landscape that gradient descent cannot reach from generic initialization.

**Prediction:** If we initialize a model *near* a hand-coded solution and train, it will stay near it and achieve 100% accuracy with fewer parameters. But if initialized randomly, SGD will find a different (higher-parameter) basin. The "trained" and "hand-coded" categories are exploring fundamentally different regions of weight space.

**Testable experiment:** Initialize our 203p model with weights derived from param_40's solution (projected into our architecture) and see if training maintains or improves the solution. If it works, this proves the degenerate basin exists and is stable — SGD just can't find it from scratch.

### Theory 2: The Carry Circuit Is the Bottleneck

Across all three hand-coded models, the most parameter-expensive component is carry detection and propagation. This is also the last thing trained models learn (grokking). The fundamental challenge:

- Detecting a carry requires knowing if `x_i + y_i + carry_in >= 10`
- Propagating a carry requires knowing if `x_i + y_i = 9` (conditional propagation)
- Chain carries (999...9 + 1) require the model to propagate information across arbitrary distances

The hand-coded models solve this with two different mechanisms:
- **Attention-based carry** (param_40, 87_torch): Attention routes digit-pair sums, FFN detects threshold crossings
- **Averaging-based carry** (TinyAdder): Uniform causal attention (Q=K=0) computes running averages, then hard-thresholds

**For trained models:** Our carry-mix curriculum explicitly addresses this by oversampling carry-heavy examples. But the hand-coded models suggest a more radical approach: the carry circuit might benefit from architectural inductive biases (like a dedicated carry dimension, or a hard-coded threshold nonlinearity) rather than hoping SGD discovers the right weight configuration for carry detection.

### Theory 3: Positional Encoding Should Match the Arithmetic Structure

Every successful model (hand-coded or trained) uses positional encoding that somehow reflects the base-10 structure of the problem:

- ALiBi slope = log(10): attention decays by a factor of 10 per position
- RoPE period = 19: matches the input sequence geometry
- RoPE theta = 3: creates frequencies that distinguish all 35 positions
- Our spiral: period-10 angular structure in the pos_dim subspace

**The deeper insight:** The "right" positional encoding for addition is not a generic sequence encoding — it's one that makes the attention pattern naturally align X[i] with Y[i] for all i simultaneously. This is fundamentally different from natural language, where the "right" positions are about recency and syntactic distance.

**Implication:** Instead of learning positions (15 params) or using generic RoPE, we should design a positional encoding that is specifically structured for addition. The hand-coded models show this is possible with 0 params. Even if we need 2-4 learnable params to tune it (like a learnable theta for RoPE), that's a 10+ param saving.

### Theory 4: Superposition Is Both Friend and Enemy

Our 203p model operates at d_model=6, which is large enough to potentially use superposition — encoding more than 6 independent features by distributing them across non-orthogonal directions. The hand-coded models suggest this is both unnecessary and harmful for addition:

- param_40 uses d_model=2 with perfect accuracy. No superposition needed.
- TinyAdder uses d_model=5 but expands to 16 dims mid-forward pass. The internal computation needs more dimensions than the residual stream.
- 87_torch uses d_model=5 with specific dimensions assigned to specific roles (dim 0 = constant, dim 1 = digit, dim 2 = accumulated sum, dim 3 = carry, dim 4 = auxiliary).

**For trained models:** SGD may be wasting parameters on superposition — learning non-orthogonal feature encodings that are harder to decode than axis-aligned ones. The hand-coded models show that addition can be solved with clean, interpretable per-dimension assignments. This suggests that training techniques that encourage axis-aligned representations (e.g., L1 regularization on activations, or explicit dimension-role assignment) might improve parameter efficiency.

### Theory 5: The Output Problem Is Separable

All hand-coded models treat digit classification (output) as a separate, solvable problem from the internal computation. The 87_torch model pre-computes exact decision boundaries in its LM head. The param_40 model uses the parabolic embedding's geometric properties.

**For trained models:** Our tied output head (`head_proj(6->3) @ tok_emb.T`) forces the token embedding to serve both encoding and classification. The hand-coded models suggest these are better separated. A small, analytically-designed output head might free the embedding to focus purely on input representation, saving parameters on the embedding side.

---

## Proposed Experiments

### Exp A: Learnable Q-Phase with Tied Q/K (target: ~186p)

**Inspired by:** param_40's tied Q/K with phase rotation.

Add a learnable phase rotation angle to Q while tying Q and K projections. This gives us the 18p saving from tied Q/K minus 1p for the phase angle = net 17p saving.

```
Current:  q_proj(18p) + k_proj(18p) = 36p
Proposed: qk_proj(18p) + q_phase(1p) = 19p  (save 17p)
```

**Implementation:** After computing `q = qk_proj(x_pos)`, apply a 2D rotation to pairs of q dimensions using a learned angle. This is equivalent to RoPE with a learned offset.

**Risk:** Our head_dim=3 (odd), so rotation pairs don't divide evenly. Need head_dim=2 or 4. Alternatively, apply the phase as a scalar complex multiplication on pairs of dims, leaving the third dim unrotated.


### Exp B: Parameterless Embedding (target: ~161p)

**Inspired by:** 87_torch's `ParameterlessEmbedding` and param_40's `[const, -digit]`.

Replace our learned 14x3 tok_emb (42p) with a fixed embedding that encodes digit value directly:

```python
embed(d) = [cos(2pi*d/10), sin(2pi*d/10), d/9]  # our current spiral init, but frozen
```

Or simpler:
```python
embed(d) = [1.0, d/9.0, 0.0]  # constant + linear digit + zero
```

Since tok_emb is tied with the output head, the output logits become `head_proj(h) @ fixed_embed.T`. The head_proj (18p) must carry all the discriminative power. This replaces 42p of embedding with 0p.

**Risk:** High. Our spiral init works as a starting point for training, but the trained embeddings drift significantly (mean drift 1.24). The model needs per-digit freedom. However, param_40 achieves 100% with a fixed 2D embedding — the question is whether our architecture can route information correctly with a rigid embedding.

**Mitigation:** Start with the spiral frozen, train everything else. If it fails, try a loose version: parametric embedding with 4-6 params instead of 42.


### Exp C: Hard-Threshold Carry Detection (architectural inductive bias)

**Inspired by:** All three hand-coded models use hard step functions for carry detection.

Our FFN (dim=2, GELU) must learn a smooth approximation to the hard carry threshold. The hand-coded models show that the carry function is fundamentally a step function: `carry = 1 if (sum >= 10) else 0`. GELU is a poor match for this.

**Proposal:** Replace GELU with a steeper nonlinearity during training:
1. Train with GELU initially for smooth gradients
2. Anneal toward a hard sigmoid or step function over training
3. Or: use `tanh(alpha * x)` with increasing alpha during training (starts smooth, ends sharp)

This is *not* hand-coding the solution — the model still learns *where* the threshold is. We're just giving it a better nonlinearity for implementing sharp decisions once it finds the right location.

**Alternative:** Use the SiLU/swish activation that the hand-coded Qwen3-like models (87_torch) use. The gated FFN structure (`gate * up`) with SiLU creates sharper transitions than GELU.


### Exp D: Explicit Dimension Roles (structured residual stream)

**Inspired by:** 87_torch's dimension assignment (dim 0 = constant, dim 1 = digit, dim 2 = accumulator, dim 3 = carry).

Our d_model=6 residual stream has implicit structure (tok_dim=3, pos_dim=3) but within each subspace, SGD must discover what each dimension represents. The hand-coded models show that explicit roles dramatically reduce the parameter count.

**Proposal:** Add soft constraints that encourage axis-aligned representations:
1. L1 penalty on off-diagonal elements of projection weight matrices (encourages sparse, axis-aligned projections)
2. Or: initialize with the hand-coded dimension assignment and use a small LR on structural weights
3. Or: reduce d_model from 6 to 4 or even 2 (param_40 proves d=2 is sufficient) and retrain

**The radical version:** d_model=2, single head, head_dim=2, with RoPE. This is essentially param_40's architecture but trained instead of hand-coded. If the trained version works at ~50-80p, it's a massive improvement over our 203p.


### Exp E: Softmax1 (Attention with Implicit "No-Op" Head)

**Inspired by:** TinyAdder's `softmax1` — softmax with +1 in the denominator.

Standard softmax forces attention weights to sum to 1. softmax1 allows them to sum to *less than* 1, effectively creating an implicit "attend to nothing" option. This is useful for addition because some positions (e.g., leading zeros) carry no useful information, and the model should be able to ignore them entirely.

```python
def softmax1(x, dim=-1):
    exp_x = x.exp()
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))
```

**Implementation:** Replace `F.softmax(att, dim=-1)` with softmax1 in our attention. Zero additional parameters. May improve gradient flow by reducing the "attention sink" phenomenon where the model wastes capacity attending to padding/delimiter tokens.


### Exp F: Uniform Causal Attention Layer (carry propagation)

**Inspired by:** TinyAdder's Layer 1, which uses Q=K=0 for uniform causal attention.

Add a second pass (or layer) where Q=K=0, giving uniform causal attention. This computes a running average of all previous positions — a natural primitive for carry accumulation. The FFN after this layer can threshold the average to detect carries.

**Implementation:** In a shared-layer (universal transformer) setup, the second pass could use fixed Q=K=0 (zero additional params) while the first pass uses learned Q/K. This would give us a 2-layer model with a carry-specialized second layer at zero extra attention params.

**Risk:** Uniform attention averages *all* previous positions, including non-digit positions (delimiters). The FFN must learn to discount these. With vocab=10 (no special tokens), this becomes cleaner.


### Exp G: Train param_40's Architecture from Scratch

**The direct test:** Take the exact param_40 architecture (d_model=2, 1 head, head_dim=2, tied Q/K with phase, tied V/O, RoPE period=19, parameterless RMSNorm, vocab=10) and try to *train* it from random initialization.

This model has ~40 parameters. If SGD can find a solution, it's an instant 5x improvement over our 203p. If it can't, the failure mode tells us exactly what SGD struggles with:
- Does it fail to find the right RoPE frequency? → Position encoding is the bottleneck
- Does it fail at carry detection? → The FFN/nonlinearity is the bottleneck
- Does it fail at Q/K asymmetry? → The tied-with-phase trick isn't SGD-discoverable
- Does it fail everywhere? → The architecture is too tight for gradient-based learning

**Variants:**
1. Train with param_40's exact architecture (40p)
2. Relax to untied Q/K (adds ~4p, removes the phase trick)
3. Relax to learned embedding (adds ~20p, tests if parameterless embedding is the bottleneck)
4. Relax all constraints simultaneously (~80-100p) and see what minimum works

This is the most informative single experiment we can run. The gap between "can represent" and "can discover" is the fundamental question.


### Exp H: Warm-Start from Hand-Coded Weights

**The convergence test:** Initialize our model with weights projected from a hand-coded solution, then fine-tune.

If the hand-coded solution lies in a good basin that SGD can maintain, fine-tuning should preserve or improve accuracy. If SGD drifts away from the hand-coded basin, it tells us the solution is at an unstable fixed point of the training dynamics.

**Implementation:**
1. Project param_40's 2D solution into our 6D architecture (pad with zeros)
2. Or: adapt 87_torch's 5D solution into our 6D architecture (close match)
3. Train with very small LR initially, then increase
4. Monitor which weights drift most from their hand-coded values — these are the "SGD-incompatible" components

---

## Bigger Questions This Research Addresses

### 1. The Representation-Learning Gap in Transformers

The AdderBoard challenge provides the cleanest possible measurement of the gap between what transformers can represent and what they can learn. For addition: representation floor ≈ 33-40p, learning floor ≈ 200p, ratio ≈ 5-6x. This ratio likely generalizes to other algorithmic tasks.

**Open question:** Is this ratio fundamental to gradient-based learning, or is it an artifact of our training techniques? If better optimizers, curricula, or architectural biases can close the gap, it has implications for all of deep learning — most transformer parameters may be "discoverability overhead" rather than capacity.

### 2. Grokking as Basin Transition

The grokking phenomenon (sharp accuracy jump after long training) can be understood through the lens of hand-coded solutions. The model must transition from a smooth, memorization-based basin to a sharp, algorithmic basin. The hand-coded solutions show that the algorithmic basin has extreme weight magnitudes and degenerate structures — exactly the kind of configurations that regularization (weight decay) discourages.

**Paradox:** Weight decay helps grokking (empirically), but the target solution has extreme weights. Resolution: weight decay prevents the model from getting stuck in the wrong sharp basin (memorization), forcing it through the smooth landscape until it finds the algorithmic circuit. Once found, the circuit amplifies itself and overcomes the decay.

### 3. Superposition vs. Clean Features

Recent mechanistic interpretability work focuses on superposition — models encoding more features than dimensions. The hand-coded addition models show that for algorithmic tasks, **superposition is unnecessary and harmful**. Clean, axis-aligned representations with 2-5 dimensions suffice. If trained models use superposition for addition, they're wasting parameters on an encoding that isn't needed.

**Broader implication:** For tasks with clean algorithmic structure, the "features as directions" framework (Anthropic's transformer circuits work) might be too generous. The optimal representation is features as *axes*, with exact zero in unused dimensions. Training techniques that encourage this (orthogonality penalties, sparse activations) could be transformative.

### 4. The Role of Precision in Algorithmic Learning

float64 vs float32 is a real difference for these models. The carry detection function is a hard step — any smoothing from finite precision introduces errors. At float32, the model needs wider networks to approximate the step function robustly. At float64, a single neuron suffices.

**Open question:** Would mixed-precision training (float32 for gradients, float64 for carry-critical weights) help trained models? Or: would quantization-aware training that explicitly plans for precision limits lead to different (possibly better) weight configurations?

### 5. Architecture Search via Hand-Coded Proofs

The hand-coded solutions serve as constructive proofs that certain architectures *can* solve addition. This inverts the usual ML workflow: instead of training and hoping, we first prove an architecture can represent the solution, then ask whether SGD can find it.

**Meta-experiment:** For each proposed architecture change (e.g., d_model=2, tied Q/K with phase, RoPE with specific theta), first hand-code a solution to prove it works, then try to train it. This eliminates wasted compute on architectures that can't represent the answer.

---

## References

- Litzenberger, A. "Building a Minimal Transformer for 10-Digit Addition." https://alexlitzenberger.com/blog/post.html?post=/building_a_minimal_transformer_for_10_digit_addition
- Weiss, G., Goldberg, Y., Yahav, E. "Thinking Like Transformers." ICML 2021. https://arxiv.org/abs/2106.06981
- Lindner, D. et al. "Tracr: Compiled Transformers as a Laboratory for Interpretability." DeepMind, 2023. https://arxiv.org/abs/2301.05062
- Nanda, N. et al. "Progress Measures for Grokking via Mechanistic Interpretability." ICLR 2023. https://arxiv.org/abs/2301.05217
- Giannou, A. et al. "Looped Transformers as Programmable Computers." 2023. https://arxiv.org/abs/2301.13196
- Shen, R. et al. "Looped Transformers for Length Generalization." ICLR 2025. https://arxiv.org/abs/2409.15647
- McLeish, S. et al. "Transformers Can Do Arithmetic with the Right Embeddings." NeurIPS 2024. https://arxiv.org/abs/2405.17399
- Stolfo, A. et al. "Understanding Addition and Subtraction in Transformers." 2024. https://arxiv.org/abs/2402.02619
- Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Anthropic, 2021. https://transformer-circuits.pub/2021/framework/index.html
- Taghadouini, S. "minimal-ten-digit-addition-transformer." 2026. https://github.com/staghado/minimal-ten-digit-addition-transformer
- AdderBoard. https://github.com/anadim/AdderBoard
