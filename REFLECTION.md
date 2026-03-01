# Reflection: 75 Parameters From Scratch

## How a Transformer Learned Perfect 10-Digit Addition in 75 Parameters

This document reflects on the research journey that compressed a trained transformer for 10-digit addition from 242 to **75 learnable parameters** — a **69% reduction** — while maintaining 100% accuracy on 10,010 test cases. Every parameter is learned from random initialization. No warm-starting, no frozen pretrained values, no inherited knowledge.

Along the way we explored a warm-start cascade down to 62p, tried and failed scaffold training (L1 annealing, freeze-in-place), swept seeds, and ultimately concluded that warm-starting with frozen learned values is a technicality — interesting research, but not a legitimate training result. The real frontier is what SGD can discover from scratch.

---

## 1. The Journey in Numbers

```
Sessions 1-2: 242p → 75p  (architectural compression, all from scratch)
Session 3:     75p → 62p  (warm-start cascade — later rejected as a technicality)
Sessions 4-5:  training innovation attempts (SAM, WD variants, seed sweep)
Session 6:     scaffold training (8 experiments, all failed)
```

**9 from-scratch breakthroughs**, each verified at 100% accuracy (10010/10010):

| Step | Params | Technique | Saving |
|------|--------|-----------|--------|
| 1 | 226p | Spiral positions | -16p |
| 2 | 214p | Rank-2 out_proj | -12p |
| 3 | 203p | Linear correction + freeze EOS | -11p |
| 4 | 187p | Tied Q/K + q-phase | -16p |
| 5 | 170p | tok_dim=2 reshape | -17p |
| 6 | 133p | Vocab=10 + d_model=5 + parametric emb | -37p |
| 7 | 100p | Rank-1 out_proj + no FFN bias | -33p |
| 8 | 78p | Tied V/O + shared norms, no pos correction | -22p |
| **9** | **75p** | **PLUS/EOS positions fixed at zero** | **-3p** |

Total: 242p → 75p, a **69% compression**, all from-scratch.

### Why We Don't Count Warm-Start Results

We also explored a warm-start cascade from 75p down to 62p by progressively freezing parameters at their trained values. Each frozen parameter was converted from `nn.Parameter` to `register_buffer` and loaded with the value from the parent checkpoint. The 62p model has 62 learnable parameters but relies on 18 frozen buffer values extracted from upstream trained models.

This is a technicality. You could take any 242p model, freeze all but one parameter, and claim a "1 parameter model." The spirit of the competition is about what SGD can discover, not what post-hoc compression can preserve. The competition rule *"Fixed/sinusoidal positional encodings are not counted"* points the right direction: fixed values with no learned information don't count, but conversely, frozen *learned* values should.

Our legitimate frontier is **75 parameters from scratch**.

---

## 2. Theories Tried, Validated, and Refuted

### Validated Theories

**"Attention is positional routing, not content-dependent."**
Our earliest structural finding at 170p: attention patterns are fixed functions of position, not token content. It predicted that vocab=10 (merging special tokens with digits) would work — and it did. This theory held through every compression stage.

**"Per-head phase rotation provides sufficient Q/K asymmetry."**
Tied Q/K (K = Q) naively caps at 39% accuracy because carry routing needs asymmetric attention. A single learnable angle per head rotates Q relative to K, breaking symmetry with 1 parameter instead of a full K projection (15p). This idea, borrowed from the hand-coded param_40 model, saved 16p and was the foundation for all subsequent compressions.

**"The model converges toward symmetry."**
At every stage, diagnostics showed the model converging toward simpler structures. At 75p, fresh diagnostics confirm this continues:
- **tok_arc A/B ratio = 1.005**: Near-perfect circle (0.5% relative difference)
- **spiral_slope = -0.058**: The z-gradient over 10 digits is only 0.52, while spiral_amp = 14.6. The z-dimension of digit positions is effectively a constant bias (offset = -7.3), not a ramp.
- **out_proj is extremely sparse**: A writes to dims 0,1,4 only; B writes almost entirely to dim 1 (2.27 vs <0.03 for all others)

Each convergence is a signal: the model tells you what's redundant.

**"Constraint interactions are scale-dependent."**
Many "hard constraints" from d_model=6 dissolved at d_model=5: norm sharing (impossible → trivial), rank-1 out_proj (dead → works), FFN bias (required → optional), tied V/O (destroys output → beneficial regularization). You can't plan a compression roadmap in advance — you have to re-test everything at each scale.

### Refuted Theories

**"All 3 RMSNorm layers are highly specialized."**
At 170p (d_model=6), pairwise norm similarity was only 0.45-0.67. At d_model=5, all three converge to identical weights. The specialization was an artifact of excess capacity, not a structural requirement. Diagnostics on our 75p model show ln1 = ln2 = ln_f to float precision: `[1.70, 3.14, 1.78, 1.80, 11.03]`.

**"Rank-2 out_proj is the minimum trainable rank."**
At d_model=6, rank-1 was dead (0.08%). At d_model=5, rank-1 works perfectly. The constraint was architecture-specific.

**"FFN bias is required for convergence."**
True at d_model=6, false at d_model=5. Saves 7p.

**"L1 scaffold training can replace warm-starting."**
Eight experiments across three approaches (standard L1, late-onset L1, freeze-in-place) all failed. L1 is fundamentally adversarial to task learning in tiny models: low lambda creates equilibrium (scaffold weights hover nonzero), medium lambda causes catastrophic collapse (carry circuit destroyed), high lambda suppresses learning entirely. You cannot remove structural capacity from a tiny model's carry circuit — it's like removing a wire from a circuit board.

**"SAM helps tiny models find flatter minima."**
SAM (rho=0.05 and 0.01) was uniformly worse than vanilla AdamW at 72p. The adversarial perturbation disrupts delicate feature learning. The issue at small sizes is basin accessibility from initialization, not basin sharpness.

**"WD scheduling can substitute for adaptive WD."**
Scheduled drops, cyclical WD, and warmup WD all produced identical ~26% tok_acc at 72p. The failure mode is seed-dependent, not WD-dependent.

---

## 3. Structural Diagnostics of the 75p Model

Fresh analysis of the from-scratch 75p checkpoint (seed 80085) reveals the model's internal structure and points toward future compression targets.

### 3.1 Token Embeddings: A Near-Perfect Circle

The parametric arc `(A*cos(s+i*d), B*sin(s+i*d))` learned:
- **A = 16.03, B = 15.95** — ratio 1.005, effectively a circle
- **start = -0.60, stride = 0.133** — 68.4° total arc span
- All 10 embedding norms are nearly identical (16.006 to 16.030)
- Minimum pairwise distance: 2.116 (digits 4↔5), maximum: 2.119 (digits 0↔1) — remarkably uniform spacing

The model wants a circle. The 0.5% A/B asymmetry is within noise. **Tying A=B saves 1p (→74p)** with near-zero risk.

### 3.2 Positions: The z-Dimension is a Constant

Spiral parameters learned:
- **amp = 14.58**: Large — the xy-plane carries the main positional signal
- **phase = -1.91**: Significant rotation from init (0.0)
- **slope = -0.058**: Tiny. Over 10 positions, z varies by only 0.52
- **offset = -7.30**: Large — the z-dimension is a constant bias, not a meaningful ramp

The digit positions live on a flat ring: circular in xy, nearly constant in z. The z-dimension's main role is separating digit positions (z ≈ -7.3) from the carry position (z_hi dim2 = -3.4) and EQUALS (z ≈ -7.8). It does this via offset alone; slope contributes almost nothing.

**Freezing slope=0 saves 1p (→73p with A=B tie)**. But this failed at 70p from scratch in our earlier attempts — freezing spiral params changes the loss landscape topology.

### 3.3 The Carry Position is Enormous

z_hi_pos = [75.85, 88.13, -3.40] with norm **116.3** — 7.1x the average digit position norm (16.4). The carry-out position is deliberately pushed far from all digit positions in the attention space. This massive separation ensures clean attention routing: the carry position can never be confused with a digit position.

### 3.4 The Norm as Feature Selector

Shared norm weights: `[1.70, 3.14, 1.78, 1.80, 11.03]`

Dim 4 is amplified **6.5x** relative to dim 0. Dim 1 gets 1.8x amplification. Dims 0, 2, 3 pass through at ~1.8x (near-equal).

This makes dim 4 the **information highway**: head_proj weights for dim 4 are [-16.05, 23.24] — the largest by far. The norm selectively amplifies the dimension that carries the most decision-relevant information to the output head. At d_model=5, the norm acts less as a normalizer and more as a learned feature gate.

### 3.5 Out_proj is Extremely Sparse

The rank-1 out_proj factors as A(5x1) @ B(1x5):
- **B is dominated by dim 1**: value -2.27, all others < 0.03
- **A reads mainly from dims 0,1,4**: [-0.16, -0.73, -0.005, -0.008, 0.26]
- Dims 2 and 3 of A are near zero (< 0.01)

The attention output is projected almost entirely onto **dim 1** of the residual stream. This makes out_proj effectively a 3-to-1 mapping (dims 0,1,4 → dim 1), wasting the capacity reserved for dims 2,3. Could a sparser parameterization exploit this?

### 3.6 FFN: Both Units Read Tokens, Write Tokens

| Unit | Token input | Position input | Token output | Position output |
|------|------------|----------------|--------------|-----------------|
| 0 | 3.80 | 0.60 | 6.37 | 0.70 |
| 1 | 2.61 | 0.87 | 2.39 | 0.60 |

Both FFN units primarily read from and write to the token subspace (dims 0-1). Position input is small but nonzero. This is different from the 170p model where unit 0 was clearly a "pos→tok bridge." At d_model=5 with tied V/O, the FFN's role has shifted — it processes token information directly, with only residual position dependence.

### 3.7 Out_proj.B ≠ Head_proj: Tying Won't Work

Cosine similarity between out_proj.B and head_proj rows: 0.26 and 0.45. These are not naturally aligned. The "most promising avenue" from our earlier analysis (tie out_proj.B with head_proj to save 5p) is **not supported by the trained weights**. The read-side (V/head_proj) and write-side (out_proj.B) serve genuinely different functions and have not converged.

---

## 4. What We Learned About Architecture

### The Irreducible Core (75p From Scratch)

| Component | Params | Why It's Essential |
|-----------|--------|--------------------|
| q_proj | 15 | Maps 3D positions to 5D attention space |
| FFN (fc1+fc2) | 20 | Carry detection: both units read tokens, write corrections |
| head_proj | 10 | Output classification + V projection (tied V/O) |
| out_proj | 10 | Rank-1 projection of attention output (writes to dim 1) |
| shared norm | 5 | Feature gate: amplifies dim 4 by 6.5x for output routing |
| tok_arc (4) | 4 | Parametric circle: A, B, start, stride |
| spiral (4) | 4 | Position encoding: amp, phase, slope, offset |
| z_hi_pos | 3 | Carry position (huge norm, far from digits) |
| equals_pos | 3 | EQUALS delimiter position |
| q_phase | 1 | Q/K asymmetry rotation (-39.5°) |

The 15p q_proj is the largest single block and seems incompressible: rank-1 fails (1D keys can't distinguish positions), rank-2 would cost 16p > 15p at these dimensions. The FFN at 20p is at minimum: ffn_dim=1 can't do carry detection. These 35p of "processing" parameters appear to be a hard floor.

### What's Different From 170p

| Property | 170p | 75p |
|----------|------|-----|
| Token embeddings | 28p learned table | 4p parametric arc (circle) |
| Positions | 14p (spiral+correction) | 11p (spiral only, PLUS/EOS=0) |
| d_model | 6 (tok=2, pos=4) | 5 (tok=2, pos=3) |
| Heads | 2 (head_dim=3) | 1 (head_dim=5 = d_model) |
| Norms | 18p (3 independent) | 5p (shared) |
| V projection | 12p (separate) | 0p (tied with head_proj) |
| Out_proj | 24p (rank-2) | 10p (rank-1) |
| FFN | 32p (with bias) | 20p (no bias) |

Every component shrank. The biggest savings came from going to d_model=5 with a single full-rank head — one head with head_dim=5 is more expressive than two heads with head_dim=3, and the reduced d_model cascades savings through every layer.

---

## 5. What We Learned About Grokking

### Grokking at Every Scale

Every from-scratch model exhibited the same pattern:
1. **Memorization**: tok_acc rises to ~50-70%, exact match stays near 0%
2. **Grokking transition**: Sudden jump in exact match (0% → 90%+ in a few thousand steps)
3. **Oscillation**: Model bounces between 50-100% exact for 50-100K steps
4. **Stabilization**: Locks in at 100% (or fails permanently)

### Adaptive Weight Decay is the Key Training Innovation

The two-stage adaptive WD schedule was essential for every sub-100p model:
- **Stage 1** (wd × 0.1): Triggered when val_exact > 1%
- **Stage 2** (wd × 0.01): Triggered when val_exact > 5%

The mechanism: the carry circuit in the FFN needs weights to grow large (toward step-function behavior). WD pushes weights toward zero. Dropping WD at the right moment lets the circuit sharpen. Without adaptive WD, grokking either doesn't happen or takes 10-100x longer.

### Seed Sensitivity is Extreme

A 10-seed sweep at 75p and 72p revealed:
- **~10-20% of random seeds** show any grokking signal
- **Only s80085** is confirmed to stably grok both 75p and 72p
- **Grokking seeds are config-specific**: s78779 groks 75p (95.8% flash) but fails 72p; s67086 fails 75p but approaches 100% at 72p
- **Flash grokking exists**: 75p s78779 spiked to 95.8% exact at 33K then crashed to 0.5% by 51K — the carry circuit crystallized briefly but dissolved in a shallow basin

Each architectural constraint (even freezing a single parameter to zero) changes the loss landscape topology, redirecting which initialization basins lead to the addition solution.

### All Training Innovations Failed

SAM, WD scheduling variants, scaffold L1 training, and scaffold freeze-in-place all failed to improve over vanilla AdamW with adaptive WD. The path from 242p to 75p was entirely architectural. Training innovations made experiments faster (adaptive WD = 18x speedup) but never reduced the parameter floor.

---

## 6. The SGD Discoverability Gap

### From 4.25x to 1.88x

The hand-coded param_40 model achieves addition in ~40 parameters. Our best from-scratch model needs 75 — a **1.88x overhead**.

| Threshold | Params | What it means |
|-----------|--------|---------------|
| Representational floor | ~40p | Hand-coded proof that 40 params suffice (param_40) |
| From-scratch frontier | **75p** | What SGD can discover from random init (this work) |
| Warm-start floor | ~62p | What can be preserved via cascaded freezing (rejected) |

The **discoverability gap** (75/40 = 1.88x) is the overhead SGD pays for not knowing the solution in advance. Closing this gap requires either smarter optimization (which we tried and failed) or architectural innovations that make the loss landscape smoother.

### What SGD Can't Do (Yet)

1. **Parameterless norms.** param_40 uses RMSNorm without learned weights. Our model needs 5 learned norm weights that serve as a feature gate (dim 4 amplified 6.5x). SGD can't discover a representation where all dimensions have equal importance.

2. **Zero-parameter positions.** param_40 uses RoPE (rotation, no learned params). We need 11p of spiral + special positions. The positions carry real information (phase, offset, carry separation) that SGD must discover.

3. **Extreme weight magnitudes.** param_40 uses values of ~60000 for hard step functions. Our FFN weights max out at ~6.3 — an order of magnitude softer. Adaptive WD helps but doesn't fully close this gap. The carry circuit works with approximate thresholds rather than crisp ones.

### What SGD *Can* Do That Surprises

1. **Learn circular embeddings from 4 parameters.** The parametric arc discovers that digits should be equally spaced on a near-perfect circle — 4 params for 10 embeddings that double as output classification boundaries.

2. **Discover weight tying spontaneously.** At d_model=5, tied V/O (v_proj = head_proj.T) works despite analysis showing these matrices are NOT naturally aligned (cosine sim = -0.30 in untied models). The tying constraint forces the model into a different, equally valid solution.

3. **Converge to minimal complexity.** A/B → circle, slope → 0, sparse out_proj → effectively writes to one dimension. The model finds the simplest structure that solves the task.

---

## 7. Research Frontiers: Pushing Below 75p From Scratch

Every approach below must produce a model that trains from random initialization to 100% accuracy with no warm-starting, no frozen learned values, and no inherited knowledge. The full parameter count is the real count.

### 7.1 Low-Hanging Fruit: Tie A=B (→74p)

The trained model converges to A/B = 1.005. Tying tok_arc_A = tok_arc_B architecturally forces a perfect circle. This is the single safest 1p reduction and should be tested first. The key risk is that tying changes the loss landscape enough to lose s80085 as a grokking seed — a pattern we saw repeatedly in the seed sweep.

### 7.2 Freeze slope=0 (→73p with A=B tie)

spiral_slope = -0.058 is functionally negligible (z varies by 0.52 over 10 digits while xy varies by ±14.6). Freezing it to zero makes digit positions a flat ring. However, this is architecturally identical to the 70p configuration that failed from scratch even with s80085 — suggesting this particular constraint breaks something subtle in the initialization dynamics. Worth retesting with A=B tie in place (the combined constraint may be different from the sum of individual constraints).

### 7.3 Structured q_proj (→potential -5 to -10p)

The q_proj is a dense 3×5 matrix (15p). Could it be replaced with a structured form?

Options:
- **SVD truncation**: q_proj has singular values [4.78, 4.25, 1.07]. The third singular value carries only 2.7% of the energy. A rank-2 q_proj would cost 2×(3+5) = 16p — actually MORE than 15p at these dimensions. Not viable via simple rank reduction.
- **Toeplitz/circulant**: Diagnostics show the trained q_proj is NOT Toeplitz (diagonals vary widely). No exploitable structure here.
- **Factored with fixed basis**: `q_proj = U @ diag(s)` where U is a fixed 5×3 matrix (e.g., DFT, Hadamard) and s is 3 learned scales. 3 params instead of 15. The risk: restricting Q to a fixed subspace may prevent the carry-lookahead routing the model needs. Worth testing.
- **Shared q_proj + head_proj structure**: Both are "position→5D" and "5D→token" mappings. A shared parameterization (e.g., q_proj = f(head_proj)) would save parameters but the functional roles are very different.

### 7.4 Sparser Out_proj (→potential -3 to -5p)

The trained out_proj is remarkably sparse:
- B writes to dim 1 only (weight -2.27, all others <0.03)
- A reads from dims 0,1,4 only (dims 2,3 weights <0.01)

This suggests out_proj could be parameterized as:
- **Scalar attention output**: A single scalar per dim, `out = alpha * attn_output[selected_dim]`, plus a target dimension. ~2-3p instead of 10p. Extreme, but the trained weights suggest the model only uses one channel.
- **Sparse rank-1**: Fix B to a one-hot [0,1,0,0,0] vector (0p) and only learn A (5p). Saves 5p but constrains the write target.

### 7.5 Absorb q_phase into q_proj (→0 savings, but clears the path)

q_phase_angle = -0.690 rad (-39.5°). This could be absorbed into q_proj initialization by pre-rotating the Q matrix. Doesn't save parameters directly (q_phase is 1p, absorbed into q_proj's 15p) but simplifies the architecture and may change optimization dynamics.

### 7.6 Novel Norm Parameterization (→potential -2 to -3p)

The shared norm weights [1.70, 3.14, 1.78, 1.80, 11.03] have interesting structure: dims 0,2,3 are near-equal (~1.8), dim 1 is 2x, dim 4 is 6.5x. Could this be captured by a simpler form?

- **2-parameter norm**: `weight = [base, base*r1, base, base, base*r2]` — 3p instead of 5p. The near-equal dims (0,2,3) share a base, dims 1 and 4 get multipliers.
- **Pattern norm**: `weight = softmax(alpha * one_hot(4) + beta * one_hot(1)) * scale` — learns which dims to amplify. 3p for which dims and how much.
- Scalar norm (1p) already failed at 50-60% tok_acc — per-dim weights are essential. But we haven't tested intermediate parameterizations.

### 7.7 Distillation Instead of Warm-Start

If we reject warm-starting with frozen learned values, distillation offers a different path. Train a larger model (e.g., 100p), then train a smaller target (e.g., 70p) with a KL-divergence loss against the teacher's output distribution. Every parameter in the student is learned from scratch — the teacher provides guidance but no frozen values end up in the student.

This is philosophically different from warm-starting: the student must discover its own internal representation. The teacher only constrains the input→output mapping, not the internal weights. If the student achieves 100% accuracy, its parameters genuinely encode the addition algorithm — it wasn't given the answer, just better training signal.

### 7.8 Loss Landscape Visualization

Project the 75p loss landscape onto 2D:
1. Take the trained 75p checkpoint as the origin
2. Choose 2 random directions in parameter space (or: direction toward a failed seed's final state, and a random orthogonal)
3. Evaluate loss on a grid
4. Visualize the basin structure

This would answer: is the 75p solution in a narrow basin (explaining seed sensitivity) or a broad one (suggesting the issue is navigating there from random init)? At different param counts (72p, 70p), does the basin shrink, fragment, or disappear?

---

## 8. Three Thresholds of Compression

The research reveals three fundamentally different compression thresholds:

### Threshold 1: Representational (40p) — "Can the architecture express it?"

Proven by hand-coded models. param_40 achieves 100% addition in ~40 parameters with human-designed weights. No learning needed; the architecture is sufficient. The techniques used (extreme weight magnitudes, RoPE, parameterless norms) are representationally elegant but not SGD-discoverable.

### Threshold 2: Learnability (75p) — "Can SGD discover it from scratch?"

Our main result. 75 parameters, trained from random initialization, 100% accuracy. The 35p overhead over the representational floor (75/40 = 1.88x) is the **price of discoverability** — extra capacity SGD needs to navigate the loss landscape. This overhead pays for: learned norms (5p), richer positions (11p vs 0p), softer weight magnitudes (carried by additional out_proj capacity).

### Threshold 3: Maintainability (62p) — "Can SGD maintain it with a hint?"

Demonstrated by warm-start cascade but rejected as a legitimate result. With the right starting point, SGD can maintain the addition circuit in 62 parameters. But those 18 frozen buffer values are learned information smuggled in from a larger model. The gap between learnability and maintainability (75p vs 62p = 13p) measures how much of the 75p model's capacity is used for *navigation* during training rather than *representation* of the final solution.

---

## 9. Conclusions

### What Made This Possible

1. **Structural diagnostics.** Analyzing what the model actually learns — attention patterns, weight magnitudes, convergence directions, parameter ratios — revealed which parameters were redundant. Each diagnostic finding (A≈B, slope≈0, norms converge) became the next compression target.

2. **Scale-dependent experimentation.** Constraints that were impossible at d_model=6 became trivial at d_model=5. We couldn't plan the compression roadmap in advance. Each step required re-testing assumptions at the new scale.

3. **Principled rejection of shortcuts.** The warm-start cascade was our most "successful" technique (75p→62p) but we rejected it on philosophical grounds. Frozen learned values are information smuggling. This constraint — every parameter must be learned from scratch — focuses research on real innovations rather than accounting tricks.

### The Philosophical Takeaway

**Trainability ≠ expressibility.** The model can *represent* addition in 40 parameters (hand-coded). It can *learn* addition from scratch in 75 parameters. It can *maintain* addition in 62 parameters (warm-start). These are three distinct thresholds revealing the optimizer as a limiting factor separate from the architecture.

**The model tells you what to compress next.** A/B converges to a circle. Slope converges to zero. Norms converge to sharing. Out_proj converges to writing one dimension. Every convergence is an invitation to add a constraint — but each constraint changes the loss landscape, so the invitation might be a trap. Flash grokking (95.8% → crash) shows the model can transiently solve the task in a basin too shallow to maintain.

**Architecture matters more than optimization tricks.** The entire path from 242p to 75p was structural: smaller d_model, fewer heads, parametric embeddings, weight tying, rank reduction. SAM, WD scheduling, scaffold training, and every other optimizer-level intervention failed. The next breakthrough will likely be architectural too.

### What's Next

The priority order for pushing below 75p:
1. **Tie A=B** (→74p): Safest single-param reduction, trained model already circular
2. **Structured norm** (→72-73p): The 5D norm has exploitable pattern (3 equal dims + 2 amplified)
3. **Sparse out_proj** (→70-72p): Train-from-scratch with constrained write target
4. **Distillation** (→<70p): Teacher-guided training without frozen values
5. **Loss landscape analysis**: Understand *why* seeds fail and *what* flash grokking reveals about basin geometry

The discoverability gap stands at 1.88x. Closing it is the open problem.

---

*Research conducted in March 2026. All from-scratch models verified at 100% accuracy on AdderBoard (10,010 samples, autoregressive evaluation, seed 2025).*
