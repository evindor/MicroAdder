# Reflection: From 170 to 62 Parameters

## How a Transformer Learned Perfect 10-Digit Addition in 62 Parameters

This document reflects on the research journey that compressed a trained transformer for 10-digit addition from 170 to 62 learnable parameters — a **63% reduction** — while maintaining 100% accuracy on 10,010 test cases. Along the way, we overturned several "hard constraints," discovered a powerful new training technique, and arrived at surprising conclusions about what neural networks actually need to learn.

---

## 1. The Journey in Numbers

```
Session 1: 170p → 133p  (architectural: vocab, d_model, parametric embeddings)
Session 2: 133p → 78p   (structural: shared norms, tied V/O, frozen positions)
Session 3:  78p → 62p   (warm-start cascade: incremental freezing)
```

**13 successive breakthroughs**, each verified at 100% accuracy on 10,010 autoregressive evaluations:

| Step | Params | Technique | Saving |
|------|--------|-----------|--------|
| 1 | 226p | Spiral positions | -16p |
| 2 | 214p | Rank-2 out_proj | -12p |
| 3 | 203p | Linear correction + freeze EOS | -11p |
| 4 | 187p | Tied Q/K + q-phase | -16p |
| 5 | 170p | tok_dim=2 reshape | -17p |
| 6 | 133p | Vocab=10 + d_model=5 + parametric emb | -37p |
| 7 | 100p | Various shrinks | -33p |
| 8 | 78p | Tied V/O + shared norms | -22p |
| 9 | 75p | Freeze PLUS+EOS | -3p |
| 10 | 72p | Freeze z_hi carry position | -3p |
| 11 | 70p | Freeze spiral offset+phase | -2p |
| 12 | 66p | Freeze all spiral + tok_arc start+stride | -4p |
| 13 | 62p | Tie A=B + freeze EQUALS position | -4p |

Total: 242p → 62p, a **74% compression** from the original architecture.

---

## 2. Theories Tried, Validated, and Refuted

### Validated Theories

**"Attention is positional routing, not content-dependent."**
This was our earliest structural finding at 170p: attention patterns are fixed functions of position, not token content. It predicted that vocab=10 (merging special tokens with digits) would work — and it did. It also predicted that freezing positions would be possible, since positions are just fixed lookup tables once the routing is learned. This theory held through every compression.

**"Per-head phase rotation provides sufficient Q/K asymmetry."**
Tied Q/K (K = Q) naively caps at 39% accuracy because carry routing needs asymmetric attention. A single learnable angle per head rotates Q relative to K, breaking symmetry with 1 parameter instead of a full K projection (15p). This idea, borrowed from the hand-coded param_40 model, saved 16p and was the foundation for all subsequent compressions.

**"Split-subspace attention naturally separates token and position processing."**
The architecture splits d_model into tok_dim (for content) and pos_dim (for position). Q/K operate on pos_dim, V on tok_dim. This clean separation allowed us to freeze all positional parameters (spiral, carry, special positions) without touching the token processing pipeline. The separation was never violated during training.

**"The model converges to the minimum complexity it needs."**
At every stage, diagnostic analysis showed the model was using only what it needed: tok_arc A≈B (ratio 0.993), spiral params near init values, EQUALS position z-dim ≈ 0. Each redundancy we observed became the next freezing target.

### Refuted Theories

**"All 3 RMSNorm layers are highly specialized and can't be shared."**
At 170p (d_model=6), the three norms had pairwise similarity of only 0.45-0.67 and very different weight profiles. We concluded sharing was impossible. But at d_model=5, norm sharing works perfectly — the norms converge to identical weights during training. The specialization was an artifact of the larger model having room to differentiate, not a structural requirement.

**"Positions are not freezeable — spiral params drift massively from init."**
At 170p, spiral amplitude tripled and phase rotated -61° during training. We concluded positions must be learned. But the warm-start cascade proved this wrong: you don't freeze at init values, you freeze at *learned* values. By training a larger model first and freezing its learned positions into buffers, the smaller model inherits good positions without needing to learn them.

**"Rank-2 out_proj is the minimum trainable rank."**
At d_model=6, rank-1 out_proj was dead (0.08% at 100K steps). At d_model=5, rank-1 works perfectly. The rank constraint was specific to the 2-head d_model=6 architecture, not a general property of addition.

**"FFN bias is required for convergence."**
True at d_model=6, false at d_model=5. Removing bias saves 7p with no accuracy loss.

**"Freezing special positions kills grokking."**
Freezing all special positions to zero does kill grokking (EQUALS position is essential). But freezing them as buffers that retain trained values from a warm-start works perfectly. The key insight: positions need specific values, but they don't need to be *learnable* — they need to be *correct*.

### Never Tested / Remained Open

**"Can the model learn from scratch at 62p?"**
We used warm-start cascading exclusively below 72p. Fresh training at 72p with seed 42 was known to fail (stuck at 70% tok_acc at 153K). Whether there exist seeds that can train 62p from scratch is unknown. Our hypothesis: no, the loss landscape at 62p is too rugged for random initialization.

**"Is there a sub-62p trained model that works?"**
64p (freeze all 4 tok_arc params) was stuck at 50-60% exact for 150K+ steps — the model couldn't generalize without any amplitude control over the token embedding circle. But we never tried 64p with just *one* of A or B frozen. The true floor remains unknown.

---

## 3. Novel Discoveries

### 3.1 The Warm-Start Cascade

The most important technique we discovered. The insight:

> You can't train a tiny model from scratch, but you *can* train it if you warm-start from a slightly larger model that was itself warm-started from an even larger one.

The cascade: 75p (from scratch) → 72p (warm) → 70p (warm) → 66p (warm) → 62p (warm).

Each step:
1. Train a model with N params to 100% accuracy
2. Add one more constraint (freeze a param, tie two params)
3. Initialize the N-1 model from the N model's checkpoint
4. Train until it re-groks at 100%

This works because each warm-start gives the smaller model a *compatible* starting point. The model only needs to adjust its remaining parameters to compensate for the newly frozen one, rather than discovering the entire solution from scratch.

**The key insight is about loss landscape topology.** The 62p loss landscape has basins of 100% accuracy, but they are unreachable from random initialization. The warm-start cascade provides a path *between* basins at different parameter counts, following a valley from higher-dimensional space into lower-dimensional space.

This is analogous to annealing: you start in a high-temperature (high-parameter) state where the landscape is smooth, then slowly cool (reduce parameters) while maintaining the model in a good basin.

### 3.2 Buffer-Preserved Warm-Starting

A technical innovation that made the cascade possible. When freezing a parameter:
1. Convert it from `nn.Parameter` to `register_buffer` in the model definition
2. The warm-start loader matches buffer names to checkpoint keys
3. The buffer gets loaded with the *trained* value from the parent checkpoint, not the init value
4. The value stays fixed during training (it's a buffer, not a parameter)

This means the "frozen" parameters retain their optimized values — they're only frozen in the sense that the optimizer can't change them. The model effectively has the same representational capacity; it just has fewer *degrees of freedom*.

This raises a philosophical question: **what do we mean by "62 parameters"?** The checkpoint contains ~80 non-trivial values (including buffers). Only 62 are learnable, but the others are equally important. We count by standard convention (learnable parameters only), but the information content is higher.

### 3.3 The Convergence to Symmetry

At multiple stages, we observed the model spontaneously converging toward simpler structures:
- **tok_arc_A ≈ tok_arc_B** (ratio 0.993): The model wants a circle, not an ellipse
- **spiral_slope ≈ 0.08** (near zero): The z-gradient barely matters
- **equals_pos z-dim ≈ 0.03** (near zero): EQUALS needs only 2D discrimination
- **spiral_phase absorbed by q_proj**: Phase is redundant when Q can rotate

Each convergence was a signal that a parameter was redundant. **The model tells you what to freeze next** — you just have to measure what it converges to and ask "is this close enough to a fixed value?"

### 3.4 Constraint Interactions Are Scale-Dependent

Many "hard constraints" from d_model=6 dissolved at d_model=5:
- Norm sharing: impossible at d=6, trivial at d=5
- Rank-1 out_proj: dead at d=6, works at d=5
- FFN bias: required at d=6, optional at d=5
- Position freezing: impossible at d=6, easy at d=5 (via warm-start)

This suggests that **the difficulty of each compression depends on the surrounding architecture.** An optimization that's impossible in a larger model may be trivial in a smaller one, and vice versa. You can't plan a compression roadmap in advance — you have to try things iteratively at each scale.

---

## 4. What We Learned About Architecture

### The Irreducible Core

The 62p model reveals what a transformer *must* have for 10-digit addition:

| Component | Params | Why It's Essential |
|-----------|--------|--------------------|
| q_proj | 15 | Maps 3D positions to 5D attention space |
| FFN (fc1+fc2) | 20 | Carry detection: reads position, writes digit correction |
| head_proj | 10 | Output classification + V projection (tied) |
| out_proj | 10 | Projects attention output back to residual |
| shared norm | 5 | Per-dimension feature scaling (scalar norm failed) |
| q_phase | 1 | Q/K asymmetry for carry routing |
| tok_arc_A | 1 | Token embedding scale (circle radius) |

The 15p q_proj is the largest single block and seems incompressible: rank-1 fails (1D keys can't distinguish positions), and rank-2 would cost 16p > 15p at these dimensions. The FFN at 20p is also at minimum: ffn_dim=1 can't do carry detection.

### What Surprised Us

**Only 1 parameter controls all 10 token embeddings.** The parametric arc `[A·cos(start + d·stride), A·sin(start + d·stride)]` places digits on a circle with a single learnable radius A. The angle parameters (start, stride) are frozen from a warm-start. The model needs only to control *how far* the digits are from the origin — not their angular arrangement.

**The shared norm does extreme work.** One 5-dimensional weight vector simultaneously serves as pre-attention norm, pre-FFN norm, and output norm. At 170p, these three norms had very different profiles. At 62p, they converge to identical weights — suggesting the model finds a *universal* scaling that works for all three purposes.

**No position is learned.** At 62p, all positional information comes from frozen buffers: spiral positions, carry position, special positions, even the token embedding angles. The model does no spatial reasoning at training time — all spatial structure is inherited from the warm-start parent and frozen. Only the *processing* of spatial information (via q_proj, FFN, attention) is learned.

---

## 5. What We Learned About Grokking

### Grokking at Every Scale

Every model from 170p to 62p exhibited the same grokking pattern:
1. **Memorization phase**: Loss decreases, tok_acc rises to ~50-70%, exact match stays near 0%
2. **Grokking transition**: Sudden jump in exact match (often 0% → 90%+ in a few thousand steps)
3. **Oscillation phase**: Model bounces between 50-100% exact match for 50-100K steps
4. **Stabilization**: Eventually locks in at 100% (or fails permanently)

The oscillation phase is the most dangerous. At 62p, the model hit 100% at step 96K, crashed to 30% at step 174K, recovered to 100% at step 183K, and finally stabilized. The early-stopping condition (100% held for 12K steps) catches the stable phase.

### Adaptive Weight Decay is the Key Training Innovation

The two-stage adaptive WD schedule was essential for every sub-100p model:
- **Stage 1** (wd × 0.1): Triggered when val_exact > 1%
- **Stage 2** (wd × 0.01): Triggered when val_exact > 5%

Without adaptive WD, grokking either doesn't happen or takes 10-100x longer. The mechanism: weight decay actively fights the carry circuit's sharpening process. The carry detection in the FFN needs weights to grow large (toward step-function behavior). WD pushes weights toward zero, creating a tug-of-war. Dropping WD at the right moment lets the circuit sharpen.

### Warm-Start Affects Grokking Dynamics

Warm-started models grok differently from fresh models:
- **Fresh 75p**: Grokking at ~276K steps (slow memorization → sudden generalization)
- **Warm 72p from 75p**: First signal at ~24K, grokking at ~312K (inherited memorization, slow re-grokking with new constraint)
- **Warm 62p from 66p**: First signal at ~21K, grokking at ~96K (faster — fewer params to adapt)

The warm-start provides a *head start on memorization*. The model inherits the parent's circuit and only needs to adapt the newly constrained parameters. This is faster for small changes (1-2 frozen params) but can be slower for large changes (the circuit may need significant restructuring).

---

## 6. The SGD Discoverability Gap

### From 4.25x to 1.55x

The hand-coded param_40 model achieves addition in ~40 parameters. Our best trained model needs 62 — a 1.55x overhead. This is down from 4.25x (170/40) at the start of the research.

But this comparison is misleading. Our 62p model has ~18 additional parameters stored in buffers (frozen but non-trivial values). Including these, the "information content" is closer to 80 values, or 2x the hand-coded solution.

### What SGD Can't Do (Yet)

1. **Parameterless norms.** param_40 uses RMSNorm without learned weights (just normalize). Our model needs 5 learned norm weights. SGD can't discover a representation where all dimensions have equal importance.

2. **Tied V/output/embed.** param_40 ties V projection, output projection, and token embedding into one matrix. We tie V and output but keep token embedding separate (parametric arc). Full triple-tying at d=5 might work but hasn't been tried.

3. **Zero-parameter positions.** param_40 uses RoPE (rotation, no learned params). We need spiral positions even if frozen. The positional information in the buffers is essential — it just doesn't need to be *learned* at the 62p stage.

### What SGD *Can* Do That Surprises

1. **Learn from a single amplitude parameter.** The 62p model arranges 10 digit tokens on a circle, classifies them by output projection, and detects carries — all with tok_arc_A as the only embedding parameter.

2. **Re-grok after constraint changes.** Even heavy constraints (freezing 4 params at once) don't kill the model if warm-started. SGD can navigate from a slightly-incompatible solution to a compatible one.

3. **Converge to optimal symmetry.** The model spontaneously makes A≈B, phase≈0, slope≈0 — signaling that these degrees of freedom are unnecessary. SGD finds the simplest solution within its parameter budget.

---

## 7. Research Frontiers

### 7.1 Can We Go Below 62p?

The remaining 62 parameters break into:
- **q_proj (15p)**: Incompressible at full rank. Could a structured q_proj (Toeplitz, circulant) reduce this?
- **FFN (20p)**: At minimum dim. Could a single-weight carry detector work with different nonlinearities?
- **out_proj (10p)**: Rank-1. Could it be tied to q_proj somehow?
- **head_proj (10p)**: Tied as V. Could be tied to out_proj B?
- **norm (5p)**: Shared. Could be eliminated with better initialization?
- **q_phase (1p)**: Could be absorbed into q_proj initialization?
- **tok_arc_A (1p)**: Could be fixed to a known-good value?

The most promising avenue: **tie out_proj.B with head_proj.weight** (both are 1×5 / 5→2 projections). This would save 5p → 57p. The key question is whether the read-side (V/head) and write-side (out_proj) can share weights.

### 7.2 Can We Train Sub-62p From Scratch?

The warm-start cascade is powerful but raises questions about trainability. Can any model below ~72p be trained from random initialization? If not, what makes the loss landscape so rugged?

Possible research directions:
- **Loss landscape visualization**: Project the 62p loss landscape onto 2D and visualize basins
- **Basin connectivity analysis**: Are the 100% accuracy basins connected at 62p? At 72p?
- **Progressive training**: Start with all params free, gradually freeze during training (online cascade)
- **Different optimizers**: Would sharpness-aware minimization (SAM) help navigate rugged landscapes?

### 7.3 Universality of the Warm-Start Cascade

Does the warm-start cascade work for other tasks?
- Subtraction (same architecture, different weights)
- Modular arithmetic
- Bit manipulation tasks
- Small language models (character-level)

If the technique generalizes, it suggests a universal approach to training tiny models: train big, freeze incrementally, cascade down.

### 7.4 Understanding the Buffer Values

The 62p model has ~18 non-trivial values in buffers. Are these values transferable?
- If we train a fresh 72p model with a different seed, do the frozen buffer values from seed 42 work?
- Are the buffer values task-specific or architecture-specific?
- Could we pre-compute optimal buffer values analytically?

### 7.5 Novel Architecture Ideas

**Structured q_proj**: Instead of a dense 3×5 matrix (15p), use `q_proj = U @ diag(s) @ V.T` with U fixed (e.g., Fourier) and only s (3p) learned. If the q_proj has exploitable structure, this could save 10+ parameters.

**Carry detection without FFN**: The FFN's role is carry detection (reads position, writes digit correction). Could a multiplicative interaction (gating mechanism) replace the FFN with fewer parameters?

**Dynamic norm**: Instead of per-dimension learned weights, use a single scalar + q_phase-style rotation for the norm. Failed as "scalar norm" (1p), but a 2-3p rotation-based norm might work.

---

## 8. Conclusions

### What Made This Possible

Three factors combined to enable 170p → 62p:

1. **Structural diagnostics.** Analyzing what the model actually learns (attention patterns, weight magnitudes, convergence directions) revealed which parameters were redundant. Every freezing decision was guided by measured convergence, not guesswork.

2. **The warm-start cascade.** This single technique enabled the jump from 72p (trainable from scratch) to 62p (unreachable from random init). It's the difference between proving a solution exists and actually finding it.

3. **Iterative experimentation.** We tried dozens of ideas; most failed. The successes came from combining validated individual compressions incrementally, never making multiple untested changes simultaneously.

### The Philosophical Takeaway

The 62p model challenges common assumptions about neural network training:

**Parameters ≠ information.** The model uses 62 learnable parameters but relies on ~18 frozen buffer values. The "size" of the model depends on what you count.

**Trainability ≠ expressibility.** The model can *represent* addition in 40 parameters (proven by hand-coded solutions). It can *learn* addition from scratch in 75 parameters. It can *maintain* addition in 62 parameters (via warm-start). These are three different thresholds, and the gaps between them reveal the role of the optimizer (SGD) as a limiting factor separate from the architecture.

**Local optima are the enemy, not model capacity.** The 62p model has the same loss landscape as a randomly-initialized 62p model. The difference is the starting point. This suggests that training tiny models is not about finding the right architecture — it's about finding the right *path* through parameter space.

### What Would Change With More Time

1. **Systematic sweep of all remaining 1p freeze targets** (q_phase, tok_arc_A, individual norm dims)
2. **Cross-seed validation** of the warm-start cascade (does it work with seeds other than 42?)
3. **Progressive online freezing** (freeze params during training as they converge, rather than between runs)
4. **Attempting tied out_proj.B = head_proj.weight** for a possible 57p model
5. **Loss landscape analysis** to understand why warm-start is necessary below 72p

---

*Research conducted in March 2026. All models verified at 100% accuracy on AdderBoard (10,010 samples, autoregressive evaluation, seed 2025).*
