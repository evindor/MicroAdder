# MicroAdder Research

### The Core Constraint: Autoregressive Transformer

The model must operate as a **genuine autoregressive transformer**. This means:

1. **Self-attention is required.** The model must contain at least one self-attention layer. This is the defining feature of a transformer — without it, you have an MLP or RNN, not a transformer.

2. **The model must be autoregressive.** It receives a token sequence as input and predicts the next token. Output digits are generated one at a time, with each new token fed back as input for predicting the next. The carry propagation must emerge from this autoregressive process — not from explicit state variables passed between steps in Python.

3. **Standard forward pass.** The model's `forward()` method must be a standard tensor-in, logits-out computation. No problem-specific control flow (for-loops over digits, explicit carry variables, string manipulation) inside `forward()`. The autoregressive generation loop lives *outside* the model, exactly as it would for any language model.

4. **The model does the work, not the code.** The inference code should be generic autoregressive decoding that would work with *any* transformer checkpoint. If your generation loop contains addition-specific logic — manually pairing digits, threading carry state, indexing into specific positions — then the Python code is solving the problem, not the model.

In short: if you can swap in a different set of weights and use the exact same inference code for a different task, your setup is legitimate. If the inference code is inseparable from the algorithm, it's not.

### What's Allowed
- Architectural variations: rank-1/low-rank projections, factorized embeddings, custom positional encodings, alternative norms
- Hand-coded weights (constructive proofs are valid — they show the architecture *can* represent addition)
- Trained weights via any generic learning algorithm (shows the solution is *learnable* — encourages creative ideas on data format, tokenization, and curriculum)
- Input formatting choices (reversed digits, delimiters, etc.) as long as the format is fixed and doesn't encode the answer

### Parameter Counting
- Count **unique** parameters (after weight tying/deduplication)
- Fixed/sinusoidal positional encodings are not counted (following the original Transformer paper convention)
- Learned positional encodings are counted

## Current Best: 62 Parameters, 100% Accuracy

A 1-layer autoregressive decoder that performs 10-digit addition with 100% accuracy (10010/10010 on AdderBoard). Built by progressively compressing JackCai's 242p split-subspace architecture through 13 successive breakthroughs.

```
d_model = 5 = tok_dim(2) + pos_dim(3)
1 layer, 1 head, head_dim = 5
FFN dim = 2 (no bias, GELU), shared RMSNorm
Tied Q/K + q-phase rotation, rank-1 out_proj, tied V/output
Parametric tok_emb (circular arc, 1 param), all positions frozen
```

### Parameter Budget (62p)

```
tok_arc_A (=B)     1  (tied circular arc amplitude)
q_phase            1  (Q rotation angle)
q_proj            15  (3 -> 5, shared with K)
out_proj          10  (5x1 + 1x5, rank-1)
FFN fc1           10  (5 -> 2, no bias)
FFN fc2           10  (2 -> 5, no bias)
head_proj         10  (5 -> 2, also used as v_proj)
RMSNorm (shared)   5  (one weight for all 3 norms)
─────────────────────
TOTAL             62

Frozen buffers (loaded from warm-start):
tok_arc_start/stride  2  (arc angle params)
spiral params         4  (amp, phase, slope, offset)
special_pos_equals    3  (EQUALS token position)
z_hi_pos              3  (carry position)
_plus_pos, _eos_pos   6  (zero vectors)
```

### Previous Best: 170p Parameter Budget

```
tok_emb           28  (14 x 2, tied with output)
spiral params      4  (amp, phase, slope, offset)
linear pos corr    2  (slope + intercept)
z_hi_pos           4  (1 x 4, carry position)
special_pos        8  (2 x 4, PLUS + EQUALS; EOS frozen to zero)
q_proj            24  (4 -> 6, shared with K)
q_phase            2  (per-head rotation angle)
v_proj            12  (2 -> 6)
out_proj          24  (6x2 + 2x6, rank-2 factorized)
ln1 + ln2 + ln_f  18  (3 x 6)
fc1 (w+b)         14  (6 -> 2)
fc2 (w+b)         18  (2 -> 6)
head_proj         12  (6 -> 2)
─────────────────────
TOTAL            170
```

### Training Recipe

- AdamW (lr=0.02), cosine decay, 500K step budget
- Adaptive weight decay: 0.01 → 0.001 → 0.0001 at grokking thresholds (val_exact 0.2, 0.5)
- Carry-mix curriculum: 30% carry-heavy examples, fading to 0% as tok_acc 0.7→0.9
- Digit curriculum: 1-3 digits for 2K steps, 1-6 for 5K, then full 1-10
- Seed 80085 grokked at ~15K steps. Seed-sensitive: 1/3 seeds grok at 170p

---

## Compression History: 242p → 62p

Each step represents a compression validated at 100% accuracy on AdderBoard (10010/10010).

```
242p  JackCai's split-subspace architecture
 |    d_model=6, split Q/K/V, shared XYZ positions, tied output, RMSNorm
 |
226p  Spiral+correction positions (-16p)
 |    Parametric spiral (4p) + per-position corrections (10p) replace 30 learned
 |
214p  Rank-2 attention output (-12p)
 |    out_proj factored as A(6x2) @ B(2x6)
 |
203p  Linear pos correction + frozen EOS (-11p)
 |    10 corrections → 2 linear params, EOS frozen to zero
 |
187p  Tied Q/K with q-phase rotation (-16p)
 |    k_proj eliminated; Q_rot = Q·cos(θ) - Q_swap·sin(θ)
 |    Adaptive weight decay introduced (18x faster grokking)
 |
170p  tok_dim=2, pos_dim=4 reshape (-17p)
 |    Token subspace 3D→2D (96% SVD energy)
 |
133p  Vocab=10 + d_model=5, 1 head (-37p)
 |    All special tokens → digit-0, d_model 6→5, 2h→1h
 |    Parametric tok_emb (4 arc params vs 20 learned)
 |    Rank-1 out_proj, no FFN bias, no pos correction
 |
100p  Shared norms (-33p from 133p baseline)
 |    All 3 RMSNorm share one weight vector. Various freezes.
 |
 78p  Tied V/output + shared norms (-22p from 100p)
 |    v_proj eliminated: V = x_tok @ head_proj.weight
 |
 75p  Freeze PLUS+EOS positions (-3p)
 |    Delimiter positions frozen at zero
 |
 72p  Freeze z_hi carry position (-3p)
 |    Warm-started from 75p — critical technique
 |
 70p  Freeze spiral_offset + spiral_phase (-2p)
 |    q_proj absorbs phase rotation; offset is z-intercept
 |
 66p  Freeze all spiral + tok_arc start+stride (-4p)
 |    Warm-started from 70p; 0 spiral params learned
 |
 65p  Tie tok_arc_A = tok_arc_B (-1p)
 |    Circular arc (A/B ratio was 0.993, nearly identical)
 |
 63p  Freeze EQUALS position (-3p from 66p)
 |    Warm-start loads trained values into buffer
 |
 62p  Tie A=B + freeze EQUALS (-4p from 66p)
      Combined approach: 62 learnable parameters
      Warm-started from 66p, grokked at ~96K steps
```

### Key Techniques (Our Contributions)

**Tied Q/K with per-head phase rotation.** The single biggest compression (-16p). Q and K share one projection matrix. A learnable per-head angle (2 params) rotates pairs of Q dimensions, giving each head a unique "viewing angle" on the shared key space. This provides the asymmetry carry routing requires, replacing an 18p K projection with 2 parameters. Inspired by the hand-coded param_40 model which uses the same trick.

**Adaptive weight decay.** Grokking is the model fighting WD to sharpen its carry circuit. Dropping WD at the right moment (val_exact crosses 2%, then 20%) gives 18x faster grokking at 187-203p and was essential for enabling the 170p configuration.

**tok_dim=2, pos_dim=4 reshape.** Token embedding subspace shrunk from 3D to 2D. Saves 17p net but makes the tied output head discrimination tight (14 classes from 2D), causing seed sensitivity.

**Spiral+correction positions.** Parametric spiral captures base-10 periodicity of digit positions. Linear correction gives per-position magnitude control. 6 params replace 30 learned.

**Carry-mix curriculum.** Oversamples carry-heavy examples (cascading nines, boundary crossings) early in training, fades to uniform by tok_acc=0.9. See [carry-mix.md](carry-mix.md).

---

## Experiment Log

### What Worked

| Experiment | Params | Result | Key Learning |
|---|---|---|---|
| Spiral+correction positions | 226p | 100%, 3/4 seeds | Parametric positions can replace learned ones |
| Rank-2 attention output | 214p | 100%, 4/10 seeds | Post-hoc SVD predicts native trainability for rank-2 |
| Linear pos correction + frozen EOS | 203p | 100%, 3/4 seeds | Linear correction sufficient; EOS pos is exactly zero |
| Tied Q/K + q-phase | 187p | 100%, 2/3 seeds | Phase rotation breaks Q=K symmetry cheaply |
| tok_dim=2, pos_dim=4 | 170p | 100%, 1/3 seeds | 2D token space is trainable but tight |
| Adaptive weight decay | (training) | 18x faster grokking | WD fights carry circuit sharpening |
| Vocab=10 + d_model=5 + parametric tok_emb | 133p | 100% | Massive combined compression works |
| Shared norms + tied V/O + no correction | 78p | 100% | 3 norms CAN share weights (contradicts 170p finding) |
| Warm-start cascade (75p→72p→70p→66p→62p) | 62p | 100% | Incremental freezing with warm-start bypasses optimization barriers |
| Tie tok_arc A=B (circular arc) | 65p | 100% | Converges to A/B ratio 0.993 anyway |
| Freeze EQUALS position as buffer | 63p | 100% | Warm-start preserves trained values in buffers |
| Freeze all positions + embeddings + tie A=B | 62p | 100% | Only 1 embedding param needed (arc amplitude) |

### What Failed

| Experiment | Params | Best Result | Why It Failed |
|---|---|---|---|
| Tied Q=K without phase | 185p | 39% max | Carry routing needs Q/K asymmetry |
| Rank-1 attention output | 191p | 0.08% at 100K | Model needs rank-2 *during* grokking even though trained solution is ~rank-1 |
| GQA (1 KV head) | 185p | 0.5% at 100K | 3x3 K projection too restrictive (near-identity) |
| tok_dim=2 without q-phase | 192p | 10.9% peak, then collapsed | 2D tok_emb too tight for tied output head without tied Q/K savings |
| RMSNorm sharing | 191p | 61.5% at 500K | All 3 norms load-bearing, highly specialized, pairwise similarity only 0.45-0.67 |
| RMSNorm fixed to 1 | 185p | 6.4% | Per-dimension scaling is essential |
| Remove pre-FFN norm (ln2) | 197p | 96.6% max, never 100% | Tantalizingly close across 8 seeds, but oscillates and never locks in |
| Freeze PAD + q-phase | 184p | 5% at 83K | Interferes with grokking dynamics |
| ALiBi (zero positions) | 190p | 20% tok_acc | Pure distance-based attention can't route content |
| ALiBi + tied Q/K on tok_dim | 172p | 20% tok_acc | Content-only Q·K insufficient even with position bias |
| Frozen PLUS position | -3p | Unstable | Norm varies 0.19-1.3 across checkpoints, load-bearing in some |
| AR training loss | (training) | 0.18% at 48K | Failed with adaptive WD; likely trigger interaction (WD never drops because val_exact never rises under AR) |
| Smooth WD (--wd-smooth) | 170p | 0.47 tok_acc | Exponential decay too aggressive for tok_dim=2 |
| Grokfast (EMA gradient filter) | 170p | 0.20 tok_acc | Completely stuck; doesn't help this architecture |
| Softmax1 (attn sum < 1) | 170p | 0% exact, 0.42 tok_acc at 189K | Stuck at memorization; "attend to nothing" hurts split-subspace carry routing |
| Tied V/O projection (SVD diagnostic) | ~146p | Not trained | V and A occupy different 2D subspaces (principal angles 17-23°); A ≈ V@R residual 37.5%; naive tie destroys 94% of output |

### Hard Constraints (Updated — Some Overturned)

| Constraint | Status | Evidence |
|---|---|---|
| ~~FFN bias=True~~ | **OVERTURNED** | No-bias works at d_model=5 — saves 7p |
| RMSNorm, not LayerNorm | Still holds | LN adds params with no gain |
| ~~All 3 norms independent~~ | **OVERTURNED** | All 3 norms CAN share one weight vector at d_model=5 |
| ~~Rank-2 out_proj minimum~~ | **OVERTURNED** | Rank-1 works at d_model=5 (the 170p result was specific to d_model=6) |
| ~~Positions not freezeable~~ | **OVERTURNED** | ALL positions can be frozen via warm-start cascade |
| ~~q_proj must be full-rank~~ | Still holds at d=5 | Rank-1 q_proj failed (1D keys can't distinguish positions) |
| Tied output head | Still holds | Untied adds too many params |
| LSB-first digit ordering | Still holds | MSB-first breaks autoregressive carry propagation |
| q_phase is essential | Confirmed | Without q_phase: dead at 21% tok_acc (tied Q/K needs asymmetry) |
| FFN dim >= 2 | Confirmed | ffn_dim=1 fails at 60% tok_acc |
| Per-dim norm weights | Confirmed | Scalar norm failed at 50-60% tok_acc |

### Structural Properties (from Exp 6 diagnostics)

| Property | Evidence |
|---|---|
| Attention is fixed positional routing | Head 0: 50/50 on X_{i+2}, Y_{i+1}. Head 1: 33/33/33 on X_{i+1}, Y_i, self. No content-dependence. |
| Carry-lookahead, not carry-chaining | No A_{i-1}→A_i attention. Carries predicted from input digit lookahead. |
| pos dim 3 is carry-exclusive | Digit positions have dim3=0; z_hi has dim3=-1.26 (largest component). Dropping dim3 hurts carry circuit. |
| Residual is 2D output + 4D scratch | ln_f amplifies dims 1,5 by 10-12x, suppresses dims 3,4 to ~0. Only 2 of 6 dims reach output head. |
| FFN unit 0 is pos→tok bridge | Reads position (carry context), writes token (digit correction). The carry detection mechanism. |

---

## Hand-Coded Models: What They Teach Us

Three hand-coded addition transformers (36p, 40p, 87p) prove the representational floor is far below our 170p. The 170/40 = **~4x overhead** is not about capacity — it's about what SGD can discover.

### The Models

**TinyAdder (36p, Litzenberger):** 2 layers, d_model=5, ALiBi slope=log(10). Layer 0 pairs digits via ALiBi attention. Layer 1 uses uniform causal attention (Q=K=0) to compute running carry averages, then hard-thresholds. Uses softmax1 (allows attention sum < 1), float64 precision, broadcast parameters.

**param_40 (40p, Wonderfall):** 1 layer, d_model=2, RoPE period=19. Tied Q/K with phase rotation. Tied V/O. Parameterless RMSNorm. Vocab=10. Parabolic embedding `[1000 - 0.001d², -d]` for dual encode/decode use. The most elegant solution — everything in a 2D plane.

**87_torch (87p):** 2 layers, d_model=5, GQA (2h/1kv), RoPE. Parameterless embedding. Rank-1 Q/K (position-only attention). Extremely large gate values (~60000) for hard step-function carry detection. Hand-designed rank-2 LM head.

### Key Insights for Trained Models

1. **Tied Q/K + phase rotation works.** param_40 uses it. We adopted this and it saved 16p. The phase trick is SGD-discoverable (unlike the hand-coded solutions' other tricks).

2. **Vocab=10 is viable.** param_40 and 87_torch both use 10-token vocabularies with no special delimiter tokens. Delimiters are identified by position alone. This is a natural fit for split-subspace attention where Q/K already operate on positional information only.

3. **Softmax1 is free and potentially useful.** TinyAdder's softmax with +1 in denominator lets attention weights sum to <1, creating an implicit "attend to nothing" option. Zero parameters, zero risk. Worth testing.

4. **d_model=2 is representationally sufficient.** param_40 proves this. The question is whether SGD can discover solutions in such a tight space.

5. **Hard carry thresholds vs smooth GELU.** Hand-coded models use extreme values (±60000) for step functions. Our model approximates this with GELU and float32 weights of magnitude ~1-10. Weight decay actively fights the sharpening process. Adaptive WD partially addresses this.

6. **Zero-parameter positions exist** but require abandoning split-subspace (RoPE rotates full embedding) or using ALiBi (which failed for us). Our spiral+correction costs 14p total — a compression target, but not via the methods hand-coded models use.

### The Discoverability Gap

| Component | 62p (ours) | 170p (prev) | param_40 (40p) |
|---|---|---|---|
| tok_emb | 1p (arc amp) | 28p (14x2) | 0p (frozen) |
| positions | 0p (frozen) | 14p (spiral+corr) | 0p (RoPE) |
| Q/K projections | 16p (q_proj+phase) | 26p (tied+phase) | ~5p |
| V projection | 0p (tied V/O) | 12p | ~4p (tied V/O) |
| out_proj | 10p (rank-1) | 24p (rank-2) | 0p (tied V/O) |
| RMSNorm | 5p (shared) | 18p (3 x 6) | 0p (parameterless) |
| FFN | 20p (no bias) | 32p | ~8p |
| head_proj | 10p | 12p | 0p (tied embed) |
| **Total** | **62p** | **170p** | **~17p** |

The discoverability gap has shrunk from 170/40 = **4.25x** to 62/40 = **1.55x**. Only ~35% overhead remains over the hand-coded solution. The warm-start cascade technique was the key breakthrough — it allows SGD to navigate optimization landscapes that are too rugged for random initialization.

---

## The Sub-100 Goal

The new target: a **general trainable architecture under 100 parameters** with 100% accuracy on 10-digit addition.

"General" means: no hand-coded weights, no task-specific initialization. The same architecture should in principle work for related tasks (e.g. subtraction) with different trained weights.

### Why Sub-100 Is Plausible

1. The representational floor is 36-40p. The gap is optimization, not capacity.
2. We've already closed the gap from 6x (242/40) to 4x (170/40). Each compression made the next one harder but also taught us more about what SGD needs.
3. The remaining 170p budget has clear redundancies: 28p on tok_emb (param_40 uses 0), 18p on norms (param_40 uses 0), 14p on positions. These won't all go to zero, but significant savings remain.

### Why Sub-100 Is Hard

1. **Seed sensitivity already severe.** At 170p only 1/3 seeds grok. Tighter budgets will make this worse.
2. **Norm removal is blocked.** All 3 RMSNorm layers (18p) are load-bearing. We can't eliminate them without a fundamentally different normalization approach.
3. **tok_dim=2 is at the limit.** The tied output head must discriminate 14 classes from 2D embeddings. Going below tok_dim=2 is impossible.
4. **Compression interactions are nonlinear.** Freeze PAD works alone but kills grokking when combined with norm removal. More compressions stacked = more interaction failures.

### Architecture-First, Then Training

The path from 242→170 was entirely architectural compression. Training innovations (adaptive WD, carry-mix) made experiments faster but didn't reduce the parameter floor. We should continue with architecture-first until we hit the wall where further compression breaks trainability. Only then escalate to progressive-constraint or basin-discovery training methods.

---

## Experiment Plan: Path to Sub-100

### Exp 1: Vocab=10 — Eliminate Special Tokens (target: ~152p)

**Thesis:** Encode PLUS/EQUALS as reused digit-0 tokens, making delimiters positional rather than lexical. Saves ~18p:
- tok_emb: 14x2 → 10x2 = -8p
- special_pos: eliminated = -8p (PLUS/EQUALS no longer need unique learned positions)
- head_proj output: tied head now discriminates 10 classes instead of 14
- EOS/PAD handling changes to position-only

**Why it should work for us:** Split-subspace attention already routes via position (Q/K from pos_dim). The model never needed PLUS/EQUALS *token identity* for attention — it only needed their *positions*. Removing the token distinction is removing something the architecture already ignores in its attention patterns.

**Risk:** V vectors for delimiter-0 and digit-0 become identical. The model must route all delimiter information through attention patterns alone. This is the intended design of split-subspace, but it's untested with ambiguous tokens.

**Variants to test:**
1. Vocab=10 on 170p baseline (target ~152p)
2. Vocab=12 (keep EOS/PAD as distinct, merge only PLUS/EQUALS with digit-0) — less aggressive, lower risk

```bash
uv run python -m src.train --run-name exp1_vocab10 \
    --vocab-size 10 --tie-qk --q-phase --tok-dim 2 --pos-dim 4 \
    --pos-mode spiral_correct --attn-out-rank 2 --pos-correction-mode linear \
    --freeze-special eos --wd-adaptive --wd-drop-exact 0.2 --wd-drop-exact-final 0.5 \
    --seed 80085 --steps 500000 --lr 0.02 --carry-mix 0.3
```

### ~~Exp 2: Softmax1~~ (FAILED — hurts grokking at 170p)

**Result:** 0% exact, 0.42 tok_acc at 189K steps (baseline grokked at 15K). Softmax1 prevents grokking entirely at 170p. The "attend to nothing" option likely dilutes the sharp attention patterns needed for carry routing in split-subspace. The `--softmax1` flag is implemented and available if architecture changes make it viable later, but it's harmful for the current design.

TinyAdder uses softmax1 but with a fundamentally different architecture (2 layers, ALiBi, uniform causal attention in layer 1). The mechanism that benefits from softmax1 there doesn't exist here.

### Exp 3: Shrink d_model (target: sub-130p)

**Thesis:** d_model=6 may be larger than needed. If vocab=10 works, try d_model=5 (tok_dim=2, pos_dim=3) or d_model=4 (tok_dim=2, pos_dim=2).

At d_model=5, tok_dim=2, pos_dim=3:
```
tok_emb          20  (10 x 2, vocab=10)
spiral params     4
linear pos corr   2
z_hi_pos          3  (1 x 3)
special_pos       0  (eliminated with vocab=10)
q_proj           15  (3 -> 5, tied with K)
q_phase           2
v_proj           10  (2 -> 5)
out_proj         15  (5x2 + 2x5, rank-2) or 10 (rank-1: 5+5)
ln1+ln2+ln_f     15  (3 x 5)
fc1 (w+b)        12  (5 -> 2)
fc2 (w+b)        15  (2 -> 5)
head_proj        10  (5 -> 2)
─────────────────────
~123p (rank-2) or ~118p (rank-1)
```

At d_model=4, tok_dim=2, pos_dim=2:
```
tok_emb          20  (10 x 2)
spiral params     4  (only 2 pos dims used in spiral)
linear pos corr   2
z_hi_pos          2  (1 x 2)
q_proj            8  (2 -> 4, tied with K)
q_phase           2
v_proj            8  (2 -> 4)
out_proj         12  (4x2 + 2x4, rank-2) or 8 (rank-1)
ln1+ln2+ln_f     12  (3 x 4)
fc1 (w+b)        10  (4 -> 2)
fc2 (w+b)        12  (2 -> 4)
head_proj         8  (4 -> 2)
─────────────────────
~100p (rank-2) or ~96p (rank-1)
```

**Key question:** Can split-subspace attention work at pos_dim=2? With only 2 positional dimensions, the spiral encoding must place 10 digit positions, a carry position, and delimiter positions in 2D space. The spiral naturally uses 2 angular dimensions, so this might be tight but feasible.

**Risk from Exp 6 diagnostics:** pos_dim shrink is more dangerous than expected. At 170p, digit positions have dim3=0 (spiral only fills dims 0-2), but z_hi (carry position) uses dim3=-1.26 as its **largest component**. Shrinking pos_dim 4→3 (d_model=5) drops the carry position's primary identifier. This likely explains d_model=5's plateau at 8-12% exact — the carry circuit can't cleanly separate carry from digit positions. At pos_dim=2 (d_model=4), both the linear ramp (dim2) and carry dimension (dim3) are lost.

**Risk:** head_dim shrinks proportionally. At d_model=4 with 2 heads, head_dim=2. The attention mechanism has very little capacity per head. However, param_40 works with d_model=2 and head_dim=2 — the question is trainability, not representability.

**Mitigation (implemented):** Decoupled inner_dim — set `--head-dim 3` with `--d-model 4` to keep attention in 6D (inner_dim=6) while shrinking the residual stream. Costs 128p vs 116p naive but preserves proven attention capacity. Ellipse mode for pos_dim=2 spiral (independent amp/phase per axis) uses all 4 spiral params instead of wasting 2.

**Early results (d_model=5, 1h, 141p):** tok_acc 87-91%, exact 8-12% with WD threshold 0.20 (never triggered). With WD threshold 0.10, stage 1 WD drop triggered on seed 42. 12-seed sweep running with WD 0.10/0.50.

### Exp 4: Parametric Token Embeddings (target: -10 to -20p)

**Thesis:** Replace learned tok_emb with a parametric form. The 10 digit embeddings could follow a curve parameterized by ~4-6 params instead of 20 (at vocab=10, tok_dim=2).

Candidate forms:
- Spiral: `[A*cos(2pi*d/10 + phi), B*sin(2pi*d/10 + phi)]` — 4 params for 10 digits
- Linear: `[a*d + b, c*d + e]` — 4 params
- Hybrid: parametric digits + 1-2 learned special tokens

Since tok_emb is tied with the output head, the parametric form must also support good classification boundaries. A spiral in 2D naturally separates 10 classes. The hand-coded param_40 uses a fixed parabolic form `[1000 - 0.001d², -d]` at tok_dim=2 — proving fixed 2D embeddings can work.

**Risk:** Trained embeddings at 170p drift significantly from spiral init (measured at earlier sizes). The model may need per-digit freedom for the dual encode/decode role. Parametric forms are rigid.

**Mitigation:** Start with parametric + small learned residual per digit (L1-penalized). If residuals stay small, freeze them to zero.

### ~~Exp 5: Tied V/Out Projection~~ (DEPRIORITIZED — SVD diagnostic negative)

**Thesis:** Tie out_proj.A (6×2) with v_proj.weight (6×2) — same shape, natural read/write duality. Optionally add a 2D bottleneck rotation (like Q-phase) to break symmetry. Would save 11-12p.

**SVD Diagnostic Result (on 170p checkpoint):**
- V and A occupy different 2D subspaces in R^6 (principal angles: 17.4°, 23.1°)
- Column cosines: -0.82, 0.93 (first column nearly *anti*-aligned)
- Best-fit 2×2 transform A ≈ V@R: 37.5% residual (R is not a rotation: SVD ratio 1.46, det=-0.67)
- Naive tie (V@B vs A@B): 94% relative difference — destroys the output projection
- Singular value profiles differ: V=[3.33, 2.16], A=[3.48, 1.57]

**Why the Q/K phase trick doesn't transfer:** Q-phase worked because Q and K naturally converge to nearly the same subspace (both read positional dims from the same source). V and A read from fundamentally different distributions (tok_dim vs full head_space), so they don't align. A bottleneck rotation can only fix angular offset within a shared subspace, but V and A don't share a subspace.

**Status:** Not worth training. param_40 ties V/O at d_model=2 where everything is trivially the same dimension. At d_model=6 with split-subspace, V and O are genuinely different projections. Revisit only if d_model shrinks to 2-3 where the subspace gap may close.

### Exp 6: Structural Diagnostics (DONE)

Full analysis of 170p checkpoint. Script: `diagnostics_170p.py`. Raw data and all findings: [structural_analysis.md](structural_analysis.md).

**Summary of findings:**

1. **Token embeddings:** Digits 0-9 form a 149° arc in 2D, not a full circle. Margins tightest at digits 3-6 (~0.28). PLUS dangerously close to digit 1 (1.8° apart, separated only by norm).

2. **Attention is fixed positional routing, not content-dependent.** Head 0: 50/50 split on X_{i+2} and Y_{i+1} (carry-lookahead). Head 1: 33/33/33 on X_{i+1}, Y_i, and self (current context). Neither head does carry-chaining via A_{i-1}. The 26p Q/K machinery learns a simple positional offset function.

3. **FFN unit 0 is the pos→tok carry bridge.** Reads position (where am I?), writes token correction (carry adjustment). Unit 1 is tok→tok (direct digit transform). Weights are soft (~2.6x) vs hand-coded ~60000x.

4. **Positions NOT freezeable.** Spiral amp tripled (1→2.95), phase rotated -61°, correction intercept grew to +2.34. Total drift 6.12 across 6 params.

5. **Effective rank:** q_proj genuinely rank-4 (no compression). fc1 nearly rank-1 but needs rank-2 for training. ln_f has extreme weights (dims 1,5 amplified 10-12x, dims 3-4 near zero) — acts as feature selector, not normalizer.

**Architectural implications derived from diagnostics:**

- **Carry-lookahead, not carry-chaining.** The model implements hardware-style carry prediction from lookahead on input digits. Optimal for 1-layer autoregressive. Any architecture change must preserve the ability to attend to offset positions in X and Y independently.

- **pos_dim=3 risks the carry circuit.** Digit positions have dim3=0.000 (spiral only fills dims 0-2), but z_hi (carry position) has dim3=-1.2559 as its *largest component*. Shrinking pos_dim 4→3 forces carry to share dimensions with digits. This likely explains d_model=5's plateau at 8-12% exact.

- **PLUS/digit-1 collision motivates vocab=10.** PLUS at -48.9° and digit-1 at -47.1° are nearly overlapping. Removing PLUS from vocab eliminates this and frees angular space.

- **ln_f as output selector.** The 6D residual is really 2D output (dims 1,5 amplified 10-12x) + 4D scratch (suppressed). At d_model=4, only 2 dims remain for scratch after reserving 2 for output — very tight.

- **Fixed-offset attention → potential for cheaper Q/K.** The 26p Q/K learns what could be expressed as ~6-8p of per-head positional offsets. ALiBi failed because it uses uniform distance slopes; the model needs group-aware offsets (different shifts for X-positions vs Y-positions). A novel "grouped offset attention" mechanism could save ~18-20p.

### Exp 7: Combined Push (target: sub-100p)

Stack the successful individual experiments. Exp 2 (softmax1) and Exp 5 (tied V/O) are eliminated.

```
170p  current baseline
-18p  vocab=10 (Exp 1)
-24p  shrink d_model to 5 (Exp 3, d_model=5 variant)
-12p  parametric tok_emb (Exp 4)
────
116p  optimistic floor

or with d_model=4:

170p  current baseline
-18p  vocab=10 (Exp 1)
-46p  shrink d_model to 4 (Exp 3, d_model=4 variant)
-12p  parametric tok_emb (Exp 4)
────
 94p  aggressive floor (likely not all trainable simultaneously)
```

Realistically, compression interactions will block some combinations. The approach:
1. Run Exp 3 first (biggest potential savings, independent of vocab change)
2. Run Exp 1 on whatever works from step 1
3. Run Exp 4 only if d_model shrink succeeds (otherwise tok_emb is already small)
4. Exp 6 runs in parallel with everything — it's checkpoint analysis, not training
5. Stack into Exp 7 combo only after individual signals are clear

---

## Training Innovation Candidates

These are orthogonal to architecture and can be tested alongside any experiment.

### Continuous WD Decay

Replace the discrete 2-stage WD drop with `wd = wd_init * exp(-alpha * steps_since_grok_onset)`. Smoother transition may avoid the instability that discrete drops can cause in tight models. (Note: `--wd-smooth` with ratcheted exponential decay failed at 170p, but that used a different schedule. Worth revisiting with different alpha or non-ratcheted variant.)

### Per-Parameter-Group WD

Different decay rates for different parameter groups. The carry circuit lives in specific weights (FFN, attention out_proj). Selectively dropping WD on carry-critical weights while maintaining it on embeddings and norms could help the circuit sharpen without destabilizing other components.

### Structural Diagnostics Logging

Log per-eval: effective rank of projections, norm weight drift from 1, carry-chain error rates by chain length. Cheap to implement, gives continuous signal about whether the model is converging toward a compressible structure.

### Progressive Constraint Training (Deferred)

The idea of training wide then compressing (start at ~150p effective, progressively freeze/tie/prune to ~80p) remains theoretically sound but premature. We don't yet know the architectural floor — no point building a progressive pipeline until we know what structure to compress toward. Revisit if architecture-only gains plateau around ~120-130p.

---

## Open Questions

1. **Is vocab=10 compatible with split-subspace?** The model must handle ambiguous tokens (digit-0 vs delimiter-0) using only positional attention routing. This is the most important near-term question.

2. **What is the minimum d_model for trainable split-subspace addition?** d_model=6 works reliably, d_model=4 is representationally sufficient (param_40 uses d_model=2). The trainability boundary is somewhere in between.

3. **Can tied V/O work at d_model > 2?** param_40 proves it at d_model=2. At d_model=4-6, V and O serve more differentiated roles. SVD analysis of the trained checkpoint will give a preliminary answer.

4. **What is the seed-sensitivity floor?** At 170p, 1/3 seeds grok. If sub-100 architectures have 1/10 or 1/20 seed success rates, is that acceptable? Or do we need training innovations specifically to improve seed robustness?

5. **Is there a role for autoregressive training?** It failed with our current adaptive WD triggers, likely because AR delays the val_exact signal that triggers WD drops. A forced WD schedule (not metric-triggered) under AR training has not been tested and could unlock faster grokking. Low priority but worth one diagnostic experiment.

---

## References

- JackCai, "smallest-addition-transformer" (242p split-subspace). https://github.com/JackCai1206/smallest-addition-transformer
- Taghadouini, "minimal-ten-digit-addition-transformer" (228p Qwen3). https://github.com/staghado/minimal-ten-digit-addition-transformer
- Litzenberger, "Building a Minimal Transformer for 10-Digit Addition" (36p TinyAdder). https://alexlitzenberger.com/blog/post.html?post=/building_a_minimal_transformer_for_10_digit_addition
- Wonderfall, param_40 (40p hand-coded). `external/param_40.py`
- 87_torch (87p hand-coded). `external/87_torch.py`
- AdderBoard challenge. https://github.com/anadim/AdderBoard
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability." ICLR 2023. https://arxiv.org/abs/2301.05217
- McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings." NeurIPS 2024. https://arxiv.org/abs/2405.17399
