# Research Journey: Path to Sub-100p

## Session 1 — 2026-03-01

### Starting Point
- **Current SOTA:** 170p, 100% accuracy (seed 80085)
- **Architecture:** 1L decoder, d_model=6 (tok_dim=2, pos_dim=4), 2h, hd=3, ff=2
- **Target:** Sub-100p trained model, 100% on 10-digit addition

### Research Plan

After deep study of RESEARCH.md and structural_analysis.md, three experiments:

#### Exp A: Vocab=10 → 162p (conservative) or 154p (freeze=all)
- PLUS/EQUALS/EOS/PAD all become digit-0, distinguished by position only
- Structural analysis confirms: attention is purely positional routing, model never needs token identity for delimiters
- PLUS and digit-1 are dangerously close (1.8° apart) — removing PLUS from vocab eliminates the collision
- Risk: V(delimiter) = V(digit-0), but split-subspace already routes via position not token content
- **Savings:** tok_emb 14×2→10×2 = -8p. Freeze all special_pos = -8p more.

#### Exp B: Fixed-Offset Attention → saves ~16p (from 26 to 10)
- Replace q_proj(24p) + q_phase(2p) with ~10 learnable offset parameters
- Based on structural analysis: Head 0 = look-ahead (X_{i+2}, Y_{i+1}), Head 1 = current (X_{i+1}, Y_i, self)
- These are just fixed relative offsets! The 26p Q/K machinery learns a simple offset function.
- Implementation: OffsetAttention with per-head (x_offset, y_offset, sharpness, self_weight, special_weight) = 10p total
- Risk: The model might need more flexible attention patterns during training even if it converges to fixed offsets
- **Savings:** q_proj(24p) + q_phase(2p) → offset params(10p) = -16p

#### Exp C: Stack Everything → target sub-100p
- Vocab=10 + Offset Attention + freeze_special=all on d_model=6
- Count: 170 - 8(tok_emb) - 8(special_pos) - 16(offset_attn) = 138p
- Then try d_model=5 or parametric embeddings for further cuts

### Implementation Status

#### Vocab=10 — IMPLEMENTED ✓
- `data.py`: `get_token_ids(vocab_size)` maps special tokens to digit-0
- `make_example()`, `sample_batch()`, `decode_answer()` all take `vocab_size` parameter
- `model.py`: `--vocab-size 10/12/14` flag, `--freeze-special all` option added
- Parameter count confirmed: 162p (freeze=eos), 154p (freeze=all)

**Training runs launched (3 parallel, max GPU):**
1. `sub100_v10_feos_s80085` — vocab=10, freeze=eos, seed 80085 (162p)
2. `sub100_v10_fall_s80085` — vocab=10, freeze=all, seed 80085 (154p)
3. `sub100_v10_feos_s42` — vocab=10, freeze=eos, seed 42 (162p)

Status at ~36K steps: tok_acc 0.51-0.59, no grokking yet. Need to be patient — the 170p baseline grokked at ~15K with seed 80085 but vocab change may alter the grokking dynamics.

#### Fixed-Offset Attention — IMPLEMENTED ✓
- `OffsetAttention` class in model.py: `--attn-mode offset`
- Per-head parameters: x_offset(2), y_offset(2), sharpness(2), self_weight(2), special_weight(2) = 10p total
- Replaces q_proj(24p) + q_phase(2p) = 26p → saves 16p
- With vocab=10 + offset + freeze=eos: 146p
- With vocab=10 + offset + freeze=all: 138p
- Forward pass and generation tested and working
- **Waiting for GPU slot** to launch training (3 runs already active)

### Key Observations

1. The model's attention is remarkably simple — just fixed positional offsets. The 26p Q/K is massive overhead for what amounts to "look at position i+2 in X and i+1 in Y".

2. The offset attention initialization matters. I initialized x_offset/y_offset to match the discovered patterns (Head0: +2/+1, Head1: +1/0). This gives the model a head start.

3. For vocab=10, the 12-token answer sequence now ends with a "0" (was EOS=12) which the model needs to predict. Since we always generate exactly 12 tokens and decode exactly 11 answer digits, this should be fine — the last "0" is just a dummy prediction at the EOS position.

### Results: Vocab=10 (d_model=6, 162p) — FAILED

All vocab=10 runs on the original d_model=6 architecture failed to grok:
- `v10_feos_s80085` (162p): 90K steps, tok_acc=0.70, val_exact=0.003 — no grokking
- `v10_feos_lowwd` (162p): 117K steps, tok_acc=0.69, val_exact=0.002 — stagnant
- `v10_aggressivewd` (162p): 48K steps, tok_acc=0.51 — early stages but declining
- `v10_fall` (154p): stuck at 54% tok_acc — freeze_all too aggressive
- `hardoff2_v10` (136p): 54K steps, tok_acc=0.24 — hard_offset dead on d6

**Conclusion:** Vocab=10 on d_model=6 significantly slows grokking. The model reaches ~70% tok_acc but can't break through to grokking. The shared "0" token for digits + special tokens may confuse the value pathway.

### Key Discovery: d_model=5, 1-head (141p) — NEAR GROKKING

Previous experiment `sub100_exp3_d5h1_141p` shows d5h1 is remarkably capable:
- **d5h1 s42 (141p, vocab=14)**: 201K steps, **14.6% exact, 87.4% tok_acc** — climbing!
- **d5h1 s80085 wd10 (141p)**: 96K steps, 90.8% tok_acc, 8.2% exact — WD Stage 1 fired
- The architecture: d_model=5, tok_dim=2, pos_dim=3, 1 head, head_dim=5, tied Q/K+q-phase, rank-2 out_proj

These models are on the cusp of grokking — ~90% tok_acc with rising exact match. The 170p model grokked sharply, but d5h1 shows a slow, gradual increase in exact match.

**Parameter savings path from d5h1 (141p):**
| Config | Params |
|--------|--------|
| d5h1 baseline | 141p |
| + vocab=10 | 133p |
| + freeze_all | 127p |
| + parametric tok_emb | 111p |
| + offset attn | 105p |

If d5h1 can grok to 100%, all these variants become viable paths to sub-100p!

## Session 2 — 2026-03-01 (continued)

### Active Experiments: Pushing d5h1 to grok

Launched 3 runs to break the d5h1 plateau:

1. **`sub100_d5h1_resume_lowwd_s80085`** — Resume from d5h1 wd10 s80085 (96K checkpoint, 90.8% tok_acc). Ultra-aggressive WD: Stage1 at val_exact>0.01, Stage2 at val_exact>0.05, tok_acc gate>0.5.
   - Status at 30K (from resume): 91.9% tok_acc, 8.9% exact. WD dropped to 1e-4 at step 9K. **Plateauing.**

2. **`sub100_d5h1_s42_lowwd`** — Resume from d5h1 s42 (201K checkpoint, 14.6% exact). Same aggressive WD.
   - Status at 30K (from resume): 87.5% tok_acc, **15.4% exact** — still climbing! Best exact match so far.

3. **`sub100_d5h1_nowd_s80085`** — Fresh d5h1 with constant WD=0.0001. Testing if the model can learn with near-zero WD throughout.
   - Status at 30K: 24.7% tok_acc — very slow, needs WD for early training.

### Observations

1. **d5h1 IS slowly grokking** — s42_lowwd shows steadily increasing exact match (7% → 15.4% over 30K steps from resume). This is a genuine grokking curve, just very gradual.

2. **WD=0.0001 from scratch fails** — The model needs regularization during the memorization phase. The constant low-WD approach doesn't work; you need adaptive WD that starts high and drops.

3. **s80085 plateaus at ~9% exact while s42 climbs past 15%** — Seed matters a lot at d5h1. The s42 seed found a better basin.

4. **The carry circuit at d_model=5 is more compressed** — With pos_dim=3 (vs 4 at d_model=6), the positional encoding has less room. The FFN sees 5-dim input instead of 6-dim. This may require more training steps to develop the same carry-propagation circuit.

### BREAKTHROUGH: d5h1 141p GROKKED TO 100%! ✓

**Step 243K: 100.0% exact match (5000 samples)**
**Full eval: 10010/10010 correct on AdderBoard test set!**

The `sub100_d5h1_s42_lowwd` run (resumed from 201K-step checkpoint with aggressive WD) grokked!

Training trajectory (from resume):
- 0-100K: ~87% tok_acc, ~15-27% exact — slow climb, WD dropped to 1e-4 at step 9K
- 108K: 27.3% exact → 111K: **83.9% exact** — GROKKING ONSET (3K steps!)
- 126K: **97.3% exact** — nearly perfect
- 165K: **99.4% exact**
- 243K: **100.0% exact** — FULLY GROKKED

The model oscillated significantly between 77-100% after the grokking onset, but ultimately stabilized.

**Submission created: `submission_141p/`**
- 141p, 100% accuracy, verified on 10010 samples
- Architecture: d_model=5, tok_dim=2, pos_dim=3, 1 head, head_dim=5, ff=2
- Config: tied Q/K + q-phase, rank-2 out_proj, spiral+linear correction positions, freeze_special=eos

### Analysis: Why d5h1 works

1. **1 head with head_dim=5 > 2 heads with head_dim=3**: The single head has full d_model-dimensional key space (5D), giving it more expressive power per head than the 2-head design (3D keys each). It sacrifices the dual-head routing (lookahead + current) for a richer single representation.

2. **Seed sensitivity**: Only s42 grokked. s80085 plateaued at 9% exact despite 90.8% tok_acc. This suggests the grokking basin is narrow at 141p — the model barely has enough capacity, and the initial weights must land near the right basin.

3. **Resume + aggressive WD is the key**: The model needed ~200K steps of regular training to develop the carry circuit, followed by aggressive WD reduction (Stage 2 at val_exact=0.05) to sharpen it. This two-phase approach was critical.

### Next Goal: Sub-100p

From 141p, the savings path:
| Change | Params saved | New total |
|--------|-------------|-----------|
| vocab=10 | 8p | 133p |
| + freeze_all | 6p | 127p |
| + parametric tok_emb | 16p | 111p |
| + offset attn (replace q_proj+q_phase) | 6p | 105p |

All these features are already implemented! But we need to verify they can grok at d5h1.

### BREAKTHROUGH #2: d5h1 + vocab=10 = 133p, GROKKED AT 15K STEPS!

The `sub100_d5h1_v10_s80085` run grokked incredibly fast:
- Step 12K: 5.2% exact, 79.0% tok_acc
- Step 15K: **85.5% exact** — instant grokking!
- Step 30K: **100.0% exact** — FULLY GROKKED
- Full eval: **10010/10010 correct on AdderBoard!**

This is remarkable:
1. **vocab=10 works at d5h1!** Despite failing on d_model=6, it works perfectly on d5h1.
2. **Grokking was FASTER than baseline** — 15K steps vs 170p's 15K steps, but at 133p!
3. **Seed 80085 grokked this time** (while it plateaued at 141p vocab=14)

The difference is likely that the aggressive WD thresholds (Stage 2 at val_exact=0.05) fire earlier with vocab=10 since the model starts generalizing at lower exact match levels.

**Submission created: `submission_133p/`**

### Next Experiments Targeting Sub-100p

From 133p, remaining savings:
- freeze_all (127p) — FAILED (dead at d5h1)
- parametric tok_emb (117p) — not yet tested at d5h1
- offset attn (needs fixing for n_heads=1)

The freeze_all failure is concerning — it means we can't eliminate special_pos (6p).
So the realistic target is: 133p - 16p(parametric) = 117p. Maybe offset attn saves a few more → ~111p.

### BREAKTHROUGH #3: 100 PARAMETERS, 100% ACCURACY!!!

**`sub100_d5h1_100p_s42`**: d5h1 + vocab=10 + parametric tok_emb + rank-1 out_proj + no FFN bias

- **100 parameters total**
- Step 30K: 80.2% exact — early grokking
- Step 36K: **98.5% exact** — massive jump
- Step 84K: **100.0% exact** (5K samples)
- **Full eval: 10010/10010 on AdderBoard!**

Parameter breakdown (100p):
```
tok_arc_A/B/start/stride: 4 (parametric embedding)
spiral_amp/phase/slope/offset: 4
pos_corr_slope/intercept: 2
z_hi_pos: 3
special_pos_learned: 6 (PLUS, EQUALS)
ln1/ln2/ln_f weights: 15 (3×5)
q_proj: 15 (3→5)
q_phase: 1
v_proj: 10 (2→5)
out_proj: 10 (rank-1: 5+5)
ffn_fc1: 10 (5→2, no bias)
ffn_fc2: 10 (2→5, no bias)
head_proj: 10 (5→2)
```

Key innovations stacked:
1. d_model=5 (from 6): Saves 29p across all layers
2. 1 head, head_dim=5: Full-rank attention in a single head
3. vocab=10: Special tokens → digit-0 (saves 8p)
4. Parametric tok_emb: 4 arc params instead of 20 learned (saves 16p)
5. Rank-1 out_proj: 10p instead of 20p (saves 10p)
6. No FFN bias: Saves 7p (fc1_bias=2, fc2_bias=5)

**Submission created: `submission_100p/`**

### Continuing: Can we go below 100p?

Remaining savings:
- freeze_all special positions: saves 6p → 94p (failed before but worth retrying)
- Remove q_phase (1p) → 99p (marginal)
- Shared norms (save 10p) → 90p

The target is now to explore how much further we can push.

### 94p (freeze_all) — FAILED

Launched `sub100_d5h1_94p_s42` with freeze_special="all" (all special positions frozen to zero).
After 93K steps: 0% exact, 26.7% tok_acc — completely dead. Same failure mode as all prior freeze_all attempts.

**Conclusion: Special position embeddings (PLUS, EQUALS) are load-bearing. They cannot be frozen.**

### Sub-100p Strategy: New Parameter Savings

Analyzed the 100p parameter budget and identified two new structural savings:

1. **Shared norms (norm_mode="shared")**: All 3 RMSNorm weight vectors (5 params each = 15p total) share one vector → 5p. Saves 10p.

2. **Tied V/output (tie_vo)**: `v_proj.weight` (5×2) and `head_proj.weight` (2×5) are transposes — both map between tok_dim and d_model. Tying them saves 10p. Implemented by removing v_proj and using `head_proj.weight` directly in the attention forward pass: `v = x_tok @ head_proj.weight` instead of `v = v_proj(x_tok)`.

Parameter configurations:
| Config | Params |
|--------|--------|
| 100p baseline (current) | 100p |
| + shared norms | 90p |
| + shared norms + no q_phase | 89p |
| + fixed norms (no norm weights) | 85p |
| + shared norms + tie_vo | 80p |
| + fixed norms + tie_vo | 75p |
| + shared + tie_vo + no q_phase | 79p |

### BREAKTHROUGH #4: 80 PARAMETERS, 100% ACCURACY!!!

**`sub100_d5h1_80p_s42`**: d5h1 + vocab=10 + parametric tok_emb + rank-1 out_proj + no FFN bias + shared norms + tie_vo

- **80 parameters total**
- Step 111K: Grokking onset (0% → 10.5% exact in 3K steps)
- Heavy post-grokking oscillation (11-94% exact) — classic pattern
- Step 240K: **100.0% exact** (5K samples) — first stable peak
- **Full eval: 10010/10010 on AdderBoard!**

Parameter breakdown (80p):
```
tok_arc_A/B/start/stride: 4     (parametric embedding)
spiral_amp/phase/slope/offset: 4 (spiral positions)
pos_corr_slope/intercept: 2     (correction)
z_hi_pos: 3                     (carry position)
special_pos_learned: 6          (PLUS=3, EQUALS=3)
shared_norm_weight: 5           (ONE vector shared by ln1, ln2, ln_f)
q_proj: 15                      (3→5, no bias)
q_phase: 1                      (angle)
out_proj: 10                    (rank-1: 5+5)
ffn_fc1: 10                     (5→2, no bias)
ffn_fc2: 10                     (2→5, no bias)
head_proj: 10                   (5→2, no bias, also used as v_proj)
Total: 80
```

Key innovations on top of 100p:
1. **Shared norms**: All 3 RMSNorm weight vectors share one parameter vector (5p vs 15p)
2. **Tied V/output (tie_vo)**: v_proj.weight = head_proj.weight.T, so `v = x_tok @ head_proj.weight`. Both map between tok_dim and d_model. Saves 10p.

Observations:
- Despite the analysis showing v_proj and head_proj.T are NOT naturally similar in the 100p model (cosine sim = -0.30), the 80p model found a solution where a single weight matrix works for both roles!
- Despite norms being dissimilar in the 100p model, the shared norm constraint doesn't prevent grokking
- The grokking onset was later (111K vs 30K for 100p) and oscillation was heavier, but the model ultimately converged

**Submission created: `submission_80p/`**

### 90p s42 — FAILED (500K steps, 7.5% exact plateau)

The `sub100_d5h1_90p_s42` run hit the 500K step limit without grokking. Stuck at 7.5% exact, 90% tok_acc. The shared norm constraint without tie_vo is harder to grok at seed 42.

### BREAKTHROUGH #5: 90p s80085 — GROKKED AT 48K STEPS!

Different seed makes all the difference. `sub100_d5h1_90p_s80085`:
- Step 48K: **100% exact** (first hit, but 10009/10010 on full eval — 1 error)
- Step 153K: **100% exact** (stable, 10010/10010 on full eval confirmed!)

90p breakdown: Same as 80p but with independent v_proj (10p extra, no tie_vo).

**Key insight:** Seed 80085 groks 90p incredibly fast (48K vs 500K for s42 failing). Meanwhile, s42 groks 80p fast but s42 fails on 90p. The tie_vo constraint acts as regularization that changes which seeds work!

**Submission created: `submission_90p/`**

### 75p (fixed norms + tie_vo) experiments

1. **75p from scratch s42**: Dead — 435K steps, 0% exact, 28.5% tok_acc
2. **75p warm-started s42**: 435K steps, 9.4% exact, 92.1% tok_acc — plateau
3. **75p from scratch s80085**: Not yet tried

The fixed norms constraint is very aggressive — the shared norm weight in the 80p model has values [-1.8, 3.8, 1.8, 2.0, 6.0], far from all-ones. This per-dimension scaling is doing real computational work.

### Active Experiments (Session 3 continued)

1. **`sub100_d5h1_80p_s80085`** — 80p with seed 80085 (to see if this seed also works for 80p)
2. **`sub100_d5h1_70p_s42`** — 70p: shared norms + tie_vo + **ffn_dim=1** (saves 10p from 80p)
3. **`sub100_d5h1_75p_warm_s42`** — 75p warm-started from 80p, still running

The 70p experiment is radical: the FFN bottleneck is reduced from 2 to 1. The MLP becomes a rank-1 "carry score" computer: project to scalar, GELU, project back. If it works, it would suggest the carry function can be computed by a single threshold.

### Comprehensive results table

| Config | Params | Seeds tried | Best result | Status |
|--------|--------|-------------|-------------|--------|
| d6h2 baseline | 170p | 80085 | 100% | GROKKED |
| d5h1 | 141p | 42 | 100% | GROKKED |
| d5h1 + vocab=10 | 133p | 80085 | 100% | GROKKED |
| d5h1 + vocab=10 + param + rank2 | 117p | 42 | 46% (500K) | PLATEAU |
| d5h1 + vocab=10 + param + rank1 + no bias | 100p | 42 | 100% | GROKKED |
| d5h1 + vocab=10 + param + rank1 + no bias + freeze_all | 94p | 42 | 0% (93K) | FAILED |
| 100p + shared norms | 90p | 42, 80085 | 100% (s80085) | GROKKED |
| 100p + shared norms + tie_vo | 80p | 42, 80085 | 100% (s42) | GROKKED |
| 80p + no pos correction | 78p | 42 | 100% | GROKKED |
| 80p + fixed norms | 75p | 42, 42(warm) | 9.4% (warm) | PLATEAU |
| 80p + ffn_dim=1 | 70p | 42, 80085 | 60% tok (s42) | FAILED |

### BREAKTHROUGH #6: 78 PARAMETERS, 100% ACCURACY!!!

**`sub100_d5h1_78p_s42`**: Same as 80p but WITHOUT position correction parameters (pos_corr_slope, pos_corr_intercept removed). 78 total params.

- Step 156K: 8.6% exact — grokking onset
- Step 234K: 63.9% exact — post-grokking oscillation begins
- Step 333K: 98.1% exact — stabilizing
- Step 414K: **100.0% exact** (best checkpoint)
- **Full eval: 10010/10010 on AdderBoard!**

78p breakdown (vs 80p: removed pos_corr_slope/intercept = -2p):
```
tok_arc_A/B/start/stride: 4
spiral_amp/phase/slope/offset: 4  (no correction!)
z_hi_pos: 3
special_pos_learned: 6
shared_norm_weight: 5
q_proj: 15
q_phase: 1
out_proj: 10 (rank-1)
ffn_fc1: 10 (no bias)
ffn_fc2: 10 (no bias)
head_proj: 10 (also v_proj via tie_vo)
Total: 78
```

Analysis of the removed correction:
- In the 80p model, pos_corr_intercept = 2.96, pos_corr_slope = 0.07
- This applies a ~4x scaling to spiral positions with slight linear variation
- The spiral_amp parameter can absorb the constant scaling factor
- The 78p model learns to work with pure spiral positions + independent amplitude

**Submission created: `submission_78p/`**

### Failed experiments update

- **80p s80085**: Dead at 28.5% tok_acc, 261K steps. Seed 80085 works for 90p (no tie_vo) but not 80p (with tie_vo).
- **70p s42**: Dead at 59.8% tok_acc, 339K steps. ffn_dim=1 too constrained.
- **70p s80085**: Dead at 33% tok_acc, 234K steps.
- **90p s42**: Hit 500K limit at 7.5% exact, never grokked. Only s80085 works.
- **75p from scratch**: Dead at 28% tok_acc. Fixed norms too aggressive.
- **75p warm-started**: Plateaued at 9-10% exact, 92% tok_acc. Close but can't break through.

### Key learnings

1. **Seed sensitivity is architecture-dependent**: s42 works for tie_vo models (80p, 78p), s80085 works for non-tie_vo (90p, 133p). The regularization effect of weight tying changes which optimization basins are accessible.

2. **Position correction is NOT necessary**: The 78p model proves the model can learn pure spiral positions. The correction was compensating for suboptimal spiral amplitude, which the model can learn directly.

3. **tie_vo acts as beneficial regularization**: 80p groks at 111K while 90p s42 never groks (500K). The forced weight sharing between V and output constrains the solution space to finding joint representations, which apparently helps grokking.

4. **ffn_dim=1 is too constrained**: The carry computation genuinely needs 2 hidden dimensions. A single scalar bottleneck can't compute the required threshold function.

## Session 3 — 2026-03-01 (continued)

### Resuming from 75p grokking

The 75p (freeze_plus_eos) run was killed mid-grok at step 168K (93.8% peak exact).
Resumed with `--resume` from last.pt (step 195K). The LR schedule restarts but model
state + optimizer state are preserved. At step 27K of the resumed run, WD Stage 2
kicked in at 7.2% exact — the grokking pattern is repeating.

### New compression: q_proj rank-1 factorization

The q_proj maps pos_dim(3) → head_dim(5) = 15 params. A rank-1 factorization
LowRankLinear(3,5,1) = 3+5 = 8 params saves 7 params.

Added `--q-proj-rank` option to both model.py and train.py.

Current experiments running:
1. **75p cont (s42)**: Resumed from 195K checkpoint, grokking in progress
2. **68p (s42)**: q_proj_rank=1 on top of 75p config → 68 params
3. **65p (s42)**: 68p + freeze_z_hi → 65 params

### 68p / 65p experiments — FAILED
Rank-1 q_proj factorization (LowRankLinear(3,5,1) = 8p instead of 15p):
- Both 68p and 65p stuck at 20.8% tok_acc (random) through 30K steps
- Rank-1 maps all 3D positions to a 1D line — can't distinguish digit positions
- Rank-2 gives 16p > 15p (full rank), so factorization doesn't help for these small dims

### Breakthrough #7: 72p — freeze_z_hi warm-started from 75p

The 72p = 75p with z_hi_pos frozen at zero (carry position = all zeros).
Fresh 72p from scratch failed (70% tok_acc at 153K), but warm-starting from the
75p best checkpoint (step 141K, 93.8% exact) bootstrapped it past the hard phase.

**72p grokking timeline (warm-started):**
- Step 24K: 58.7% exact (fast pickup from warm-start)
- Step 42K: 88.9% exact (first near-peak)
- Step 66K: 94.4% exact
- Step 273K: 99.86% exact
- Step 312K: **100.0% exact** (first perfect)
- Step 345K: 100.0% exact
- Step 360K: 100.0% exact

**Full eval: 10010/10010 correct (autoregressive, seed 2025)**

Config: `--norm-mode shared --tie-vo --attn-out-rank 1 --no-ffn-bias --pos-correction-mode none --freeze-special plus_eos --freeze-z-hi --tok-emb-mode parametric --vocab-size 10 --tie-qk --q-phase --warm-start <75p_best.pt>`

**Submission created: `submission_72p/`**

### 72p parameter breakdown
```
Component           Params
tok_emb (arc)            4  (A, B, start, stride)
spiral_pos               4  (amp, phase, slope, offset)
z_hi_pos              [0]  (frozen buffer)
equals_pos               3  (EQUALS only; PLUS+EOS frozen)
q_proj                  15  (3->5, full rank)
out_proj (rank-1)       10  (5x1 + 1x5)
q_phase                  1
FFN fc1                 10  (5x2, no bias)
FFN fc2                 10  (2x5, no bias)
head_proj               10  (5->2, tied as v_proj)
RMSNorm (shared)         5
TOTAL                   72
```

### 75p seed 80085 — also GROKKING
While 72p was converging, 75p with seed 80085 also started grokking:
- Step 69K: 73.4% exact
- Step 159K: 90.5% exact
- Step 201K: 95.6% exact
Confirms s80085 works for this architecture too.

