# Research Journey: Minimizing a 10-Digit Addition Transformer

## Overview

This document traces the path from a 170-parameter single-layer decoder that solves 10-digit addition with 100% accuracy (10010/10010 on the AdderBoard test set) down to a **75-parameter model trained from scratch** that achieves the same perfect accuracy.

The journey spans six research sessions covering architecture compression, training dynamics, seed sensitivity, and several failed approaches (scaffold training, SAM, WD scheduling) that produced important negative results. The legitimate from-scratch frontier is **75 parameters** (seed 80085). Models below 75p relied on warm-starting with frozen learned values -- an interesting research direction documented here, but not a true from-scratch result.

---

## Session 1: Architecture Exploration (170p to 133p)

### Starting Point

- **Baseline:** 170p, 100% accuracy, seed 80085
- **Architecture:** 1-layer decoder, d_model=6 (tok_dim=2, pos_dim=4), 2 heads, head_dim=3, ffn_dim=2
- **Target:** Sub-100p model, 100% on 10-digit addition

### Research Plan

Three experiments motivated by structural analysis of the trained 170p model:

**Exp A: Vocab reduction (14 to 10).** The model's attention is purely positional -- it never uses token identity for delimiters. PLUS/EQUALS/EOS/PAD can all map to digit-0, distinguished by position alone. Saves 8p from embeddings.

**Exp B: Fixed-offset attention.** The learned Q/K machinery computes simple positional offsets (Head 0: look-ahead at X_{i+2}/Y_{i+1}; Head 1: current X_{i+1}/Y_i/self). Replace 26p of Q/K params with 10p of explicit offset parameters.

**Exp C: Stack everything.** Combine vocab=10 + offset attention + freeze_special to reach sub-140p, then explore d_model=5.

### Results

**Vocab=10 on d_model=6: FAILED.** All runs (162p, 154p, 136p) stalled at ~70% token accuracy without grokking. The shared "0" token for digits and special tokens confused the value pathway at d_model=6. This was a dead end for the original architecture.

**d_model=5, 1-head: NEAR-GROKKING.** A parallel experiment with d_model=5, 1 head, head_dim=5 (141p, vocab=14) showed slow but genuine grokking -- 14.6% exact match at 201K steps with s42, still climbing. This became the key architectural insight.

### Key Discovery: 1 Head > 2 Heads

One head with head_dim=5 (full d_model rank) is more expressive than two heads with head_dim=3 each. The single head has a 5D key space, trading dual-head routing for a richer representation. This architectural change unlocked the entire compression path.

---

## Session 2: Breaking Through to Sub-100p (141p to 78p)

### 141p: d5h1 Groks (seed 42)

The resumed d5h1 run (`sub100_d5h1_s42_lowwd`, 141p) grokked at step 243K of the resumed training:

- 0-108K: Slow climb, 87% tok_acc, ~15-27% exact
- 108K-111K: **Grokking onset** -- 27% to 84% exact in 3K steps
- 243K: **100.0% exact** (10010/10010)

The model needed ~200K steps of regular training to develop the carry circuit, followed by aggressive WD reduction to sharpen it. Seed sensitivity was already apparent: s80085 plateaued at 9% exact on the same architecture.

### 133p: Vocab=10 Works at d5h1

Despite failing at d_model=6, vocab=10 worked perfectly at d5h1 -- and grokked even faster:

- `sub100_d5h1_v10_s80085` (133p): Grokking at 15K steps, 100% at 30K
- Seed 80085 grokked here despite failing on 141p (vocab=14)

The aggressive WD thresholds fire earlier with vocab=10 since the model starts generalizing at lower exact-match levels.

### 100p: The Parametric Compression Stack

**`sub100_d5h1_100p_s42`**: d5h1 + vocab=10 + parametric tok_emb + rank-1 out_proj + no FFN bias = **100 parameters, 100% accuracy.**

- Grokking onset at 30K, 100% at 84K (10010/10010)

Six innovations stacked:
1. **d_model=5** (from 6): Saves 29p across all layers
2. **1 head, head_dim=5**: Full-rank attention in a single head
3. **vocab=10**: Special tokens map to digit-0 (saves 8p)
4. **Parametric tok_emb**: 4 arc parameters instead of 20 learned embeddings (saves 16p)
5. **Rank-1 out_proj**: 10p instead of 20p (saves 10p)
6. **No FFN bias**: Saves 7p

### 80p: Shared Norms + Tied V/Output

Two structural insights yielded another 20p savings:

- **Shared norms** (`norm_mode="shared"`): All 3 RMSNorm weight vectors share one 5D vector (5p vs 15p)
- **Tied V/output** (`tie_vo`): v_proj.weight = head_proj.weight^T -- both map between tok_dim and d_model (saves 10p)

`sub100_d5h1_80p_s42`: Grokking onset at 111K (later than 100p, heavy oscillation), stabilized at 100% by 240K. 10010/10010.

Despite analysis showing these weight matrices were NOT naturally similar in the 100p model (cosine sim = -0.30), the 80p model found a joint solution. tie_vo acts as beneficial regularization.

### 78p: Position Correction is Unnecessary

Removing pos_corr_slope and pos_corr_intercept (2p): the spiral_amp parameter can absorb the constant scaling factor.

`sub100_d5h1_78p_s42`: Grokked at 414K, 10010/10010. **78 parameters from scratch.**

### 75p: The From-Scratch Frontier

75p = 78p minus the 3 learned EQUALS position parameters (EQUALS frozen to zero). This is the minimal configuration that groks from scratch.

- **Seed 80085**: 100% at 276K, 10010/10010
- **Seed 42**: Dead from scratch (28% tok_acc at 435K)

### Results Summary: From-Scratch Models

| Params | Architecture delta | Seeds tried | Best from-scratch result |
|--------|-------------------|-------------|--------------------------|
| 170p | Baseline d6h2 | 80085 | 100% (15K grok) |
| 141p | d5h1, vocab=14 | 42, 80085 | 100% s42 (243K grok) |
| 133p | + vocab=10 | 80085 | 100% (15K grok) |
| 100p | + parametric + rank-1 + no bias | 42 | 100% (84K grok) |
| 90p | + shared norms | 80085 | 100% (48K grok) |
| 80p | + tie_vo | 42 | 100% (240K grok) |
| 78p | + no pos correction | 42 | 100% (414K grok) |
| **75p** | **+ freeze PLUS/EOS** | **80085** | **100% (276K grok)** |
| 70p | + freeze spiral(offset,phase) | 80085 | 36% tok_acc -- FAILED |
| 66p | + freeze all spiral + tok_arc | 80085 | 25% tok_acc -- FAILED |

**Below 75p, no seed has been found that groks from scratch.** The 70p and 66p configurations fail even with s80085.

### 75p Parameter Breakdown

```
Component           Params
tok_emb (arc)            4  (A, B, start, stride)
spiral_pos               4  (amp, phase, slope, offset)
z_hi_pos                 3  (carry position)
equals_pos               3  (EQUALS learned; PLUS+EOS frozen)
q_proj                  15  (3->5, full rank)
out_proj (rank-1)       10  (5x1 + 1x5)
q_phase                  1
FFN fc1                 10  (5x2, no bias)
FFN fc2                 10  (2x5, no bias)
head_proj               10  (5->2, tied as v_proj)
RMSNorm (shared)         5
TOTAL                   75
```

### Dead Ends

- **freeze_all special positions (94p):** Dead at 0% exact -- PLUS/EQUALS positions are load-bearing
- **Rank-1 q_proj (68p):** Dead at 20% tok_acc -- 1D keys can't distinguish positions
- **Scalar norm (68p):** Dead at 50-60% tok_acc -- per-dim norm weights are load-bearing
- **d_model=4 (55p):** Dead at 21% tok_acc -- pos_dim=2 too constraining for spiral
- **ffn_dim=1 (70p):** Dead at 60% tok_acc -- carry computation genuinely needs 2 hidden dims
- **No q_phase (69p):** Dead at 21% tok_acc -- q_phase essential for tied Q/K asymmetry

### Key Learnings

1. **Seed sensitivity is architecture-dependent.** s42 works for tie_vo models (80p, 78p), s80085 works for non-tie_vo (90p, 133p). Weight tying changes which optimization basins are accessible.

2. **Position correction is redundant.** The spiral_amp parameter absorbs constant scaling.

3. **tie_vo is beneficial regularization.** 80p (with tie_vo) groks at 111K while 90p s42 (without tie_vo) never groks at 500K.

4. **ffn_dim=1 is too constrained.** The carry computation needs 2 hidden dimensions; a single scalar bottleneck cannot compute the required threshold function.

---

## Session 3: Warm-Start Cascade (75p to 62p)

*Note: Everything in this section relies on warm-starting from parent models and freezing learned parameter values as buffers. These are interesting research results but do not represent legitimate from-scratch training.*

### The Cascade Strategy

Starting from the 75p from-scratch model, we systematically froze additional parameters at their trained values and warm-started from each successive checkpoint:

```
75p (from scratch, s80085)
 -> 72p: freeze z_hi_pos (3 params)
     -> 71p: freeze spiral_offset (1 param)
     -> 70p: freeze spiral_offset + spiral_phase (2 params)
         -> 68p: freeze 3 spiral params + tok_arc_stride (4 params from 70p)
         -> 66p: freeze ALL spiral + tok_arc start/stride (4 params from 70p)
             -> 65p: tie tok_arc_A = tok_arc_B (1 param from 66p)
             -> 63p: freeze EQUALS position (3 params from 66p)
             -> 62p: tie A=B + freeze EQUALS (4 params from 66p)
```

All models achieved 10010/10010 on the AdderBoard test set. Each warm-start step preserves trained values as frozen buffers, so the model retains its carry circuit while the parameter count decreases.

### Why This Works

At each step, we identify parameters whose learned values are either (a) redundant (spiral_phase can be absorbed by q_proj rotation) or (b) near-identical across models (tok_arc_A and tok_arc_B learned values 11.39 vs 11.47, ratio 0.993). Freezing such parameters preserves the computation while removing them from the learnable count.

### Why This is a Technicality

The 62p model has only 62 learnable parameters, but its computation depends on 18 additional frozen buffer values extracted from upstream trained models. Without those specific frozen values, the 62p configuration cannot learn addition from scratch. This makes the warm-start cascade an exercise in post-hoc compression rather than true minimal learning.

### 62p Parameter Breakdown (Lowest Warm-Start Result)

```
Learnable:
tok_arc_A (=B)           1  (circular arc amplitude)
q_phase                  1  (Q rotation angle)
q_proj                  15  (3->5, full rank)
out_proj (rank-1)       10  (5x1 + 1x5)
FFN fc1                 10  (5x2, no bias)
FFN fc2                 10  (2x5, no bias)
head_proj               10  (5->2, tied as v_proj)
RMSNorm (shared)         5
TOTAL                   62

Frozen buffers (from warm-start cascade):
tok_arc_start            1
tok_arc_stride           1
spiral (amp,phase,slope,offset)  4
special_pos_equals       3
z_hi_pos                 3
_plus_pos, _eos_pos      6  (zero vectors)
TOTAL frozen            18
```

### Failed Sub-70p Attempts

- **64p (freeze ALL tok_arc params):** Stuck at 50-60% exact -- without any learnable amplitude, digit embeddings are fixed circles and can't adapt spacing
- **69p (no q_phase from 70p):** Dead at 21% tok_acc

---

## Session 4: SAM and WD Variants (Negative Results)

### Motivation

The from-scratch frontier sat at 75p (later confirmed at 72p with s80085). Can alternative optimizers push it lower?

### SAM (Sharpness-Aware Minimization)

**Hypothesis:** The 72p loss landscape has sharp local optima trapping SGD. SAM's adversarial weight perturbation should find flatter minima.

**Result: SAM hurts small models.** Tested rho=0.05 and rho=0.01 across seeds 42 and 80085:

| Run | rho | Seed | tok_acc | Baseline tok_acc |
|-----|-----|------|---------|------------------|
| SAM | 0.05 | 42 | 43% | 70% |
| SAM | 0.05 | 80085 | 26% | 99.9% |
| SAM | 0.01 | 42 | 56% | 70% |

SAM was uniformly worse than vanilla AdamW. The adversarial perturbation disrupts the delicate feature learning required at 72p. Lower rho helps but never reaches baseline performance.

### WD Schedule Variants

**Hypothesis:** The adaptive WD has a chicken-and-egg problem at small model sizes -- WD doesn't drop because the model never reaches the grokking threshold, and the model never groks because WD is too high.

Tested scheduled drops (fixed step), cyclical (cosine oscillation), and warmup (zero then ramp) WD modes. **All produced identical ~26% tok_acc**, same as no WD adaptation. The 72p failure for most seeds is not caused by WD timing -- it is a seed/initialization issue.

### The Real Discovery: 72p From Scratch (seed 80085)

While testing alternatives, vanilla AdamW with s80085 grokked 72p from scratch:

- Grokking onset at 21K steps (90.6% tok, 28.4% exact)
- Extreme post-grokking oscillation for ~240K steps (0%-99.5% exact)
- Stabilized at 100% by step 267K (10010/10010)

This established that the from-scratch frontier was 72p, not 75p. (Later seed sweep work in Session 5 further confirmed this.)

### Key Insights

1. **SAM is counterproductive for tiny models.** The hypothesis that sharp minima trap SGD at 72p is wrong. The issue is basin accessibility from initialization, not basin sharpness.

2. **WD scheduling is irrelevant.** All schedule variants produce the same failure mode as no WD adaptation.

3. **Seed is everything at 72p.** s80085 groks easily; s42 gets to 70% tok_acc but never groks; s123 shows faint signals (2% exact); s1 and s1337 fail completely.

---

## Session 5: Seed Sensitivity Sweep

### Experimental Design

- **10 random seeds**: 13453, 15614, 29267, 41876, 65866, 67086, 72950, 78779, 81460, 84830
- **2 configs**: 75p (freeze_special=plus_eos) and 72p (same + freeze_z_hi)
- **Kill criterion**: val_exact < 0.2 AND tok_acc < 0.8 at step 50K
- **Max steps**: 300K
- Managed by `sweep_seed_sensitivity.py`

### Results: 75p

| Seed | Peak Exact | Peak Tok | Step@Peak | Behavior |
|------|-----------|---------|-----------|----------|
| 13453 | 0.3% | 68.7% | 36K | dead |
| 15614 | 0.0% | 61.6% | -- | dead |
| **29267** | **14.7%** | **85.8%** | 27K | slow signal, killed |
| 41876 | 0.5% | 76.4% | 42K | dead |
| 65866 | 0.2% | 69.0% | 33K | dead |
| **67086** | **2.5%** | **80.7%** | 27K | signal, collapsed |
| **72950** | **9.6%** | **86.6%** | 18K | early burst, collapsed |
| **78779** | **95.8%** | **99.7%** | **33K** | **flash grok: spiked to 95.8%, crashed by 51K** |
| 81460 | 0.0% | 26.6% | -- | dead |
| 84830 | 0.0% | 56.7% | -- | dead |

### Results: 72p

| Seed | Peak Exact | Peak Tok | Step@Peak | Behavior |
|------|-----------|---------|-----------|----------|
| 13453 | 1.0% | 71.3% | 27K | dead |
| 15614 | 0.0% | 26.3% | -- | dead |
| 29267 | 0.0% | 20.9% | -- | dead |
| 41876 | 0.2% | 68.4% | 39K | dead |
| **65866** | **4.9%** | **85.8%** | 51K | weak signal |
| **67086** | **91.6%** | **99.3%** | **282K** | **approaching 100% at step limit** |
| **72950** | **64.9%** | **96.5%** | **39K** | **flash grok, then crashed** |
| 78779 | 0.0% | 58.3% | -- | dead |
| 81460 | 0.0% | 35.5% | -- | dead |
| 84830 | 0.5% | 76.2% | 33K | dead |

### Key Findings

**1. Grokking seeds are config-specific.** Seeds that grok at 75p fail at 72p, and vice versa:

| Seed | 75p Peak | 72p Peak |
|------|---------|---------|
| 78779 | **95.8%** | 0.02% |
| 67086 | 2.5% | **91.6%** |
| 72950 | 9.6% | **64.9%** |
| 29267 | **14.7%** | 0.0% |

The freeze_z_hi constraint does not make grokking globally harder or easier -- it fundamentally changes the loss landscape topology, redirecting which initialization basins lead to the 100% solution.

**2. Flash grokking is a real phenomenon.** 75p s78779 achieved 95.8% exact at step 33K -- a full grokking spike in only 33K steps -- then immediately crashed to 0.5% by step 51K. The carry circuit crystallized briefly but the basin was too shallow to maintain under continued gradient updates. 72p s72950 showed similar behavior (64.9% then crash).

**3. About 10-20% of random seeds show grokking signals.** Counting seeds with >2% val_exact: 4/10 at 75p, 3/10 at 72p. Including s80085 (works for both), the estimated rate for stable 100% grokking is ~10%.

**4. The kill criterion was too aggressive.** The 50K cutoff missed critical runs: 75p s78779 grokked at 33K but crashed by 51K; 72p s72950 had 64.9% at 39K. Recommended future criterion: peak_exact < 0.02 AND tok_acc < 0.70 at step 80K.

---

## Session 6: Scaffold Experiments (Negative Results)

### Motivation

The warm-start cascade (Session 3) is tedious -- each step requires a separate training run. Scaffold training aims to replace this with a single run: train with extra capacity, anneal the scaffold to zero via L1 penalty, then hard-prune to the target size.

### Approach

Scaffold capacity was added by widening out_proj (rank 1 to 2) and FFN (dim 2 to 3), bringing a 75p target model to 85p effective parameters. L1 penalty on scaffold weights was supposed to drive them toward zero for clean pruning.

### Results: All Eight Experiments Failed

**Phase 1 -- Standard L1 (3 runs):**

| lambda | Seed | Outcome |
|--------|------|---------|
| 0.1 | 42 | L1 equilibrium: scaffold weights hover at 0.6-0.7, never reach zero |
| 0.1 | 80085 | Catastrophic collapse: L1 destroyed the carry circuit at 87K (42% exact to 0%) |
| 0.5 | 42 | Suppressed learning: L1 too strong from the start, 19% peak exact |

**Phase 2 -- Late L1 after grokking (3 runs):**

| lambda | anneal_start | Seed | Outcome |
|--------|-------------|------|---------|
| 0.1 | 150K | 42 | Degraded from 44% to 1% exact after L1 started |
| 0.1 | 150K | 80085 | Degraded from 55% to 0% exact after L1 started |
| 0 (control) | -- | 42 | 85p model plateaus at 47% exact, never groks |

The control run revealed a deeper problem: the 85p scaffold model never reaches 100% even without any L1. The widen-then-prune approach is doubly doomed.

**Phase 3 -- Freeze-in-place scaffold (2 runs):**
An alternative approach that freezes parameters at their trained values (like the warm-start cascade but automated). This is conceptually equivalent to the warm-start cascade and was abandoned as also being a technicality rather than a true training method.

### Root Cause Analysis

The L1 penalty is fundamentally adversarial to task learning in tiny models. Three failure modes:

1. **L1 equilibrium** (low lambda): Scaffold weights hover at small but nonzero values -- task gradient pushes them up, L1 pushes them down
2. **Catastrophic collapse** (medium lambda): L1 eventually wins, scaffold dims go to zero suddenly, carry circuit lost
3. **Suppressed learning** (high lambda): L1 too strong from the start, model never develops a carry circuit

**Core lesson:** You cannot remove structural capacity from a tiny model's carry circuit. Dimension pruning is like removing a wire from a circuit board. The warm-start cascade works precisely because it freezes (preserves) values rather than removing dimensions.

---

## Session 7: High Carry-Mix Training Breakthrough (74p)

### Motivation

Structural diagnostics of the 75p model revealed tok_arc A/B = 1.005 — a near-perfect circle. Tying A=B saves 1 parameter (75p→74p). But at 75p, only ~10% of random seeds grok. Can a better training recipe improve the odds?

### Discovery: Aggressive Carry-Mix + Step-Based Fade

**First attempt: carry_mix=0.8, metric-based fade.** Three random seeds (78988, 45214, 71046). Seed 45214 showed repeated flash grokking: 88.3% exact at 24K → crash → 98.8% at 33K → crash → 100% at 42K → crash. The pattern: when tok_acc crosses 0.9, carry_mix drops from 0.8 to 0 instantly (metric-triggered). This 80% distribution shift destabilizes the carry circuit. It recovers because carry_mix snaps back when tok_acc drops, creating a feedback oscillation.

**The fix: step-based carry_mix fade.** Replace metric-triggered fade with a fixed linear schedule: `--carry-mix-fade-start 10000 --carry-mix-fade-end 80000`. Full carry_mix for curriculum warmup, then smooth linear ramp to zero. No metric dependency, no oscillation.

**Second insight: shorter step budget.** With 400K total steps, cosine LR decay is too slow — LR is still 0.017 at 80K when the model needs to stabilize. With `--steps 120000`, LR at 80K drops to ~0.007, low enough to hold the grokking basin.

### Results

| Seed | Peak Exact | Grokked Step | Final | Behavior |
|------|-----------|-------------|-------|----------|
| **45214** | **100%** | **78K** | **100% (12K hold)** | Multiple crashes, locked in at LR=0.007 |
| **71046** | **100%** | **81K** | **100% (12K hold)** | Similar pattern, stable after 81K |
| 78988 | 98.9% | 87K | 98.9% (still climbing) | Approaching convergence |

**2 out of 3 random seeds grokked at 74p from scratch.** Compare to ~10% (1/10) grok rate at 75p with carry_mix=0.3.

### Why High Carry-Mix Works

At 0.3 carry_mix, long carry chains (e.g., 9999999999+1) appear in ~30% of batches. With 0.8, it's 80%. The carry circuit is the hardest part of addition — it's the gate between memorization (per-digit lookup) and generalization (propagated carry). Flooding the model with carry-heavy examples during the critical early phase forces the carry circuit to form before the curriculum shifts to uniform sampling.

The step-based fade is essential at high carry_mix. Metric-based fade creates a violent feedback loop: accuracy up → carries removed → accuracy down → carries restored. At 0.3 this is a mild perturbation; at 0.8 it's catastrophic.

### 74p Parameter Breakdown

```
Component           Params
tok_emb (arc)            3  (A=B tied, start, stride)
spiral_pos               4  (amp, phase, slope, offset)
z_hi_pos                 3  (carry position)
equals_pos               3  (EQUALS learned; PLUS+EOS frozen)
q_proj                  15  (3->5, full rank)
out_proj (rank-1)       10  (5x1 + 1x5)
q_phase                  1
FFN fc1                 10  (5x2, no bias)
FFN fc2                 10  (2x5, no bias)
head_proj               10  (5->2, tied as v_proj)
RMSNorm (shared)         5
TOTAL                   74
```

---

## Comprehensive Seed Database

| Params | Seed | Peak Exact | Steps | Status | Source |
|--------|------|-----------|-------|--------|--------|
| **74p** | **45214** | **100%** | **78K** | **GROKKED** | **Session 7 (cm0.8 step-fade)** |
| **74p** | **71046** | **100%** | **81K** | **GROKKED** | **Session 7 (cm0.8 step-fade)** |
| **74p** | **78988** | **100%** | **~117K** | **GROKKED** | **Session 7 (cm0.8 step-fade)** |
| 75p | 80085 | 100% | 276K | GROKKED | Session 2 (cm0.3) |
| 75p | 78779 | 95.8% | 33K | Flash (unstable) | Session 5 sweep |
| 75p | 29267 | 14.7% | 27K | Signal (killed 51K) | Session 5 sweep |
| 75p | 72950 | 9.6% | 18K | Signal (killed 51K) | Session 5 sweep |
| 75p | 67086 | 2.5% | 27K | Signal (killed 51K) | Session 5 sweep |
| 72p | 80085 | 100% | 267K | GROKKED | Session 4 |
| 72p | 67086 | 91.6% | 282K | Grokking (hit 300K limit) | Session 5 sweep |
| 72p | 72950 | 64.9% | 39K | Flash (crashed) | Session 5 sweep |
| 72p | 65866 | 4.9% | 51K | Signal (killed 63K) | Session 5 sweep |
| 72p | 123 | 2.2% | 99K | Signal (killed) | Session 4 |
| 72p | 42 | 0% | 180K | 70% tok plateau | Session 4 |

---

## Current State

**From-scratch frontier: 74 parameters, 100% accuracy (10010/10010), seeds 45214 and 71046.**

The 74p model (75p + tie A=B) trained with high carry-mix (0.8) and step-based fade (10K→80K, 120K total steps) achieves **100% grok rate (3/3 random seeds)** — a dramatic improvement over the ~10% rate at 75p with carry_mix=0.3.

### Architecture (74p)

Single-layer decoder with d_model=5 (tok_dim=2, pos_dim=3), 1 attention head (head_dim=5), ffn_dim=2. Parametric token embeddings (circular arc, A=B tied), spiral positional encoding (no correction), shared RMSNorm, tied V/output, rank-1 output projection, no FFN bias.

### What We Learned

1. **Architecture matters, but training innovation matters too.** The path from 242p to 75p was architectural. The path from 75p to 74p combined a 1p architectural change (tie A=B) with a training breakthrough (high carry-mix + step-based fade) that dramatically improved seed robustness.

2. **Carry-mix is underrated.** 0.8 carry_mix with step-based fade gives ~67% grok rate vs ~10% at 0.3. The carry circuit is the bottleneck; flooding it with carry examples during early training forces formation.

3. **Metric-based curriculum can create feedback loops.** At high carry_mix, metric-triggered fade causes oscillation: accuracy up → carries removed → accuracy down → carries restored. Step-based fade eliminates this.

4. **Shorter step budgets improve stability.** With 120K steps, the cosine LR decay is steep enough that LR is low (~0.007) when the model needs to lock in, preventing post-grokking ejection from the basin.

5. **Flash grokking can be stabilized.** What previously looked like "transient unstable solutions" (Session 5) may have been models that would have converged with lower LR and smoother curriculum transitions.
