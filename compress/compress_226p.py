"""Post-training compression of the 226p MicroAdder checkpoint.

Compression targets identified from weight analysis:
  1. out_proj rank-2 SVD:        36 -> 24  saves 12  (97.2% energy)
  2. pos_correction -> linear:   10 ->  2  saves  8  (1.0% rel err on positions)
  3. special_pos PLUS -> zero:    3 ->  0  saves  3  (norm 0.09)
  4. special_pos EOS -> zero:     3 ->  0  saves  3  (already exactly zero)
  5. z_hi_pos xy -> zero:         3 ->  1  saves  2  (xy: -0.01, 0.08 vs z: -5.26)

Best case: 226 - 28 = 198p

Usage:
    cd microadder
    uv run python compress/compress_226p.py                         # analyze all, eval full combo
    uv run python compress/compress_226p.py --eval-each             # eval each compression individually
    uv run python compress/compress_226p.py --skip out_proj_rank2   # skip risky out_proj compression
    uv run python compress/compress_226p.py --save compressed.pt    # save compressed checkpoint
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# We import the submission model directly so we can load, modify, and eval
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submission_226p.submission_226p import (
    MicroAdder,
    build_model,
    add,
    VOCAB_SIZE,
    D_MODEL,
    TOK_DIM,
    POS_DIM,
    N_HEADS,
    HEAD_DIM,
    FFN_DIM,
    MAX_SEQ_LEN,
    MAX_DIGITS,
    ANSWER_LEN,
    PLUS_TOKEN,
    EQUALS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
)


# ── Evaluation ────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_model(model: MicroAdder, n_samples: int = 10_000, seed: int = 2025) -> float:
    """Autoregressive exact-match evaluation. Returns fraction correct."""
    device = next(model.parameters()).device
    model.eval()
    rng = random.Random(seed)
    correct = 0
    for _ in range(n_samples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        pred = add(model, a, b)
        if pred == a + b:
            correct += 1
    return correct / n_samples


def count_params(model: nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


# ── Compression functions ─────────────────────────────────────────────────
# Each takes a state_dict (or model), modifies it in place, and returns
# the parameter savings.


def compress_eos_special_pos(sd: dict) -> int:
    """Fix EOS special position to zero. Already ~0 in checkpoint."""
    sp = sd["special_pos"].clone()
    sp[2] = 0.0
    sd["special_pos"] = sp
    return 3  # saves 3 params


def compress_plus_special_pos(sd: dict) -> int:
    """Fix PLUS special position to zero. Norm is only 0.09."""
    sp = sd["special_pos"].clone()
    sp[0] = 0.0
    sd["special_pos"] = sp
    return 3  # saves 3 params


def compress_z_hi_pos(sd: dict) -> int:
    """Fix z_hi_pos x,y dims to zero, keep only z. x=-0.01, y=0.08 vs z=-5.26."""
    zhp = sd["z_hi_pos"].clone()
    zhp[0, 0] = 0.0
    zhp[0, 1] = 0.0
    sd["z_hi_pos"] = zhp
    return 2  # saves 2 params


def compress_pos_correction_linear(sd: dict) -> int:
    """Replace 10 pos_correction values with best-fit linear (2 params)."""
    pc = sd["pos_correction"].float().cpu().numpy()
    idx = np.arange(10, dtype=np.float64)
    coeffs = np.polyfit(idx, pc.astype(np.float64), 1)  # [slope, intercept]
    pc_linear = np.polyval(coeffs, idx).astype(np.float32)
    sd["pos_correction"] = torch.tensor(pc_linear, device=sd["pos_correction"].device)
    return 8  # saves 10 - 2 = 8 params


def compress_out_proj_rank2(sd: dict) -> int:
    """Replace out_proj (6x6=36) with rank-2 SVD approximation (6x2+2x6=24).

    The rank-2 approximation captures 97.2% of the energy (singular values).
    We store it as the full 6x6 materialized matrix — the param count reduction
    is structural (the effective rank is 2, meaning 24 free parameters).

    For a proper submission this would need a LowRankLinear module, but for
    evaluation we just materialize the rank-2 approximation as a dense matrix.
    """
    key = "out_proj.weight"
    W = sd[key].float().cpu().numpy()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = 2
    W_approx = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    sd[key] = torch.tensor(W_approx, device=sd[key].device)
    return 12  # saves 36 - 24 = 12 params


# ── Compression registry ──────────────────────────────────────────────────

COMPRESSIONS = {
    "eos_zero": ("EOS special_pos → 0", compress_eos_special_pos, 3),
    "plus_zero": ("PLUS special_pos → 0", compress_plus_special_pos, 3),
    "z_hi_xy_zero": ("z_hi_pos xy → 0", compress_z_hi_pos, 2),
    "pos_correction_linear": (
        "pos_correction → linear",
        compress_pos_correction_linear,
        8,
    ),
    "out_proj_rank2": ("out_proj rank-2 SVD", compress_out_proj_rank2, 12),
}


# ── Main ──────────────────────────────────────────────────────────────────


def load_base_model():
    """Load the 226p model and return (model, original_state_dict)."""
    model, meta = build_model()
    orig_sd = copy.deepcopy(model.state_dict())
    return model, orig_sd


def apply_compressions(model: MicroAdder, orig_sd: dict, names: list[str]) -> int:
    """Apply a set of compressions and load into model. Returns total savings."""
    sd = copy.deepcopy(orig_sd)
    total_saved = 0
    for name in names:
        desc, fn, expected_savings = COMPRESSIONS[name]
        saved = fn(sd)
        total_saved += saved
    model.load_state_dict(sd)
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="Compress 226p MicroAdder checkpoint")
    parser.add_argument(
        "--eval-each",
        action="store_true",
        help="Evaluate each compression individually",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Compression names to skip"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Only apply these compressions"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10_000, help="Number of eval samples"
    )
    parser.add_argument("--seed", type=int, default=2025, help="Eval RNG seed")
    parser.add_argument(
        "--save", type=str, default=None, help="Save compressed checkpoint to this path"
    )
    args = parser.parse_args()

    print("Loading 226p model...")
    model, orig_sd = load_base_model()
    print(f"Base params: {count_params(model)}")

    # Baseline accuracy
    print(f"\nEvaluating baseline ({args.n_samples} samples, seed={args.seed})...")
    t0 = time.time()
    base_acc = eval_model(model, args.n_samples, args.seed)
    print(
        f"  Baseline accuracy: {base_acc:.6f} ({base_acc * args.n_samples:.0f}/{args.n_samples}) [{time.time() - t0:.1f}s]"
    )

    # Determine which compressions to apply
    if args.only is not None:
        active = [n for n in args.only if n in COMPRESSIONS]
    else:
        active = [n for n in COMPRESSIONS if n not in args.skip]

    # ── Individual evaluation ─────────────────────────────────────────
    if args.eval_each:
        print("\n" + "=" * 60)
        print("INDIVIDUAL COMPRESSIONS")
        print("=" * 60)
        for name in active:
            desc, fn, expected_savings = COMPRESSIONS[name]
            sd = copy.deepcopy(orig_sd)
            saved = fn(sd)
            model.load_state_dict(sd)
            t0 = time.time()
            acc = eval_model(model, args.n_samples, args.seed)
            elapsed = time.time() - t0
            delta = acc - base_acc
            status = "OK" if acc >= 0.99 else "FAIL"
            print(
                f"  [{status}] {desc:35s}  saves {saved:2d}p  "
                f"acc={acc:.6f} (Δ={delta:+.6f})  [{elapsed:.1f}s]"
            )

    # ── Combined compression ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMBINED COMPRESSION")
    print("=" * 60)
    print(f"Applying: {', '.join(active)}")
    total_saved = apply_compressions(model, orig_sd, active)
    effective_params = 226 - total_saved
    print(f"Savings: {total_saved}p → {effective_params}p effective")
    print(
        f"Actual nn.Module params: {count_params(model)} (dense matrices, same shapes)"
    )

    t0 = time.time()
    acc = eval_model(model, args.n_samples, args.seed)
    elapsed = time.time() - t0
    print(
        f"Accuracy: {acc:.6f} ({acc * args.n_samples:.0f}/{args.n_samples}) [{elapsed:.1f}s]"
    )
    print(f"Qualified (≥99%): {'YES' if acc >= 0.99 else 'NO'}")

    # ── Incremental: find the best safe subset ────────────────────────
    # Try removing one compression at a time if combined fails
    if acc < 0.99 and len(active) > 1:
        print("\n" + "=" * 60)
        print("ABLATION: removing one compression at a time")
        print("=" * 60)
        for drop in active:
            subset = [n for n in active if n != drop]
            saved = apply_compressions(model, orig_sd, subset)
            sub_acc = eval_model(model, args.n_samples, args.seed)
            status = "OK" if sub_acc >= 0.99 else "FAIL"
            eff = 226 - saved
            desc = COMPRESSIONS[drop][0]
            print(f"  [{status}] Without {desc:35s}  {eff}p  acc={sub_acc:.6f}")

    # ── Save ──────────────────────────────────────────────────────────
    if args.save and acc >= 0.99:
        # Save the compressed state dict along with metadata
        save_data = {
            "model_state_dict": model.state_dict(),
            "compressions_applied": active,
            "effective_params": effective_params,
            "accuracy": acc,
            "eval_samples": args.n_samples,
            "eval_seed": args.seed,
            "base_params": 226,
            "savings": total_saved,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save(save_data, args.save)
        print(f"\nSaved compressed checkpoint to {args.save}")
    elif args.save and acc < 0.99:
        print(f"\nNOT saving — accuracy {acc:.4f} below 99% threshold")


if __name__ == "__main__":
    main()
