"""Post-training compression of BD 230p checkpoints.

BD already has rank-2 out_proj natively (trained with attn_out_rank=2).
Remaining compression targets:
  1. EOS special_pos → 0:       saves 3p  (already exactly zero)
  2. PLUS special_pos → 0:      saves 3p  (norm ~0.51, risky)
  3. pos_correction → linear:   saves 8p  (7.5% rel error, worse than 226p)

Best safe: 230 - 3 - 8 = 219p.  Aggressive: 230 - 14 = 216p.

Usage:
    cd microadder
    uv run python compress/compress_bd.py --all-seeds --n-samples 1000
    uv run python compress/compress_bd.py --checkpoint results/runs/expBD_.../checkpoints/best.pt --eval-each
"""

import argparse
import copy
import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ModelConfig, MicroAdder, count_parameters
from src.data import (
    MAX_DIGITS,
    ANSWER_LEN,
    PROMPT_LEN,
    PLUS,
    EQUALS,
    EOS,
    encode_number,
    decode_answer,
    make_example,
)


# ── Evaluation ────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_model(
    model: MicroAdder,
    n_samples: int = 1000,
    seed: int = 2025,
    device: torch.device = None,
) -> float:
    """Autoregressive exact-match evaluation."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rng = random.Random(seed)
    correct = 0
    for _ in range(n_samples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, _ = make_example(a, b)
        prompt = torch.tensor([inp[:PROMPT_LEN]], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=ANSWER_LEN + 1)
        pred = decode_answer(generated[0].tolist())
        if pred == a + b:
            correct += 1
    return correct / n_samples


def load_bd_checkpoint(path: str, device: torch.device = None):
    """Load a BD training checkpoint. Returns (model, state_dict_copy, config)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = MicroAdder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    orig_sd = copy.deepcopy(model.state_dict())
    return model, orig_sd, cfg


# ── Compression functions ─────────────────────────────────────────────────


def compress_eos_special_pos(sd: dict) -> int:
    """Fix EOS special position to zero."""
    sp = sd["special_pos"].clone()
    sp[2] = 0.0
    sd["special_pos"] = sp
    return 3


def compress_plus_special_pos(sd: dict) -> int:
    """Fix PLUS special position to zero."""
    sp = sd["special_pos"].clone()
    sp[0] = 0.0
    sd["special_pos"] = sp
    return 3


def compress_pos_correction_linear(sd: dict) -> int:
    """Replace 10 pos_correction values with best-fit linear (2 params)."""
    pc = sd["pos_correction"].float().cpu().numpy()
    idx = np.arange(10, dtype=np.float64)
    coeffs = np.polyfit(idx, pc.astype(np.float64), 1)
    pc_linear = np.polyval(coeffs, idx).astype(np.float32)
    sd["pos_correction"] = torch.tensor(pc_linear, device=sd["pos_correction"].device)
    return 8


COMPRESSIONS = {
    "eos_zero": ("EOS special_pos → 0", compress_eos_special_pos, 3),
    "plus_zero": ("PLUS special_pos → 0", compress_plus_special_pos, 3),
    "pos_correction_linear": (
        "pos_correction → linear",
        compress_pos_correction_linear,
        8,
    ),
}


def apply_compressions(model, orig_sd, names):
    """Apply compressions to a copy of state_dict, load into model. Returns savings."""
    sd = copy.deepcopy(orig_sd)
    total = 0
    for name in names:
        _, fn, _ = COMPRESSIONS[name]
        total += fn(sd)
    model.load_state_dict(sd)
    return total


def analyze_one(path, args):
    """Run compression analysis on a single checkpoint."""
    seed_name = path.split("/")[-3] if "checkpoints" in path else os.path.basename(path)
    print(f"\n{'=' * 65}")
    print(f"  {seed_name}")
    print(f"{'=' * 65}")

    device = torch.device(args.device)
    model, orig_sd, cfg = load_bd_checkpoint(path, device)
    base_params = count_parameters(model)
    print(f"Base params: {base_params}")

    # Quick weight analysis
    sp = orig_sd["special_pos"].float().cpu().numpy()
    pc = orig_sd["pos_correction"].float().cpu().numpy()
    idx = np.arange(10, dtype=np.float64)
    coeffs = np.polyfit(idx, pc.astype(np.float64), 1)
    pc_fit = np.polyval(coeffs, idx).astype(np.float32)
    pc_err = np.linalg.norm(pc - pc_fit) / np.linalg.norm(pc)
    print(
        f"  PLUS norm: {np.linalg.norm(sp[0]):.4f}  EOS norm: {np.linalg.norm(sp[2]):.6f}  "
        f"pos_corr linear err: {pc_err:.4f}"
    )

    # Baseline
    t0 = time.time()
    base_acc = eval_model(model, args.n_samples, args.eval_seed, device)
    print(
        f"  Baseline: {base_acc:.4f} ({base_acc * args.n_samples:.0f}/{args.n_samples}) [{time.time() - t0:.1f}s]"
    )

    if base_acc < 0.99:
        print(f"  SKIP — baseline below 99%")
        return None

    # Individual compressions
    if args.eval_each:
        for name in COMPRESSIONS:
            desc, fn, expected = COMPRESSIONS[name]
            sd = copy.deepcopy(orig_sd)
            saved = fn(sd)
            model.load_state_dict(sd)
            t0 = time.time()
            acc = eval_model(model, args.n_samples, args.eval_seed, device)
            delta = acc - base_acc
            status = "OK" if acc >= 0.99 else "FAIL"
            print(
                f"  [{status}] {desc:30s}  saves {saved:2d}p  "
                f"acc={acc:.4f} (Δ={delta:+.4f})  [{time.time() - t0:.1f}s]"
            )

    # Combined: all compressions
    total_saved = apply_compressions(model, orig_sd, list(COMPRESSIONS.keys()))
    eff = base_params - total_saved
    t0 = time.time()
    acc_all = eval_model(model, args.n_samples, args.eval_seed, device)
    print(
        f"  ALL combined: {eff}p  acc={acc_all:.4f} [{time.time() - t0:.1f}s]  "
        f"{'QUALIFIED' if acc_all >= 0.99 else 'FAILED'}"
    )

    # Safe combo (no PLUS zeroing)
    safe = ["eos_zero", "pos_correction_linear"]
    safe_saved = apply_compressions(model, orig_sd, safe)
    eff_safe = base_params - safe_saved
    t0 = time.time()
    acc_safe = eval_model(model, args.n_samples, args.eval_seed, device)
    print(
        f"  SAFE (eos+linpos): {eff_safe}p  acc={acc_safe:.4f} [{time.time() - t0:.1f}s]  "
        f"{'QUALIFIED' if acc_safe >= 0.99 else 'FAILED'}"
    )

    return {
        "seed": seed_name,
        "base_params": base_params,
        "all_eff": eff,
        "all_acc": acc_all,
        "safe_eff": eff_safe,
        "safe_acc": acc_safe,
    }


def main():
    parser = argparse.ArgumentParser(description="Compress BD 230p checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Single checkpoint path"
    )
    parser.add_argument(
        "--all-seeds", action="store_true", help="Process all qualified BD seeds"
    )
    parser.add_argument(
        "--eval-each",
        action="store_true",
        help="Evaluate each compression individually",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.checkpoint:
        paths = [args.checkpoint]
    elif args.all_seeds:
        paths = sorted(
            glob.glob("results/runs/expBD_230p_atnrankoutspiral_s*/checkpoints/best.pt")
        )
    else:
        parser.error("Specify --checkpoint or --all-seeds")

    print(f"Processing {len(paths)} checkpoint(s), {args.n_samples} samples each")

    results = []
    for p in paths:
        r = analyze_one(p, args)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n{'=' * 65}")
        print("SUMMARY")
        print(f"{'=' * 65}")
        for r in results:
            print(
                f"  {r['seed']:55s}  "
                f"all={r['all_eff']}p/{r['all_acc']:.4f}  "
                f"safe={r['safe_eff']}p/{r['safe_acc']:.4f}"
            )


if __name__ == "__main__":
    main()
