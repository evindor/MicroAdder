"""Post-training compression of 203p BDcombo checkpoints.

The 203p model already has:
  - Rank-2 out_proj (native)
  - EOS special_pos frozen to zero (native, buffer _eos_pos)
  - Linear pos_correction (2 params: slope + intercept)
  - Spiral positional encoding (4 params: amp, offset, phase, slope)

Remaining compression targets:
  1. pos_corr_slope → 0:        saves 1p  (s69420: 0.001, s1995: 0.036)
  2. spiral_slope → 0:          saves 1p  (s69420: 0.020, s1995: 0.065)
  3. spiral_offset → 0:         saves 1p  (s69420: 0.074, s1995: 0.504)
  4. EOS tok_emb → 0:           saves 3p  (s69420: norm 0.005, s1995: norm 0.53)
  5. PAD tok_emb → 0:           saves 3p  (s69420: norm 0.71, s1995: norm 0.56)
  6. PLUS special_pos → 0:      saves 3p  (s69420: norm 2.09, s1995: norm 0.51)
  7. v_proj → rank 2:           saves 6p  (18→12, risky)
  8. out_proj → rank 1:         saves 6p  (24→12, risky)

Usage:
    cd microadder
    uv run python compress/compress_203p.py --checkpoint results/runs/expBDcombo_203p_s69420/checkpoints/best.pt --eval-each
    uv run python compress/compress_203p.py --all-seeds --n-samples 1000
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ModelConfig, MicroAdder, count_parameters
from src.data import (
    ANSWER_LEN,
    PROMPT_LEN,
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


def load_checkpoint(path: str, device: torch.device = None):
    """Load a training checkpoint. Returns (model, state_dict_copy, config)."""
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
# Each returns the number of effective params saved.


def compress_pos_corr_slope_zero(sd: dict) -> int:
    """Fix pos_corr_slope to zero (constant correction)."""
    sd["pos_corr_slope"] = torch.tensor(0.0, device=sd["pos_corr_slope"].device)
    return 1


def compress_spiral_slope_zero(sd: dict) -> int:
    """Fix spiral_slope to zero."""
    sd["spiral_slope"] = torch.tensor(0.0, device=sd["spiral_slope"].device)
    return 1


def compress_spiral_offset_zero(sd: dict) -> int:
    """Fix spiral_offset to zero."""
    sd["spiral_offset"] = torch.tensor(0.0, device=sd["spiral_offset"].device)
    return 1


def compress_eos_tok_emb_zero(sd: dict) -> int:
    """Fix EOS token embedding to zero."""
    te = sd["tok_emb.weight"].clone()
    te[12] = 0.0  # EOS = token 12
    sd["tok_emb.weight"] = te
    return 3


def compress_pad_tok_emb_zero(sd: dict) -> int:
    """Fix PAD token embedding to zero."""
    te = sd["tok_emb.weight"].clone()
    te[13] = 0.0  # PAD = token 13
    sd["tok_emb.weight"] = te
    return 3


def compress_plus_tok_emb_zero(sd: dict) -> int:
    """Fix PLUS token embedding to zero."""
    te = sd["tok_emb.weight"].clone()
    te[10] = 0.0  # PLUS = token 10
    sd["tok_emb.weight"] = te
    return 3


def compress_plus_special_pos_zero(sd: dict) -> int:
    """Fix PLUS special position to zero (row 0 of special_pos_learned)."""
    sp = sd["special_pos_learned"].clone()
    sp[0] = 0.0
    sd["special_pos_learned"] = sp
    return 3


def compress_v_proj_rank2(sd: dict) -> int:
    """Compress v_proj from rank-3 (18p) to rank-2 (12p) via SVD."""
    w = sd["blocks.0.attn.v_proj.weight"].float().cpu().numpy()
    U, S, Vt = np.linalg.svd(w, full_matrices=False)
    w_r2 = (U[:, :2] * S[:2]) @ Vt[:2, :]
    sd["blocks.0.attn.v_proj.weight"] = torch.tensor(
        w_r2,
        dtype=sd["blocks.0.attn.v_proj.weight"].dtype,
        device=sd["blocks.0.attn.v_proj.weight"].device,
    )
    return 6  # 18 -> 12


def compress_out_proj_rank1(sd: dict) -> int:
    """Compress out_proj from rank-2 to effective rank-1 (keep shape, zero second component)."""
    A = sd["blocks.0.attn.out_proj.A"].float().cpu().numpy()
    B = sd["blocks.0.attn.out_proj.B"].float().cpu().numpy()
    AB = A @ B
    U, S, Vt = np.linalg.svd(AB, full_matrices=False)
    # Keep shape [6,2] and [2,6] but make rank-1 effective
    A_new = np.zeros_like(A)
    B_new = np.zeros_like(B)
    A_new[:, 0] = U[:, 0] * np.sqrt(S[0])
    B_new[0, :] = np.sqrt(S[0]) * Vt[0, :]
    sd["blocks.0.attn.out_proj.A"] = torch.tensor(
        A_new,
        dtype=sd["blocks.0.attn.out_proj.A"].dtype,
        device=sd["blocks.0.attn.out_proj.A"].device,
    )
    sd["blocks.0.attn.out_proj.B"] = torch.tensor(
        B_new,
        dtype=sd["blocks.0.attn.out_proj.B"].dtype,
        device=sd["blocks.0.attn.out_proj.B"].device,
    )
    return 12  # 24 -> 12 effective (shape preserved but rank-1)


def compress_head_proj_rank2(sd: dict) -> int:
    """Compress head_proj from rank-3 (18p) to rank-2 (12p) via SVD."""
    w = sd["head_proj.weight"].float().cpu().numpy()
    U, S, Vt = np.linalg.svd(w, full_matrices=False)
    w_r2 = (U[:, :2] * S[:2]) @ Vt[:2, :]
    sd["head_proj.weight"] = torch.tensor(
        w_r2, dtype=sd["head_proj.weight"].dtype, device=sd["head_proj.weight"].device
    )
    return 6  # 18 -> 12


# Registry: name -> (description, function, expected_savings)
COMPRESSIONS = {
    "pos_corr_slope_zero": ("pos_corr_slope → 0", compress_pos_corr_slope_zero, 1),
    "spiral_slope_zero": ("spiral_slope → 0", compress_spiral_slope_zero, 1),
    "spiral_offset_zero": ("spiral_offset → 0", compress_spiral_offset_zero, 1),
    "eos_tok_emb_zero": ("EOS tok_emb → 0", compress_eos_tok_emb_zero, 3),
    "pad_tok_emb_zero": ("PAD tok_emb → 0", compress_pad_tok_emb_zero, 3),
    "plus_tok_emb_zero": ("PLUS tok_emb → 0", compress_plus_tok_emb_zero, 3),
    "plus_special_zero": ("PLUS special_pos → 0", compress_plus_special_pos_zero, 3),
    "v_proj_rank2": ("v_proj → rank 2", compress_v_proj_rank2, 6),
    "out_proj_rank1": ("out_proj → rank 1", compress_out_proj_rank1, 12),
    "head_proj_rank2": ("head_proj → rank 2", compress_head_proj_rank2, 6),
}


def apply_compressions(model, orig_sd, names):
    """Apply compressions to a copy of state_dict, load into model. Returns savings."""
    sd = copy.deepcopy(orig_sd)
    total = 0
    for name in names:
        _, fn, _ = COMPRESSIONS[name]
        total += fn(sd)
    model.load_state_dict(sd, strict=False)
    return total


def analyze_one(path, args):
    """Run compression analysis on a single checkpoint."""
    seed_name = path.split("/")[-3] if "checkpoints" in path else os.path.basename(path)
    print(f"\n{'=' * 65}")
    print(f"  {seed_name}")
    print(f"{'=' * 65}")

    device = torch.device(args.device)
    model, orig_sd, cfg = load_checkpoint(path, device)
    base_params = count_parameters(model)
    print(f"Base params: {base_params}")
    print(
        f"Config: freeze_special={cfg.freeze_special}, pos_correction_mode={cfg.pos_correction_mode}"
    )

    # Quick weight analysis
    print(f"\n  --- Weight Norms ---")
    for key in [
        "pos_corr_slope",
        "pos_corr_intercept",
        "spiral_amp",
        "spiral_offset",
        "spiral_phase",
        "spiral_slope",
    ]:
        if key in orig_sd:
            print(f"  {key:25s} = {orig_sd[key].item():.6f}")

    if "special_pos_learned" in orig_sd:
        sp = orig_sd["special_pos_learned"].float().cpu().numpy()
        for i, name in enumerate(["PLUS", "EQUALS"]):
            print(f"  special_pos {name:8s} norm={np.linalg.norm(sp[i]):.4f}")

    te = orig_sd["tok_emb.weight"].float().cpu().numpy()
    token_names = [f"d{i}" for i in range(10)] + ["PAD", "PLUS", "EQ", "EOS"]
    for i in [10, 11, 13]:  # PAD, PLUS, EOS
        print(f"  tok_emb {token_names[i]:4s} norm={np.linalg.norm(te[i]):.6f}")

    # SVD analysis
    for proj_name in ["v_proj", "q_proj", "k_proj"]:
        w = orig_sd[f"blocks.0.attn.{proj_name}.weight"].float().cpu().numpy()
        _, S, _ = np.linalg.svd(w, full_matrices=False)
        energy = S**2 / (S**2).sum() * 100
        print(
            f"  {proj_name} SVD energy: [{energy[0]:.1f}%, {energy[1]:.1f}%, {energy[2]:.1f}%]"
        )

    hp = orig_sd["head_proj.weight"].float().cpu().numpy()
    _, S, _ = np.linalg.svd(hp, full_matrices=False)
    energy = S**2 / (S**2).sum() * 100
    print(
        f"  head_proj SVD energy: [{energy[0]:.1f}%, {energy[1]:.1f}%, {energy[2]:.1f}%]"
    )

    if "blocks.0.attn.out_proj.A" in orig_sd:
        A = orig_sd["blocks.0.attn.out_proj.A"].float().cpu().numpy()
        B = orig_sd["blocks.0.attn.out_proj.B"].float().cpu().numpy()
        AB = A @ B
        _, S, _ = np.linalg.svd(AB, full_matrices=False)
        S_nz = S[S > 1e-6]
        energy = S_nz**2 / (S_nz**2).sum() * 100
        print(f"  out_proj (A@B) SVD energy: [{energy[0]:.1f}%, {energy[1]:.1f}%]")

    # Baseline eval
    t0 = time.time()
    base_acc = eval_model(model, args.n_samples, args.eval_seed, device)
    print(
        f"\n  Baseline: {base_acc:.4f} ({base_acc * args.n_samples:.0f}/{args.n_samples}) [{time.time() - t0:.1f}s]"
    )

    if base_acc < 0.99:
        print(f"  SKIP — baseline below 99%")
        return None

    results = {"seed": seed_name, "base_params": base_params, "base_acc": base_acc}

    # Individual compressions
    if args.eval_each:
        print(f"\n  --- Individual Compressions ---")
        for name in COMPRESSIONS:
            desc, fn, expected = COMPRESSIONS[name]
            sd = copy.deepcopy(orig_sd)
            try:
                saved = fn(sd)
                model.load_state_dict(sd, strict=False)
                t0 = time.time()
                acc = eval_model(model, args.n_samples, args.eval_seed, device)
                delta = acc - base_acc
                status = "OK" if acc >= 0.99 else "FAIL"
                print(
                    f"  [{status}] {desc:30s}  saves {saved:2d}p → {base_params - saved}p  "
                    f"acc={acc:.4f} (Δ={delta:+.4f})  [{time.time() - t0:.1f}s]"
                )
                results[name] = acc
            except Exception as e:
                print(f"  [ERR] {desc:30s}  {e}")
                results[name] = -1

    # Safe combo: things that are near-zero
    safe = [
        "pos_corr_slope_zero",
        "spiral_slope_zero",
        "spiral_offset_zero",
        "eos_tok_emb_zero",
    ]
    safe_saved = apply_compressions(model, orig_sd, safe)
    t0 = time.time()
    acc_safe = eval_model(model, args.n_samples, args.eval_seed, device)
    eff_safe = base_params - safe_saved
    print(
        f"\n  SAFE (slope+spiral_slope+spiral_offset+eos_tok): "
        f"{eff_safe}p  acc={acc_safe:.4f} [{time.time() - t0:.1f}s]  "
        f"{'QUALIFIED' if acc_safe >= 0.99 else 'FAILED'}"
    )
    results["safe_eff"] = eff_safe
    results["safe_acc"] = acc_safe

    # Aggressive combo: safe + pad + plus_special
    agg = safe + ["pad_tok_emb_zero", "plus_special_zero"]
    agg_saved = apply_compressions(model, orig_sd, agg)
    t0 = time.time()
    acc_agg = eval_model(model, args.n_samples, args.eval_seed, device)
    eff_agg = base_params - agg_saved
    print(
        f"  AGGRESSIVE (safe+pad+plus_sp): "
        f"{eff_agg}p  acc={acc_agg:.4f} [{time.time() - t0:.1f}s]  "
        f"{'QUALIFIED' if acc_agg >= 0.99 else 'FAILED'}"
    )
    results["agg_eff"] = eff_agg
    results["agg_acc"] = acc_agg

    return results


def main():
    parser = argparse.ArgumentParser(description="Compress 203p BDcombo checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Single checkpoint path"
    )
    parser.add_argument(
        "--all-seeds", action="store_true", help="Process all BDcombo seeds"
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
        paths = sorted(glob.glob("results/runs/expBDcombo_203p_s*/checkpoints/best.pt"))
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
                f"  {r['seed']:40s}  safe={r['safe_eff']}p/{r['safe_acc']:.4f}  "
                f"agg={r['agg_eff']}p/{r['agg_acc']:.4f}"
            )


if __name__ == "__main__":
    main()
