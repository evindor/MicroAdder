"""Standalone evaluation: load checkpoint and test on held-out samples.

Usage:
    uv run python -m src.eval --checkpoint results/runs/exp001/checkpoints/best.pt
    uv run python -m src.eval --checkpoint results/runs/exp001/checkpoints/best.pt --autoregressive
    uv run python -m src.eval --checkpoint results/runs/exp001/checkpoints/best.pt --seed 2025 --n-samples 10010
"""

import argparse
import random
import sys
import time

import torch

from .model import ModelConfig, MicroAdder, count_parameters
from .data import (
    PROMPT_LEN, ANSWER_LEN, make_example, decode_answer, encode_number,
    PLUS, EQUALS, EOS, MAX_DIGITS,
)
from .train import evaluate, evaluate_autoregressive


def main():
    p = argparse.ArgumentParser(description="MicroAdder evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--seed", type=int, default=2025, help="Eval RNG seed")
    p.add_argument("--n-samples", type=int, default=10010)
    p.add_argument("--autoregressive", action="store_true", default=False,
                   help="Use autoregressive generation (slower, accurate)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show-errors", type=int, default=10,
                   help="Print this many error examples (0 to disable)")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = MicroAdder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = count_parameters(model)
    print(f"Model: {n_params} parameters")
    print(f"Checkpoint step: {ckpt.get('step', '?')}")
    print(f"Checkpoint metrics: {ckpt.get('metrics', {})}")
    print()

    # ── Edge cases ──────────────────────────────────────────────────────
    if args.autoregressive:
        edge_pass, edge_total, edge_failures = _run_edge_cases(model, device, vocab_size=cfg.vocab_size)
        print(f"Edge cases: {edge_pass}/{edge_total}")
        for a, b, expected, got in edge_failures:
            print(f"  FAIL: {a} + {b} = {expected}, got {got}")
        print()

    # ── Random eval ───────────────────────────────────────────────────
    t0 = time.time()
    if args.autoregressive:
        print(f"Autoregressive evaluation on {args.n_samples} samples (seed={args.seed})...")
        metrics = evaluate_autoregressive(model, args.n_samples, args.seed, device,
                                          vocab_size=cfg.vocab_size)
    else:
        print(f"Teacher-forced evaluation on {args.n_samples} samples (seed={args.seed})...")
        metrics = evaluate(model, args.n_samples, args.seed, device,
                          vocab_size=cfg.vocab_size)

    elapsed = time.time() - t0
    print(f"Results ({elapsed:.1f}s):")
    print(f"  Exact match:    {metrics['exact_match']:.6f} "
          f"({int(metrics['exact_match'] * args.n_samples)}/{args.n_samples})")
    if "token_accuracy" in metrics:
        print(f"  Token accuracy: {metrics['token_accuracy']:.4f}")

    # Show errors (autoregressive only, skip if perfect)
    n_wrong = args.n_samples - int(metrics['exact_match'] * args.n_samples)
    if args.show_errors > 0 and args.autoregressive and n_wrong > 0:
        print(f"\nFirst {min(args.show_errors, n_wrong)} errors:")
        _show_errors(model, args.n_samples, args.seed, device, args.show_errors,
                    vocab_size=cfg.vocab_size)

    # Qualification check
    qualified = metrics["exact_match"] >= 0.99
    print(f"\nQualified (>=99%): {'YES' if qualified else 'NO'}")


EDGE_CASES = [
    (0, 0),
    (0, 1),
    (9_999_999_999, 0),
    (9_999_999_999, 1),
    (9_999_999_999, 9_999_999_999),
    (5_000_000_000, 5_000_000_000),
    (1_111_111_111, 8_888_888_889),
    (1_234_567_890, 9_876_543_210),
    (1, 9_999_999_999),
]


@torch.no_grad()
def _run_edge_cases(model, device, vocab_size=14):
    passed = 0
    failures = []
    for a, b in EDGE_CASES:
        inp, _ = make_example(a, b, vocab_size=vocab_size)
        prompt = torch.tensor([inp[:PROMPT_LEN]], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=ANSWER_LEN + 1)
        predicted = decode_answer(generated[0].tolist(), vocab_size=vocab_size)
        expected = a + b
        if predicted == expected:
            passed += 1
        else:
            failures.append((a, b, expected, predicted))
    return passed, len(EDGE_CASES), failures


@torch.no_grad()
def _show_errors(model, n_samples, seed, device, max_errors, vocab_size=14):
    rng = random.Random(seed)
    shown = 0
    for _ in range(n_samples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, _ = make_example(a, b, vocab_size=vocab_size)
        prompt = torch.tensor([inp[:PROMPT_LEN]], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=ANSWER_LEN + 1)
        predicted = decode_answer(generated[0].tolist(), vocab_size=vocab_size)
        expected = a + b
        if predicted != expected:
            print(f"  {a} + {b} = {expected}, got {predicted}")
            shown += 1
            if shown >= max_errors:
                break


if __name__ == "__main__":
    main()
