"""Standalone evaluation: load checkpoint and test on random examples.

Usage:
    uv run python -m microadder.eval --checkpoint results/runs/my_run/checkpoints/best.pt
    uv run python -m microadder.eval --checkpoint best.pt --n-samples 10010
"""

import argparse
import random
import time

import torch

from .model import MicroAdder, ModelConfig, count_params
from .data import PROMPT_LEN, ANSWER_LEN, make_example, decode_answer


# ── Edge cases ───────────────────────────────────────────────────────────

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
def run_edge_cases(model, device):
    """Test model on known edge cases using autoregressive generation."""
    passed = 0
    failures = []
    for a, b in EDGE_CASES:
        inp, _ = make_example(a, b)
        prompt = torch.tensor([inp[:PROMPT_LEN]], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=ANSWER_LEN + 1)
        predicted = decode_answer(generated[0].tolist())
        expected = a + b
        if predicted == expected:
            passed += 1
        else:
            failures.append((a, b, expected, predicted))
    return passed, len(EDGE_CASES), failures


@torch.no_grad()
def evaluate_autoregressive(model, n_samples, seed, device, show_errors=0):
    """Full autoregressive evaluation on random examples."""
    model.eval()
    rng = random.Random(seed)
    correct = 0
    errors = []

    for i in range(n_samples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, _ = make_example(a, b)
        prompt = torch.tensor([inp[:PROMPT_LEN]], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=ANSWER_LEN + 1)
        predicted = decode_answer(generated[0].tolist())
        expected = a + b
        if predicted == expected:
            correct += 1
        elif len(errors) < show_errors:
            errors.append((a, b, expected, predicted))
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_samples}, "
                  f"{correct}/{i+1} correct ({correct/(i+1):.4f})")

    return correct, n_samples, errors


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MicroAdder evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n-samples", type=int, default=10010)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show-errors", type=int, default=10,
                   help="Number of error examples to print")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    raw_cfg = ckpt["config"]
    # Handle both old (flat) and new (nested) config formats
    if "model" in raw_cfg and isinstance(raw_cfg["model"], dict):
        raw_cfg = raw_cfg["model"]
    cfg = ModelConfig.from_dict(raw_cfg)
    model = MicroAdder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = count_params(model)
    print(f"Model: {n_params} parameters")
    print(f"Checkpoint step: {ckpt.get('step', '?')}")
    print(f"Checkpoint metrics: {ckpt.get('metrics', {})}")
    print()

    # Edge cases
    print("Edge cases (autoregressive):")
    passed, total, failures = run_edge_cases(model, device)
    print(f"  {passed}/{total} passed")
    for a, b, expected, got in failures:
        print(f"  FAIL: {a} + {b} = {expected}, got {got}")
    print()

    # Random evaluation
    print(f"Autoregressive evaluation on {args.n_samples} samples "
          f"(seed={args.seed})...")
    t0 = time.time()
    correct, n, errors = evaluate_autoregressive(
        model, args.n_samples, args.seed, device,
        show_errors=args.show_errors,
    )
    elapsed = time.time() - t0

    exact_match = correct / n
    print(f"\nResults ({elapsed:.1f}s):")
    print(f"  Exact match: {exact_match:.6f} ({correct}/{n})")

    if errors:
        print(f"\nFirst {len(errors)} errors:")
        for a, b, expected, got in errors:
            print(f"  {a} + {b} = {expected}, got {got}")

    qualified = exact_match >= 0.99
    print(f"\nQualified (>=99%): {'YES' if qualified else 'NO'}")


if __name__ == "__main__":
    main()
