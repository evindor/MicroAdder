"""Training script for the MicroAdder.

Usage:
    # 67p (default): sinusoidal positions, qk_dim=4
    uv run python -m microadder.train --run-name sub100_my_run --seed 71046

    # 74p: learned spiral, qk_dim=5
    uv run python -m microadder.train --run-name sub100_my_run --seed 45214 \
        --qk-dim 0 --freeze-spiral "" --wd-adaptive --steps 120000 \
        --carry-mix-fade-start 10000 --carry-mix-fade-end 80000
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import MicroAdder, ModelConfig, count_params, parameter_breakdown
from .data import (
    PROMPT_LEN, ANSWER_LEN, SEQ_LEN,
    sample_batch, parse_curriculum, get_digit_range,
    make_example, decode_answer,
)


# ── LR schedule ──────────────────────────────────────────────────────────

def get_lr(step: int, args) -> float:
    """Cosine decay with linear warmup."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + (args.lr - min_lr) * cosine


# ── Adaptive weight decay ────────────────────────────────────────────────

def effective_weight_decay(
    base_wd: float, val_exact: float, tok_acc: float, args
) -> float:
    """Adaptive weight decay with two-stage drop triggered by grokking signals.

    Stage 1 (onset):  val_exact > wd_drop_exact AND tok_acc > wd_drop_tok_acc
                      -> WD *= wd_drop_factor (e.g. 0.1)
    Stage 2 (locked): val_exact > wd_drop_exact_final
                      -> WD *= wd_drop_factor^2 (e.g. 0.01)
    """
    if not args.wd_adaptive:
        return base_wd

    # Stage 2: deep drop once val_exact is solidly rising
    if val_exact >= args.wd_drop_exact_final and tok_acc >= args.wd_drop_tok_acc:
        return base_wd * args.wd_drop_factor * args.wd_drop_factor

    # Stage 1: first drop when grokking onset detected
    if val_exact >= args.wd_drop_exact and tok_acc >= args.wd_drop_tok_acc:
        return base_wd * args.wd_drop_factor

    return base_wd


# ── Step-based carry-mix fade ────────────────────────────────────────────

def effective_carry_mix(base_mix: float, step: int, args) -> float:
    """Linear fade of carry_mix from base_mix to 0 over [fade_start, fade_end].

    No metric dependency -- avoids the oscillation feedback loop that metric-based
    fading causes at high carry_mix values.
    """
    if base_mix <= 0:
        return 0.0
    if args.carry_mix_fade_start < 0:
        # No fade configured, use constant carry_mix
        return base_mix
    if step <= args.carry_mix_fade_start:
        return base_mix
    if step >= args.carry_mix_fade_end:
        return 0.0
    frac = (step - args.carry_mix_fade_start) / (
        args.carry_mix_fade_end - args.carry_mix_fade_start
    )
    return base_mix * (1.0 - frac)


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: MicroAdder, n_samples: int, seed: int, device: torch.device,
    batch_size: int = 512,
) -> dict:
    """Teacher-forced evaluation: exact match and token accuracy."""
    model.eval()
    rng = random.Random(seed)
    total = 0
    exact_correct = 0
    token_correct = 0
    token_total = 0

    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        inputs, targets = [], []
        for _ in range(bs):
            a = rng.randint(0, 10**10 - 1)
            b = rng.randint(0, 10**10 - 1)
            inp, tgt = make_example(a, b)
            inputs.append(inp)
            targets.append(tgt)

        input_t = torch.tensor(inputs, dtype=torch.long, device=device)
        target_t = torch.tensor(targets, dtype=torch.long, device=device)

        logits, _ = model(input_t)
        preds = logits.argmax(dim=-1)

        mask = target_t != -100
        matches = (preds == target_t) & mask
        example_correct = matches.sum(dim=1) == mask.sum(dim=1)
        exact_correct += example_correct.sum().item()
        token_correct += matches.sum().item()
        token_total += mask.sum().item()
        total += bs

    model.train()
    return {
        "exact_match": exact_correct / total,
        "token_accuracy": token_correct / token_total if token_total > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_autoregressive(
    model: MicroAdder, n_samples: int, seed: int, device: torch.device,
) -> dict:
    """Autoregressive evaluation (accurate but slower)."""
    model.eval()
    rng = random.Random(seed)
    correct = 0

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
        if (i + 1) % 500 == 0:
            print(f"  AR eval: {i+1}/{n_samples}, {correct}/{i+1} "
                  f"({correct/(i+1):.4f})")

    model.train()
    return {"exact_match": correct / n_samples, "n_samples": n_samples}


# ── Checkpoint ───────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, step, metrics, args, path):
    """Save model checkpoint."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.cfg.to_dict(),
        "metrics": metrics,
        "args": vars(args),
    }, path)


# ── Training loop ────────────────────────────────────────────────────────

def train(model, optimizer, curriculum, args, run_dir, device):
    """Main training loop."""
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "log.jsonl"

    def log(msg):
        print(msg, flush=True)

    def log_metrics(step, loss_val, lr, metrics, t0, min_d, max_d, wd, carry_mix):
        entry = {
            "step": step,
            "loss": round(loss_val, 6),
            "lr": lr,
            "val_exact": round(metrics["exact_match"], 6),
            "val_tok_acc": round(metrics["token_accuracy"], 4),
            "wall_time": round(time.time() - t0, 1),
            "min_digits": min_d,
            "max_digits": max_d,
            "wd": wd,
            "carry_mix": round(carry_mix, 4),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    rng_data = random.Random(args.seed)
    best_exact = -1.0
    last_tok_acc = 0.0
    last_val_exact = 0.0
    perfect_since_step = None
    t0 = time.time()
    running_loss = 0.0
    log_interval = min(args.eval_interval, 1000)

    model.train()
    for step in range(args.steps):
        # ── Curriculum digit range ───────────────────────────────────
        min_d, max_d = get_digit_range(step, curriculum)

        # ── Carry mix (step-based fade) ──────────────────────────────
        carry_mix = effective_carry_mix(args.carry_mix, step, args)

        # ── Sample batch ─────────────────────────────────────────────
        batch_input, batch_target = sample_batch(
            args.batch_size, min_d, max_d, rng_data, device,
            carry_mix=carry_mix,
        )

        # ── LR + weight decay update ────────────────────────────────
        lr = get_lr(step, args)
        wd = effective_weight_decay(
            args.weight_decay, last_val_exact, last_tok_acc, args
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = wd

        # ── Forward / backward / step ────────────────────────────────
        _, loss = model(batch_input, batch_target)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # ── Console logging ──────────────────────────────────────────
        if step > 0 and step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t0
            log(f"step {step:>7d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"wd {wd:.2e} | cm {carry_mix:.3f} | "
                f"digits {min_d}-{max_d} | {elapsed:.0f}s")
            running_loss = 0.0

        # ── Evaluation + checkpointing ───────────────────────────────
        if step > 0 and step % args.eval_interval == 0:
            metrics = evaluate(model, args.eval_samples, seed=2025, device=device)
            last_tok_acc = metrics["token_accuracy"]
            last_val_exact = metrics["exact_match"]
            elapsed = time.time() - t0

            log(f"  EVAL step {step:>7d} | exact {metrics['exact_match']:.6f} | "
                f"tok_acc {metrics['token_accuracy']:.4f} | "
                f"wd {wd:.2e} | cm {carry_mix:.3f} | {elapsed:.0f}s")

            log_metrics(step, loss.item(), lr, metrics, t0, min_d, max_d,
                        wd, carry_mix)

            # Save last
            _save_checkpoint(model, optimizer, step, metrics, args,
                             ckpt_dir / "last.pt")

            # Save best
            if metrics["exact_match"] > best_exact:
                best_exact = metrics["exact_match"]
                _save_checkpoint(model, optimizer, step, metrics, args,
                                 ckpt_dir / "best.pt")
                log(f"  NEW BEST: {best_exact:.6f}")

            # Early stopping: val_exact=1.0 sustained for 10K steps
            if metrics["exact_match"] >= 1.0 - 1e-9:
                if perfect_since_step is None:
                    perfect_since_step = step
                    log(f"  [early-stop] val_exact=1.0 first seen at step {step}")
                elif step - perfect_since_step >= 10_000:
                    log(f"  [early-stop] val_exact=1.0 held for "
                        f"{step - perfect_since_step} steps. Stopping.")
                    break
            else:
                if perfect_since_step is not None:
                    log(f"  [early-stop] val_exact dropped, resetting "
                        f"(was perfect since step {perfect_since_step})")
                perfect_since_step = None

    # Final eval
    final = evaluate(model, args.eval_samples, seed=2025, device=device)
    log(f"FINAL | exact {final['exact_match']:.6f} | "
        f"tok_acc {final['token_accuracy']:.4f}")
    _save_checkpoint(model, optimizer, args.steps, final, args,
                     ckpt_dir / "last.pt")


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MicroAdder training")

    # Run
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture
    p.add_argument("--d-model", type=int, default=5)
    p.add_argument("--tok-dim", type=int, default=2)
    p.add_argument("--pos-dim", type=int, default=3)
    p.add_argument("--head-dim", type=int, default=0, help="0=d_model")
    p.add_argument("--qk-dim", type=int, default=4,
                   help="Q/K projection dim (0=head_dim, 4=67p, 5=74p)")
    p.add_argument("--qk-input", default="pos", choices=["pos", "full"],
                   help="Q/K input: 'pos' (position subspace) or 'full' (full d_model)")
    p.add_argument("--norm-mode", default="weighted",
                   choices=["weighted", "parameterless", "structured", "spiral"],
                   help="Norm: 'weighted' (5p), 'structured' (3p), 'spiral' (0p), or 'parameterless' (0p)")
    p.add_argument("--tie-fc2-head", action="store_true", default=False,
                   help="Tie fc2 to head_proj.T (triple-duty, saves 10p)")
    p.add_argument("--freeze-tok-arc", default="",
                   help="Tok arc params to freeze, e.g. 'A,start' (saves 2p)")
    p.add_argument("--tok-arc-init-A", type=float, default=2.5)
    p.add_argument("--tok-arc-init-start", type=float, default=-1.2)
    p.add_argument("--tok-arc-init-stride", type=float, default=0.29)
    p.add_argument("--freeze-spiral", default="amp,phase,slope,offset",
                   help="Spiral params to freeze (empty=learn all, 'amp,phase,slope,offset'=sinusoidal)")
    p.add_argument("--equals-spiral-idx", type=float, default=-1.0,
                   help="Freeze EQUALS pos as spiral(idx). -1=learned (3p), 9.5=frozen sinusoidal (0p)")
    p.add_argument("--spiral-init-amp", type=float, default=3.5)
    p.add_argument("--spiral-init-phase", type=float, default=0.0)
    p.add_argument("--spiral-init-slope", type=float, default=0.15)
    p.add_argument("--spiral-init-offset", type=float, default=0.0)

    # Training
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Adaptive weight decay
    p.add_argument("--wd-adaptive", action="store_true", default=False,
                   help="Enable adaptive WD (drop when grokking detected)")
    p.add_argument("--wd-drop-exact", type=float, default=0.02,
                   help="Val exact threshold for first WD drop")
    p.add_argument("--wd-drop-exact-final", type=float, default=0.2,
                   help="Val exact threshold for second WD drop")
    p.add_argument("--wd-drop-tok-acc", type=float, default=0.7,
                   help="Token accuracy gate for WD drops")
    p.add_argument("--wd-drop-factor", type=float, default=0.1,
                   help="WD multiplier per stage")

    # Carry-mix with step-based fade
    p.add_argument("--carry-mix", type=float, default=0.8,
                   help="Fraction of carry-focused examples")
    p.add_argument("--carry-mix-fade-start", type=int, default=15_000,
                   help="Step where carry_mix starts fading (-1 = no fade)")
    p.add_argument("--carry-mix-fade-end", type=int, default=45_000,
                   help="Step where carry_mix reaches 0")

    # Curriculum
    p.add_argument("--curriculum", default="1-3:2000,1-6:5000,1-10:rest")

    # Evaluation
    p.add_argument("--eval-interval", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=5000)

    return p.parse_args()


def main():
    args = parse_args()

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Model
    cfg = ModelConfig(
        d_model=args.d_model,
        tok_dim=args.tok_dim,
        pos_dim=args.pos_dim,
        head_dim=args.head_dim if args.head_dim > 0 else args.d_model,
        qk_dim=args.qk_dim,
        qk_input=args.qk_input,
        norm_mode=args.norm_mode,
        tie_fc2_head=args.tie_fc2_head,
        freeze_tok_arc=args.freeze_tok_arc,
        tok_arc_init_A=args.tok_arc_init_A,
        tok_arc_init_start=args.tok_arc_init_start,
        tok_arc_init_stride=args.tok_arc_init_stride,
        equals_spiral_idx=args.equals_spiral_idx,
        freeze_spiral=args.freeze_spiral,
        spiral_init_amp=args.spiral_init_amp,
        spiral_init_phase=args.spiral_init_phase,
        spiral_init_slope=args.spiral_init_slope,
        spiral_init_offset=args.spiral_init_offset,
    )
    model = MicroAdder(cfg).to(device)
    n_params = count_params(model)
    print(f"Model: {n_params} parameters")
    for name, count in parameter_breakdown(model).items():
        print(f"  {name}: {count}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Curriculum
    curriculum = parse_curriculum(args.curriculum)
    print(f"Curriculum: {curriculum}")
    print(f"Carry-mix: {args.carry_mix:.0%}, fade [{args.carry_mix_fade_start}, "
          f"{args.carry_mix_fade_end}]")
    print(f"Adaptive WD: {args.wd_adaptive}")

    # Run directory
    run_dir = Path("results/runs") / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    # Train
    train(model, optimizer, curriculum, args, run_dir, device)


if __name__ == "__main__":
    main()
