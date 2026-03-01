"""Training loop with integrated jiggle support.

Usage:
    uv run python -m src.train --run-name exp001_242p_s42 --seed 42 --steps 500000
    uv run python -m src.train --run-name exp002_226p_spiral --pos-mode spiral_correct --seed 42
    uv run python -m src.train --run-name exp003_jiggle --jiggle --jiggle-interval 50000
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import ModelConfig, MicroAdder, count_parameters, parameter_breakdown
from .data import (
    VOCAB_SIZE, PROMPT_LEN, ANSWER_LEN, SEQ_LEN,
    sample_batch, parse_curriculum, get_digit_range, make_example, decode_answer,
    EOS, MAX_DIGITS,
)


# ── LR schedule ────────────────────────────────────────────────────────────

def get_lr(step: int, args) -> float:
    """Cosine decay with linear warmup."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + (args.lr - min_lr) * cosine


def effective_weight_decay(base_wd: float, val_exact: float, tok_acc: float, args) -> float:
    """Adaptive weight decay: drop WD when grokking signals appear.

    Uses val_exact as the primary signal (rising val_exact = circuit found)
    with tok_acc as a secondary gate (must be above threshold too).

    Two-stage drop:
      - Stage 1 (onset): val_exact > wd_drop_exact AND tok_acc > wd_drop_tok_acc
        → multiply WD by wd_drop_factor (e.g. 0.1)
      - Stage 2 (locked): val_exact > wd_drop_exact_final
        → multiply WD by wd_drop_factor² (e.g. 0.01)
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


def smooth_weight_decay(base_wd: float, step: int, onset_step: int, alpha: float, floor: float) -> float:
    """Smooth exponential WD decay after grokking onset (ratcheted — never goes back up).

    wd = base_wd * exp(-alpha * (step - onset_step)), clamped to floor.
    """
    decay = math.exp(-alpha * (step - onset_step))
    return max(base_wd * decay, floor)


def check_wd_onset(val_exact: float, tok_acc: float, args) -> bool:
    """Check if grokking onset condition is met (shared by discrete and smooth WD)."""
    return val_exact >= args.wd_drop_exact and tok_acc >= args.wd_drop_tok_acc


def effective_carry_mix(base_mix: float, step: int, tok_acc: float, args) -> float:
    """Fade carry_mix to 0 based on token accuracy and step count.

    - Below fade threshold: full base_mix
    - Between fade and zero thresholds: linear ramp to 0
    - Above zero threshold OR past max_steps: 0
    """
    if base_mix <= 0:
        return 0.0
    if step >= args.carry_mix_max_steps:
        return 0.0
    if tok_acc >= args.carry_mix_tok_acc_zero:
        return 0.0
    if tok_acc <= args.carry_mix_tok_acc_fade:
        return base_mix
    # Linear fade
    span = args.carry_mix_tok_acc_zero - args.carry_mix_tok_acc_fade
    return base_mix * (args.carry_mix_tok_acc_zero - tok_acc) / span


# ── Autoregressive training loss ──────────────────────────────────────────

def ar_training_loss(
    model: MicroAdder,
    batch_input: torch.Tensor,
    batch_target: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss using autoregressive (own-prediction) feeding.

    Instead of teacher forcing (feeding ground truth at every position),
    we feed the model's own predictions for answer positions. This eliminates
    the train/inference distribution mismatch.

    Gradients flow through the cross-entropy loss at each step. The fed-back
    tokens are detached (argmax is non-differentiable), so each step's gradient
    is independent — but the *input distribution* each step sees reflects the
    model's actual behavior, not teacher-forced ground truth.

    Args:
        model: the MicroAdder model
        batch_input: (B, 33) token ids — full teacher-forced input sequence
        batch_target: (B, 33) target ids — with -100 on prompt positions
    """
    B, T = batch_input.shape
    device = batch_input.device
    prompt_len = PROMPT_LEN  # 22: X(10) + PLUS(1) + Y(10) + EQ(1)

    # Collect predicted tokens for AR feeding (detached from graph).
    # Start with a copy of the teacher-forced input; overwrite answer positions
    # as we go with our own predictions.
    ar_tokens = batch_input.clone()

    # Collect per-position losses
    total_loss = torch.tensor(0.0, device=device)
    n_tokens = 0

    # target[t] = ground truth for position t+1 in the full sequence.
    # target[t] is -100 for t < PROMPT_LEN - 1 (prompt positions).
    # First non-masked target is target[PROMPT_LEN - 1] = Z_0.
    #
    # At each step t, we run the model on ar_tokens[:, :t+1], read
    # logits[:, t] (prediction for token at position t+1), compute loss
    # against target[t], then set ar_tokens[:, t+1] = argmax(logits[:, t]).

    for t in range(prompt_len - 1, T):
        # Clone the prefix so in-place edits to ar_tokens don't invalidate
        # the autograd graph from this forward pass.
        step_input = ar_tokens[:, :t + 1].clone()
        logits, _ = model(step_input)

        tgt = batch_target[:, t]  # (B,)
        mask = tgt != -100
        if mask.any():
            pos_logits = logits[:, t]  # (B, vocab_size)
            loss_t = F.cross_entropy(pos_logits[mask], tgt[mask], reduction='sum')
            total_loss = total_loss + loss_t
            n_tokens += mask.sum().item()

        # Feed back our own prediction for the next position's input
        if t + 1 < T:
            with torch.no_grad():
                pred_tok = logits[:, t].argmax(dim=-1)  # (B,)
                ar_tokens[:, t + 1] = pred_tok

    if n_tokens > 0:
        total_loss = total_loss / n_tokens
    return total_loss


# ── Grokfast gradient filter ───────────────────────────────────────────

def gradfilter_ema(
    model: torch.nn.Module,
    grads: dict | None = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> dict:
    """Grokfast EMA gradient filter (Lee et al., 2024).

    Maintains an EMA of gradients and amplifies slow-varying components.
    Call after loss.backward(), before optimizer.step().
    """
    if grads is None:
        grads = {n: p.grad.data.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    else:
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
                p.grad.data += grads[n] * lamb
    return grads


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: MicroAdder,
    n_samples: int,
    seed: int,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    """Teacher-forced evaluation: exact match + token accuracy."""
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
    model: MicroAdder,
    n_samples: int,
    seed: int,
    device: torch.device,
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
        if predicted == a + b:
            correct += 1
        if (i + 1) % 500 == 0:
            print(f"  AR eval: {i+1}/{n_samples} done, {correct}/{i+1} correct ({correct/(i+1):.4f})")

    model.train()
    return {"exact_match": correct / n_samples, "n_samples": n_samples}


# ── Jiggle ─────────────────────────────────────────────────────────────────

def perturb_model(
    model: MicroAdder,
    sigma: float,
    fraction: float,
    scope: str,
    seed: int,
) -> None:
    """Apply deterministic Gaussian perturbation in-place.

    sigma is absolute scale (typically jiggle_strength * current_lr * rms(W)).
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    for name, param in model.named_parameters():
        if param.dim() == 0:
            continue

        # Scope filtering
        if scope == "ffn" and ".attn." in name:
            continue
        if scope == "attn" and ".ffn." in name:
            continue

        # Generate perturbation on CPU, then move
        delta = torch.randn(param.shape, generator=rng)

        # Masking
        if fraction < 1.0:
            mask = torch.rand(param.shape, generator=rng) < fraction
            delta = delta * mask

        # Scale relative to parameter RMS
        rms = param.data.float().pow(2).mean().sqrt().item()
        if rms > 1e-12:
            delta_rms = delta.float().pow(2).mean().sqrt().item()
            if delta_rms > 1e-12:
                delta = delta * (sigma * rms / delta_rms)

        param.data.add_(delta.to(device=param.device, dtype=param.dtype))


def _settle_and_track(
    model: MicroAdder,
    lr: float,
    settle_steps: int,
    eval_interval: int,
    args,
    curriculum,
    main_step: int,
    data_rng: random.Random,
    device: torch.device,
) -> list:
    """Train for settle_steps, eval periodically, return trajectory.

    All candidates in an event share the same data_rng seed so differences
    come purely from the perturbation, not batch randomness.
    """
    temp_optim = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay,
    )
    trajectory = []
    model.train()

    for s in range(settle_steps):
        min_d, max_d = get_digit_range(main_step, curriculum)
        batch_input, batch_target = sample_batch(
            args.batch_size, min_d, max_d, data_rng, device,
            carry_mix=args.carry_mix,
        )
        _, loss = model(batch_input, batch_target)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        temp_optim.step()
        temp_optim.zero_grad()

        # Periodic eval (always include the final step)
        if (s + 1) % eval_interval == 0 or s == settle_steps - 1:
            metrics = evaluate(model, args.eval_samples, seed=2025, device=device)
            trajectory.append({
                "settle_step": s + 1,
                "exact_match": metrics["exact_match"],
                "token_accuracy": metrics["token_accuracy"],
            })

    return trajectory


def _compute_score(trajectory: list) -> tuple:
    """Score a candidate from its trajectory: composite performance * stability.

    composite = exact_match + token_accuracy  (both in [0,1])
    stability = min(composite_during_settle) / final_composite  (in [0,1])
    score     = composite * (0.5 + 0.5 * stability)

    A stable high-performing candidate scores higher than an unstable one
    that briefly spiked.  When composite is near-zero everywhere, stability
    defaults to 1.0 so we fall back to raw composite ranking.
    """
    if not trajectory:
        return 0.0, {}

    composites = [t["exact_match"] + t["token_accuracy"] for t in trajectory]
    final = composites[-1]
    worst = min(composites)

    if final > 1e-10:
        stability = min(worst / final, 1.0)
    else:
        stability = 1.0

    score = final * (0.5 + 0.5 * stability)

    details = {
        "final_exact": trajectory[-1]["exact_match"],
        "final_tok_acc": trajectory[-1]["token_accuracy"],
        "composite": final,
        "worst_composite": worst,
        "stability": stability,
        "score": score,
    }
    return score, details


def run_jiggle_event(
    model: MicroAdder,
    optimizer: torch.optim.Optimizer,
    step: int,
    args,
    curriculum,
    device: torch.device,
    log_fn,
) -> dict:
    """Run a jiggle event: settle k perturbed candidates AND a control,
    then promote the best.

    Control = unperturbed continued training for settle_steps (same data,
    same fresh optimizer).  This is the fair baseline — jiggle only wins
    if perturbation genuinely finds a better basin than normal training.
    """
    current_lr = get_lr(step, args)
    sigma = args.jiggle_strength * current_lr
    settle_eval_every = max(1, args.jiggle_settle_steps // 5)

    # Save base state
    base_model_state = copy.deepcopy(model.state_dict())
    base_optim_state = copy.deepcopy(optimizer.state_dict())

    # Fixed data seed so every candidate (including control) sees the
    # exact same batch sequence — differences come only from weights.
    data_seed = args.seed + step * 7919

    log_fn(f"[jiggle] step={step} lr={current_lr:.2e} sigma={sigma:.2e} "
           f"candidates={args.jiggle_candidates}+control "
           f"settle={args.jiggle_settle_steps}")

    candidates = []

    # ── Control: normal continued training (no perturbation) ───────────
    model.load_state_dict(copy.deepcopy(base_model_state))
    trajectory = _settle_and_track(
        model, current_lr, args.jiggle_settle_steps, settle_eval_every,
        args, curriculum, step, random.Random(data_seed), device,
    )
    score, details = _compute_score(trajectory)
    candidates.append({
        "label": "control",
        "score": score,
        "details": details,
        "model_state": copy.deepcopy(model.state_dict()),
    })
    log_fn(f"[jiggle]   control:   score={score:.4f}  "
           f"exact={details['final_exact']:.6f}  "
           f"tok={details['final_tok_acc']:.4f}  "
           f"stab={details['stability']:.3f}")

    # ── Jiggle candidates ──────────────────────────────────────────────
    for cand_id in range(args.jiggle_candidates):
        model.load_state_dict(copy.deepcopy(base_model_state))
        perturb_model(
            model, sigma=sigma, fraction=args.jiggle_fraction,
            scope=args.jiggle_scope, seed=step * 1000 + cand_id,
        )
        trajectory = _settle_and_track(
            model, current_lr, args.jiggle_settle_steps, settle_eval_every,
            args, curriculum, step, random.Random(data_seed), device,
        )
        score, details = _compute_score(trajectory)
        candidates.append({
            "label": f"jiggle_{cand_id}",
            "score": score,
            "details": details,
            "model_state": copy.deepcopy(model.state_dict()),
        })
        log_fn(f"[jiggle]   jiggle_{cand_id}: score={score:.4f}  "
               f"exact={details['final_exact']:.6f}  "
               f"tok={details['final_tok_acc']:.4f}  "
               f"stab={details['stability']:.3f}")

    # ── Promote best ───────────────────────────────────────────────────
    best = max(candidates, key=lambda c: c["score"])
    model.load_state_dict(best["model_state"])
    optimizer.load_state_dict(base_optim_state)

    log_fn(f"[jiggle] promoted={best['label']}  score={best['score']:.4f}")

    return {
        "promoted": best["label"],
        "promoted_score": best["score"],
        "control_score": candidates[0]["score"],
        "all_candidates": [
            {"label": c["label"], **c["details"]} for c in candidates
        ],
    }


# ── Training loop ──────────────────────────────────────────────────────────

def train(model, optimizer, curriculum, args, run_dir, device):
    metrics_path = run_dir / "metrics.csv"
    ckpt_dir = run_dir / "checkpoints"

    # CSV logger
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "lr", "val_exact", "val_token_acc",
                         "wall_time", "min_digits", "max_digits"])

    def log(msg):
        print(msg, flush=True)

    def log_metrics(step, loss_val, lr, metrics, t0, min_d, max_d):
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, f"{loss_val:.6f}", f"{lr:.2e}",
                f"{metrics['exact_match']:.6f}", f"{metrics['token_accuracy']:.4f}",
                f"{time.time() - t0:.1f}", min_d, max_d,
            ])

    rng_data = random.Random(args.seed)
    best_exact = -1.0
    last_tok_acc = 0.0
    last_val_exact = 0.0
    wd_stage = 0  # 0=full, 1=first drop, 2=second drop
    wd_onset_step = None  # for --wd-smooth: step when grokking onset detected
    grokfast_grads = None  # for --grokfast: EMA state
    perfect_since_step = None  # for early stopping: step when val_exact first hit 1.0
    t0 = time.time()
    running_loss = 0.0
    log_interval = min(args.eval_interval, 1000)

    model.train()
    for step in range(args.steps):
        # ── Jiggle event ───────────────────────────────────────────────
        if (
            args.jiggle
            and step > 0
            and step % args.jiggle_interval == 0
        ):
            jiggle_result = run_jiggle_event(
                model, optimizer, step, args, curriculum, device, log,
            )
            # Log jiggle event to separate file
            jiggle_log = run_dir / "jiggle_events.csv"
            header_needed = not jiggle_log.exists()
            with open(jiggle_log, "a", newline="") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow([
                        "step", "promoted", "promoted_score",
                        "control_score", "candidates_json",
                    ])
                writer.writerow([
                    step,
                    jiggle_result["promoted"],
                    f"{jiggle_result['promoted_score']:.6f}",
                    f"{jiggle_result['control_score']:.6f}",
                    json.dumps(jiggle_result["all_candidates"]),
                ])

        # ── Curriculum digit range ─────────────────────────────────────
        min_d, max_d = get_digit_range(step, curriculum)

        # ── Forward / backward ─────────────────────────────────────────
        carry_mix = effective_carry_mix(args.carry_mix, step, last_tok_acc, args)
        batch_input, batch_target = sample_batch(
            args.batch_size, min_d, max_d, rng_data, device,
            carry_mix=carry_mix,
        )
        if args.ar_loss:
            loss = ar_training_loss(model, batch_input, batch_target)
        else:
            _, loss = model(batch_input, batch_target)
        loss.backward()

        # Grokfast: amplify slow-varying gradient components
        if args.grokfast:
            grokfast_grads = gradfilter_ema(
                model, grokfast_grads,
                alpha=args.grokfast_alpha, lamb=args.grokfast_lamb,
            )

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Update LR + weight decay
        lr = get_lr(step, args)
        if args.wd_smooth and wd_onset_step is not None:
            wd_floor = args.weight_decay * args.wd_drop_factor * args.wd_drop_factor
            wd = smooth_weight_decay(args.weight_decay, step, wd_onset_step,
                                     args.wd_smooth_alpha, wd_floor)
        else:
            wd = effective_weight_decay(args.weight_decay, last_val_exact, last_tok_acc, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = wd

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # ── Logging ────────────────────────────────────────────────────
        if step > 0 and step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t0
            log(f"step {step:>7d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"digits {min_d}-{max_d} | {elapsed:.0f}s")
            running_loss = 0.0

        # ── Evaluation + checkpointing ─────────────────────────────────
        if step > 0 and step % args.eval_interval == 0:
            metrics = evaluate(model, args.eval_samples, seed=2025, device=device)
            last_tok_acc = metrics["token_accuracy"]
            last_val_exact = metrics["exact_match"]
            elapsed = time.time() - t0
            carry_str = f" | carry_mix {carry_mix:.3f}" if args.carry_mix > 0 else ""
            wd_str = f" | wd {wd:.2e}" if (args.wd_adaptive or args.wd_smooth) else ""
            log(f"  EVAL step {step:>7d} | exact {metrics['exact_match']:.6f} | "
                f"tok_acc {metrics['token_accuracy']:.4f}{carry_str}{wd_str} | {elapsed:.0f}s")
            log_metrics(step, loss.item(), lr, metrics, t0, min_d, max_d)

            # Log WD transitions
            if args.wd_smooth:
                if wd_onset_step is None and check_wd_onset(last_val_exact, last_tok_acc, args):
                    wd_onset_step = step
                    log(f"  [wd-smooth] ONSET at step {step}: beginning exponential decay "
                        f"(alpha={args.wd_smooth_alpha}, val_exact={last_val_exact:.4f}, "
                        f"tok_acc={last_tok_acc:.4f})")
            elif args.wd_adaptive:
                new_wd = effective_weight_decay(args.weight_decay, last_val_exact, last_tok_acc, args)
                new_stage = 0
                if new_wd <= args.weight_decay * args.wd_drop_factor * args.wd_drop_factor * 1.01:
                    new_stage = 2
                elif new_wd <= args.weight_decay * args.wd_drop_factor * 1.01:
                    new_stage = 1
                if new_stage > wd_stage:
                    wd_stage = new_stage
                    log(f"  [wd-adaptive] STAGE {wd_stage}: wd {args.weight_decay:.2e} -> {new_wd:.2e} "
                        f"(val_exact={last_val_exact:.4f}, tok_acc={last_tok_acc:.4f})")

            # Save last
            _save_checkpoint(model, optimizer, step, metrics, args,
                             ckpt_dir / "last.pt")

            # Save best
            if metrics["exact_match"] > best_exact:
                best_exact = metrics["exact_match"]
                _save_checkpoint(model, optimizer, step, metrics, args,
                                 ckpt_dir / "best.pt")
                log(f"  NEW BEST: {best_exact:.6f}")

            # Early stopping: stop if val_exact == 1.0 for 10K steps
            if metrics["exact_match"] >= 1.0 - 1e-9:
                if perfect_since_step is None:
                    perfect_since_step = step
                    log(f"  [early-stop] val_exact=1.0 first seen at step {step}")
                elif step - perfect_since_step >= 10_000:
                    log(f"  [early-stop] val_exact=1.0 held for {step - perfect_since_step} steps. Stopping.")
                    break
            else:
                if perfect_since_step is not None:
                    log(f"  [early-stop] val_exact dropped below 1.0, resetting (was perfect since step {perfect_since_step})")
                perfect_since_step = None

    # Final eval
    final = evaluate(model, args.eval_samples, seed=2025, device=device)
    log(f"FINAL | exact {final['exact_match']:.6f} | tok_acc {final['token_accuracy']:.4f}")
    _save_checkpoint(model, optimizer, args.steps, final, args, ckpt_dir / "last.pt")


def _save_checkpoint(model, optimizer, step, metrics, args, path):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.cfg.to_dict(),
        "metrics": metrics,
        "args": vars(args),
    }, path)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MicroAdder training")

    # Run
    p.add_argument("--run-name", required=True, help="Name for results/runs/<name>/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Model
    p.add_argument("--d-model", type=int, default=6)
    p.add_argument("--tok-dim", type=int, default=3)
    p.add_argument("--pos-dim", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--ffn-dim", type=int, default=2)
    p.add_argument("--ffn-bias", action="store_true", default=True)
    p.add_argument("--no-ffn-bias", dest="ffn_bias", action="store_false")
    p.add_argument("--pos-mode", default="learned", choices=["learned", "spiral_correct", "zero"])
    p.add_argument("--pos-correction-mode", default="full", choices=["full", "linear"],
                   help="Position correction: 'full' (10 params) or 'linear' (2 params)")
    p.add_argument("--freeze-special", default="none", choices=["none", "eos", "plus_eos"],
                   help="Freeze special positions to zero: 'eos' saves 3p, 'plus_eos' saves 6p")
    p.add_argument("--alibi", action="store_true", default=False,
                   help="Add ALiBi attention bias with learned per-head slopes")
    p.add_argument("--qk-source", default="pos", choices=["pos", "tok"],
                   help="Q,K input: 'pos' (position subspace) or 'tok' (token subspace)")
    p.add_argument("--tie-qk", action="store_true", default=False)
    p.add_argument("--attn-out-rank", type=int, default=0)
    p.add_argument("--num-kv-heads", type=int, default=0,
                   help="Number of KV heads for GQA (0 = same as n_heads = standard MHA)")
    p.add_argument("--q-phase", action="store_true", default=False,
                   help="Add learnable per-head phase rotation to Q (for tied Q/K asymmetry)")
    p.add_argument("--share-layers", action="store_true", default=False)
    p.add_argument("--norm-mode", default="full", choices=["full", "shared", "fixed", "no_ln2"],
                   help="RMSNorm mode: full (18p), shared (6p), fixed (0p), no_ln2 (12p)")
    p.add_argument("--freeze-pad", action="store_true", default=False,
                   help="Freeze PAD token embedding to zero (saves tok_dim params)")
    p.add_argument("--softmax1", action="store_true", default=False,
                   help="Use softmax1 (add 1 to denominator, allows attention sum < 1)")
    p.add_argument("--token-init", default="spiral", choices=["spiral", "normal"])

    # Training
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--wd-adaptive", action="store_true", default=False,
                   help="Enable adaptive weight decay (drop WD when grokking detected)")
    p.add_argument("--wd-drop-exact", type=float, default=0.02,
                   help="Val exact match threshold for first WD drop (stage 1)")
    p.add_argument("--wd-drop-exact-final", type=float, default=0.20,
                   help="Val exact match threshold for second WD drop (stage 2)")
    p.add_argument("--wd-drop-tok-acc", type=float, default=0.70,
                   help="Token accuracy gate: WD drop only fires if tok_acc also above this")
    p.add_argument("--wd-drop-factor", type=float, default=0.1,
                   help="WD multiplier per stage (stage1=factor, stage2=factor²)")
    p.add_argument("--wd-smooth", action="store_true", default=False,
                   help="Smooth exponential WD decay after grokking onset (ratcheted)")
    p.add_argument("--wd-smooth-alpha", type=float, default=0.0001,
                   help="Exponential decay rate for smooth WD (reaches 0.1x at ~23K steps)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grokfast", action="store_true", default=False,
                   help="Enable Grokfast EMA gradient filter (amplify slow gradient components)")
    p.add_argument("--grokfast-alpha", type=float, default=0.98,
                   help="Grokfast EMA decay rate (higher = longer memory)")
    p.add_argument("--grokfast-lamb", type=float, default=2.0,
                   help="Grokfast amplification factor for slow gradient components")
    p.add_argument("--ar-loss", action="store_true", default=False,
                   help="Use autoregressive training loss (feed own predictions, not teacher forcing)")

    # Curriculum
    p.add_argument("--curriculum", default="1-3:2000,1-6:5000,1-10:rest")
    p.add_argument("--carry-mix", type=float, default=0.0,
                   help="Fraction of carry-focused examples (0-1)")
    p.add_argument("--carry-mix-tok-acc-fade", type=float, default=0.7,
                   help="Token accuracy where carry_mix starts fading to 0")
    p.add_argument("--carry-mix-tok-acc-zero", type=float, default=0.9,
                   help="Token accuracy where carry_mix reaches 0")
    p.add_argument("--carry-mix-max-steps", type=int, default=100_000,
                   help="Hard step cutoff: carry_mix=0 after this many steps")

    # Evaluation
    p.add_argument("--eval-interval", type=int, default=3000)
    p.add_argument("--eval-samples", type=int, default=5000)

    # Jiggle
    p.add_argument("--jiggle", action="store_true", default=False)
    p.add_argument("--jiggle-interval", type=int, default=50_000)
    p.add_argument("--jiggle-candidates", type=int, default=3)
    p.add_argument("--jiggle-settle-steps", type=int, default=10_000)
    p.add_argument("--jiggle-strength", type=float, default=1.0,
                   help="sigma = jiggle_strength * current_lr")
    p.add_argument("--jiggle-fraction", type=float, default=0.5)
    p.add_argument("--jiggle-scope", default="ffn", choices=["all", "attn", "ffn"])

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Model config
    cfg = ModelConfig(
        d_model=args.d_model,
        tok_dim=args.tok_dim,
        pos_dim=args.pos_dim,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        ffn_bias=args.ffn_bias,
        pos_mode=args.pos_mode,
        pos_correction_mode=args.pos_correction_mode,
        freeze_special=args.freeze_special,
        alibi=args.alibi,
        qk_source=args.qk_source,
        tie_qk=args.tie_qk,
        attn_out_rank=args.attn_out_rank,
        num_kv_heads=args.num_kv_heads,
        q_phase=args.q_phase,
        share_layers=args.share_layers,
        norm_mode=args.norm_mode,
        freeze_pad=args.freeze_pad,
        softmax1=args.softmax1,
        token_init=args.token_init,
    )

    model = MicroAdder(cfg).to(device)
    n_params = count_parameters(model)
    inner_dim = cfg.n_heads * cfg.head_dim
    print(f"Model: {n_params} parameters")
    if inner_dim != cfg.d_model:
        print(f"  ** Decoupled: d_model={cfg.d_model}, inner_dim={inner_dim} "
              f"(n_heads={cfg.n_heads} x head_dim={cfg.head_dim})")
    for name, count in parameter_breakdown(model).items():
        print(f"  {name}: {count}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {args.resume} (step {ckpt.get('step', '?')})")

    # Curriculum
    curriculum = parse_curriculum(args.curriculum)
    print(f"Curriculum: {curriculum}")
    if args.carry_mix > 0:
        print(f"Carry-focused mix: {args.carry_mix:.0%}")
    if args.ar_loss:
        print("Training loss: AUTOREGRESSIVE (feed own predictions)")
    if args.wd_smooth:
        print(f"Smooth WD: exponential decay (alpha={args.wd_smooth_alpha}) after "
              f"val_exact>{args.wd_drop_exact} & tok_acc>{args.wd_drop_tok_acc}, "
              f"floor={args.weight_decay * args.wd_drop_factor**2:.2e}")
    elif args.wd_adaptive:
        print(f"Adaptive WD: drop {args.wd_drop_factor}x at val_exact>{args.wd_drop_exact}, "
              f"{args.wd_drop_factor}²x at val_exact>{args.wd_drop_exact_final}, "
              f"tok_acc gate>{args.wd_drop_tok_acc}")
    if args.grokfast:
        print(f"Grokfast: EMA alpha={args.grokfast_alpha}, lambda={args.grokfast_lamb}")
    if args.jiggle:
        print(f"Jiggle: every {args.jiggle_interval} steps, "
              f"{args.jiggle_candidates} candidates, "
              f"{args.jiggle_settle_steps} settle steps, "
              f"strength={args.jiggle_strength}, "
              f"fraction={args.jiggle_fraction}, scope={args.jiggle_scope}")

    # Run directory (auto-append seed)
    args.run_name = f"{args.run_name}_s{args.seed}"
    run_dir = Path("results/runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    # Train
    train(model, optimizer, curriculum, args, run_dir, device)


if __name__ == "__main__":
    main()
