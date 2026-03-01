#!/usr/bin/env python3
"""Seed sensitivity sweep: 75p vs 72p from scratch, 10 random seeds."""

import csv
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
MAX_PARALLEL = 3
MAX_STEPS = 300_000  # enough for grokking; 500K is overkill for a sweep
KILL_STEP = 50_000
KILL_VAL_EXACT = 0.20
KILL_TOK_ACC = 0.80
EVAL_INTERVAL = 3000
POLL_INTERVAL = 30  # seconds between checks
NUM_SEEDS = 10

RESULTS_DIR = Path("results/runs")
REPORT_FILE = Path("seed_sensitivity_report.md")

# Generate 10 random seeds (fixed RNG for reproducibility of the sweep itself)
rng = random.Random(2026)
SEEDS = sorted(rng.sample(range(1, 100_000), NUM_SEEDS))

# ── Base configs ────────────────────────────────────────────────────
BASE_ARGS = [
    "uv", "run", "python", "-m", "src.train",
    "--d-model", "5", "--tok-dim", "2", "--pos-dim", "3",
    "--n-heads", "1", "--head-dim", "5", "--ffn-dim", "2", "--no-ffn-bias",
    "--tie-qk", "--q-phase", "--attn-out-rank", "1",
    "--vocab-size", "10", "--tok-emb-mode", "parametric",
    "--pos-mode", "spiral_correct", "--pos-correction-mode", "none",
    "--freeze-special", "plus_eos",
    "--norm-mode", "shared", "--tie-vo",
    "--lr", "0.02", "--carry-mix", "0.3",
    "--wd-adaptive", "--wd-drop-exact", "0.01", "--wd-drop-exact-final", "0.05",
    "--steps", str(MAX_STEPS),
]

CONFIGS = {
    "75p": [],  # base config is 75p
    "72p": ["--freeze-z-hi"],  # 72p = 75p + freeze z_hi
}


@dataclass
class Job:
    config_name: str
    seed: int
    run_name: str
    process: subprocess.Popen | None = None
    metrics_path: Path = Path("")
    status: str = "queued"  # queued, running, killed, finished, grokked
    final_step: int = 0
    final_val_exact: float = 0.0
    final_tok_acc: float = 0.0
    grok_step: int | None = None  # step where val_exact first > 0.9
    peak_val_exact: float = 0.0
    peak_tok_acc: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0


def make_job(config_name: str, seed: int) -> Job:
    run_name = f"sub100_sweep_{config_name}_s{seed}"
    return Job(
        config_name=config_name,
        seed=seed,
        run_name=run_name,
        metrics_path=RESULTS_DIR / f"{run_name}_s{seed}" / "metrics.csv",
    )


def launch_job(job: Job) -> None:
    args = BASE_ARGS + CONFIGS[job.config_name] + [
        "--run-name", job.run_name,
        "--seed", str(job.seed),
    ]
    log_path = RESULTS_DIR / f"{job.run_name}_s{job.seed}" / "stdout.log"
    os.makedirs(log_path.parent, exist_ok=True)
    log_file = open(log_path, "w")
    job.process = subprocess.Popen(
        args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    job.status = "running"
    job.start_time = time.time()
    print(f"  LAUNCHED {job.run_name} (pid={job.process.pid})")


def read_metrics(job: Job) -> tuple[int, float, float]:
    """Read latest metrics from CSV. Returns (step, val_exact, val_tok_acc)."""
    if not job.metrics_path.exists():
        return 0, 0.0, 0.0
    try:
        with open(job.metrics_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return 0, 0.0, 0.0
        last = rows[-1]
        step = int(last["step"])
        val_exact = float(last["val_exact"])
        tok_acc = float(last["val_token_acc"])
        # Track peak and grok step
        for row in rows:
            ve = float(row["val_exact"])
            ta = float(row["val_token_acc"])
            if ve > job.peak_val_exact:
                job.peak_val_exact = ve
            if ta > job.peak_tok_acc:
                job.peak_tok_acc = ta
            if ve > 0.9 and job.grok_step is None:
                job.grok_step = int(row["step"])
        return step, val_exact, tok_acc
    except Exception:
        return 0, 0.0, 0.0


def kill_job(job: Job, reason: str) -> None:
    if job.process and job.process.poll() is None:
        try:
            os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    job.status = "killed"
    job.end_time = time.time()
    print(f"  KILLED {job.run_name} — {reason}")


def check_job(job: Job) -> bool:
    """Check job status. Returns True if job is still running."""
    if job.process is None:
        return False
    ret = job.process.poll()
    if ret is not None:
        # Process ended
        step, ve, ta = read_metrics(job)
        job.final_step = step
        job.final_val_exact = ve
        job.final_tok_acc = ta
        if job.peak_val_exact >= 0.999:
            job.status = "grokked"
        else:
            job.status = "finished"
        job.end_time = time.time()
        print(f"  DONE {job.run_name}: step={step}, exact={ve:.4f}, tok={ta:.4f} → {job.status}")
        return False

    step, ve, ta = read_metrics(job)
    job.final_step = step
    job.final_val_exact = ve
    job.final_tok_acc = ta

    # Kill criterion: past 50K and below thresholds
    if step >= KILL_STEP and ve < KILL_VAL_EXACT and ta < KILL_TOK_ACC:
        kill_job(job, f"step={step}, exact={ve:.4f}<{KILL_VAL_EXACT}, tok={ta:.4f}<{KILL_TOK_ACC}")
        return False

    return True


def generate_report(jobs: list[Job]) -> str:
    lines = []
    lines.append("# Seed Sensitivity Study: 75p vs 72p From Scratch")
    lines.append("")
    lines.append(f"**Seeds tested:** {SEEDS}")
    lines.append(f"**Kill criterion:** val_exact < {KILL_VAL_EXACT} AND val_tok_acc < {KILL_TOK_ACC} at step {KILL_STEP}")
    lines.append(f"**Max steps:** {MAX_STEPS}")
    lines.append("")

    for config_name in ["75p", "72p"]:
        config_jobs = [j for j in jobs if j.config_name == config_name]
        lines.append(f"## {config_name} Results")
        lines.append("")
        lines.append("| Seed | Status | Final Step | Peak Exact | Peak Tok | Grok Step | Wall Time |")
        lines.append("|------|--------|-----------|-----------|---------|-----------|-----------|")

        grokked = 0
        for j in sorted(config_jobs, key=lambda x: x.seed):
            wall = f"{(j.end_time - j.start_time)/60:.0f}m" if j.end_time else "—"
            grok = str(j.grok_step) if j.grok_step else "—"
            status_icon = {"grokked": "✓ GROKKED", "killed": "✗ killed", "finished": "— finished", "running": "⋯ running", "queued": "queued"}
            lines.append(
                f"| {j.seed} | {status_icon.get(j.status, j.status)} | {j.final_step} | "
                f"{j.peak_val_exact:.4f} | {j.peak_tok_acc:.4f} | {grok} | {wall} |"
            )
            if j.status == "grokked":
                grokked += 1
        lines.append("")
        lines.append(f"**Grok rate: {grokked}/{len(config_jobs)} seeds ({100*grokked/max(len(config_jobs),1):.0f}%)**")
        lines.append("")

    # Comparative analysis
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Seed | 75p Status | 75p Grok Step | 72p Status | 72p Grok Step |")
    lines.append("|------|-----------|--------------|-----------|--------------|")
    for seed in SEEDS:
        j75 = next((j for j in jobs if j.config_name == "75p" and j.seed == seed), None)
        j72 = next((j for j in jobs if j.config_name == "72p" and j.seed == seed), None)
        s75 = j75.status if j75 else "—"
        g75 = str(j75.grok_step) if j75 and j75.grok_step else "—"
        s72 = j72.status if j72 else "—"
        g72 = str(j72.grok_step) if j72 and j72.grok_step else "—"
        lines.append(f"| {seed} | {s75} | {g75} | {s72} | {g72} |")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"=== Seed Sensitivity Sweep: 75p vs 72p ===")
    print(f"Seeds: {SEEDS}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print()

    # Build job queue — interleave 75p and 72p for same seed to run them "together"
    queue: list[Job] = []
    for seed in SEEDS:
        queue.append(make_job("75p", seed))
        queue.append(make_job("72p", seed))

    running: list[Job] = []
    completed: list[Job] = []
    qi = 0  # queue index

    while qi < len(queue) or running:
        # Launch jobs up to MAX_PARALLEL
        while qi < len(queue) and len(running) < MAX_PARALLEL:
            job = queue[qi]
            qi += 1
            launch_job(job)
            running.append(job)

        # Wait before checking
        time.sleep(POLL_INTERVAL)

        # Check all running jobs
        still_running = []
        for job in running:
            if check_job(job):
                still_running.append(job)
            else:
                completed.append(job)
        running = still_running

        # Status summary
        n_grokked = sum(1 for j in completed if j.status == "grokked")
        n_killed = sum(1 for j in completed if j.status == "killed")
        n_finished = sum(1 for j in completed if j.status == "finished")
        run_info = ", ".join(f"{j.run_name}(s{j.final_step})" for j in running)
        print(f"[{time.strftime('%H:%M:%S')}] running={len(running)} done={len(completed)} "
              f"(grokked={n_grokked}, killed={n_killed}, finished={n_finished}) "
              f"queue={len(queue)-qi} | {run_info}")

    # Final report
    all_jobs = completed
    report = generate_report(all_jobs)

    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\n=== SWEEP COMPLETE ===")
    print(f"Report written to {REPORT_FILE}")
    print(report)


if __name__ == "__main__":
    main()
