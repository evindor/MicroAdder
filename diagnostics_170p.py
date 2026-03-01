"""Structural diagnostics on the 170p checkpoint (Exp 6)."""

import math
import random

import numpy as np
import torch

from src.model import ModelConfig, MicroAdder
from src.data import make_example


def load_checkpoint():
    ckpt = torch.load("submission_170p/checkpoint_170p.pt", map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    return state


def diag1_token_geometry(state):
    print("=" * 70)
    print("1. TOKEN EMBEDDING GEOMETRY (14 tokens in 2D)")
    print("=" * 70)

    tok_emb = state["tok_emb.weight"].numpy()  # [14, 2]
    names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "PLUS", "EQUALS", "EOS", "PAD"]

    print(f"{'Token':>8s} | {'dim0':>8s} {'dim1':>8s} | {'norm':>6s} | {'angle':>7s}")
    print("-" * 50)
    for i in range(14):
        x, y = tok_emb[i]
        norm = np.sqrt(x**2 + y**2)
        angle = np.degrees(np.arctan2(y, x))
        print(f"{names[i]:>8s} | {x:8.4f} {y:8.4f} | {norm:6.3f} | {angle:7.1f}°")

    print("\nAngular separation between consecutive digits:")
    for i in range(9):
        a1 = np.arctan2(tok_emb[i, 1], tok_emb[i, 0])
        a2 = np.arctan2(tok_emb[i + 1, 1], tok_emb[i + 1, 0])
        diff = np.degrees(a2 - a1)
        if diff > 180: diff -= 360
        if diff < -180: diff += 360
        print(f"  {i}->{i+1}: {diff:+.1f}°")

    print("\nPairwise L2 distances (digits 0-9):")
    dists = []
    for i in range(10):
        for j in range(i + 1, 10):
            d = np.linalg.norm(tok_emb[i] - tok_emb[j])
            dists.append((d, i, j))
    dists.sort()
    print(f"  Closest:  digits {dists[0][1]},{dists[0][2]}  dist={dists[0][0]:.4f}")
    print(f"  2nd:      digits {dists[1][1]},{dists[1][2]}  dist={dists[1][0]:.4f}")
    print(f"  3rd:      digits {dists[2][1]},{dists[2][2]}  dist={dists[2][0]:.4f}")
    print(f"  Farthest: digits {dists[-1][1]},{dists[-1][2]}  dist={dists[-1][0]:.4f}")

    # Classification margin: for tied output head, logit_i = head_proj(x) · tok_emb[i]
    # The decision boundary between class i and j is the perpendicular bisector of
    # the line segment between tok_emb[i] and tok_emb[j] in 2D.
    print("\nClassification margin (min dist to nearest other token / 2):")
    for i in range(14):
        min_d = float("inf")
        nearest = -1
        for j in range(14):
            if j != i:
                d = np.linalg.norm(tok_emb[i] - tok_emb[j])
                if d < min_d:
                    min_d = d
                    nearest = j
        print(f"  {names[i]:>7s}: margin={min_d / 2:.4f}  nearest={names[nearest]}")


def diag2_attention_patterns(state):
    print("\n" + "=" * 70)
    print("2. ATTENTION PATTERNS (2 heads)")
    print("=" * 70)

    cfg = ModelConfig(d_model=6, tok_dim=2, pos_dim=4, n_heads=2, head_dim=3,
                      pos_mode="spiral_correct", pos_correction_mode="linear",
                      tie_qk=True, q_phase=True, attn_out_rank=2, freeze_special="eos")
    model = MicroAdder(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Generate batch of examples
    rng = random.Random(42)
    examples = []
    for _ in range(200):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, _ = make_example(a, b)
        examples.append(inp)

    batch = torch.stack([torch.tensor(ex) for ex in examples])
    B, T = batch.shape

    with torch.no_grad():
        x = model.tok_emb(batch)
        pos = model._get_positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x, pos], dim=-1)

        x_ln = model.blocks[0].ln1(x)
        attn = model.blocks[0].attn

        x_pos = x_ln[:, :, attn.tok_dim:]
        q = attn.q_proj(x_pos)
        k = attn.q_proj(x_pos)[:, :, :attn.kv_inner_dim]

        q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = k.view(B, T, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

        q = attn._apply_q_phase(q)
        k = attn._repeat_kv(k)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        att = att.masked_fill(attn.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

    avg_att = att.mean(0).numpy()  # (2, T, T)

    # Position labels
    pos_labels = ([f"X{i}" for i in range(10)] + ["Zhi", "+"]
                  + [f"Y{i}" for i in range(10)] + ["="]
                  + [f"A{i}" for i in range(11)] + ["EOS"])

    angles = model.blocks[0].attn.q_phase_angle.detach().numpy()
    print(f"Q-phase angles: {angles} rad = {np.degrees(angles)} degrees")

    print(f"\nSequence length: {T}, attention shape: {avg_att.shape}")
    print("\nAverage attention from answer positions:")
    for h in range(2):
        print(f"\n  Head {h}:")
        for a_i in range(11):
            a_idx = 23 + a_i
            if a_idx >= T:
                break
            a_label = pos_labels[a_idx] if a_idx < len(pos_labels) else f"pos{a_idx}"
            row = avg_att[h, a_idx, :a_idx + 1]
            top_k = np.argsort(row)[::-1][:5]
            top_str = ", ".join(
                f"{pos_labels[j] if j < len(pos_labels) else f'pos{j}'}={row[j]:.3f}"
                for j in top_k
            )
            print(f"    {a_label:>3s} -> {top_str}")

    n_answer = min(10, T - 23)
    print(f"\nHead specialization summary (for A0-A{n_answer-1}):")
    for h in range(2):
        digit_pair = 0
        carry_chain = 0
        self_attn = 0
        for i in range(n_answer):
            a_pos = 23 + i
            x_pos = i
            y_pos = 12 + i
            prev_a = 23 + i - 1 if i > 0 else None

            row = avg_att[h, a_pos, :a_pos + 1]
            digit_pair += row[x_pos] + row[y_pos]
            self_attn += row[a_pos]
            if prev_a is not None:
                carry_chain += row[prev_a]

        n_carry = max(1, n_answer - 1)
        print(f"  Head {h}: digit-pair={digit_pair / n_answer:.3f}, "
              f"carry-chain={carry_chain / n_carry:.3f}, self={self_attn / n_answer:.3f}")

    # Also show attention for non-answer positions (what does + and = attend to?)
    print("\nDelimiter attention:")
    for h in range(2):
        # + is at position 11
        row_plus = avg_att[h, 11, :12]
        top = np.argsort(row_plus)[::-1][:3]
        print(f"  Head {h}, + -> " + ", ".join(f"{pos_labels[j]}={row_plus[j]:.3f}" for j in top))

        # = is at position 22
        row_eq = avg_att[h, 22, :23]
        top = np.argsort(row_eq)[::-1][:3]
        print(f"  Head {h}, = -> " + ", ".join(f"{pos_labels[j]}={row_eq[j]:.3f}" for j in top))


def diag3_ffn_analysis(state):
    print("\n" + "=" * 70)
    print("3. FFN ACTIVATION ANALYSIS")
    print("=" * 70)

    fc1_w = state["blocks.0.ffn.fc1.weight"].numpy()  # [2, 6]
    fc1_b = state["blocks.0.ffn.fc1.bias"].numpy()     # [2]
    fc2_w = state["blocks.0.ffn.fc2.weight"].numpy()  # [6, 2]
    fc2_b = state["blocks.0.ffn.fc2.bias"].numpy()     # [6]

    print("FFN: x -> fc1 (6->2, +bias) -> GELU -> fc2 (2->6, +bias)")
    print(f"\nfc1 weights [2x6]:\n{fc1_w}")
    print(f"fc1 bias: {fc1_b}")
    print(f"\nfc2 weights [6x2]:\n{fc2_w}")
    print(f"fc2 bias: {fc2_b}")

    print("\nPer hidden unit analysis:")
    for i in range(2):
        w = fc1_w[i]
        b = fc1_b[i]
        tok_w = w[:2]
        pos_w = w[2:]
        print(f"  Unit {i}:")
        print(f"    fc1: tok=[{tok_w[0]:.3f}, {tok_w[1]:.3f}], "
              f"pos=[{pos_w[0]:.3f}, {pos_w[1]:.3f}, {pos_w[2]:.3f}, {pos_w[3]:.3f}], "
              f"bias={b:.3f}")
        print(f"    ||w_tok||={np.linalg.norm(tok_w):.3f}, ||w_pos||={np.linalg.norm(pos_w):.3f}")
        print(f"    → reads mostly from {'tok' if np.linalg.norm(tok_w) > np.linalg.norm(pos_w) else 'pos'} subspace")

        out_col = fc2_w[:, i]
        tok_out = out_col[:2]
        pos_out = out_col[2:]
        print(f"    fc2: writes tok=[{tok_out[0]:.3f}, {tok_out[1]:.3f}], "
              f"pos=[{pos_out[0]:.3f}, {pos_out[1]:.3f}, {pos_out[2]:.3f}, {pos_out[3]:.3f}]")
        print(f"    → writes mostly to {'tok' if np.linalg.norm(tok_out) > np.linalg.norm(pos_out) else 'pos'} subspace")

    # Sharpness analysis
    print(f"\nSharpness: max |weight| in fc1: {np.abs(fc1_w).max():.3f}")
    print(f"  (hand-coded models use ~60000 for hard step functions)")
    print(f"  Our transitions are soft — GELU slope ≈ {np.abs(fc1_w).max():.1f}x at threshold")

    # Full FFN as a map: what does it do to the residual?
    # The bottleneck is 2D, so the FFN can only add a rank-2 correction to the residual
    print(f"\nFFN effective rank: 2 (bottleneck dim)")
    print(f"fc2 bias (additive constant to residual): [{', '.join(f'{b:.3f}' for b in fc2_b)}]")


def diag4_positions(state):
    print("\n" + "=" * 70)
    print("4. POSITION ENCODING ANALYSIS")
    print("=" * 70)

    params = {
        "amp": (state["spiral_amp"].item(), 1.0),
        "phase": (state["spiral_phase"].item(), 0.0),
        "slope": (state["spiral_slope"].item(), 1.0 / 9),
        "offset": (state["spiral_offset"].item(), 0.0),
        "corr_slope": (state["pos_corr_slope"].item(), 0.0),
        "corr_intercept": (state["pos_corr_intercept"].item(), 0.0),
    }

    print("Spiral parameters (trained vs init):")
    for name, (val, init) in params.items():
        pct = abs(val - init) / max(abs(init), 1e-6) * 100
        print(f"  {name:15s}: {val:+10.6f}  (init: {init:+.6f}, drift: {val - init:+.6f}, {pct:.0f}%)")

    # Compute positions
    idx = np.arange(10, dtype=np.float32)
    amp, phase = params["amp"][0], params["phase"][0]
    slope, offset = params["slope"][0], params["offset"][0]
    corr_s, corr_i = params["corr_slope"][0], params["corr_intercept"][0]

    angle = 2 * np.pi * idx / 10.0 + phase
    base = np.zeros((10, 4))
    base[:, 0] = amp * np.cos(angle)
    base[:, 1] = amp * np.sin(angle)
    base[:, 2] = slope * idx + offset
    correction = corr_i + corr_s * idx
    positions = base * (1.0 + correction)[:, None]

    print(f"\nDigit positions [10x4]:")
    print(f"{'pos':>3s} | {'dim0':>8s} {'dim1':>8s} {'dim2':>8s} {'dim3':>8s} | {'corr':>6s} | {'norm':>6s}")
    print("-" * 65)
    for i in range(10):
        norm = np.linalg.norm(positions[i])
        print(f"{i:>3d} | {positions[i, 0]:8.4f} {positions[i, 1]:8.4f} "
              f"{positions[i, 2]:8.4f} {positions[i, 3]:8.4f} | {correction[i]:6.3f} | {norm:6.3f}")

    # Special positions
    z_hi = state["z_hi_pos"].numpy().flatten()
    special = state["special_pos_learned"].numpy()  # [2, 4]
    eos = state["_eos_pos"].numpy().flatten()

    print(f"\nSpecial positions:")
    print(f"  z_hi (carry): [{', '.join(f'{v:.4f}' for v in z_hi)}]  norm={np.linalg.norm(z_hi):.4f}")
    print(f"  PLUS:         [{', '.join(f'{v:.4f}' for v in special[0])}]  norm={np.linalg.norm(special[0]):.4f}")
    print(f"  EQUALS:       [{', '.join(f'{v:.4f}' for v in special[1])}]  norm={np.linalg.norm(special[1]):.4f}")
    print(f"  EOS (frozen): [{', '.join(f'{v:.4f}' for v in eos)}]")

    # Freezeability: if params are close to init, they could be frozen
    total_drift = sum(abs(val - init) for val, init in params.values())
    print(f"\nFreezeability: total param drift from init = {total_drift:.4f}")
    print(f"  Largest drift: {max((abs(v - i), n) for n, (v, i) in params.items())[1]}")


def diag5_effective_rank(state):
    print("\n" + "=" * 70)
    print("5. EFFECTIVE RANK OF ALL PROJECTIONS")
    print("=" * 70)

    def analyze(name, W):
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        total = (S**2).sum()
        cum = np.cumsum(S**2) / total
        s_norm = S / S.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        eff_rank = np.exp(entropy)
        r95 = int(np.searchsorted(cum, 0.95)) + 1
        r99 = int(np.searchsorted(cum, 0.99)) + 1
        sv_str = "  ".join(f"{s:.3f}" for s in S)
        en_str = "  ".join(f"{c:.1%}" for c in cum)
        cond = S[0] / S[-1] if S[-1] > 1e-10 else float("inf")
        print(f"  {name:25s} {str(list(W.shape)):>10s}")
        print(f"    singular values: [{sv_str}]")
        print(f"    cumulative energy: [{en_str}]")
        print(f"    eff_rank={eff_rank:.2f}, rank@95%={r95}, rank@99%={r99}, cond={cond:.1f}")
        print()

    matrices = [
        ("q_proj", state["blocks.0.attn.q_proj.weight"].numpy()),
        ("v_proj", state["blocks.0.attn.v_proj.weight"].numpy()),
        ("out_proj.A", state["blocks.0.attn.out_proj.A"].numpy()),
        ("out_proj.B", state["blocks.0.attn.out_proj.B"].numpy()),
        ("out_proj (A@B)", state["blocks.0.attn.out_proj.A"].numpy()
         @ state["blocks.0.attn.out_proj.B"].numpy()),
        ("fc1", state["blocks.0.ffn.fc1.weight"].numpy()),
        ("fc2", state["blocks.0.ffn.fc2.weight"].numpy()),
        ("head_proj", state["head_proj.weight"].numpy()),
        ("tok_emb (digits 0-9)", state["tok_emb.weight"].numpy()[:10]),
    ]

    for name, W in matrices:
        analyze(name, W)

    # RMSNorm analysis
    print("RMSNorm weights (deviation from 1.0):")
    for norm_name in ["blocks.0.ln1.weight", "blocks.0.ln2.weight", "ln_f.weight"]:
        w = state[norm_name].numpy()
        delta = w - 1.0
        print(f"  {norm_name}:")
        print(f"    values: [{', '.join(f'{v:.3f}' for v in w)}]")
        print(f"    delta:  [{', '.join(f'{d:+.3f}' for d in delta)}]")
        print(f"    range: [{w.min():.3f}, {w.max():.3f}], std: {w.std():.4f}")
        print()


if __name__ == "__main__":
    state = load_checkpoint()
    diag1_token_geometry(state)
    diag2_attention_patterns(state)
    diag3_ffn_analysis(state)
    diag4_positions(state)
    diag5_effective_rank(state)
