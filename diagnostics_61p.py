"""Structural diagnostics on the 61p (qk_dim=2) near-grokked checkpoint."""

import math
import random

import numpy as np
import torch

from microadder.model import ModelConfig, MicroAdder, count_params, parameter_breakdown
from microadder.data import make_example, PROMPT_LEN, ANSWER_LEN, SEQ_LEN, MAX_DIGITS


CKPT_PATH = "results/runs/sub100_61p_qk2_s7777/checkpoints/best.pt"


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = MicroAdder(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def diag1_token_geometry(model):
    print("=" * 70)
    print("1. TOKEN EMBEDDING GEOMETRY (10 digits in 2D)")
    print("=" * 70)

    tok_emb = model._compute_tok_emb().detach().numpy()  # [10, 2]
    names = [str(i) for i in range(10)]

    print(f"\nArc params: A={model.tok_arc_A.item():.4f}, "
          f"start={model.tok_arc_start.item():.4f} ({math.degrees(model.tok_arc_start.item()):.1f}°), "
          f"stride={model.tok_arc_stride.item():.4f} ({math.degrees(model.tok_arc_stride.item()):.1f}°)")
    total_arc = 9 * model.tok_arc_stride.item()
    print(f"Total arc span: {math.degrees(total_arc):.1f}° (9 * stride)")

    print(f"\n{'Token':>8s} | {'dim0':>8s} {'dim1':>8s} | {'norm':>6s} | {'angle':>7s}")
    print("-" * 50)
    for i in range(10):
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
    print(f"  Closest:  {dists[0][1]}↔{dists[0][2]}  dist={dists[0][0]:.4f}")
    print(f"  2nd:      {dists[1][1]}↔{dists[1][2]}  dist={dists[1][0]:.4f}")
    print(f"  3rd:      {dists[2][1]}↔{dists[2][2]}  dist={dists[2][0]:.4f}")
    print(f"  Farthest: {dists[-1][1]}↔{dists[-1][2]}  dist={dists[-1][0]:.4f}")

    print("\nClassification margin (min dist to nearest other digit / 2):")
    for i in range(10):
        min_d = float("inf")
        nearest = -1
        for j in range(10):
            if j != i:
                d = np.linalg.norm(tok_emb[i] - tok_emb[j])
                if d < min_d:
                    min_d = d
                    nearest = j
        print(f"  {names[i]:>7s}: margin={min_d / 2:.4f}  nearest={names[nearest]}")


def diag2_attention_patterns(model, ckpt):
    print("\n" + "=" * 70)
    print("2. ATTENTION PATTERNS (1 head, qk_dim=2)")
    print("=" * 70)

    cfg = model.cfg
    qk_dim = cfg.effective_qk_dim

    # Q phase
    angle_rad = model.q_phase_angle.detach().numpy()
    print(f"Q-phase angle: {angle_rad[0]:.4f} rad = {np.degrees(angle_rad[0]):.1f}°")

    # Generate batch of examples
    rng = random.Random(42)
    examples = []
    for _ in range(500):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, tgt = make_example(a, b)
        examples.append(inp)

    batch = torch.tensor(examples)
    B, T = batch.shape

    with torch.no_grad():
        tok_emb_table = model._compute_tok_emb()
        tok = tok_emb_table[batch]
        pos = model._get_positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        h = model.norm1(x)
        pos_h = h[:, :, cfg.tok_dim:]  # (B, T, 3)

        Q = model.q_proj(pos_h)  # (B, T, 2)
        K = model.q_proj(pos_h)  # (B, T, 2)

        Q = Q.view(B, T, 1, qk_dim).transpose(1, 2)
        K = K.view(B, T, 1, qk_dim).transpose(1, 2)

        Q = model._apply_q_phase(Q)

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(qk_dim)
        att = att.masked_fill(model.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

    avg_att = att.mean(0).squeeze(0).numpy()  # (T, T) since 1 head

    # Position labels for 10-digit addition
    # Sequence: X0..X9 Zhi + Y0..Y9 = A0..A10 EOS
    pos_labels = ([f"X{i}" for i in range(10)] + ["Zhi", "+"]
                  + [f"Y{i}" for i in range(10)] + ["="]
                  + [f"A{i}" for i in range(11)] + ["EOS"])

    print(f"\nSequence length: {T}, attention shape: {avg_att.shape}")

    print("\nAverage attention from answer positions (top 5 sources):")
    for a_i in range(11):
        a_idx = 23 + a_i
        if a_idx >= T:
            break
        a_label = pos_labels[a_idx] if a_idx < len(pos_labels) else f"pos{a_idx}"
        row = avg_att[a_idx, :a_idx + 1]
        top_k = np.argsort(row)[::-1][:5]
        top_str = ", ".join(
            f"{pos_labels[j] if j < len(pos_labels) else f'pos{j}'}={row[j]:.3f}"
            for j in top_k
        )
        print(f"  {a_label:>3s} -> {top_str}")

    # Specialization summary
    print(f"\nSpecialization summary:")
    digit_pair = 0
    carry_chain = 0
    self_attn = 0
    z_hi_attn = 0
    n_answer = min(11, T - 23)
    for i in range(n_answer):
        a_pos = 23 + i
        x_pos = i        # X_i
        y_pos = 12 + i   # Y_i
        prev_a = 23 + i - 1 if i > 0 else None
        z_hi_idx = 10  # Zhi position

        row = avg_att[a_pos, :a_pos + 1]
        digit_pair += row[x_pos] + row[y_pos]
        self_attn += row[a_pos]
        z_hi_attn += row[z_hi_idx]
        if prev_a is not None:
            carry_chain += row[prev_a]

    n_carry = max(1, n_answer - 1)
    print(f"  digit-pair(Xi+Yi)={digit_pair / n_answer:.3f}, "
          f"carry-chain(A_i-1)={carry_chain / n_carry:.3f}, "
          f"self={self_attn / n_answer:.3f}, "
          f"z_hi={z_hi_attn / n_answer:.3f}")

    # Also check if it's doing offset patterns like 67p
    print("\nOffset analysis (which X and Y does each A attend to most?):")
    for a_i in range(min(11, T - 23)):
        a_idx = 23 + a_i
        row = avg_att[a_idx, :a_idx + 1]
        # Find best X (positions 0-9)
        best_x = np.argmax(row[:10])
        best_x_val = row[best_x]
        # Find best Y (positions 12-21)
        if a_idx > 12:
            best_y = np.argmax(row[12:min(22, a_idx+1)]) + 12
            best_y_val = row[best_y]
        else:
            best_y = -1
            best_y_val = 0
        x_offset = best_x - a_i
        y_offset = (best_y - 12) - a_i if best_y >= 0 else None
        label = pos_labels[a_idx]
        print(f"  {label}: best_X={pos_labels[best_x]}({best_x_val:.3f}, offset={x_offset:+d}), "
              f"best_Y={pos_labels[best_y] if best_y >= 0 else 'N/A'}({best_y_val:.3f}, offset={y_offset if y_offset is not None else 'N/A'})")

    # Delimiter attention
    print("\nDelimiter attention:")
    row_plus = avg_att[11, :12]
    top = np.argsort(row_plus)[::-1][:3]
    print(f"  + -> " + ", ".join(f"{pos_labels[j]}={row_plus[j]:.3f}" for j in top))
    row_eq = avg_att[22, :23]
    top = np.argsort(row_eq)[::-1][:3]
    print(f"  = -> " + ", ".join(f"{pos_labels[j]}={row_eq[j]:.3f}" for j in top))

    # Show full attention heatmap for key positions
    print("\nFull attention distribution from A0, A5, A9, A10:")
    for a_i in [0, 5, 9, 10]:
        a_idx = 23 + a_i
        if a_idx >= T:
            break
        row = avg_att[a_idx, :a_idx + 1]
        label = pos_labels[a_idx]
        nonzero = [(pos_labels[j], row[j]) for j in range(len(row)) if row[j] > 0.01]
        nonzero.sort(key=lambda x: -x[1])
        entries = ", ".join(f"{n}={v:.3f}" for n, v in nonzero)
        print(f"  {label}: {entries}")


def diag3_ffn_analysis(model):
    print("\n" + "=" * 70)
    print("3. FFN ANALYSIS (no bias)")
    print("=" * 70)

    fc1_w = model.fc1.weight.detach().numpy()  # [2, 5]
    fc2_w = model.fc2.weight.detach().numpy()  # [5, 2]

    print(f"FFN: x -> fc1 (5->2) -> GELU -> fc2 (2->5)")
    print(f"\nfc1 weights [2x5]:\n{fc1_w}")
    print(f"\nfc2 weights [5x2]:\n{fc2_w}")

    print("\nPer hidden unit analysis:")
    for i in range(2):
        w = fc1_w[i]
        tok_w = w[:2]
        pos_w = w[2:]
        print(f"  Unit {i}:")
        print(f"    fc1: tok=[{tok_w[0]:.4f}, {tok_w[1]:.4f}], "
              f"pos=[{pos_w[0]:.4f}, {pos_w[1]:.4f}, {pos_w[2]:.4f}]")
        print(f"    ||w_tok||={np.linalg.norm(tok_w):.4f}, ||w_pos||={np.linalg.norm(pos_w):.4f}")
        ratio = np.linalg.norm(tok_w) / (np.linalg.norm(pos_w) + 1e-8)
        print(f"    → reads {ratio:.2f}x more from {'tok' if ratio > 1 else 'pos'} subspace")

        out_col = fc2_w[:, i]
        tok_out = out_col[:2]
        pos_out = out_col[2:]
        print(f"    fc2: writes tok=[{tok_out[0]:.4f}, {tok_out[1]:.4f}], "
              f"pos=[{pos_out[0]:.4f}, {pos_out[1]:.4f}, {pos_out[2]:.4f}]")
        out_ratio = np.linalg.norm(tok_out) / (np.linalg.norm(pos_out) + 1e-8)
        print(f"    → writes {out_ratio:.2f}x more to {'tok' if out_ratio > 1 else 'pos'} subspace")

    # Full FFN matrix (fc2 @ fc1 without GELU -- linear approximation)
    linear_approx = fc2_w @ fc1_w
    print(f"\nLinear approx fc2@fc1 [5x5]:")
    print(linear_approx)
    print(f"Max |weight| in fc1: {np.abs(fc1_w).max():.4f}")


def diag4_positions(model):
    print("\n" + "=" * 70)
    print("4. POSITION ENCODING ANALYSIS")
    print("=" * 70)

    spiral_params = {
        "amp": model.spiral_amp.item(),
        "phase": model.spiral_phase.item(),
        "slope": model.spiral_slope.item(),
        "offset": model.spiral_offset.item(),
    }

    frozen = set(model.cfg.freeze_spiral.split(",")) if model.cfg.freeze_spiral else set()
    print("Spiral parameters (all frozen for 61p):")
    for name, val in spiral_params.items():
        is_frozen = name in frozen
        print(f"  {name:10s}: {val:+10.6f}  {'[FROZEN]' if is_frozen else '[LEARNED]'}")

    # Compute positions
    digit_pos = model._get_digit_positions().detach().numpy()  # (10, 3)

    print(f"\nDigit positions [10x3]:")
    print(f"{'pos':>3s} | {'dim0':>8s} {'dim1':>8s} {'dim2':>8s} | {'norm':>6s} | {'angle_01':>8s}")
    print("-" * 60)
    for i in range(10):
        norm = np.linalg.norm(digit_pos[i])
        angle = np.degrees(np.arctan2(digit_pos[i, 1], digit_pos[i, 0]))
        print(f"{i:>3d} | {digit_pos[i, 0]:8.4f} {digit_pos[i, 1]:8.4f} "
              f"{digit_pos[i, 2]:8.4f} | {norm:6.3f} | {angle:8.1f}°")

    # Special positions
    z_hi = model.z_hi_pos.detach().numpy().flatten()
    eq_pos = model.special_pos_equals.detach().numpy().flatten()
    plus_pos = model._plus_pos.detach().numpy().flatten()
    eos_pos = model._eos_pos.detach().numpy().flatten()

    print(f"\nSpecial positions:")
    print(f"  z_hi (carry):  [{', '.join(f'{v:.4f}' for v in z_hi)}]  norm={np.linalg.norm(z_hi):.4f}")
    print(f"  PLUS (frozen): [{', '.join(f'{v:.4f}' for v in plus_pos)}]  norm={np.linalg.norm(plus_pos):.4f}")
    print(f"  EQUALS:        [{', '.join(f'{v:.4f}' for v in eq_pos)}]  norm={np.linalg.norm(eq_pos):.4f}")
    print(f"  EOS (frozen):  [{', '.join(f'{v:.4f}' for v in eos_pos)}]  norm={np.linalg.norm(eos_pos):.4f}")

    # Analyze Q projection to understand how positions are mapped to qk_dim=2
    q_w = model.q_proj.weight.detach().numpy()  # [2, 3]
    print(f"\nQ/K projection (3 -> 2):")
    print(f"  q_proj.weight [2x3]:\n{q_w}")
    U, S, Vh = np.linalg.svd(q_w, full_matrices=False)
    print(f"  Singular values: {S}")
    print(f"  Condition number: {S[0]/S[1]:.2f}")

    # Project digit positions through Q to see what Q/K space looks like
    q_digit = digit_pos @ q_w.T  # (10, 2)
    print(f"\nDigit positions projected to Q/K space (before phase):")
    for i in range(10):
        angle = np.degrees(np.arctan2(q_digit[i, 1], q_digit[i, 0]))
        norm = np.linalg.norm(q_digit[i])
        print(f"  pos{i}: [{q_digit[i, 0]:7.4f}, {q_digit[i, 1]:7.4f}]  norm={norm:.4f}  angle={angle:.1f}°")

    # Also project special positions
    q_zhi = z_hi @ q_w.T
    q_eq = eq_pos @ q_w.T
    print(f"\n  z_hi in Q/K: [{q_zhi[0]:.4f}, {q_zhi[1]:.4f}]  norm={np.linalg.norm(q_zhi):.4f}")
    print(f"  EQUALS in Q/K: [{q_eq[0]:.4f}, {q_eq[1]:.4f}]  norm={np.linalg.norm(q_eq):.4f}")
    print(f"  PLUS (zero) in Q/K: [0, 0]")

    # Q phase rotation effect
    angle_rad = model.q_phase_angle.detach().numpy()[0]
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    q_digit_rotated = q_digit @ R.T  # After phase rotation

    print(f"\nQ phase rotation: {np.degrees(angle_rad):.1f}°")
    print(f"Q-K dot products per digit pair (Q_i · K_j, normalized by sqrt(2)):")
    dot_matrix = q_digit_rotated @ q_digit.T / math.sqrt(2)

    # Show the attention logit pattern for consecutive digits
    print("\nAttention logits A_i -> X_j (diagonal shows offset pattern):")
    print("     " + "  ".join(f"X{j}" for j in range(10)))
    for i in range(10):
        vals = [f"{dot_matrix[i, j]:5.2f}" for j in range(10)]
        print(f"  p{i} " + " ".join(vals))


def diag5_effective_rank(model):
    print("\n" + "=" * 70)
    print("5. EFFECTIVE RANK & WEIGHT MAGNITUDES")
    print("=" * 70)

    state = {k: v.detach().numpy() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}

    def analyze(name, W):
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        total = (S**2).sum()
        cum = np.cumsum(S**2) / total
        s_norm = S / S.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        eff_rank = np.exp(entropy)
        cond = S[0] / S[-1] if S[-1] > 1e-10 else float("inf")
        sv_str = "  ".join(f"{s:.4f}" for s in S)
        print(f"  {name:25s} {str(list(W.shape)):>10s}")
        print(f"    singular values: [{sv_str}]")
        print(f"    eff_rank={eff_rank:.2f}, cond={cond:.1f}")
        print()

    matrices = [
        ("q_proj (3->2)", state["q_proj.weight"]),
        ("out_proj.A (5->1)", state["out_proj.A"]),
        ("out_proj.B (1->5)", state["out_proj.B"]),
        ("fc1 (5->2)", state["fc1.weight"]),
        ("fc2 (2->5)", state["fc2.weight"]),
        ("head_proj (5->2)", state["head_proj.weight"]),
    ]

    for name, W in matrices:
        analyze(name, W)

    # Rank-1 out_proj combined
    A = state["out_proj.A"]
    B = state["out_proj.B"]
    combined = A @ B
    print(f"  out_proj combined (A@B) [5x5]:")
    print(f"    A: [{', '.join(f'{v:.4f}' for v in A.flatten())}]")
    print(f"    B: [{', '.join(f'{v:.4f}' for v in B.flatten())}]")
    print(f"    A norms by dim: {np.abs(A.flatten())}")
    print(f"    B norms by dim: {np.abs(B.flatten())}")
    print()

    # RMSNorm analysis
    print("RMSNorm weights (shared across norm1/norm2/norm_f):")
    w = state["norm1.weight"]
    print(f"  values: [{', '.join(f'{v:.4f}' for v in w)}]")
    delta = w - 1.0
    print(f"  delta:  [{', '.join(f'{d:+.4f}' for d in delta)}]")
    print(f"  range: [{w.min():.4f}, {w.max():.4f}], std: {w.std():.4f}")
    print(f"  tok dims (0,1): [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"  pos dims (2,3,4): [{w[2]:.4f}, {w[3]:.4f}, {w[4]:.4f}]")

    # Head proj analysis (also serves as V projection)
    hp = state["head_proj.weight"]  # [2, 5]
    print(f"\nhead_proj (also V-proj) [2x5]:\n{hp}")
    tok_part = hp[:, :2]
    pos_part = hp[:, 2:]
    print(f"  tok subspace [2x2]:\n{tok_part}")
    print(f"  pos subspace [2x3]:\n{pos_part}")
    print(f"  ||tok_part||={np.linalg.norm(tok_part):.4f}, ||pos_part||={np.linalg.norm(pos_part):.4f}")


def diag6_error_analysis(model, ckpt):
    """Analyze WHERE the model fails - which digit positions, which carry patterns."""
    print("\n" + "=" * 70)
    print("6. ERROR ANALYSIS (where does 73% exact fail?)")
    print("=" * 70)

    rng = random.Random(42)
    n_total = 3000
    n_exact = 0
    digit_errors = [0] * 11  # per answer position
    carry_errors = {0: 0, 1: 0}  # errors on no-carry vs carry digits
    carry_totals = {0: 0, 1: 0}
    error_examples = []

    for _ in range(n_total):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, tgt = make_example(a, b)
        inp_t = torch.tensor([inp])

        with torch.no_grad():
            logits, _ = model.forward(inp_t)

        pred = logits[0].argmax(dim=-1).numpy()
        tgt = np.array(tgt)

        # Answer starts at PROMPT_LEN+1=22 (after = token), goes for ANSWER_LEN=11
        ans_start = 22
        ans_end = ans_start + ANSWER_LEN
        pred_ans = pred[ans_start:ans_end]
        tgt_ans = np.array(tgt[ans_start:ans_end])

        exact = True
        s = a + b
        s_str = str(s).zfill(11)
        a_str = str(a).zfill(10)
        b_str = str(b).zfill(10)

        # Compute carries per answer position
        carry = 0
        carries = []
        for i in range(10):
            d_a = int(a_str[9 - i])
            d_b = int(b_str[9 - i])
            total = d_a + d_b + carry
            carries.append(carry)
            carry = total // 10
        carries.append(carry)  # carry into MSB+1

        for pos in range(ANSWER_LEN):
            if tgt_ans[pos] >= 0:
                expected = tgt_ans[pos]
                got = pred_ans[pos]
                c = carries[pos] if pos < len(carries) else 0
                carry_totals[min(c, 1)] += 1
                if got != expected:
                    exact = False
                    digit_errors[pos] += 1
                    carry_errors[min(c, 1)] += 1
                    if len(error_examples) < 10:
                        error_examples.append((a, b, pos, expected, got, c))

        if exact:
            n_exact += 1

    print(f"\nExact match: {n_exact}/{n_total} = {n_exact/n_total:.1%}")

    print(f"\nPer-position error rate (answer digit 0=LSB, 10=MSB+1):")
    for pos in range(11):
        rate = digit_errors[pos] / n_total * 100
        bar = "#" * int(rate * 2)
        print(f"  A{pos:>2d}: {digit_errors[pos]:>4d}/{n_total} = {rate:5.1f}%  {bar}")

    print(f"\nCarry vs no-carry errors:")
    for c in [0, 1]:
        t = carry_totals[c]
        e = carry_errors[c]
        print(f"  carry={c}: {e}/{t} = {e/t*100:.2f}%" if t > 0 else f"  carry={c}: no samples")

    print(f"\nSample errors (first 10):")
    for a, b, pos, exp, got, c in error_examples:
        print(f"  {a}+{b} A{pos}: expected={exp} got={got} carry_in={c}")


def diag7_qk_dim_bottleneck(model):
    """Analyze if qk_dim=2 is sufficient for the attention routing needed."""
    print("\n" + "=" * 70)
    print("7. QK_DIM=2 BOTTLENECK ANALYSIS")
    print("=" * 70)

    # With qk_dim=2, the Q and K vectors live in 2D
    # The model needs to create attention patterns that distinguish:
    # - 10 digit positions (each X_i should attend differently)
    # - z_hi, +, =, EOS special positions
    # - answer positions A_0..A_10 attending to appropriate sources

    # In the 67p model (qk_dim=4), positions project to 4D Q/K space
    # In this 61p model (qk_dim=2), they project to 2D
    # This means attention patterns are determined by angle in 2D

    q_w = model.q_proj.weight.detach().numpy()  # [2, 3]
    digit_pos = model._get_digit_positions().detach().numpy()  # (10, 3)

    # All positions in the sequence
    all_pos = model._get_positions(SEQ_LEN).detach().numpy()  # (34, 3)

    # Project all positions to Q/K space
    all_qk = all_pos @ q_w.T  # (34, 2)

    # Phase rotation
    angle_rad = model.q_phase_angle.detach().item()
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    all_q = all_qk @ R.T  # Q vectors (after rotation)
    all_k = all_qk  # K vectors (no rotation)

    pos_labels = ([f"X{i}" for i in range(10)] + ["Zhi", "+"]
                  + [f"Y{i}" for i in range(10)] + ["="]
                  + [f"A{i}" for i in range(11)] + ["EOS"])

    print(f"\nAll positions in K-space (2D):")
    for i in range(min(34, len(pos_labels))):
        angle = np.degrees(np.arctan2(all_k[i, 1], all_k[i, 0]))
        norm = np.linalg.norm(all_k[i])
        print(f"  {pos_labels[i]:>4s}: [{all_k[i, 0]:7.4f}, {all_k[i, 1]:7.4f}]  "
              f"norm={norm:.4f}  angle={angle:.1f}°")

    # Key question: can 2D Q/K separate the positions that need different attention?
    # For answer tokens, we need Xi, Yi, and possibly Ai-1 to be distinguishable
    print(f"\nK-space angular separations (critical pairs):")
    for i in range(10):
        x_k = all_k[i]         # X_i
        y_k = all_k[12 + i]    # Y_i
        angle_x = np.degrees(np.arctan2(x_k[1], x_k[0]))
        angle_y = np.degrees(np.arctan2(y_k[1], y_k[0]))
        diff = angle_y - angle_x
        if diff > 180: diff -= 360
        if diff < -180: diff += 360
        print(f"  X{i} vs Y{i}: X={angle_x:.1f}° Y={angle_y:.1f}° diff={diff:.1f}°")

    # Are X and Y positions distinguishable in K-space?
    # They share the same digit positions but in different sequence positions
    # Since Q/K only looks at pos subspace, and X_i and Y_i share the same
    # digit position index, they should have IDENTICAL K vectors!
    print(f"\n  NOTE: X_i and Y_i have the SAME digit position index (both use pos[i])")
    print(f"  So they have IDENTICAL K vectors! The model CANNOT distinguish Xi from Yi via K.")
    print(f"  Verification: ||K(X0) - K(Y0)|| = {np.linalg.norm(all_k[0] - all_k[12]):.6f}")
    print(f"                ||K(X5) - K(Y5)|| = {np.linalg.norm(all_k[5] - all_k[17]):.6f}")

    # What about A_i positions? They also share digit indices
    print(f"\n  A_i also shares digit index i with X_i and Y_i:")
    print(f"  ||K(X0) - K(A0)|| = {np.linalg.norm(all_k[0] - all_k[23]):.6f}")

    # Effective attention resolution with qk_dim=2
    # With 2D Q/K, the model has only 1 angular degree of freedom for routing
    # (the other is magnitude). Compare to qk_dim=4 which has 3 angular DOF.
    print(f"\nCapacity analysis:")
    print(f"  qk_dim=2: attention routing is 2D -> 1 angular DOF for selectivity")
    print(f"  qk_dim=4 (67p): 4D -> 3 angular DOFs")
    print(f"  The q_phase angle rotates Q relative to K, creating asymmetric attention")
    print(f"  With only 1 angular DOF, the model can implement AT MOST 1 offset pattern")


if __name__ == "__main__":
    model, ckpt = load_model()

    print(f"Model: {count_params(model)} parameters")
    breakdown = parameter_breakdown(model)
    for name, n in breakdown.items():
        print(f"  {name}: {n}")
    print()

    diag1_token_geometry(model)
    diag2_attention_patterns(model, ckpt)
    diag3_ffn_analysis(model)
    diag4_positions(model)
    diag5_effective_rank(model)
    diag6_error_analysis(model, ckpt)
    diag7_qk_dim_bottleneck(model)
