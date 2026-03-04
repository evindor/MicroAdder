"""Structural analysis of the 57-parameter MicroAdder checkpoint.

Analyzes tie_fc2_head=True model where head_proj does triple duty:
  V projection, output classification head, AND FFN expansion matrix.
"""

import math
import random
import sys

import torch
import torch.nn.functional as F
import numpy as np

# Add project root
sys.path.insert(0, ".")
from microadder.model import MicroAdder, ModelConfig, count_params, parameter_breakdown
from microadder.data import (
    VOCAB_SIZE, SEQ_LEN, MAX_DIGITS, ANSWER_LEN, PROMPT_LEN,
    Z_START, Z_HI_POS, EOS_POS, PLUS_POS, EQ_POS,
    make_example, encode_number, decode_answer,
)

CKPT_PATH = "results/runs/sub100_57p_tiefc2/checkpoints/last.pt"

def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def subheader(title):
    print(f"\n--- {title} ---\n")

def main():
    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = MicroAdder(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Training step: {ckpt.get('step', '?')}")
    print(f"Seed: {ckpt['config'].get('seed', ckpt.get('args', {}).get('seed', '?'))}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"Final metrics: loss={m.get('loss','?')}, exact_acc={m.get('exact_acc','?')}, digit_acc={m.get('digit_acc','?')}")

    # ================================================================
    # 1. PARAMETER BUDGET VERIFICATION
    # ================================================================
    header("1. PARAMETER BUDGET VERIFICATION")

    total = count_params(model)
    bd = parameter_breakdown(model)
    print(f"Total learnable parameters: {total}")
    print(f"\nBreakdown:")
    for name, n in bd.items():
        print(f"  {name:30s}  {n:3d}")
    print(f"  {'TOTAL':30s}  {sum(bd.values()):3d}")

    # Verify tie
    assert cfg.tie_fc2_head, "Expected tie_fc2_head=True"
    print(f"\ntie_fc2_head = {cfg.tie_fc2_head}")
    print(f"  -> fc2 reuses head_proj.weight, saving 10 params (67p -> 57p)")

    # ================================================================
    # 2. TOKEN EMBEDDING GEOMETRY
    # ================================================================
    header("2. TOKEN EMBEDDING GEOMETRY")

    A = model.tok_arc_A.item()
    start = model.tok_arc_start.item()
    stride = model.tok_arc_stride.item()
    print(f"Arc parameters:")
    print(f"  A (radius)      = {A:.6f}")
    print(f"  start (radians) = {start:.6f}  ({math.degrees(start):.2f} deg)")
    print(f"  stride (radians)= {stride:.6f}  ({math.degrees(stride):.2f} deg)")

    tok_emb = model._compute_tok_emb().detach()
    print(f"\nToken embeddings (tok_dim=2):")
    for d in range(VOCAB_SIZE):
        angle = start + d * stride
        print(f"  digit {d}: [{tok_emb[d,0]:+.4f}, {tok_emb[d,1]:+.4f}]  "
              f"angle={math.degrees(angle):+.1f}deg  r={torch.norm(tok_emb[d]).item():.4f}")

    subheader("Angular spacing & pairwise distances")
    print("Digit pair  |  Angular sep (deg)  |  Euclidean dist")
    for i in range(VOCAB_SIZE):
        for j in range(i+1, min(i+2, VOCAB_SIZE)):
            ang_sep = math.degrees(stride)
            dist = torch.norm(tok_emb[i] - tok_emb[j]).item()
            print(f"  {i}-{j}        |  {ang_sep:+.2f}             |  {dist:.4f}")

    # Full pairwise distance matrix
    subheader("Pairwise distance matrix (digits 0-9)")
    dists = torch.cdist(tok_emb.unsqueeze(0), tok_emb.unsqueeze(0)).squeeze(0)
    print("     " + "".join(f"  {d:5d}" for d in range(10)))
    for i in range(10):
        row = f"  {i}: " + "".join(f"  {dists[i,j].item():5.2f}" for j in range(10))
        print(row)

    # ================================================================
    # 3. ATTENTION PATTERNS
    # ================================================================
    header("3. ATTENTION PATTERNS")

    # Sample: 1234567890 + 9876543210
    a_val = 1234567890
    b_val = 9876543210
    s_val = a_val + b_val
    inp, tgt = make_example(a_val, b_val)
    inp_t = torch.tensor([inp], dtype=torch.long)

    # Forward with attention capture
    with torch.no_grad():
        B, T = inp_t.shape
        qk_dim = cfg.effective_qk_dim
        tok_emb_table = model._compute_tok_emb()
        tok = tok_emb_table[inp_t]
        pos = model._get_positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)
        h = model.norm1(x)
        tok_h = h[:, :, :cfg.tok_dim]
        qk_in = h[:, :, cfg.tok_dim:]
        Q = model.q_proj(qk_in)
        K = model.q_proj(qk_in)
        V = tok_h @ model.head_proj.weight
        Q = Q.view(B, T, 1, qk_dim).transpose(1, 2)
        K = K.view(B, T, 1, qk_dim).transpose(1, 2)
        V = V.view(B, T, 1, cfg.head_dim).transpose(1, 2)
        Q = model._apply_q_phase(Q)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(qk_dim)
        att = att.masked_fill(model.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att, dim=-1)

    att_np = att_weights[0, 0].numpy()  # (T, T)

    print(f"Input: {a_val} + {b_val} = {s_val}")
    x_digits = encode_number(a_val, MAX_DIGITS)
    y_digits = encode_number(b_val, MAX_DIGITS)
    z_digits = encode_number(s_val, ANSWER_LEN)
    print(f"X (LSB-first): {x_digits}")
    print(f"Y (LSB-first): {y_digits}")
    print(f"Z (LSB-first): {z_digits}")

    # Position labels
    labels = []
    for i in range(10):
        labels.append(f"X{i}={x_digits[i]}")
    labels.append("PLUS")
    for i in range(10):
        labels.append(f"Y{i}={y_digits[i]}")
    labels.append("EQ")
    # inp is length 33, Z runs from pos 22..32 (11 digits), then no EOS in input
    for i in range(11):
        if i < len(z_digits):
            labels.append(f"Z{i}={z_digits[i]}")

    subheader("Attention weights for answer positions (Z0..Z10)")
    print("For each answer position, showing top-5 attended positions:")
    for z_idx in range(ANSWER_LEN):
        seq_pos = Z_START - 1 + z_idx  # -1 because input is shifted
        if seq_pos >= T:
            break
        row = att_np[seq_pos]
        top5 = np.argsort(row)[::-1][:5]
        print(f"\n  Z{z_idx} (seq pos {seq_pos}, label={labels[seq_pos] if seq_pos < len(labels) else '?'}):")
        for t in top5:
            lbl = labels[t] if t < len(labels) else f"pos{t}"
            print(f"    -> {lbl:12s} (pos {t:2d}): {row[t]:.4f}")

    subheader("Carry-lookahead pattern check")
    print("For each Z position, does it attend to the NEXT digit position?")
    print("(+1 lookahead means attending to X_{i+1} and Y_{i+1} for carry detection)")
    for z_idx in range(min(10, ANSWER_LEN)):
        seq_pos = Z_START - 1 + z_idx
        if seq_pos >= T:
            break
        row = att_np[seq_pos]
        # Attention to X_i, Y_i, X_{i+1}, Y_{i+1}
        x_i = min(z_idx, 9)
        y_i = min(z_idx, 9) + 11  # Y_START=11
        attn_xi = row[x_i] if x_i < T else 0
        attn_yi = row[y_i] if y_i < T else 0
        if z_idx + 1 < 10:
            attn_xi1 = row[x_i + 1]
            attn_yi1 = row[y_i + 1]
        else:
            attn_xi1 = 0
            attn_yi1 = 0
        print(f"  Z{z_idx}: X{z_idx}={attn_xi:.4f}, Y{z_idx}={attn_yi:.4f}, "
              f"X{z_idx+1}={attn_xi1:.4f}, Y{z_idx+1}={attn_yi1:.4f}")

    # ================================================================
    # 4. HEAD_PROJ TRIPLE-DUTY ANALYSIS
    # ================================================================
    header("4. HEAD_PROJ TRIPLE-DUTY ANALYSIS")

    hp_w = model.head_proj.weight.detach()  # (2, 5)
    print(f"head_proj.weight shape: {hp_w.shape}")
    print(f"head_proj.weight:\n{hp_w}")
    print(f"\nSingular values: {torch.linalg.svdvals(hp_w).tolist()}")
    print(f"Frobenius norm: {torch.norm(hp_w).item():.4f}")

    subheader("4a. As V projection: tok_h @ head_proj.weight")
    print("V = tok_h @ head_proj.weight  ->  (B,T,2) @ (2,5) = (B,T,5)")
    print("This maps 2D token space to 5D head_dim space for value computation.")
    print("\nWhat each token digit maps to as a V vector:")
    for d in range(VOCAB_SIZE):
        v = tok_emb[d] @ hp_w  # (5,)
        print(f"  digit {d}: V = [{', '.join(f'{x:+.4f}' for x in v.tolist())}]  |V|={torch.norm(v).item():.4f}")

    subheader("4b. As fc2 expansion: ffn_hidden @ head_proj.weight")
    print("fc2(h) = h @ head_proj.weight  ->  (B,T,2) @ (2,5) = (B,T,5)")
    print("Same matrix! FFN expansion writes into the SAME subspace as V values.")
    print("\nhead_proj rows (each row is a basis vector for 2D -> 5D):")
    for i in range(hp_w.shape[0]):
        print(f"  row {i}: [{', '.join(f'{x:+.4f}' for x in hp_w[i].tolist())}]  |row|={torch.norm(hp_w[i]).item():.4f}")

    subheader("4c. As output head: head_proj(x) @ tok_emb.T")
    print("logits = head_proj(norm(x)) @ tok_emb.T")
    print("       = norm(x) @ head_proj.weight.T @ tok_emb.T")
    print("       = norm(x) @ (tok_emb @ head_proj.weight).T")
    print("\nClassification prototypes (tok_emb @ head_proj.weight):")
    protos = tok_emb @ hp_w  # (10, 5)
    for d in range(VOCAB_SIZE):
        print(f"  digit {d}: [{', '.join(f'{x:+.4f}' for x in protos[d].tolist())}]")

    print("\nPairwise cosine similarity of classification prototypes:")
    protos_norm = F.normalize(protos, dim=1)
    cos_sim = protos_norm @ protos_norm.T
    print("     " + "".join(f"  {d:5d}" for d in range(10)))
    for i in range(10):
        row = f"  {i}: " + "".join(f"  {cos_sim[i,j].item():+5.2f}" for j in range(10))
        print(row)

    subheader("4d. Comparison to 67p (conceptual)")
    print("In 67p: fc2 (2->5) and head_proj (5->2) are SEPARATE matrices (10p each).")
    print("In 57p: fc2 IS head_proj.weight, saving 10 params.")
    print("Constraint: FFN expansion and V projection MUST share the same 2D->5D map.")
    print("This means the FFN's output subspace is forced to align with the")
    print("token-to-value mapping, creating a structural coupling between")
    print("the attention pathway and the FFN pathway.")

    # ================================================================
    # 5. FFN ANALYSIS
    # ================================================================
    header("5. FFN ANALYSIS")

    fc1_w = model.fc1.weight.detach()  # (2, 5)
    print(f"fc1.weight shape: {fc1_w.shape}")
    print(f"fc1.weight:\n{fc1_w}")
    print(f"\nSingular values: {torch.linalg.svdvals(fc1_w).tolist()}")

    subheader("fc1 reading subspace")
    print("fc1 reads from 5D -> 2D. Which dimensions does it emphasize?")
    for i in range(fc1_w.shape[0]):
        row = fc1_w[i]
        abs_row = row.abs()
        dominant = torch.argmax(abs_row).item()
        print(f"  neuron {i}: [{', '.join(f'{x:+.4f}' for x in row.tolist())}]  "
              f"dominant_dim={dominant}  |row|={torch.norm(row).item():.4f}")

    subheader("FFN pipeline: fc1 -> GELU -> fc2 (=head_proj.weight)")
    print("fc1.weight (2x5) reads from x, GELU activates, then head_proj.weight (2x5) writes back.")
    print("\nComposite fc2 @ fc1 (linear part, ignoring GELU nonlinearity):")
    composite = hp_w.T @ fc1_w  # (5,2) @ (2,5) = (5,5)
    print(f"  head_proj.weight.T @ fc1.weight = (5x5):\n{composite}")
    print(f"  Eigenvalues: {torch.linalg.eigvals(composite).tolist()}")
    print(f"  Trace: {composite.trace().item():.4f}")

    subheader("fc2=head_proj subspace verification")
    print("Since fc2=head_proj.weight, FFN output lives in span(head_proj rows).")
    print("head_proj row 0 and row 1 span a 2D subspace of R^5.")
    # Check angle between the two rows
    row0 = hp_w[0]
    row1 = hp_w[1]
    cos_angle = F.cosine_similarity(row0.unsqueeze(0), row1.unsqueeze(0)).item()
    print(f"  Cosine between hp rows: {cos_angle:.4f} (angle={math.degrees(math.acos(np.clip(cos_angle,-1,1))):.1f} deg)")
    print(f"  -> {'Nearly orthogonal' if abs(cos_angle) < 0.3 else 'Correlated'} basis for FFN output")

    # ================================================================
    # 6. NORM WEIGHTS
    # ================================================================
    header("6. NORM WEIGHTS")

    norm_w = model.norm1.weight.detach()
    print(f"Shared RMSNorm weight (5 dims): [{', '.join(f'{x:.4f}' for x in norm_w.tolist())}]")
    print(f"\nDimension labels: [tok0, tok1, pos0, pos1, pos2]")
    print(f"  tok dims: [{norm_w[0].item():.4f}, {norm_w[1].item():.4f}]")
    print(f"  pos dims: [{norm_w[2].item():.4f}, {norm_w[3].item():.4f}, {norm_w[4].item():.4f}]")

    ratio_1_0 = norm_w[1].item() / (norm_w[0].item() + 1e-8)
    print(f"\n  dim1/dim0 ratio: {ratio_1_0:.4f}")
    print(f"  max/min ratio: {norm_w.max().item() / (norm_w.min().item() + 1e-8):.4f}")

    print(f"\n67p comparison: dim1/dim0 ~ 1.7x, max/min ~ varies")
    print(f"57p:            dim1/dim0 = {ratio_1_0:.2f}x")

    # ================================================================
    # 7. POSITION ENCODING
    # ================================================================
    header("7. POSITION ENCODING")

    z_hi = model.z_hi_pos.detach().squeeze()
    eq_pos = model.special_pos_equals.detach().squeeze()
    print(f"z_hi_pos:             [{', '.join(f'{x:+.4f}' for x in z_hi.tolist())}]  |z_hi|={torch.norm(z_hi).item():.4f}")
    print(f"special_pos_equals:   [{', '.join(f'{x:+.4f}' for x in eq_pos.tolist())}]  |eq|={torch.norm(eq_pos).item():.4f}")

    subheader("Spiral parameters (all frozen)")
    print(f"  amp   = {model.spiral_amp.item():.4f}")
    print(f"  phase = {model.spiral_phase.item():.4f}")
    print(f"  slope = {model.spiral_slope.item():.4f}")
    print(f"  offset= {model.spiral_offset.item():.4f}")

    subheader("Digit positions (spiral, 10 positions)")
    digit_pos = model._get_digit_positions().detach()
    for i in range(MAX_DIGITS):
        p = digit_pos[i]
        print(f"  pos {i}: [{', '.join(f'{x:+.4f}' for x in p.tolist())}]  |p|={torch.norm(p).item():.4f}")

    print(f"\n67p comparison:")
    print(f"  67p z_hi norm: 16.2, 57p z_hi norm: {torch.norm(z_hi).item():.2f}")

    # ================================================================
    # 8. Q/K AND PHASE
    # ================================================================
    header("8. Q/K AND PHASE ANALYSIS")

    q_proj_w = model.q_proj.weight.detach()  # (4, 3)
    q_phase = model.q_phase_angle.detach()
    print(f"q_proj.weight shape: {q_proj_w.shape}")
    print(f"q_proj.weight:\n{q_proj_w}")
    print(f"\nq_phase_angle: {q_phase.item():.6f} rad = {math.degrees(q_phase.item()):.2f} deg")
    print(f"  cos(phase) = {math.cos(q_phase.item()):.6f}")
    print(f"  sin(phase) = {math.sin(q_phase.item()):.6f}")

    subheader("Q/K projection of digit positions")
    # K = q_proj(pos_dim)  (no phase rotation)
    # Q = phase_rotate(q_proj(pos_dim))
    print("K vectors (no phase):")
    for i in range(MAX_DIGITS):
        k = (digit_pos[i] @ q_proj_w.T)
        print(f"  pos {i}: [{', '.join(f'{x:+.4f}' for x in k.tolist())}]  |k|={torch.norm(k).item():.4f}")

    print("\nQ vectors (with phase rotation):")
    for i in range(MAX_DIGITS):
        q = (digit_pos[i] @ q_proj_w.T)
        # Apply phase rotation to pairs (0,1), (2,3)
        q_rot = q.clone()
        cos_a = math.cos(q_phase.item())
        sin_a = math.sin(q_phase.item())
        q_rot[0] = q[0] * cos_a - q[1] * sin_a
        q_rot[1] = q[0] * sin_a + q[1] * cos_a
        q_rot[2] = q[2] * cos_a - q[3] * sin_a
        q_rot[3] = q[2] * sin_a + q[3] * cos_a
        print(f"  pos {i}: [{', '.join(f'{x:+.4f}' for x in q_rot.tolist())}]  |q|={torch.norm(q_rot).item():.4f}")

    subheader("Effective Q-K dot products (attention routing)")
    print("Q[i] . K[j] matrix (tells us what each position attends to):")
    qs = []
    ks = []
    for i in range(MAX_DIGITS):
        k = digit_pos[i] @ q_proj_w.T
        ks.append(k)
        q = k.clone()
        cos_a = math.cos(q_phase.item())
        sin_a = math.sin(q_phase.item())
        q_rot = q.clone()
        q_rot[0] = q[0] * cos_a - q[1] * sin_a
        q_rot[1] = q[0] * sin_a + q[1] * cos_a
        q_rot[2] = q[2] * cos_a - q[3] * sin_a
        q_rot[3] = q[2] * sin_a + q[3] * cos_a
        qs.append(q_rot)

    qs_t = torch.stack(qs)
    ks_t = torch.stack(ks)
    qk_dots = qs_t @ ks_t.T / math.sqrt(qk_dim)
    print("     " + "".join(f"  K{d:d}   " for d in range(10)))
    for i in range(10):
        row = f"Q{i}: " + "".join(f"  {qk_dots[i,j].item():+5.2f}" for j in range(10))
        print(row)

    # Check +1 lookahead
    print("\nDiagonal vs +1 offset (lookahead check):")
    for i in range(9):
        print(f"  Q{i}.K{i} = {qk_dots[i,i].item():+.3f},  Q{i}.K{i+1} = {qk_dots[i,i+1].item():+.3f}  "
              f"(diff = {qk_dots[i,i+1].item() - qk_dots[i,i].item():+.3f})")

    # ================================================================
    # 9. OUT_PROJ (Rank-1)
    # ================================================================
    header("9. OUT_PROJ (Rank-1)")

    out_A = model.out_proj.A.detach().squeeze()  # (5,)
    out_B = model.out_proj.B.detach().squeeze()  # (5,)
    print(f"out_proj.A (read vector, 5D): [{', '.join(f'{x:+.4f}' for x in out_A.tolist())}]  |A|={torch.norm(out_A).item():.4f}")
    print(f"out_proj.B (write vector, 5D): [{', '.join(f'{x:+.4f}' for x in out_B.tolist())}]  |B|={torch.norm(out_B).item():.4f}")

    full_mat = out_A.unsqueeze(1) @ out_B.unsqueeze(0)  # (5,5)
    print(f"\nRank-1 matrix A @ B.T:\n{full_mat}")
    print(f"\nSingular values: {torch.linalg.svdvals(full_mat).tolist()}")

    # What dimension does it primarily write to?
    abs_B = out_B.abs()
    dominant_write = torch.argmax(abs_B).item()
    print(f"\nDominant write dimension: {dominant_write} (|B[{dominant_write}]| = {abs_B[dominant_write].item():.4f})")
    print(f"Write vector breakdown: tok=[{out_B[0].item():.4f}, {out_B[1].item():.4f}], "
          f"pos=[{out_B[2].item():.4f}, {out_B[3].item():.4f}, {out_B[4].item():.4f}]")

    abs_A = out_A.abs()
    dominant_read = torch.argmax(abs_A).item()
    print(f"\nDominant read dimension: {dominant_read} (|A[{dominant_read}]| = {abs_A[dominant_read].item():.4f})")
    print(f"Read vector breakdown: tok=[{out_A[0].item():.4f}, {out_A[1].item():.4f}], "
          f"pos=[{out_A[2].item():.4f}, {out_A[3].item():.4f}, {out_A[4].item():.4f}]")

    # Effective gain
    gain = (out_A @ out_B).item()  # trace of the rank-1 matrix would be this if aligned
    print(f"\nEffective gain (A.B): {gain:.4f}")
    print(f"67p comparison: out_proj gain ~ 0.58")

    # ================================================================
    # 10. ERROR ANALYSIS
    # ================================================================
    header("10. ERROR ANALYSIS (1000 random examples)")

    rng = random.Random(42)
    n_test = 1000
    correct_per_pos = [0] * ANSWER_LEN
    total_per_pos = [0] * ANSWER_LEN
    exact_correct = 0
    errors = []

    with torch.no_grad():
        for _ in range(n_test):
            a = rng.randint(0, 10**10 - 1)
            b = rng.randint(0, 10**10 - 1)
            s = a + b
            inp, tgt = make_example(a, b)
            inp_t = torch.tensor([inp], dtype=torch.long)
            logits, _ = model(inp_t)

            z_true = encode_number(s, ANSWER_LEN)
            z_pred = []
            all_correct = True
            for zi in range(ANSWER_LEN):
                pos = Z_START - 1 + zi  # shifted input
                pred_digit = logits[0, pos].argmax().item()
                z_pred.append(pred_digit)
                total_per_pos[zi] += 1
                if pred_digit == z_true[zi]:
                    correct_per_pos[zi] += 1
                else:
                    all_correct = False
            if all_correct:
                exact_correct += 1
            else:
                if len(errors) < 10:
                    errors.append((a, b, s, z_true, z_pred))

    print(f"Exact match accuracy: {exact_correct}/{n_test} = {100*exact_correct/n_test:.1f}%")
    print(f"\nPer-position accuracy:")
    for zi in range(ANSWER_LEN):
        acc = correct_per_pos[zi] / total_per_pos[zi] * 100
        bar = '#' * int(acc / 2)
        label = f"Z{zi:2d}" if zi < 10 else "carry"
        print(f"  {label}: {acc:6.2f}% ({correct_per_pos[zi]}/{total_per_pos[zi]})  {bar}")

    if errors:
        print(f"\nFirst {len(errors)} errors:")
        for a, b, s, zt, zp in errors:
            wrong_pos = [i for i in range(ANSWER_LEN) if zt[i] != zp[i]]
            print(f"  {a} + {b} = {s}")
            print(f"    true:  {zt}  pred: {zp}  wrong at: {wrong_pos}")
    else:
        print(f"\nNo errors found in {n_test} samples!")

    # ================================================================
    # SUMMARY
    # ================================================================
    header("SUMMARY")
    print(f"Model: 57-parameter MicroAdder (tie_fc2_head=True)")
    print(f"Parameters: {total}")
    print(f"Key structural difference from 67p: fc2 = head_proj.weight (saves 10p)")
    print(f"  head_proj does triple duty: V projection, fc2 expansion, output head")
    print(f"Accuracy: {exact_correct}/{n_test} ({100*exact_correct/n_test:.1f}%) exact match")
    print(f"q_phase: {math.degrees(q_phase.item()):.1f} deg")
    print(f"Token radius: {A:.4f}")
    print(f"z_hi norm: {torch.norm(z_hi).item():.2f}")
    print(f"out_proj gain: {gain:.4f}")
    print(f"Norm dim1/dim0: {ratio_1_0:.2f}x")


if __name__ == "__main__":
    main()
