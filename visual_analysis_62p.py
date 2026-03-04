"""Visual structural analysis of the 62p spiral-norm model."""

import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from microadder.model import ModelConfig, MicroAdder, count_params
from microadder.data import make_example, PROMPT_LEN, ANSWER_LEN, SEQ_LEN, MAX_DIGITS

CKPT = "results/runs/sub100_62p_qk4_spiralnorm/checkpoints/last.pt"
OUT = "analysis_62p.png"

POS_LABELS = ([f"X{i}" for i in range(10)] + ["Zhi", "+"]
              + [f"Y{i}" for i in range(10)] + ["="]
              + [f"A{i}" for i in range(11)] + ["EOS"])


def load():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = MicroAdder(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def get_attention(model, n=500):
    rng = random.Random(42)
    examples = []
    for _ in range(n):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, _ = make_example(a, b)
        examples.append(inp)
    batch = torch.tensor(examples)
    B, T = batch.shape
    cfg = model.cfg
    qk_dim = cfg.effective_qk_dim

    with torch.no_grad():
        tok_emb_table = model._compute_tok_emb()
        tok = tok_emb_table[batch]
        pos = model._get_positions(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)
        h = model.norm1(x)
        pos_h = h[:, :, cfg.tok_dim:]
        Q = model.q_proj(pos_h).view(B, T, 1, qk_dim).transpose(1, 2)
        K = model.q_proj(pos_h).view(B, T, 1, qk_dim).transpose(1, 2)
        Q = model._apply_q_phase(Q)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(qk_dim)
        att = att.masked_fill(model.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
    return att.mean(0).squeeze(0).numpy()  # (T, T)


def error_analysis(model, n=3000):
    rng = random.Random(42)
    digit_errors = [0] * ANSWER_LEN
    digit_totals = [0] * ANSWER_LEN
    carry_err = {0: 0, 1: 0}
    carry_tot = {0: 0, 1: 0}
    pos_carry_err = {}
    pos_carry_tot = {}
    n_exact = 0
    off_by = [0] * 10

    for _ in range(n):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp, tgt = make_example(a, b)
        with torch.no_grad():
            logits, _ = model(torch.tensor([inp]))
        pred = logits[0].argmax(-1).numpy()
        tgt = np.array(tgt)
        ans_start = 22
        pred_ans = pred[ans_start:ans_start + ANSWER_LEN]
        tgt_ans = tgt[ans_start:ans_start + ANSWER_LEN]

        a_str = str(a).zfill(10)
        b_str = str(b).zfill(10)
        carry = 0
        carries = []
        for i in range(10):
            d_a = int(a_str[9 - i])
            d_b = int(b_str[9 - i])
            total = d_a + d_b + carry
            carries.append(carry)
            carry = total // 10
        carries.append(carry)

        exact = True
        for pos in range(ANSWER_LEN):
            if tgt_ans[pos] >= 0:
                exp, got = tgt_ans[pos], pred_ans[pos]
                c = carries[pos] if pos < len(carries) else 0
                key = (pos, min(c, 1))
                digit_totals[pos] += 1
                carry_tot[min(c, 1)] = carry_tot.get(min(c, 1), 0) + 1
                pos_carry_tot[key] = pos_carry_tot.get(key, 0) + 1
                if got != exp:
                    exact = False
                    digit_errors[pos] += 1
                    carry_err[min(c, 1)] = carry_err.get(min(c, 1), 0) + 1
                    pos_carry_err[key] = pos_carry_err.get(key, 0) + 1
                    off_by[(got - exp) % 10] += 1
        if exact:
            n_exact += 1

    return {
        "n": n, "n_exact": n_exact,
        "digit_errors": digit_errors, "digit_totals": digit_totals,
        "carry_err": carry_err, "carry_tot": carry_tot,
        "pos_carry_err": pos_carry_err, "pos_carry_tot": pos_carry_tot,
        "off_by": off_by,
    }


def plot_all(model, ckpt):
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ── 1. Token embeddings (2D) ──
    ax1 = fig.add_subplot(gs[0, 0])
    tok_emb = model._compute_tok_emb().detach().numpy()
    for i in range(10):
        ax1.annotate(str(i), (tok_emb[i, 0], tok_emb[i, 1]),
                     fontsize=14, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="C0", alpha=0.3))
    ax1.plot(tok_emb[:, 0], tok_emb[:, 1], "o-", alpha=0.4, color="C0")
    ax1.set_title(f"Token Embeddings (2D)\nA={model.tok_arc_A.item():.1f}, "
                  f"stride={math.degrees(model.tok_arc_stride.item()):.1f}°")
    ax1.set_xlabel("dim 0")
    ax1.set_ylabel("dim 1")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ── 2. Positions in Q/K space ──
    ax2 = fig.add_subplot(gs[0, 1])
    cfg = model.cfg
    qk_dim = cfg.effective_qk_dim
    all_pos = model._get_positions(SEQ_LEN).detach().numpy()
    q_w = model.q_proj.weight.detach().numpy()
    all_qk = all_pos @ q_w.T  # (34, qk_dim)

    # Color by group
    colors_map = {"X": "C0", "Z": "C3", "+": "gray", "Y": "C1", "=": "gray", "A": "C2", "E": "gray"}
    for i, label in enumerate(POS_LABELS[:len(all_qk)]):
        c = colors_map.get(label[0], "gray")
        ax2.plot(all_qk[i, 0], all_qk[i, 1], "o", color=c, markersize=5, alpha=0.6)
        ax2.annotate(label, (all_qk[i, 0], all_qk[i, 1]), fontsize=5,
                     ha="center", va="bottom", color=c)
    ax2.set_title(f"Positions in Q/K Space (dims 0,1 of {qk_dim}D)\nBlue=X, Orange=Y, Green=A, Red=Zhi")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # ── 3. Positions in Q/K space dims 2,3 ──
    ax3 = fig.add_subplot(gs[0, 2])
    if qk_dim >= 4:
        for i, label in enumerate(POS_LABELS[:len(all_qk)]):
            c = colors_map.get(label[0], "gray")
            ax3.plot(all_qk[i, 2], all_qk[i, 3], "o", color=c, markersize=5, alpha=0.6)
            ax3.annotate(label, (all_qk[i, 2], all_qk[i, 3]), fontsize=5,
                         ha="center", va="bottom", color=c)
        ax3.set_title(f"Positions in Q/K Space (dims 2,3)")
    else:
        ax3.text(0.5, 0.5, f"qk_dim={qk_dim}\n(only {qk_dim} dims)", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=14)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # ── 4. Attention heatmap (answer rows only) ──
    ax4 = fig.add_subplot(gs[0, 3])
    avg_att = get_attention(model)
    T = avg_att.shape[0]
    ans_att = avg_att[23:min(34, T), :]  # A0-A10 rows
    im = ax4.imshow(ans_att, aspect="auto", cmap="hot", vmin=0, vmax=0.5)
    ax4.set_yticks(range(11))
    ax4.set_yticklabels([f"A{i}" for i in range(11)])
    # x-axis: show selected labels
    tick_pos = list(range(0, len(POS_LABELS), 2))
    ax4.set_xticks(tick_pos)
    ax4.set_xticklabels([POS_LABELS[i] for i in tick_pos], rotation=90, fontsize=7)
    ax4.set_title("Avg Attention (answer → all)")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # ── 5. Attention offset pattern ──
    ax5 = fig.add_subplot(gs[1, 0])
    T = avg_att.shape[0]
    for a_i in range(min(11, T - 23)):
        a_idx = 23 + a_i
        if a_idx >= T:
            break
        row = avg_att[a_idx, :a_idx + 1]
        top3 = np.argsort(row)[::-1][:3]
        for rank, j in enumerate(top3):
            marker = "o" if rank == 0 else ("s" if rank == 1 else "^")
            alpha = max(0.2, row[j])
            c = "C3" if POS_LABELS[j].startswith("Z") else ("C2" if POS_LABELS[j].startswith("A") else "C0")
            ax5.scatter(a_i, j, s=row[j] * 300, marker=marker, color=c, alpha=alpha)
            if row[j] > 0.1:
                ax5.annotate(f"{POS_LABELS[j]}\n{row[j]:.0%}", (a_i, j),
                             fontsize=6, ha="center", va="bottom")
    ax5.set_xlabel("Answer position (A_i)")
    ax5.set_ylabel("Attended position")
    ax5.set_yticks(range(0, 34, 2))
    ax5.set_yticklabels([POS_LABELS[i] for i in range(0, 34, 2)], fontsize=7)
    ax5.set_title("Top-3 Attention Sources per Answer")
    ax5.grid(True, alpha=0.2)

    # ── 6. Error rate per position ──
    ax6 = fig.add_subplot(gs[1, 1])
    errs = error_analysis(model)
    rates = [errs["digit_errors"][i] / max(1, errs["digit_totals"][i]) * 100
             for i in range(ANSWER_LEN)]
    colors = ["C3" if r > 5 else ("C1" if r > 1 else "C2") for r in rates]
    ax6.bar(range(ANSWER_LEN), rates, color=colors)
    ax6.set_xticks(range(ANSWER_LEN))
    ax6.set_xticklabels([f"A{i}" for i in range(ANSWER_LEN)])
    ax6.set_ylabel("Error rate (%)")
    ax6.set_title(f"Per-Position Error Rate\nExact: {errs['n_exact']}/{errs['n']} "
                  f"({errs['n_exact']/errs['n']:.1%})")
    for i, r in enumerate(rates):
        if r > 0.5:
            ax6.text(i, r + 0.3, f"{r:.1f}%", ha="center", fontsize=8)
    ax6.grid(True, alpha=0.3, axis="y")

    # ── 7. Error by position + carry ──
    ax7 = fig.add_subplot(gs[1, 2])
    x_pos = np.arange(ANSWER_LEN)
    width = 0.35
    rates_c0 = []
    rates_c1 = []
    for pos in range(ANSWER_LEN):
        t0 = errs["pos_carry_tot"].get((pos, 0), 0)
        e0 = errs["pos_carry_err"].get((pos, 0), 0)
        t1 = errs["pos_carry_tot"].get((pos, 1), 0)
        e1 = errs["pos_carry_err"].get((pos, 1), 0)
        rates_c0.append(e0 / t0 * 100 if t0 > 0 else 0)
        rates_c1.append(e1 / t1 * 100 if t1 > 0 else 0)
    ax7.bar(x_pos - width/2, rates_c0, width, label="carry=0", color="C0", alpha=0.7)
    ax7.bar(x_pos + width/2, rates_c1, width, label="carry=1", color="C3", alpha=0.7)
    ax7.set_xticks(range(ANSWER_LEN))
    ax7.set_xticklabels([f"A{i}" for i in range(ANSWER_LEN)])
    ax7.set_ylabel("Error rate (%)")
    ax7.set_title("Error by Position + Carry")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis="y")

    # ── 8. Off-by histogram ──
    ax8 = fig.add_subplot(gs[1, 3])
    total_errs = sum(errs["off_by"])
    if total_errs > 0:
        pcts = [errs["off_by"][i] / total_errs * 100 for i in range(10)]
        ax8.bar(range(10), pcts, color="C1")
        for i, p in enumerate(pcts):
            if p > 2:
                ax8.text(i, p + 0.5, f"{p:.0f}%", ha="center", fontsize=8)
    ax8.set_xlabel("(predicted - target) mod 10")
    ax8.set_ylabel("% of errors")
    ax8.set_title(f"Error Direction (total errors: {total_errs})")
    ax8.set_xticks(range(10))
    ax8.set_xticklabels([f"+{i}" for i in range(10)])
    ax8.grid(True, alpha=0.3, axis="y")

    # ── 9. A·V scores (rank-1 readout) ──
    ax9 = fig.add_subplot(gs[2, 0])
    tok_emb = model._compute_tok_emb().detach().numpy()
    hp = model.head_proj.weight.detach().numpy()
    A = model.out_proj.A.detach().numpy().flatten()
    B = model.out_proj.B.detach().numpy().flatten()
    V_per_digit = tok_emb @ hp
    scores = [V_per_digit[d] @ A for d in range(10)]
    ax9.bar(range(10), scores, color="C0")
    ax9.set_xlabel("Digit")
    ax9.set_ylabel("A·V score")
    ax9.set_title("Rank-1 Readout (A·V per digit)\nShould be monotonic for clean digit separation")
    ax9.grid(True, alpha=0.3, axis="y")

    # ── 10. out_proj directions ──
    ax10 = fig.add_subplot(gs[2, 1])
    dims = ["tok0", "tok1", "pos0", "pos1", "pos2"]
    x_d = np.arange(5)
    ax10.bar(x_d - 0.15, np.abs(A), 0.3, label="|A| (read)", color="C0", alpha=0.7)
    ax10.bar(x_d + 0.15, np.abs(B), 0.3, label="|B| (write)", color="C1", alpha=0.7)
    ax10.set_xticks(x_d)
    ax10.set_xticklabels(dims)
    ax10.set_title("out_proj: |A| (read from V) vs |B| (write to residual)")
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3, axis="y")

    # ── 11. Norm weights comparison ──
    ax11 = fig.add_subplot(gs[2, 2])
    spiral_w = model.norm1.weight.detach().numpy()
    learned_67 = np.array([0.74, 2.83, 4.00, 4.01, 5.10])
    ax11.bar(x_d - 0.15, spiral_w, 0.3, label="Spiral norm (frozen)", color="C0", alpha=0.7)
    ax11.bar(x_d + 0.15, learned_67, 0.3, label="67p learned", color="C2", alpha=0.7)
    ax11.set_xticks(x_d)
    ax11.set_xticklabels(dims)
    ax11.set_title("Norm Weights: Spiral vs 67p Learned")
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3, axis="y")

    # ── 12. Training trajectory ──
    ax12 = fig.add_subplot(gs[2, 3])
    import json
    log_path = "results/runs/sub100_62p_qk4_spiralnorm/log.jsonl"
    with open(log_path) as f:
        log = [json.loads(l) for l in f]
    steps = [l["step"] for l in log]
    exact = [l["val_exact"] for l in log]
    tok = [l["val_tok_acc"] for l in log]
    ax12.plot(steps, exact, "o-", label="Exact match", color="C0", markersize=3)
    ax12.plot(steps, tok, "s-", label="Token acc", color="C2", markersize=3)
    ax12.set_xlabel("Step")
    ax12.set_ylabel("Accuracy")
    ax12.set_title("Training Trajectory")
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)
    ax12.set_ylim(0, 1.05)

    fig.suptitle(f"62p Spiral Norm Analysis (step {ckpt['step']}, "
                 f"exact={ckpt['metrics']['exact_match']:.1%})",
                 fontsize=16, fontweight="bold")
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    model, ckpt = load()
    print(f"Model: {count_params(model)}p, step {ckpt['step']}")
    plot_all(model, ckpt)
