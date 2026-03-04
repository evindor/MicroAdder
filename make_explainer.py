"""Generate comprehensive visual explainer for the 62-parameter MicroAdder."""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import torch

# ── Load checkpoint ──────────────────────────────────────────────────────
ckpt = torch.load(
    'results/runs/sub100_62p_qk4_spiralnorm/checkpoints/best.pt',
    map_location='cpu', weights_only=False,
)
sd = ckpt['model_state_dict']

# Extract actual weight values
tok_A = sd['tok_arc_A'].item()
tok_start = sd['tok_arc_start'].item()
tok_stride = sd['tok_arc_stride'].item()
z_hi = sd['z_hi_pos'].flatten().numpy()
eq_pos = sd['special_pos_equals'].flatten().numpy()
q_phase = sd['q_phase_angle'].item()
q_proj_w = sd['q_proj.weight'].numpy()  # (4,3)
out_A = sd['out_proj.A'].flatten().numpy()  # (5,)
out_B = sd['out_proj.B'].flatten().numpy()  # (5,)
fc1_w = sd['fc1.weight'].numpy()  # (2,5)
fc2_w = sd['fc2.weight'].numpy()  # (5,2)
head_w = sd['head_proj.weight'].numpy()  # (2,5)
norm_w = sd['norm1.weight'].numpy()  # (5,)
spiral_amp = sd['spiral_amp'].item()
spiral_phase = sd['spiral_phase'].item()
spiral_slope = sd['spiral_slope'].item()
spiral_offset = sd['spiral_offset'].item()


# ── Style ────────────────────────────────────────────────────────────────
BG = '#0f1318'
FG = '#d0d7de'
ACCENT = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#7ee787'
ACCENT4 = '#d2a8ff'
ACCENT5 = '#ff7b72'
GRID = '#21262d'
BOX_BG = '#161b22'
BOX_EDGE = '#30363d'
MUTED = '#8b949e'
PANEL_TITLE_SIZE = 17
LABEL_SIZE = 11.5
SMALL_SIZE = 10
TINY_SIZE = 8.5

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'axes.edgecolor': BOX_EDGE,
    'axes.labelcolor': FG,
    'text.color': FG,
    'xtick.color': FG,
    'ytick.color': FG,
    'grid.color': GRID,
    'font.family': 'monospace',
})

# Digit colors (0-9)
DCOLS = plt.cm.hsv(np.linspace(0, 0.85, 10))


# ── Create figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 26))

gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.35,
                        left=0.04, right=0.96, top=0.925, bottom=0.03,
                        height_ratios=[0.85, 1.2, 1.3, 0.85])


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 1: THE BIG PICTURE (top, full width)
# ═══════════════════════════════════════════════════════════════════════════
ax0 = fig.add_subplot(gs[0, :])
ax0.set_xlim(0, 100)
ax0.set_ylim(0, 18)
ax0.axis('off')
ax0.set_title('THE BIG PICTURE:  Data Flow Through the 62-Parameter Transformer',
              fontsize=20, fontweight='bold', color=ACCENT, pad=8)

def draw_box(ax, x, y, w, h, label, sublabel='', color=BOX_EDGE, fc=BOX_BG, fontsize=10.5, lw=1.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                          facecolor=fc, edgecolor=color, linewidth=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2 + (0.35 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold',
            color=FG, zorder=3)
    if sublabel:
        ax.text(x+w/2, y+h/2 - 0.55, sublabel,
                ha='center', va='center', fontsize=TINY_SIZE, color=MUTED, zorder=3)

def draw_arrow_h(ax, x1, x2, y, color=FG, lw=1.3):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)

# Input example
ax0.text(2, 16, 'Input:  1 2 3 4 5 6 7 8 9 0  +  9 8 7 6 5 4 3 2 1 0  =  ? ? ? ? ? ? ? ? ? ? ?',
         fontsize=12, fontweight='bold', color=FG, family='monospace',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#1c2128', edgecolor=ACCENT2, lw=1.3))

# Row 1: Embed → concat → Norm → Attention → Rank-1 → +residual
y1 = 8.5
bh = 3.2

draw_box(ax0, 1, y1, 9.5, bh, 'Token Embed', '3p (arc on circle)', color=ACCENT3, fc='#0d2818')
draw_arrow_h(ax0, 7, 7, 14.5, ACCENT3)

ax0.text(11.5, y1+bh/2, '+', fontsize=15, fontweight='bold', color=FG, ha='center', va='center')

draw_box(ax0, 13, y1, 9.5, bh, 'Pos Encode', '0p (frozen spiral)', color=ACCENT4, fc='#1a0d2e')
draw_arrow_h(ax0, 18, 18, 14.5, ACCENT4)

ax0.text(23.5, y1+bh/2, '→', fontsize=15, fontweight='bold', color=FG, ha='center', va='center')

draw_box(ax0, 25, y1, 7.5, bh, '5D Vector', 'tok(2)|pos(3)', color=ACCENT, fc='#0d1f30')

draw_arrow_h(ax0, 32.5, 34, y1+bh/2, FG)

draw_box(ax0, 34, y1, 6.5, bh, 'Norm₁', '0p (spiral)', color='#ffa657', fc='#2d1a00')
draw_arrow_h(ax0, 40.5, 42, y1+bh/2, FG)

draw_box(ax0, 42, y1, 9, bh, 'Attention', 'Q,K:pos  V:tok', color=ACCENT5, fc='#2d0d0d')
draw_arrow_h(ax0, 51, 52.5, y1+bh/2, FG)

draw_box(ax0, 52.5, y1, 8, bh, 'Rank-1 Proj', '10p (A·V → ×B)', color=ACCENT2, fc='#2d1500')

# +residual
ax0.text(61.5, y1+bh/2, '+', fontsize=13, fontweight='bold', color=ACCENT, ha='center', va='center')
ax0.plot([28.75, 28.75, 61.5, 61.5], [y1, y1-1.2, y1-1.2, y1+0.2], color=ACCENT, lw=1, ls='--', alpha=0.7)
ax0.text(45, y1-1.8, 'residual', fontsize=TINY_SIZE, color=ACCENT, fontstyle='italic', ha='center')

# Row 2: → Norm → FFN → +residual → Norm → Head → Logits
y2 = 2.5
# Connect row 1 to row 2 with an L-shaped path
ax0.plot([61.5, 63, 63, 4, 4], [y1+bh/2, y1+bh/2, y2+bh/2, y2+bh/2, y2+bh/2],
         '-', color=FG, lw=1.3)
ax0.annotate('', xy=(4, y2+bh/2), xytext=(3, y2+bh/2),
             arrowprops=dict(arrowstyle='->', color=FG, lw=1.3))

draw_box(ax0, 4, y2, 6.5, bh, 'Norm₂', '(shared)', color='#ffa657', fc='#2d1a00')
draw_arrow_h(ax0, 10.5, 12, y2+bh/2, FG)

draw_box(ax0, 12, y2, 9, bh, 'FFN', '20p (GELU, dim=2)', color=ACCENT3, fc='#0d2818')

# +residual
ax0.text(22, y2+bh/2, '+', fontsize=13, fontweight='bold', color=ACCENT, ha='center', va='center')

draw_arrow_h(ax0, 23, 24.5, y2+bh/2, FG)

draw_box(ax0, 24.5, y2, 6.5, bh, 'Norm_f', '(shared)', color='#ffa657', fc='#2d1a00')
draw_arrow_h(ax0, 31, 32.5, y2+bh/2, FG)

draw_box(ax0, 32.5, y2, 9, bh, 'Output Head', '10p (= V proj)', color=ACCENT4, fc='#1a0d2e')
draw_arrow_h(ax0, 41.5, 43, y2+bh/2, FG)

ax0.text(43.5, y2+bh/2, '@ tok_emb.T →', fontsize=SMALL_SIZE, color=MUTED, va='center')

draw_box(ax0, 53, y2, 8, bh, 'Logits', '→ argmax', color=ACCENT, fc='#0d1f30')

# Output
ax0.text(63, y2+bh/2, '→', fontsize=14, fontweight='bold', color=ACCENT3, ha='center', va='center')
ax0.text(65, y2+bh/2, '1 1 1 1 1 1 1 1 0 1 0',
         fontsize=11.5, fontweight='bold', color=ACCENT3, va='center', family='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d2818', edgecolor=ACCENT3, lw=1))

# Param summary
ax0.text(83, 10, '62 total params\n─────────────\n'
         'Embeddings:  6p\nAttention:  23p\nFFN:        20p\nOutput:     10p\nNorm:        0p\nPositions:   0p',
         fontsize=SMALL_SIZE, color=FG, va='center', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=BOX_BG, edgecolor=BOX_EDGE, lw=1))


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 2: TOKEN EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[1, 0:2])
ax1.set_title('Token Embeddings: 10 Digits on a Circular Arc\n3 learnable params: amplitude A, start angle, stride angle',
              fontsize=PANEL_TITLE_SIZE-2, fontweight='bold', color=ACCENT3, pad=8)

# Compute digit positions on circle
digits = np.arange(10)
angles = tok_start + digits * tok_stride
xs = tok_A * np.cos(angles)
ys = tok_A * np.sin(angles)

# Draw the full arc
arc_t = np.linspace(tok_start - 0.05, tok_start + 9*tok_stride + 0.05, 200)
ax1.plot(tok_A*np.cos(arc_t), tok_A*np.sin(arc_t),
         color=ACCENT3, lw=2, ls='--', alpha=0.25, zorder=1)

# Draw the full circle as reference
circle_t = np.linspace(0, 2*np.pi, 200)
ax1.plot(tok_A*np.cos(circle_t), tok_A*np.sin(circle_t),
         color='#21262d', lw=1, ls=':', alpha=0.35, zorder=0)

# Connect consecutive digits with lines
for d in range(9):
    ax1.plot([xs[d], xs[d+1]], [ys[d], ys[d+1]],
             color=ACCENT3, lw=1, alpha=0.3, zorder=2)

# Plot digits
for d in range(10):
    ax1.scatter(xs[d], ys[d], c=[DCOLS[d]], s=400, zorder=5,
                edgecolors='white', linewidths=1.5)
    ax1.text(xs[d], ys[d], str(d), ha='center', va='center',
             fontsize=14, fontweight='bold', color='white', zorder=6)

# Annotation box
info = (f'A (radius)  = {tok_A:.2f}\n'
        f'start angle = {np.degrees(tok_start):.1f} deg\n'
        f'stride      = {np.degrees(tok_stride):.1f} deg\n\n'
        'Digits are equally spaced along\n'
        'a circular arc in 2D.\n\n'
        'SAME embedding table used for:\n'
        '  - Input token lookup\n'
        '  - Output logit computation\n'
        '    (logits = h @ emb_table.T)')
ax1.text(0.03, 0.03, info,
         transform=ax1.transAxes, fontsize=SMALL_SIZE-0.5, color=FG,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor=BOX_BG, edgecolor=ACCENT3, lw=1))

# Draw angle annotation
ax1.annotate('', xy=(xs[0], ys[0]), xytext=(0, 0),
             arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, ls='--'))

pad = 3
ax1.set_xlim(-tok_A-pad, tok_A+pad)
ax1.set_ylim(-tok_A-pad, tok_A+pad)
ax1.set_aspect('equal')
ax1.set_xlabel('dim 0', fontsize=LABEL_SIZE)
ax1.set_ylabel('dim 1', fontsize=LABEL_SIZE)
ax1.grid(True, alpha=0.12)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 3: POSITION ENCODINGS (3D spiral)
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 2:4], projection='3d')
ax2.set_facecolor('#0a0e14')
ax2.set_title('Position Encodings: 3D Spiral\n0 learnable params (all frozen at init values)',
              fontsize=PANEL_TITLE_SIZE-2, fontweight='bold', color=ACCENT4, pad=12)

# Compute digit positions on spiral
d_idx = np.arange(10)
sp_angle = 2*np.pi*d_idx/10.0 + spiral_phase
sp_x = spiral_amp * np.cos(sp_angle)
sp_y = spiral_amp * np.sin(sp_angle)
sp_z = spiral_slope * d_idx + spiral_offset

# Smooth spiral
t_smooth = np.linspace(0, 9, 300)
a_smooth = 2*np.pi*t_smooth/10.0 + spiral_phase
ax2.plot(spiral_amp*np.cos(a_smooth), spiral_amp*np.sin(a_smooth),
         spiral_slope*t_smooth + spiral_offset,
         color=ACCENT4, lw=2, alpha=0.3, zorder=1)

# Plot digit positions
for d in range(10):
    ax2.scatter(sp_x[d], sp_y[d], sp_z[d], c=[DCOLS[d]], s=250, zorder=5,
                edgecolors='white', linewidths=1.2, depthshade=False)
    offset_r = 1.5
    ax2.text(sp_x[d]+offset_r*np.cos(sp_angle[d]),
             sp_y[d]+offset_r*np.sin(sp_angle[d]),
             sp_z[d], str(d),
             fontsize=13, fontweight='bold', color=FG, zorder=6)

# Plot z_hi (carry) position
ax2.scatter([z_hi[0]], [z_hi[1]], [z_hi[2]], c=[ACCENT5], s=350, marker='*',
            zorder=5, depthshade=False)
ax2.text(z_hi[0]+1.5, z_hi[1], z_hi[2]+0.2, 'CARRY (z_hi)\n3 learned params',
         fontsize=SMALL_SIZE-1, fontweight='bold', color=ACCENT5, zorder=6)

# Plot EQUALS position
ax2.scatter([eq_pos[0]], [eq_pos[1]], [eq_pos[2]], c=[ACCENT], s=200, marker='D',
            zorder=5, depthshade=False)
ax2.text(eq_pos[0]+1, eq_pos[1]+1, eq_pos[2], 'EQUALS\n3 learned params',
         fontsize=SMALL_SIZE-1, fontweight='bold', color=ACCENT, zorder=6)

# Style 3D axes
ax2.set_xlabel('dim 0 (cos)', fontsize=SMALL_SIZE-1, color=FG, labelpad=2)
ax2.set_ylabel('dim 1 (sin)', fontsize=SMALL_SIZE-1, color=FG, labelpad=2)
ax2.set_zlabel('dim 2 (linear ramp)', fontsize=SMALL_SIZE-1, color=FG, labelpad=2)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('#1a1f26')
ax2.yaxis.pane.set_edgecolor('#1a1f26')
ax2.zaxis.pane.set_edgecolor('#1a1f26')
ax2.tick_params(colors='#484f58', labelsize=7)
ax2.view_init(elev=22, azim=-55)

# Key insight
ax2.text2D(0.02, 0.02,
           'X₀..X₉ share positions with Y₀..Y₉ and A₀..A₉\n'
           'Same digit = same spiral point!\n'
           f'spiral: amp={spiral_amp}, slope={spiral_slope}\n'
           'PLUS & EOS = zero vector (frozen)',
           transform=ax2.transAxes, fontsize=TINY_SIZE, color=FG,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round,pad=0.35', facecolor=BOX_BG, edgecolor=ACCENT4, lw=1))


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 4: THE NORM TRICK
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 4:6])
ax3.set_title('The Norm Trick: Adaptive Weights at 0 Extra Cost\nReuses z_hi_pos to boost position dimensions',
              fontsize=PANEL_TITLE_SIZE-2, fontweight='bold', color='#ffa657', pad=8)

dims = np.arange(5)
dim_labels = ['tok₀', 'tok₁', 'pos₀', 'pos₁', 'pos₂']
base_w = spiral_amp * np.sin(2*np.pi*dims/10.0) + 1.0

# Boost from z_hi reuse
z_abs = np.abs(z_hi)
boost_raw = z_abs / (z_abs.sum() + 1.0) * spiral_amp
boost = np.zeros(5)
boost[2:] = boost_raw
final_w = base_w + boost

# 67p learned norm for comparison
learned_67p = norm_w

bar_width = 0.22
x = np.arange(5)

bars1 = ax3.bar(x - bar_width, base_w, bar_width, label='Frozen base: amp·sin(2πd/10)+1',
                color='#484f58', edgecolor='#6e7681', linewidth=0.8)
bars2 = ax3.bar(x, final_w, bar_width, label='+ z_hi boost (0 extra params!)',
                color='#ffa657', edgecolor='#d29922', linewidth=0.8)
bars3 = ax3.bar(x + bar_width, learned_67p, bar_width, label='67p learned norm (5 params)',
                color=ACCENT4, edgecolor='#bc8cff', linewidth=0.8, alpha=0.65)

# Value labels on the orange bars
for i in range(5):
    ax3.text(x[i], final_w[i]+0.08, f'{final_w[i]:.2f}', ha='center', va='bottom',
             fontsize=TINY_SIZE-1, color='#ffa657', fontweight='bold')

ax3.set_xticks(x)
ax3.set_xticklabels(dim_labels, fontsize=LABEL_SIZE)
ax3.set_ylabel('Norm Weight', fontsize=LABEL_SIZE)
ax3.legend(fontsize=SMALL_SIZE-1.5, loc='upper left',
           facecolor=BOX_BG, edgecolor=BOX_EDGE, labelcolor=FG)
ax3.grid(True, alpha=0.12, axis='y')

# Formula annotation
formula_text = (
    'FORMULA\n'
    '──────────────────────\n'
    'base[d] = 3.5 · sin(2πd/10) + 1\n\n'
    'z_hi_pos is already learned (3p)\n'
    'for carry detection. Reuse it:\n\n'
    'boost = |z_hi| / (Σ|z_hi| + 1) · 3.5\n'
    'w[pos_dims] += boost\n\n'
    '→ Position dims get amplified\n'
    '→ Matches what the 67p model\n'
    '   learns with 5 free params!'
)
ax3.text(0.98, 0.98, formula_text,
         transform=ax3.transAxes, fontsize=SMALL_SIZE-1, color=FG,
         verticalalignment='top', ha='right', family='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor=BOX_BG, edgecolor='#ffa657', lw=1))


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 5: ATTENTION ROUTING
# ═══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[2, 0:4])
ax4.set_title('Attention Routing: The Phase Trick Creates +1 / +2 Lookahead\n'
              'q_proj maps 3D positions → 4D (two rotation planes). q_phase rotates Q but not K.',
              fontsize=PANEL_TITLE_SIZE-1, fontweight='bold', color=ACCENT5, pad=8)
ax4.axis('off')
ax4.set_xlim(-1, 100)
ax4.set_ylim(-3, 24)

# Draw three rows: X, Y, Answer
y_x = 20
y_y = 14.5
y_a = 5
bsz = 2.3
gap = 0.18

# Helper to draw a row of boxes
def draw_row(ax, y, n, prefix, color, fc):
    for i in range(n):
        bx = 3 + i*(bsz+gap)
        rect = FancyBboxPatch((bx, y), bsz, bsz, boxstyle="round,pad=0.08",
                               facecolor=fc, edgecolor=color, lw=1.2)
        ax.add_patch(rect)
        ax.text(bx+bsz/2, y+bsz/2, f'{prefix}{i}', ha='center', va='center',
                fontsize=SMALL_SIZE-0.5, fontweight='bold', color=color)

# Row labels
ax4.text(1.5, y_x+bsz/2, 'X:', fontsize=LABEL_SIZE, fontweight='bold', color=ACCENT3, ha='right', va='center')
ax4.text(1.5, y_y+bsz/2, 'Y:', fontsize=LABEL_SIZE, fontweight='bold', color=ACCENT4, ha='right', va='center')
ax4.text(1.5, y_a+bsz/2, 'A:', fontsize=LABEL_SIZE, fontweight='bold', color=ACCENT, ha='right', va='center')

draw_row(ax4, y_x, 10, 'X', ACCENT3, '#0d2818')
draw_row(ax4, y_y, 10, 'Y', ACCENT4, '#1a0d2e')
draw_row(ax4, y_a, 10, 'A', ACCENT, '#0d1f30')

# CARRY box
bx_c = 3 + 10*(bsz+gap) + 1.5
rect = FancyBboxPatch((bx_c, (y_y + y_x)/2 - 0.5), 4, 3,
                       boxstyle="round,pad=0.15",
                       facecolor='#2d0d0d', edgecolor=ACCENT5, lw=2)
ax4.add_patch(rect)
ax4.text(bx_c+2, (y_y+y_x)/2+1, 'CARRY', ha='center', va='center',
         fontsize=LABEL_SIZE, fontweight='bold', color=ACCENT5)

# Draw a few representative attention arrows
# A_i reads Y_{i+1} and X_{i+2}
examples = [0, 3, 6]  # show a subset to avoid clutter
for i in examples:
    a_cx = 3 + i*(bsz+gap) + bsz/2
    a_top = y_a + bsz

    # A_i → Y_{i+1}
    if i + 1 < 10:
        y_cx = 3 + (i+1)*(bsz+gap) + bsz/2
        ax4.annotate('', xy=(y_cx, y_y), xytext=(a_cx, a_top),
                     arrowprops=dict(arrowstyle='->', color=ACCENT4, lw=1.8,
                                     connectionstyle='arc3,rad=0.12', alpha=0.8))

    # A_i → X_{i+2}
    if i + 2 < 10:
        x_cx = 3 + (i+2)*(bsz+gap) + bsz/2
        ax4.annotate('', xy=(x_cx, y_x), xytext=(a_cx, a_top),
                     arrowprops=dict(arrowstyle='->', color=ACCENT3, lw=1.8,
                                     connectionstyle='arc3,rad=0.15', alpha=0.8))

# Label the pattern on one example
a0_cx = 3 + 0*(bsz+gap) + bsz/2
ax4.text(3 + 1*(bsz+gap) + bsz/2, y_y - 0.8, 'Y₁', fontsize=TINY_SIZE, color=ACCENT4, ha='center', fontweight='bold')
ax4.text(3 + 2*(bsz+gap) + bsz/2, y_x + bsz + 0.3, 'X₂', fontsize=TINY_SIZE, color=ACCENT3, ha='center', fontweight='bold')

# Special: A8 → CARRY
a8_cx = 3 + 8*(bsz+gap) + bsz/2
ax4.annotate('', xy=(bx_c+2, (y_y+y_x)/2-0.5), xytext=(a8_cx, a_top),
             arrowprops=dict(arrowstyle='->', color=ACCENT5, lw=2.5,
                             connectionstyle='arc3,rad=-0.2'))

# A8 label
ax4.text(a8_cx+3, 9, 'A₈ reads CARRY\n(the 10th output digit\nneeds the carry bit!)',
         fontsize=SMALL_SIZE, fontweight='bold', color=ACCENT5,
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#2d0d0d', edgecolor=ACCENT5, lw=1))

# Legend
ly = 1.2
ax4.plot([3, 5], [ly, ly], color=ACCENT3, lw=2)
ax4.text(5.5, ly, 'A_i reads X at position i+2', fontsize=SMALL_SIZE, color=ACCENT3, va='center')
ax4.plot([22, 24], [ly, ly], color=ACCENT4, lw=2)
ax4.text(24.5, ly, 'A_i reads Y at position i+1', fontsize=SMALL_SIZE, color=ACCENT4, va='center')
ax4.plot([44, 46], [ly, ly], color=ACCENT5, lw=2.5)
ax4.text(46.5, ly, 'A₈ reads carry flag', fontsize=SMALL_SIZE, color=ACCENT5, va='center')

# Why explanation
explanation = (
    'WHY qk_dim=4 IS THE MINIMUM\n'
    '────────────────────────────────────\n'
    'Positions live in 3D → projected to 4D by q_proj\n\n'
    '4D = two rotation planes: (d₀,d₁) and (d₂,d₃)\n\n'
    'q_phase rotates Q in BOTH planes at once:\n'
    f'  angle = {np.degrees(q_phase):.1f}° ≈ 23.4°\n\n'
    'Plane 1: 36°/step on spiral → phase shifts by +2 steps\n'
    '  → A₀ lines up with X₂, A₁ with X₃, ...\n\n'
    'Plane 2: different rotation → +1 step offset\n'
    '  → A₀ lines up with Y₁, A₁ with Y₂, ...\n\n'
    'With qk_dim=3 you only get ONE rotation plane\n'
    '→ cannot independently control X vs Y routing.\n'
    'qk_dim=4 is the sweet spot!'
)
ax4.text(58, 23.5, explanation, fontsize=SMALL_SIZE-0.5, color=FG,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=BOX_BG, edgecolor=ACCENT5, lw=1))


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 6: RANK-1 BOTTLENECK
# ═══════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, 4:6])
ax5.set_title('The Rank-1 Output Bottleneck\n10 params: A(5) + B(5) vectors',
              fontsize=PANEL_TITLE_SIZE-2, fontweight='bold', color=ACCENT2, pad=8)
ax5.axis('off')
ax5.set_xlim(-4, 22)
ax5.set_ylim(-1, 24)

# Diagram of rank-1 decomposition
# Step 1: V (5D) → dot with A → scalar
# Step 2: scalar × B → 5D output

y_top = 22

# Flow diagram
ax5.text(5, y_top, 'Attention output V (5D per token)', fontsize=LABEL_SIZE,
         color=FG, fontweight='bold', ha='center')

# Arrow down
ax5.annotate('', xy=(5, y_top-2.5), xytext=(5, y_top-0.5),
             arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1.5))

# A vector visualization
y_a_vec = 16
ax5.text(5, y_a_vec+2.2, 'Step 1: score = V · A', fontsize=LABEL_SIZE,
         color=ACCENT2, fontweight='bold', ha='center')

dim_labels_5d = ['d₀', 'd₁', 'd₂', 'd₃', 'd₄']
for i, val in enumerate(out_A):
    bar_len = abs(val) * 2.5
    color = ACCENT5 if val < 0 else ACCENT3
    x0 = 0 if val >= 0 else -bar_len
    ax5.barh(y_a_vec - i*0.85, bar_len, height=0.6,
             left=x0, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax5.text(-2.5, y_a_vec - i*0.85, dim_labels_5d[i], fontsize=TINY_SIZE,
             color=MUTED, ha='right', va='center')
    ax5.text(bar_len + 0.3 if val >= 0 else -bar_len - 0.3,
             y_a_vec - i*0.85, f'{val:.2f}',
             fontsize=TINY_SIZE, color=FG, ha='left' if val >= 0 else 'right', va='center')

# Note about d4 dominance
ax5.annotate('d₄ dominates!\n(−2.75)',
             xy=(out_A[4]*2.5, y_a_vec - 4*0.85),
             xytext=(10, y_a_vec - 3*0.85),
             fontsize=TINY_SIZE, color=ACCENT5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=ACCENT5, lw=1))

# Arrow: scalar
ax5.annotate('', xy=(5, y_a_vec - 5.5), xytext=(5, y_a_vec - 4.5),
             arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1.5))
ax5.text(5, y_a_vec - 5.2, 'scalar score', fontsize=SMALL_SIZE, color=ACCENT2,
         ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#2d1500', edgecolor=ACCENT2))

# Arrow: multiply by B
ax5.annotate('', xy=(5, y_a_vec - 8.5), xytext=(5, y_a_vec - 6.5),
             arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1.5))
ax5.text(7, y_a_vec - 7.5, '× B', fontsize=LABEL_SIZE, color=ACCENT2, fontweight='bold')

# B vector
y_b_vec = 5
ax5.text(5, y_b_vec+2.2, 'Step 2: output = score × B', fontsize=LABEL_SIZE,
         color=ACCENT2, fontweight='bold', ha='center')
for i, val in enumerate(out_B):
    bar_len = abs(val) * 2.5
    color = ACCENT5 if val < 0 else ACCENT3
    x0 = 0 if val >= 0 else -bar_len
    ax5.barh(y_b_vec - i*0.85, bar_len, height=0.6,
             left=x0, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax5.text(-2.5, y_b_vec - i*0.85, dim_labels_5d[i], fontsize=TINY_SIZE,
             color=MUTED, ha='right', va='center')
    ax5.text(bar_len + 0.3 if val >= 0 else -bar_len - 0.3,
             y_b_vec - i*0.85, f'{val:.2f}',
             fontsize=TINY_SIZE, color=FG, ha='left' if val >= 0 else 'right', va='center')

# Result
ax5.text(5, 0.2, '→ 5D output added as residual',
         fontsize=SMALL_SIZE, color=ACCENT2, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d1500', edgecolor=ACCENT2))

# Score per digit chart (inset)
ax5r = fig.add_axes([0.81, 0.255, 0.12, 0.14])
ax5r.set_facecolor(BOX_BG)
for spine in ax5r.spines.values():
    spine.set_color(BOX_EDGE)

tok_emb = np.stack([tok_A*np.cos(tok_start + np.arange(10)*tok_stride),
                     tok_A*np.sin(tok_start + np.arange(10)*tok_stride)], axis=1)
V_per_digit = tok_emb @ head_w  # (10,2) @ (2,5) = (10,5)
scores = (V_per_digit @ out_A).flatten()  # (10,)

ax5r.bar(np.arange(10), scores, color=[DCOLS[i] for i in range(10)],
         edgecolor='white', linewidth=0.5)
ax5r.set_xticks(np.arange(10))
ax5r.set_xlabel('digit', fontsize=TINY_SIZE-1, color=FG)
ax5r.set_ylabel('A·V', fontsize=TINY_SIZE-1, color=FG)
ax5r.set_title('Score per digit\n(≈ monotonic → digit readout)',
               fontsize=TINY_SIZE, color=ACCENT2, fontweight='bold')
ax5r.tick_params(labelsize=6, colors=FG)
ax5r.grid(True, alpha=0.12, axis='y')


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 7: PARAMETER BUDGET
# ═══════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[3, :])
ax6.set_title('Parameter Budget: Every Single One of the 62 Parameters',
              fontsize=PANEL_TITLE_SIZE+1, fontweight='bold', color=ACCENT, pad=10)
ax6.axis('off')
ax6.set_xlim(0, 100)
ax6.set_ylim(-2, 16)

# Stacked horizontal bar with clear segments
param_groups = [
    ('tok_arc',     3,  ACCENT3, 'A, start,\nstride'),
    ('z_hi_pos',    3,  ACCENT5, 'carry pos\n(reused!)'),
    ('equals_pos',  3,  ACCENT4, 'EQUALS\nposition'),
    ('q_phase',     1,  '#ffa657', 'Q/K\nbreak'),
    ('q_proj',      12, '#ff7b72', '3→4 matrix\n(routing)'),
    ('out_proj',    10, ACCENT2,  'rank-1\n(A+B vecs)'),
    ('fc1',         10, ACCENT3,  'FFN down\n(5→2)'),
    ('fc2',         10, '#56d364', 'FFN up\n(2→5)'),
    ('head_proj',   10, ACCENT4,  'output head\n(= V proj)'),
]

total = 62
x_start = 3
bar_h = 5
y_bar = 6

# Draw total label
ax6.text(50, y_bar + bar_h + 3.5, 'TOTAL: 62 learnable parameters',
         fontsize=18, fontweight='bold', color=ACCENT, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor=BOX_BG, edgecolor=ACCENT, lw=2))

for i, (name, count, color, note) in enumerate(param_groups):
    width = count / total * 82
    rect = FancyBboxPatch((x_start, y_bar), max(width-0.15, 0.8), bar_h,
                           boxstyle="round,pad=0.04",
                           facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.85)
    ax6.add_patch(rect)

    cx = x_start + width/2

    if width > 4:
        ax6.text(cx, y_bar + bar_h/2 + 1, name, ha='center', va='center',
                 fontsize=SMALL_SIZE-0.5, fontweight='bold', color='white')
        ax6.text(cx, y_bar + bar_h/2 - 0.4, f'{count}p', ha='center', va='center',
                 fontsize=LABEL_SIZE+2, fontweight='bold', color='white')
    elif width > 2:
        ax6.text(cx, y_bar + bar_h + 0.3, name, ha='center', va='bottom',
                 fontsize=TINY_SIZE, fontweight='bold', color=color)
        ax6.text(cx, y_bar + bar_h/2, f'{count}p', ha='center', va='center',
                 fontsize=LABEL_SIZE, fontweight='bold', color='white')
    else:
        ax6.text(cx, y_bar + bar_h + 0.3, f'{name}\n{count}p', ha='center', va='bottom',
                 fontsize=TINY_SIZE-1, fontweight='bold', color=color)

    # Note below
    ax6.text(cx, y_bar - 0.5, note, ha='center', va='top',
             fontsize=TINY_SIZE-0.5, color=MUTED, fontstyle='italic')

    x_start += width

# Frozen section
x_f = 88
ax6.text(x_f, y_bar + bar_h + 2, 'FROZEN', fontsize=LABEL_SIZE,
         fontweight='bold', color='#484f58', ha='center')
ax6.text(x_f, y_bar + bar_h + 0.5, '(0 extra params)', fontsize=SMALL_SIZE,
         color='#484f58', ha='center')

frozen = [('Spiral positions', '4 values, all frozen'),
          ('Norm weights', 'sinusoidal + z_hi'),
          ('PLUS / EOS pos', 'zero vectors')]

for i, (name, desc) in enumerate(frozen):
    yf = y_bar + bar_h - 1 - i*2
    rect = FancyBboxPatch((x_f-5, yf-0.4), 10, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor=BOX_BG, edgecolor='#484f58',
                           linewidth=1, linestyle='--')
    ax6.add_patch(rect)
    ax6.text(x_f, yf+0.35, f'{name}: {desc}',
             fontsize=TINY_SIZE-0.5, color='#6e7681', ha='center', va='center')

# Fun comparison
ax6.text(50, -0.5,
         'For perspective: GPT-4 has about 1.8 trillion parameters. '
         'This model has 62. That is a ratio of roughly 29,000,000,000 to 1.',
         fontsize=SMALL_SIZE, color='#484f58', ha='center', fontstyle='italic')


# ── Main title ───────────────────────────────────────────────────────────
fig.suptitle('MicroAdder: A 62-Parameter Transformer That Adds Two 10-Digit Numbers',
             fontsize=26, fontweight='bold', color='white', y=0.985)
fig.text(0.5, 0.955,
         '2nd place, Adder Challenge   |   100% accuracy on all 10,010 test cases   |   1 layer, 1 head, d_model = 5',
         fontsize=13, color=MUTED, ha='center')


# ── Save ─────────────────────────────────────────────────────────────────
fig.savefig('explainer_62p.png', dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print("Saved explainer_62p.png")
