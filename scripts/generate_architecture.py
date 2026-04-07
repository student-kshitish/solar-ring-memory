import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
import numpy as np
import os
os.makedirs('docs', exist_ok=True)

BG     = '#0d1117'
BORDER = '#30363d'
WHITE  = '#f0f6fc'
GRAY   = '#8b949e'
BLUE   = '#58a6ff'
GREEN  = '#3fb950'
PURPLE = '#bc8cff'
ORANGE = '#ffa657'
RED    = '#ff7b72'
GOLD   = '#e3b341'
TEAL   = '#39d353'

# ════════════════════════════════════════════
# FIGURE 1 — Layer Stack (tall, clean)
# ════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(10, 14))
fig1.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

def rbox(ax, x, y, w, h, text, sub=None,
         fc='#161b22', ec=BORDER, tc=WHITE,
         fs=9, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=ec,
        linewidth=1.5, zorder=3
    ))
    if sub:
        ax.text(x+w/2, y+h*0.62, text,
                ha='center', va='center',
                color=tc, fontsize=fs,
                fontweight='bold' if bold else 'normal',
                zorder=4)
        ax.text(x+w/2, y+h*0.25, sub,
                ha='center', va='center',
                color=GRAY, fontsize=fs-1.5,
                zorder=4)
    else:
        ax.text(x+w/2, y+h/2, text,
                ha='center', va='center',
                color=tc, fontsize=fs,
                fontweight='bold' if bold else 'normal',
                zorder=4)

def arr(ax, x, y1, y2, c=BLUE):
    ax.annotate('',
        xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(
            arrowstyle='->', color=c,
            lw=1.8, mutation_scale=14
        ), zorder=5)

# Title
ax.text(5, 13.6, 'Solar Ring Memory',
        ha='center', color=WHITE,
        fontsize=16, fontweight='bold')
ax.text(5, 13.2, 'Architecture — Layer Stack',
        ha='center', color=GRAY, fontsize=11)

# Input token
rbox(ax, 2.5, 12.2, 5, 0.7,
     'Token Input  xt in R^300',
     fc='#161b22', ec=BLUE, tc=BLUE, fs=10)
arr(ax, 5, 12.2, 11.85)

# POS tagger
rbox(ax, 1.5, 11.1, 7, 0.65,
     'spaCy POS Tagger',
     'SUBJ · VERB · OBJ · PREP · CONJ · ADJ · ADV · DET',
     fc='#2d1b00', ec=ORANGE, tc=ORANGE, fs=9)
arr(ax, 5, 11.1, 10.75)

# Layer stack outline
ax.add_patch(FancyBboxPatch(
    (0.4, 3.5), 9.2, 7.1,
    boxstyle="round,pad=0.2",
    facecolor='#0d1f3c',
    edgecolor=BLUE, linewidth=1.5,
    alpha=0.4, zorder=1
))
ax.text(5, 10.5, '8-Layer Processing Stack',
        ha='center', color=BLUE,
        fontsize=9, fontweight='bold', zorder=4)

# 8 layers
LAYERS = [
    ('L1', 'POS Slot Fill',
     'Write token into correct ring slot', '#1f3d6e', BLUE),
    ('L2', 'Context Correction',
     'Fix slot based on surrounding context', '#1f3d6e', BLUE),
    ('L3', 'Spawn Controller',
     'CONJ detected -> spawn child ring', '#3d1f6e', PURPLE),
    ('L4', 'Orbital Link Encoder',
     'Connect parent -> child ring via gravity', '#3d1f6e', PURPLE),
    ('L5', 'Solar Spring Attention',
     'F = G_micro + G_macro + F_spring + F_bh', '#0d3d2e', GREEN),
    ('L6', 'Pronoun Resolution',
     'Orbital walk: Pluto -> antecedent ring', '#3d2a0d', ORANGE),
    ('L7', 'Relation Encoder',
     'Encode S x V x O triple', '#3d0d1f', RED),
    ('L8', 'Output Projection',
     'Flatten rings -> context vector c in R^d', '#1f3d6e', BLUE),
]

for i, (num, name, desc, fc, ec) in enumerate(LAYERS):
    y = 10.0 - i * 0.82
    rbox(ax, 0.7, y, 8.6, 0.70,
         f'{num}  {name}', desc,
         fc=fc, ec=ec, tc=WHITE, fs=8.5)
    if i < 7:
        arr(ax, 5, y, y - 0.12, c=ec)

arr(ax, 5, 3.5, 3.2)

# Output
rbox(ax, 1.5, 2.5, 7, 0.65,
     'Context Vector  c in R^300',
     'Fixed size · Same shape regardless of sentence length',
     fc='#0d3d2e', ec=GREEN, tc=GREEN, fs=9)

# Results strip
ax.add_patch(FancyBboxPatch(
    (0.4, 0.15), 9.2, 2.1,
    boxstyle="round,pad=0.12",
    facecolor='#161b22', edgecolor=GOLD,
    linewidth=1.5, zorder=3
))
ax.text(5, 2.05, 'Benchmark Results',
        ha='center', color=GOLD,
        fontsize=9, fontweight='bold', zorder=4)

results = [
    ('Winograd Schema',   '80.7%', 'BERT ~70%',   '+10.7%'),
    ('Pronoun Direct',    '76.7%', 'BERT ~70%',   '+6.7%'),
    ('Nested Depth 4',    '50.0%', 'BERT ~38%',   '+12%'),
    ('Memory',            '27MB',  'BERT 418MB',  '15x less'),
    ('Context Window',    'inf',   'BERT 512 tok','unlimited'),
]
for i, (task, sr, bert, gain) in enumerate(results):
    y = 1.72 - i * 0.30
    ax.text(0.8,  y, task, color=GRAY,
            fontsize=7, va='center', zorder=4)
    ax.text(4.3,  y, sr,   color=GREEN,
            fontsize=7, va='center',
            fontweight='bold', zorder=4)
    ax.text(6.0,  y, bert, color=GRAY,
            fontsize=7, va='center', zorder=4)
    ax.text(8.3,  y, gain, color=GOLD,
            fontsize=7, va='center',
            fontweight='bold', zorder=4)

plt.tight_layout(pad=0.5)
plt.savefig('docs/architecture_layers.png',
            dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("Saved docs/architecture_layers.png")

# ════════════════════════════════════════════
# FIGURE 2 — Ring Hierarchy (orbital diagram)
# ════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 14))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
ax2.set_xlim(-7, 7)
ax2.set_ylim(-7, 8)
ax2.set_aspect('equal')
ax2.axis('off')

ax2.text(0, 7.6, 'Solar Ring Memory',
         ha='center', color=WHITE,
         fontsize=16, fontweight='bold')
ax2.text(0, 7.1, 'Ring Hierarchy — Orbital Structure',
         ha='center', color=GRAY, fontsize=11)

# ── Sun ──
sun_c = Circle((0,0), 1.1,
               color='#f59e0b', alpha=0.95, zorder=6)
ax2.add_patch(sun_c)
ax2.text(0,  0.22, 'SUN', ha='center', va='center',
         color='#1a1a2e', fontsize=11,
         fontweight='bold', zorder=7)
ax2.text(0, -0.22, 'Main Clause', ha='center', va='center',
         color='#1a1a2e', fontsize=7, zorder=7)

# Sun pole labels
ax2.text( 0,  1.4, 'SUBJ: John',
          ha='center', color='#34d399',
          fontsize=8, fontweight='bold')
ax2.text( 0, -1.5, 'OBJ: Mary',
          ha='center', color='#f87171',
          fontsize=8, fontweight='bold')
ax2.text(-1.3, 0.1, 'VERB\ntold',
          ha='center', color='#a78bfa', fontsize=7)
ax2.text( 1.3, 0.1, 'CONJ\nthat',
          ha='center', color='#94a3b8', fontsize=7)

# ── Planet 1 orbit ──
th = np.linspace(0, 2*np.pi, 300)
ax2.plot(3.0*np.cos(th), 3.0*np.sin(th),
         color=BLUE, lw=1.2, ls='--', alpha=0.4, zorder=2)

# Planet 1
p1 = Circle((2.6, 1.6), 0.82,
             color='#1f4e79', alpha=0.95, zorder=6)
ax2.add_patch(p1)
ax2.text(2.6, 1.72, 'PLANET 1', ha='center', va='center',
         color=WHITE, fontsize=7, fontweight='bold', zorder=7)
ax2.text(2.6, 1.42, 'depth = 1', ha='center', va='center',
         color='#93c5fd', fontsize=6, zorder=7)

ax2.text(3.7, 2.7, 'SUBJ: cat',
         ha='center', color='#34d399',
         fontsize=7, fontweight='bold')
ax2.text(3.7, 0.5, 'OBJ: dog',
         ha='center', color='#f87171',
         fontsize=7, fontweight='bold')

# ── Planet 2 orbit (same) ──
p2 = Circle((-2.6, 1.6), 0.82,
             color='#4c1d95', alpha=0.95, zorder=6)
ax2.add_patch(p2)
ax2.text(-2.6, 1.72, 'PLANET 2', ha='center', va='center',
         color=WHITE, fontsize=7, fontweight='bold', zorder=7)
ax2.text(-2.6, 1.42, 'depth = 1', ha='center', va='center',
         color='#c4b5fd', fontsize=6, zorder=7)
ax2.text(-3.7, 2.7, 'SUBJ: she',
         ha='center', color='#34d399',
         fontsize=7, fontweight='bold')
ax2.text(-3.7, 0.5, 'OBJ: it',
         ha='center', color='#f87171',
         fontsize=7, fontweight='bold')

# ── Moon orbit around Planet 1 ──
cx, cy = 2.6, 1.6
ax2.plot(cx+1.5*np.cos(th), cy+1.5*np.sin(th),
         color=ORANGE, lw=1.0, ls=':', alpha=0.4, zorder=2)

moon = Circle((3.8, 2.9), 0.52,
              color='#7c2d12', alpha=0.95, zorder=6)
ax2.add_patch(moon)
ax2.text(3.8, 2.92, 'MOON', ha='center', va='center',
         color=WHITE, fontsize=6, fontweight='bold', zorder=7)
ax2.text(3.8, 2.68, 'depth=2', ha='center', va='center',
         color='#fed7aa', fontsize=5, zorder=7)

# ── Pluto — pronoun ──
ax2.plot(5.0*np.cos(th)-1.2, 2.2*np.sin(th),
         color='#4b5563', lw=0.8, ls=':',
         alpha=0.35, zorder=2)
pluto = Circle((3.8, 0.0), 0.42,
               color='#1f2937', alpha=0.95, zorder=6)
ax2.add_patch(pluto)
ax2.add_patch(Circle((3.8, 0.0), 0.42,
                      fill=False, color='#4b5563',
                      lw=1.5, zorder=7))
ax2.text(3.8,  0.08, 'it', ha='center', va='center',
         color='#9ca3af', fontsize=9,
         fontweight='bold', zorder=8)
ax2.text(3.8, -0.18, 'Pluto', ha='center', va='center',
         color='#6b7280', fontsize=5, zorder=8)
ax2.text(3.8, -0.55, 'e=0.85', ha='center',
         color='#4b5563', fontsize=6, zorder=4)

# ── Spring force arrow (pronoun -> antecedent) ──
ax2.annotate('',
    xy=(0, 1.2), xytext=(3.4, 0.1),
    arrowprops=dict(arrowstyle='->',
                   color=GOLD, lw=2.2,
                   connectionstyle='arc3,rad=-0.35'),
    zorder=8)
ax2.text(1.9, 0.3, 'Spring\nforce', ha='center',
         color=GOLD, fontsize=7, fontweight='bold')

# ── Conjunction labels on orbits ──
ax2.text( 1.3, -2.3, '"that"', ha='center',
          color='#fbbf24', fontsize=8, fontstyle='italic')
ax2.text(-1.3, -2.3, '"because"', ha='center',
          color='#fbbf24', fontsize=8, fontstyle='italic')
ax2.text( 4.3,  1.6, '"which"',  ha='center',
          color='#fb923c', fontsize=7, fontstyle='italic')

# ── Black hole ──
bh = Circle((-4.0, -2.5), 0.5,
             color='#000000', alpha=1.0, zorder=6)
ax2.add_patch(bh)
ax2.add_patch(Circle((-4.0, -2.5), 0.7,
                      fill=False, color='#f59e0b',
                      lw=2.0, alpha=0.7, zorder=7))
ax2.text(-4.0, -2.5, 'X', ha='center', va='center',
         color='#f59e0b', fontsize=12,
         fontweight='bold', zorder=8)
ax2.text(-4.0, -3.4, 'Black Hole', ha='center',
         color='#f59e0b', fontsize=7, fontweight='bold')
ax2.text(-4.0, -3.75, 'conf < 0.10 -> collapse', ha='center',
         color=GRAY, fontsize=6)
ax2.text(-4.0, -4.05, 'Hawking -> Sun State', ha='center',
         color=GRAY, fontsize=6)

# ── White hole ──
wh = Circle((4.0, -2.5), 0.5,
             color='#dbeafe', alpha=0.95, zorder=6)
ax2.add_patch(wh)
ax2.text(4.0, -2.5, '+', ha='center', va='center',
         color=BLUE, fontsize=16,
         fontweight='bold', zorder=7)
ax2.text(4.0, -3.4, 'White Hole', ha='center',
         color=BLUE, fontsize=7, fontweight='bold')
ax2.text(4.0, -3.75, 'Orphan pronoun -> spawn', ha='center',
         color=GRAY, fontsize=6)
ax2.text(4.0, -4.05, 'Seeded from Sun State', ha='center',
         color=GRAY, fontsize=6)

# Wormhole arrow between BH and WH
ax2.annotate('',
    xy=(3.3, -2.5), xytext=(-3.3, -2.5),
    arrowprops=dict(arrowstyle='->', color=PURPLE,
                   lw=1.5, connectionstyle='arc3,rad=-0.3'),
    zorder=8)
ax2.text(0, -3.1, 'Wormhole', ha='center',
         color=PURPLE, fontsize=7, fontweight='bold')

# ── 6-level hierarchy strip at bottom ──
levels = [
    ('Multiverse',   'parallel ambiguous interpretations', '#4c1d95'),
    ('Galaxy',       'all documents on same topic',        '#1e3a5f'),
    ('Sub-Galaxy',   'document clusters · beta=0.1 fusion','#064e3b'),
    ('Multi-Solar',  'one document · gravity waves',       '#7c2d12'),
    ('Solar System', 'one sentence · sun+planets+moons',   '#1f3d6e'),
    ('Sub-Planet',   'animacy · case · size per token',    '#3d2a0d'),
]
ax2.text(0, -4.7, 'Complete 6-Level Hierarchy',
         ha='center', color=GOLD,
         fontsize=9, fontweight='bold')
for i, (name, desc, fc) in enumerate(levels):
    y = -5.1 - i*0.38
    ax2.add_patch(FancyBboxPatch(
        (-6.5, y-0.16), 13, 0.30,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor='none',
        alpha=0.75, zorder=3
    ))
    ax2.text(-6.2, y+0.0, name, color=WHITE,
             fontsize=7, fontweight='bold',
             va='center', zorder=4)
    ax2.text(-2.5, y+0.0, desc, color='#cbd5e1',
             fontsize=6, va='center', zorder=4)

# ── Legend ──
legend_items = [
    (mpatches.Patch(color='#f59e0b'), 'Sun — main clause'),
    (mpatches.Patch(color='#1f4e79'), 'Planet — nested (depth 1)'),
    (mpatches.Patch(color='#7c2d12'), 'Moon — deep nested (depth 2)'),
    (mpatches.Patch(color='#1f2937'), 'Pluto — pronoun (e=0.85)'),
    (Line2D([0],[0],color='#34d399',lw=2), 'SUBJ pole (protected)'),
    (Line2D([0],[0],color='#f87171',lw=2), 'OBJ pole (protected)'),
    (Line2D([0],[0],color=GOLD,lw=2),      'Spring force'),
    (Line2D([0],[0],color=PURPLE,lw=2),    'Wormhole'),
]
ax2.legend(
    [h for h,_ in legend_items],
    [l for _,l in legend_items],
    loc='upper left',
    facecolor='#161b22', edgecolor=BORDER,
    labelcolor=WHITE, fontsize=6.5,
    framealpha=0.9, ncol=2
)

plt.tight_layout(pad=0.5)
plt.savefig('docs/architecture_rings.png',
            dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("Saved docs/architecture_rings.png")

# ════════════════════════════════════════════
# Combine side by side into architecture.png
# ════════════════════════════════════════════
from PIL import Image
img1 = Image.open('docs/architecture_layers.png')
img2 = Image.open('docs/architecture_rings.png')

h = max(img1.height, img2.height)

def pad(img, target_h, bg=(13, 17, 23)):
    if img.height == target_h:
        return img
    new = Image.new('RGB', (img.width, target_h), bg)
    new.paste(img, (0, (target_h - img.height) // 2))
    return new

img1 = pad(img1, h)
img2 = pad(img2, h)

combined = Image.new('RGB', (img1.width + img2.width, h),
                     (13, 17, 23))
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width, 0))
combined.save('docs/architecture.png')
print("Saved docs/architecture.png (combined)")
