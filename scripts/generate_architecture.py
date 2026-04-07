import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os
os.makedirs('docs', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 14))
fig.patch.set_facecolor('#0d1117')

# ─── LEFT PANEL: Full Architecture ───────────────────────────
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis('off')
ax.set_title('Solar Ring Memory — Architecture',
             color='white', fontsize=16, fontweight='bold', pad=15)

def draw_box(ax, x, y, w, h, label, sublabel='',
             color='#1f6feb', text_color='white', alpha=0.9):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    if sublabel:
        ax.text(x+w/2, y+h*0.65, label,
                ha='center', va='center',
                color=text_color, fontsize=9,
                fontweight='bold')
        ax.text(x+w/2, y+h*0.25, sublabel,
                ha='center', va='center',
                color=text_color, fontsize=7, alpha=0.85)
    else:
        ax.text(x+w/2, y+h/2, label,
                ha='center', va='center',
                color=text_color, fontsize=9,
                fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color='#58a6ff'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->',
                               color=color, lw=2))

# Input
draw_box(ax, 2, 16.5, 6, 0.9, 'Token Input  xₜ ∈ ℝᵈ',
         color='#21262d', text_color='#58a6ff')
draw_arrow(ax, 5, 16.5, 5, 15.8)

# POS Tagger
draw_box(ax, 1.5, 14.8, 7, 0.9,
         'POS Tagger + spaCy',
         'SUBJ=1  OBJ=2  VERB=3  CONJ=4  ADJ=5',
         color='#b45309', text_color='white')
draw_arrow(ax, 5, 14.8, 5, 14.1)

# 8 Layers box
layer_box = FancyBboxPatch((0.3, 6.5), 9.4, 7.4,
                            boxstyle="round,pad=0.2",
                            facecolor='#0d419d',
                            edgecolor='#58a6ff',
                            linewidth=2, alpha=0.3)
ax.add_patch(layer_box)
ax.text(5, 13.7, '8-Layer Solar Ring Stack',
        ha='center', va='center',
        color='#58a6ff', fontsize=10, fontweight='bold')

layers = [
    ('L1', 'POS Detection + Raw Slot Fill', '#1f6feb'),
    ('L2', 'Context Window Correction', '#1f6feb'),
    ('L3', 'Spawn Controller — CONJ Detection', '#7c3aed'),
    ('L4', 'Orbital Link Encoder', '#7c3aed'),
    ('L5', 'Solar Spring Unified Field Attention', '#047857'),
    ('L6', 'Pronoun Resolution — Orbital Walk', '#b45309'),
    ('L7', 'Relation Encoder  S×V×O', '#9d174d'),
    ('L8', 'Flatten + Project → ℝᵈ', '#1f6feb'),
]

for i, (lnum, ldesc, lcolor) in enumerate(layers):
    y = 13.1 - i * 0.82
    draw_box(ax, 0.6, y, 8.8, 0.72,
             f'{lnum}  {ldesc}',
             color=lcolor, text_color='white', alpha=0.85)
    if i < 7:
        draw_arrow(ax, 5, y, 5, y - 0.1)

draw_arrow(ax, 5, 6.5, 5, 5.8)

# Output
draw_box(ax, 1.5, 4.8, 7, 0.9,
         'Context Vector  c ∈ ℝᵈ',
         'Fixed size — always same regardless of sentence length',
         color='#047857', text_color='white')
draw_arrow(ax, 5, 4.8, 5, 4.1)

# Results box
draw_box(ax, 0.5, 2.8, 9, 1.1,
         'Winograd: 80.7%  |  Pronoun: 76.7%  |  Memory: 27MB',
         'Beats BERT-base (70%) · Unlimited context · Runs on Android',
         color='#1a1a2e', text_color='#ffd700', alpha=1.0)

# Physics label
ax.text(9.7, 10.5, 'Solar Spring\nPhysics',
        ha='center', color='#047857',
        fontsize=8, rotation=90, fontweight='bold')

# ─── RIGHT PANEL: Ring Structure ─────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_xlim(-6, 6)
ax2.set_ylim(-6, 7)
ax2.axis('off')
ax2.set_title('Solar Ring Memory — Clause Hierarchy',
              color='white', fontsize=16, fontweight='bold', pad=15)

# Sun
sun = plt.Circle((0, 0), 1.2, color='#f59e0b', alpha=0.9, zorder=5)
ax2.add_patch(sun)
ax2.text(0, 0.15, 'SUN', ha='center', va='center',
         color='#1a1a2e', fontsize=11, fontweight='bold', zorder=6)
ax2.text(0, -0.3, 'Main Clause', ha='center', va='center',
         color='#1a1a2e', fontsize=7, zorder=6)

# Sun ring slots
ax2.text(0, 1.5, 'SUBJ: John', ha='center',
         color='#34d399', fontsize=8, fontweight='bold')
ax2.text(0, -1.6, 'OBJ: Mary', ha='center',
         color='#f87171', fontsize=8, fontweight='bold')
ax2.text(-1.5, 0, 'VERB\ntold', ha='center',
         color='#a78bfa', fontsize=7)
ax2.text(1.5, 0, 'PREP\nin', ha='center',
         color='#94a3b8', fontsize=7)

# Planet 1 orbit
orbit1 = plt.Circle((0, 0), 3.2, fill=False,
                     color='#58a6ff', linewidth=1.5,
                     linestyle='--', alpha=0.5, zorder=3)
ax2.add_patch(orbit1)

# Planet 1
planet1 = plt.Circle((2.8, 1.8), 0.85,
                      color='#1f6feb', alpha=0.9, zorder=5)
ax2.add_patch(planet1)
ax2.text(2.8, 1.95, 'PLANET 1', ha='center', va='center',
         color='white', fontsize=7, fontweight='bold', zorder=6)
ax2.text(2.8, 1.6, 'depth=1', ha='center', va='center',
         color='#93c5fd', fontsize=6, zorder=6)

# Planet 1 slots
ax2.text(3.6, 2.8, 'SUBJ: cat', ha='center',
         color='#34d399', fontsize=7, fontweight='bold')
ax2.text(3.6, 0.8, 'OBJ: dog', ha='center',
         color='#f87171', fontsize=7, fontweight='bold')

# Planet 2
planet2 = plt.Circle((-2.8, 1.8), 0.85,
                      color='#7c3aed', alpha=0.9, zorder=5)
ax2.add_patch(planet2)
ax2.text(-2.8, 1.95, 'PLANET 2', ha='center', va='center',
         color='white', fontsize=7, fontweight='bold', zorder=6)
ax2.text(-2.8, 1.6, 'depth=1', ha='center', va='center',
         color='#c4b5fd', fontsize=6, zorder=6)
ax2.text(-3.8, 2.8, 'SUBJ: she', ha='center',
         color='#34d399', fontsize=7, fontweight='bold')
ax2.text(-3.8, 0.8, 'OBJ: it', ha='center',
         color='#f87171', fontsize=7, fontweight='bold')

# Moon orbit around planet 1
moon_orbit = plt.Circle((2.8, 1.8), 1.6, fill=False,
                         color='#f97316', linewidth=1,
                         linestyle=':', alpha=0.5, zorder=3)
ax2.add_patch(moon_orbit)

# Moon
moon = plt.Circle((4.2, 3.0), 0.55,
                   color='#ea580c', alpha=0.9, zorder=5)
ax2.add_patch(moon)
ax2.text(4.2, 3.05, 'MOON', ha='center', va='center',
         color='white', fontsize=6, fontweight='bold', zorder=6)
ax2.text(4.2, 2.78, 'depth=2', ha='center', va='center',
         color='#fed7aa', fontsize=5, zorder=6)

# Conjunction labels on orbits
ax2.text(1.4, -2.2, '"that"', ha='center',
         color='#fbbf24', fontsize=8, fontstyle='italic')
ax2.text(-1.4, -2.2, '"because"', ha='center',
         color='#fbbf24', fontsize=8, fontstyle='italic')
ax2.text(4.5, 1.8, '"which"', ha='center',
         color='#fb923c', fontsize=7, fontstyle='italic')

# Pluto pronoun (eccentric orbit)
theta = np.linspace(0, 2*np.pi, 200)
pluto_x = 4.5 * np.cos(theta) - 1.0
pluto_y = 1.8 * np.sin(theta)
ax2.plot(pluto_x, pluto_y, color='#6b7280',
         linewidth=1, linestyle=':', alpha=0.4)

pluto = plt.Circle((3.5, 0), 0.45,
                    color='#374151', alpha=0.9, zorder=5)
ax2.add_patch(pluto)
ax2.text(3.5, 0.05, 'it', ha='center', va='center',
         color='#9ca3af', fontsize=8, fontweight='bold', zorder=6)
ax2.text(3.5, -0.25, 'Pluto', ha='center', va='center',
         color='#6b7280', fontsize=5, zorder=6)
ax2.text(3.5, -0.75, 'e=0.85', ha='center',
         color='#4b5563', fontsize=6)

# Spring force arrows (pronoun → antecedent)
ax2.annotate('', xy=(0, 1.3), xytext=(3.1, 0.1),
             arrowprops=dict(arrowstyle='->',
                            color='#ffd700', lw=2.5,
                            connectionstyle='arc3,rad=-0.3'))
ax2.text(1.8, 0.2, '🌀 Spring\nforce', ha='center',
         color='#ffd700', fontsize=7, fontweight='bold')

# Black hole indicator
bh = plt.Circle((-3.5, -2.5), 0.4,
                color='#111827', alpha=1.0, zorder=5)
ax2.add_patch(bh)
bh_ring = plt.Circle((-3.5, -2.5), 0.6, fill=False,
                      color='#f59e0b', linewidth=2,
                      alpha=0.7, zorder=4)
ax2.add_patch(bh_ring)
ax2.text(-3.5, -2.5, '✕', ha='center', va='center',
         color='#f59e0b', fontsize=10, fontweight='bold', zorder=6)
ax2.text(-3.5, -3.2, 'Black Hole\n(collapsed ring)', ha='center',
         color='#f59e0b', fontsize=6)

# White hole indicator
wh = plt.Circle((3.5, -3.5), 0.4,
                color='#eff6ff', alpha=0.9, zorder=5)
ax2.add_patch(wh)
ax2.text(3.5, -3.5, '+', ha='center', va='center',
         color='#1f6feb', fontsize=14, fontweight='bold', zorder=6)
ax2.text(3.5, -4.2, 'White Hole\n(new ring spawn)', ha='center',
         color='#93c5fd', fontsize=6)

# Legend
legend_items = [
    (mpatches.Patch(color='#f59e0b'), 'Sun — main clause'),
    (mpatches.Patch(color='#1f6feb'), 'Planet — nested clause (depth 1)'),
    (mpatches.Patch(color='#ea580c'), 'Moon — deep nested (depth 2)'),
    (mpatches.Patch(color='#374151'), 'Pluto — pronoun (e=0.85)'),
    (plt.Line2D([0],[0], color='#34d399', linewidth=2), 'SUBJ pole (protected)'),
    (plt.Line2D([0],[0], color='#f87171', linewidth=2), 'OBJ pole (protected)'),
    (plt.Line2D([0],[0], color='#ffd700', linewidth=2), 'Spring force (grows with distance)'),
]
handles = [h for h, _ in legend_items]
labels  = [l for _, l in legend_items]
ax2.legend(handles, labels, loc='lower left',
           facecolor='#161b22', edgecolor='#30363d',
           labelcolor='white', fontsize=7,
           framealpha=0.9)

# Equation at bottom
ax2.text(0, -5.5,
         'F(i,j) = G_micro/r² + G_macro/r² + k·Δpos + F_spring + F_bh',
         ha='center', color='#58a6ff', fontsize=8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#161b22',
                   edgecolor='#30363d', alpha=0.8))

plt.tight_layout(pad=1.5)
plt.savefig('docs/architecture.png', dpi=150,
            bbox_inches='tight',
            facecolor='#0d1117')
print("Saved docs/architecture.png")
plt.close()
