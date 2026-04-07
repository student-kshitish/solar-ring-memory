import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
os.makedirs('docs', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')

# ─── LEFT: 6-Level Hierarchy ───────────────────────────────
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.axis('off')
ax.set_title('Solar Ring Memory — Complete 6-Level Hierarchy',
             color='white', fontsize=14, fontweight='bold', pad=12)

def box(ax, x, y, w, h, label, sub='',
        color='#1f6feb', tc='white', alpha=0.9, fs=9):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.1",
                        facecolor=color, edgecolor='white',
                        linewidth=1.2, alpha=alpha)
    ax.add_patch(p)
    if sub:
        ax.text(x+w/2, y+h*0.65, label, ha='center',
                va='center', color=tc, fontsize=fs,
                fontweight='bold')
        ax.text(x+w/2, y+h*0.25, sub, ha='center',
                va='center', color=tc, fontsize=fs-2,
                alpha=0.85)
    else:
        ax.text(x+w/2, y+h/2, label, ha='center',
                va='center', color=tc, fontsize=fs,
                fontweight='bold')

def arrow(ax, x1, y1, x2, y2, c='#58a6ff'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',
                               color=c, lw=1.8))

# Level 6 — Multiverse
box(ax, 0.3, 20.2, 9.4, 1.2,
    'LEVEL 6 — Multiverse',
    'Parallel ambiguous interpretations · collapse when P > 0.80',
    color='#4c1d95', tc='#e9d5ff', fs=10)
arrow(ax, 5, 20.2, 5, 19.5)

# Level 5 — Galaxy
box(ax, 0.3, 18.5, 9.4, 1.2,
    'LEVEL 5 — Galaxy',
    'All documents on same topic · Galactic Core = topic embedding',
    color='#1e3a5f', tc='#bfdbfe', fs=10)
arrow(ax, 5, 18.5, 5, 17.8)

# Level 4 — Sub-Galaxy
box(ax, 0.3, 16.8, 9.4, 1.2,
    'LEVEL 4 — Sub-Galaxy',
    'Document clusters · inter-cluster gravity waves · beta=0.1 fusion',
    color='#064e3b', tc='#a7f3d0', fs=10)
arrow(ax, 5, 16.8, 5, 16.1)

# Level 3 — Multi-Solar System
box(ax, 0.3, 15.1, 9.4, 1.2,
    'LEVEL 3 — Multi-Solar System',
    'One document · paragraphs -> solar systems · gravitational waves',
    color='#7c2d12', tc='#fed7aa', fs=10)
arrow(ax, 5, 15.1, 5, 14.4)

# Level 2 — Solar System (8 layers)
level2 = FancyBboxPatch((0.3, 6.8), 9.4, 7.8,
                         boxstyle="round,pad=0.15",
                         facecolor='#0d419d',
                         edgecolor='#58a6ff',
                         linewidth=2, alpha=0.25)
ax.add_patch(level2)
ax.text(5, 14.2, 'LEVEL 2 — Solar System (8-Layer Stack)',
        ha='center', va='center',
        color='#58a6ff', fontsize=10, fontweight='bold')

layers = [
    ('L1', 'POS Detection + Raw Slot Fill',          '#1f6feb'),
    ('L2', 'Context Window Correction',               '#1f6feb'),
    ('L3', 'Spawn Controller — CONJ Detection',       '#7c3aed'),
    ('L4', 'Orbital Link Encoder',                    '#7c3aed'),
    ('L5', 'Solar Spring Unified Field Attention',    '#047857'),
    ('L6', 'Pronoun Resolution — Orbital Walk',       '#b45309'),
    ('L7', 'Relation Encoder  S x V x O',             '#9d174d'),
    ('L8', 'Flatten + Project -> Context Vector Rd',  '#1f6feb'),
]
for i, (ln, ld, lc) in enumerate(layers):
    y = 13.6 - i * 0.84
    box(ax, 0.6, y, 8.8, 0.74, f'{ln}  {ld}',
        color=lc, tc='white', alpha=0.85, fs=8)
    if i < 7:
        arrow(ax, 5, y, 5, y-0.1)

arrow(ax, 5, 6.8, 5, 6.2)

# Level 1 — Sub-planet
box(ax, 0.3, 5.2, 9.4, 1.2,
    'LEVEL 1 — Sub-Planet (Micro-Worker)',
    'Animacy · Case · Size detected simultaneously per token',
    color='#92400e', tc='#fde68a', fs=10)

# Spring equation at bottom
ax.text(5, 4.5,
        'F(i,j) = G_micro + G_macro + F_spring + F_bh + F_ns + F_cp - F_cf',
        ha='center', color='#58a6ff', fontsize=8,
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#161b22',
                  edgecolor='#30363d'))

# Results
box(ax, 0.3, 0.3, 9.4, 3.8,
    '',
    color='#0d1117', tc='white', alpha=1.0)

results = [
    ('Winograd Schema',  '80.7%',     'BERT 70%',   '+10.7%', '#ffd700'),
    ('Pronoun Direct',   '76.7%',     'BERT 70%',   '+6.7%',  '#34d399'),
    ('Nested Depth 4',   '50.0%',     'BERT 38%',   '+12%',   '#34d399'),
    ('Context Window',   'unlimited', 'BERT 17',    '+inf',   '#34d399'),
    ('Memory',           '27MB',      'BERT 418MB', '15x',    '#34d399'),
]
ax.text(5, 4.0, 'KEY RESULTS', ha='center',
        color='white', fontsize=9, fontweight='bold')
for i, (task, sr, bert, delta, c) in enumerate(results):
    y = 3.5 - i * 0.58
    ax.text(0.6, y, task, color='#94a3b8', fontsize=7)
    ax.text(4.2, y, sr,   color=c, fontsize=7, fontweight='bold')
    ax.text(6.0, y, bert, color='#64748b', fontsize=7)
    ax.text(8.2, y, delta,color=c, fontsize=7, fontweight='bold')

# ─── RIGHT: Physics Equations ──────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 22)
ax2.axis('off')
ax2.set_title('Solar Ring Memory — Physics & Math',
              color='white', fontsize=14, fontweight='bold', pad=12)

sections = [
    (20.0, 'Solar Spring Unified Field', '#047857',
     ['F(i,j) = G_micro + G_macro + F_spring + F_bh + F_ns',
      'G_micro = G_k x mi x mj / r2_slot',
      'G_macro = G_O x mi x mj / r2_orbital',
      'F_spring = k x |pos_i - pos_j|  <- grows with distance',
      'F_bh = -G_bh / (conf - horizon)^2']),
    (16.8, 'Isotope Decay (POS half-life)', '#b45309',
     ['conf(t) = conf(0) x e^(-lambda_pos x t)',
      'lambda_SUBJ = 0.005  half-life = 138 tokens',
      'lambda_VERB = 0.030  half-life = 23 tokens',
      'lambda_DET  = 0.350  half-life = 2 tokens',
      'Nouns persist · Articles ejected instantly']),
    (13.6, 'Neutron Star + Centripetal/Centrifugal', '#7c3aed',
     ['F_ns = G_ns x (1/conf) x mi x mj / r^2',
      'F_centripetal = m x v^2 / r  (keeps concept in orbit)',
      'F_centrifugal = -m x w^2 x r  (ejects low-mass)',
      'Lagrange point r* = (2G/k)^(1/3)',
      'Maximum attraction at Lagrange distance']),
    (10.4, 'Sun State + Galactic Core', '#1f6feb',
     ['Sun(t+1) = (1-a)·Sun(t) + a·sum(Planet slots)',
      'alpha = 0.3  (sentence-level fusion rate)',
      'GalacticCore(t+1) = (1-b)·Core(t) + b·Sun',
      'beta = 0.1  (document-level, very slow)',
      'Sub-galaxy gravity = G x sim(i,core) x sim(j,core)']),
    (7.2, 'Multiverse Parallelism', '#9d174d',
     ['Spawn when: token in AMBIGUOUS_WORDS',
      'P(universe_k) = softmax(Sun_k · context / sqrt(d))',
      'Collapse when: max(P) > 0.80',
      'Both universes run in parallel until collapse',
      'Like quantum superposition -> observation']),
    (4.0, 'Black Hole / White Hole', '#374151',
     ['Black hole: conf < EVENT_HORIZON (0.1) -> collapse',
      'Hawking radiation: ring_state -> Sun State (a=0.3)',
      'White hole: orphan pronoun -> spawn new ring',
      'Wormhole: destroyed info reappears next paragraph',
      'Neutron star: collapsed ring still exerts gravity']),
]

for (y_start, title, color, lines) in sections:
    box(ax2, 0.3, y_start-1.0, 9.4, 0.65,
        title, color=color, tc='white', fs=9)
    for i, line in enumerate(lines):
        ax2.text(0.6, y_start-1.55-i*0.46, line,
                 color='#e2e8f0', fontsize=7,
                 fontfamily='monospace')

# Hierarchy ladder on right
ax2.text(5, 2.8, 'COMPLETE HIERARCHY', ha='center',
         color='#ffd700', fontsize=10, fontweight='bold')
hierarchy = [
    ('Multiverse',    'parallel ambiguous interpretations', '#7c3aed'),
    ('Galaxy',        'all documents on same topic',        '#1e3a5f'),
    ('Sub-Galaxy',    'document clusters',                  '#064e3b'),
    ('Multi-Solar',   'one document / conversation',        '#7c2d12'),
    ('Solar System',  'one sentence / clause group',        '#1f6feb'),
    ('Sub-Planet',    'animacy · case · size slots',        '#92400e'),
]
for i, (name, desc, c) in enumerate(hierarchy):
    y = 2.3 - i * 0.38
    ax2.add_patch(FancyBboxPatch(
        (0.4, y-0.15), 9.2, 0.30,
        boxstyle="round,pad=0.05",
        facecolor=c, edgecolor='none', alpha=0.7
    ))
    ax2.text(1.0, y+0.0, name, color='white',
             fontsize=7, fontweight='bold', va='center')
    ax2.text(4.2, y+0.0, desc, color='#cbd5e1',
             fontsize=6, va='center')

plt.tight_layout(pad=1.5)
plt.savefig('docs/architecture.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
print("Saved docs/architecture.png")
plt.close()
