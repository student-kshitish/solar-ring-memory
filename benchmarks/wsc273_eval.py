"""
WSC273 zero-shot evaluation — frozen checkpoint, run once.

Checkpoint : checkpoints/winograd_final.pt  (weights_only=True)
Data       : WillHeld/wsc273, split='test', 273 examples
Adapter    : text + ' ' + get_entity(option, text)
             (pronoun lives inside text; no context dropped)
Output     : results/wsc273_eval.jsonl  + printed summary
"""

import sys, os, json, math, re
sys.path.insert(0, '.')

import torch

# ── 1. Load model (frozen) ─────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT   = 'checkpoints/winograd_final.pt'

from benchmarks.winograd_80_ls import WinogradSpringModel, get_entity, find_pronoun_idx
from benchmarks.winograd_95plus import gender_ensemble_score, pronoun_from_context

model = WinogradSpringModel().to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.spring.load_state_dict(ckpt['spring'], strict=False)
model.head.load_state_dict(ckpt['head'])
model.spring.eval()
model.head.eval()
print(f"Checkpoint loaded: {CKPT}  (acc field stored={ckpt.get('acc','n/a')})")
print(f"Device: {DEVICE}")

# ── 2. Load WSC273 ─────────────────────────────────────────────────────────

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: 'datasets' not found. Activate /tmp/eval_venv.")

print("\nLoading WillHeld/wsc273 ...")
wsc = load_dataset('WillHeld/wsc273', trust_remote_code=False, split='test')
print(f"Loaded {len(wsc)} examples.")

# ── 3. Build contamination index ───────────────────────────────────────────

from benchmarks.winograd_full import WINOGRAD_SCHEMAS

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', s.lower().strip())

train_texts = {_norm(ctx) for ctx, _, _ in WINOGRAD_SCHEMAS}

def is_contaminated(text: str) -> bool:
    n = _norm(text)
    if n in train_texts:
        return True
    # near-duplicate: share >=7 aligned words out of min length
    words_n = n.split()
    if len(words_n) < 6:
        return False
    for t in train_texts:
        words_t = t.split()
        if len(words_t) < 6:
            continue
        aligned = sum(a == b for a, b in zip(words_n, words_t))
        if aligned >= min(len(words_n), len(words_t)) - 2 and aligned >= 7:
            return True
    return False

# ── 4. Pre-embed all sentences (single batch, no gradient) ─────────────────

print("\nBuilding scored sentences ...")
all_sents = []
rows = []

for ex in wsc:
    text    = ex['text']
    label   = int(ex['label'])      # index of correct option
    options = ex['options']
    pronoun = ex['pronoun'].lower().rstrip('.,;:!?')

    ent_c = get_entity(options[label],     text)   # correct entity head
    ent_w = get_entity(options[1 - label], text)   # wrong entity head

    sent_c = text + ' ' + ent_c
    sent_w = text + ' ' + ent_w

    rows.append({
        'text':           text,
        'pronoun':        pronoun,
        'option_correct': options[label],
        'option_wrong':   options[1 - label],
        'entity_correct': ent_c,
        'entity_wrong':   ent_w,
        'sent_c':         sent_c,
        'sent_w':         sent_w,
        'contaminated':   is_contaminated(text),
        'source':         ex.get('source', ''),
    })
    all_sents.extend([sent_c, sent_w])

unique_sents = list(dict.fromkeys(all_sents))
print(f"Pre-computing embeddings for {len(unique_sents)} unique sentences ...")

with torch.no_grad():
    emb_cache = model.embedder.embed_words_batch(unique_sents)

print(f"Embeddings cached: {len(emb_cache)} entries.")

# ── 5. Score each example (frozen, no_grad) ────────────────────────────────

print("\nScoring ...")
results = []

PRONOUN_CAT = {
    'it': 'IT', 'its': 'IT',
    'he': 'HE', 'him': 'HE', 'his': 'HE',
    'she': 'SHE', 'her': 'SHE', 'hers': 'SHE',
    'they': 'THEY', 'them': 'THEY', 'their': 'THEY', 'theirs': 'THEY',
}

with torch.no_grad():
    for i, row in enumerate(rows):
        sc = gender_ensemble_score(
            model, row['text'], row['entity_correct'],
            row['pronoun'], emb_cache,
        )
        sw = gender_ensemble_score(
            model, row['text'], row['entity_wrong'],
            row['pronoun'], emb_cache,
        )
        predicted_correct = bool(sc > sw)
        cat = PRONOUN_CAT.get(row['pronoun'], 'POSS')

        results.append({
            'idx':            i,
            'text':           row['text'],
            'pronoun':        row['pronoun'],
            'option_correct': row['option_correct'],
            'option_wrong':   row['option_wrong'],
            'entity_correct': row['entity_correct'],
            'entity_wrong':   row['entity_wrong'],
            'score_correct':  round(float(sc), 5),
            'score_wrong':    round(float(sw), 5),
            'predicted':      predicted_correct,
            'contaminated':   row['contaminated'],
            'source':         row['source'],
            'pronoun_cat':    cat,
        })

# ── 6. Dump JSONL ──────────────────────────────────────────────────────────

os.makedirs('results', exist_ok=True)
out_path = 'results/wsc273_eval.jsonl'
with open(out_path, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f"\nPer-example predictions saved: {out_path}")

# ── 7. Compute metrics ─────────────────────────────────────────────────────

def wilson_ci(correct: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    center = (correct + z*z/2) / (n + z*z)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, center - margin), min(1.0, center + margin)

all_correct   = sum(r['predicted'] for r in results)
all_n         = len(results)
novel         = [r for r in results if not r['contaminated']]
novel_correct = sum(r['predicted'] for r in novel)
novel_n       = len(novel)

ci_all   = wilson_ci(all_correct,   all_n)
ci_novel = wilson_ci(novel_correct, novel_n)

# Per-pronoun category
cats = {}
for r in results:
    c = r['pronoun_cat']
    cats.setdefault(c, [0, 0])
    cats[c][1] += 1
    cats[c][0] += int(r['predicted'])

# ── 8. Print summary ───────────────────────────────────────────────────────

print()
print("=" * 65)
print("WSC273 ZERO-SHOT EVALUATION — Solar Ring (winograd_final.pt)")
print("=" * 65)
print(f"Checkpoint    : {CKPT}  [FROZEN — no tuning]")
print(f"Total schemas : {all_n}  ({all_n - novel_n} contaminated, {novel_n} novel)")
print()
print(f"{'Metric':<30} {'Acc':>7}  {'95% CI':>17}  {'N':>5}")
print("-" * 65)

acc_all = all_correct / all_n * 100
print(f"{'All 273':<30} {acc_all:>6.1f}%  "
      f"[{ci_all[0]*100:5.1f}% – {ci_all[1]*100:5.1f}%]  {all_n:>5}")

acc_novel = novel_correct / novel_n * 100
print(f"{'Novel 267 (no contamination)':<30} {acc_novel:>6.1f}%  "
      f"[{ci_novel[0]*100:5.1f}% – {ci_novel[1]*100:5.1f}%]  {novel_n:>5}")

print()
print(f"{'Category':<10} {'Correct':>8} {'Total':>6} {'Acc':>8}")
print("-" * 36)
for cat in ['IT', 'HE', 'SHE', 'THEY', 'POSS']:
    if cat in cats:
        c, t = cats[cat]
        lo, hi = wilson_ci(c, t)
        print(f"  {cat:<8} {c:>8} {t:>6} {c/t:>8.1%}  [{lo:.1%}–{hi:.1%}]")

print()
print("-" * 65)
print("HEADLINE (novel-267):")
print(f"  Accuracy : {acc_novel:.1f}%  "
      f"(95% CI: {ci_novel[0]*100:.1f}%–{ci_novel[1]*100:.1f}%)")
print(f"  Correct  : {novel_correct}/{novel_n}")
print()
print("Baselines:")
print("  Random (50/50 binary choice) : 50.0%")
print("  BERT-base zero-shot WSC      : no verified baseline on hand")
print("    (Published figures for BERT on WSC vary by paper and setup;")
print("     citing without a specific paper would be guessing.)")
print()
print(f"vs random: {acc_novel - 50.0:+.1f} pp  "
      f"({'above' if acc_novel > 50 else 'below'} chance)")

ci_excludes_50 = ci_novel[0] * 100 > 50.0
print(f"CI entirely above 50%: {ci_excludes_50}")
print("-" * 65)
