"""World-knowledge rules for pronoun / antecedent resolution.

Applied as a post-hoc score adjustment on top of classifier scores —
no retraining required.

Rule logic
----------
HE / HIM / HIS  →  prefer ANIMATE candidates, penalise INANIMATE
SHE / HER / HERS →  prefer ANIMATE candidates, penalise INANIMATE
IT / ITS         →  prefer INANIMATE candidates, penalise ANIMATE
THEY / THEM      →  neutral (both animate collectives are common)
"""

from typing import List

# ── World-knowledge dictionaries ──────────────────────────────────────────────

OBJECT_SIZES = {
    'trophy': 'large', 'suitcase': 'large',
    'cup': 'small',    'plate': 'medium',
    'ball': 'medium',  'window': 'large',
    'car': 'large',    'tree': 'large',
    'cat': 'small',    'dog': 'medium',
    'elephant': 'huge','ant': 'tiny',
    'book': 'small',   'box': 'medium',
    'bottle': 'small', 'table': 'large',
    'bag': 'medium',   'vase': 'medium',
    'jar': 'small',    'bucket': 'medium',
    'rock': 'medium',  'tile': 'small',
    'fence': 'large',  'pipe': 'medium',
    'glass': 'small',  'mat': 'medium',
}

ANIMATE = {
    'john', 'mary', 'tom', 'lisa', 'mike', 'anna',
    'bob',  'sarah','alex','beth', 'chris','emma',
    'paul', 'dave', 'nick','jake', 'mark', 'sam',
    'tim',  'steve','george','susan','carol','diana',
    'rachel','alice','amy', 'joan', 'bill', 'george',
    'cat',  'dog',  'bird', 'fish', 'wolf', 'deer',
    'hawk', 'rabbit','elephant','ant',
    'man',  'woman','boy',  'girl','child',
    'person','people','student','students',
    'teacher','teachers','doctor','nurse','lawyer',
    'worker','workers','manager','managers',
    'police','army','rebels','director','employee',
    'scientist','judge','defendant','intern','professor',
    'he', 'she', 'they', 'him', 'her', 'his', 'hers',
}

INANIMATE = {
    'trophy', 'suitcase', 'cup', 'plate', 'ball',
    'window', 'car',  'tree',  'book',  'box',
    'bottle', 'table','rock',  'vase',  'bag',
    'jar',    'bucket','pipe', 'glass', 'mat',
    'fence',  'tile', 'sand',  'water', 'theory',
    'essay',  'score','letter','prize', 'gift',
    'present','token','hammer','chicken',
    'it', 'its',
}

# Pronouns that must refer to animate entities
_ANIMATE_PRONOUNS   = {'he', 'him', 'his', 'she', 'her', 'hers'}
# Pronouns that must refer to inanimate entities
_INANIMATE_PRONOUNS = {'it', 'its'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(word: str) -> str:
    return word.lower().strip(".,!?;:\"'()-[]")


def extract_pronoun(context: str) -> str:
    """Return the first pronoun found in context, lower-cased, or ''."""
    all_pronouns = _ANIMATE_PRONOUNS | _INANIMATE_PRONOUNS | {'they', 'them', 'their'}
    for w in context.split():
        c = _clean(w)
        if c in all_pronouns:
            return c
    return ''


def extract_candidate(answer_text: str) -> str:
    """
    Extract the likely antecedent noun from an answer string.
    Answers look like:
        'The trophy was too big.'   → 'trophy'
        'Susan had given help.'     → 'susan'
        'The police feared riots.'  → 'police'
    Returns first content word after stripping stop words.
    """
    skip = {'the', 'a', 'an', 'was', 'were', 'is', 'are', 'had', 'have',
            'has', 'would', 'will', 'should', 'could', 'did', 'does', 'do'}
    for w in answer_text.split():
        c = _clean(w)
        if c and c not in skip:
            return c
    return ''


# ── Core scoring function ─────────────────────────────────────────────────────

def knowledge_score(pronoun: str, candidate: str, context: str = '') -> float:
    """
    Return a score adjustment for (pronoun, candidate) pair.

    Positive → candidate is more likely the correct referent.
    Negative → candidate is less likely.
    Zero     → no applicable rule.
    """
    p = pronoun.lower().strip()
    c = candidate.lower().strip()

    if p in _ANIMATE_PRONOUNS:
        if c in ANIMATE:
            return +0.3
        if c in INANIMATE:
            return -0.5

    if p in _INANIMATE_PRONOUNS:
        if c in INANIMATE:
            return +0.3
        if c in ANIMATE:
            return -0.3

    return 0.0


# ── Knowledge-adjusted Winograd evaluator ────────────────────────────────────

def knowledge_adjusted_eval(model, word2id: dict, device, dtype,
                             schemas: List = None) -> tuple:
    """
    Evaluate model on Winograd schemas with knowledge score adjustment.

    For each schema:
      adjusted_score = classifier_score + knowledge_score(pronoun, candidate)

    Returns (accuracy, per-category dict, n_rules_applied).
    """
    from benchmarks.winograd_full import WINOGRAD_SCHEMAS, _pronoun_category
    from benchmarks.direct_train  import encode
    import torch

    if schemas is None:
        schemas = WINOGRAD_SCHEMAS

    model.eval()
    cat_correct: dict = {}
    cat_total:   dict = {}
    n_rules = 0

    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        for ctx, corr, wrong in schemas:
            cat = _pronoun_category(ctx)
            cat_total[cat]   = cat_total.get(cat, 0) + 1
            cat_correct[cat] = cat_correct.get(cat, 0)

            ids_c = encode(ctx + " " + corr,  word2id).to(device)
            ids_w = encode(ctx + " " + wrong, word2id).to(device)

            sc = model(ids_c).item() if ids_c.numel() > 0 else 0.5
            sw = model(ids_w).item() if ids_w.numel() > 0 else 0.5

            # Knowledge adjustment
            pronoun    = extract_pronoun(ctx)
            cand_corr  = extract_candidate(corr)
            cand_wrong = extract_candidate(wrong)

            adj_c = knowledge_score(pronoun, cand_corr,  ctx)
            adj_w = knowledge_score(pronoun, cand_wrong, ctx)

            if adj_c != 0.0 or adj_w != 0.0:
                n_rules += 1

            sc_adj = sc + adj_c
            sw_adj = sw + adj_w

            if sc_adj > sw_adj:
                cat_correct[cat] += 1

    total         = len(schemas)
    total_correct = sum(cat_correct.values())
    acc           = total_correct / total
    cat_accs      = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}

    return acc, cat_accs, n_rules
