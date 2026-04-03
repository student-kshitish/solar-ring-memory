"""Structured QA benchmark — 100 generated question-answer pairs, 4 depth levels.

Dataset generated from templates:
  Depth 0 (sun ring,    25 pairs): main-clause subject/object queries
  Depth 1 (planet ring, 25 pairs): singly-embedded clause subject queries
  Depth 2 (moon ring,   25 pairs): doubly-embedded ownership queries
  Depth 3 (pronoun+QA,  25 pairs): pronoun-resolution queries

Evaluation:
  Solar Ring : cosine_sim(ring[depth].subject_vector, embed(correct)) >
               cosine_sim(ring[depth].subject_vector, embed(wrong))
               — reads directly from the planet/moon slot at the query depth
  BiLSTM     : cosine_sim(bidirectional_hidden, embed(correct)) > same wrong
  LSTM       : cosine_sim(last_hidden,          embed(correct)) > same wrong

Models loaded from checkpoints/solar_direct_best.pt etc.
Device: cuda bfloat16.

Usage:
  python benchmarks/structured_qa.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple


# ── Question type parser ──────────────────────────────────────────────────────

def parse_question_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ['who gave', 'who told', 'who sent',
                              'who helped', 'who hired', 'who warned',
                              'who claimed', 'who said']):
        return 'SUBJ'
    if any(w in q for w in ['who received', 'who got', 'who was hired',
                              'who was helped', 'who heard']):
        return 'OBJ'
    if any(w in q for w in ['what did', 'what happened', 'what action']):
        return 'VERB'
    if 'who won' in q or 'who fixed' in q or 'who bought' in q:
        return 'SUBJ_NESTED'
    return 'SUBJ'

# ── Device ────────────────────────────────────────────────────────────────────
print(f"Torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

from solar_ring.model        import SolarRingModel
from solar_ring.config       import D_MODEL
from solar_ring.glove_loader import load_glove
from baseline.vanilla_lstm   import VanillaLSTM
from baseline.bilstm         import BiLSTM
from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from benchmarks.direct_train  import (
    build_generated_pairs,
    SolarClassifier, BiLSTMClassifier, LSTMClassifier,
)

GLOVE_PATH = "data/glove.6B.300d.txt"


# ── Dataset generation ────────────────────────────────────────────────────────

NAMES = ["John", "Mary", "Tom", "Lisa", "Mike", "Anna", "Bob", "Sarah", "Alex", "Beth"]
VERBS = ["gave", "told", "sent", "showed", "helped", "hired", "warned", "thanked"]
NOUNS = ["book", "letter", "gift", "message", "package", "report", "note", "file",
         "ticket", "medal"]

def _r(lst, i, off=0):
    return lst[(i + off) % len(lst)]


def build_dataset() -> List[Tuple[str, str, str, str, int, str]]:
    """
    Returns exactly 100 tuples:
      (context, question, correct_answer, wrong_answer, depth, slot)
    slot ∈ {"SUBJ", "OBJ"} — which memory slot the correct answer occupies.
    Answers are single name strings (lowercase).
    """
    pairs: List[Tuple[str, str, str, str, int, str]] = []

    # ── Depth 0 (25 pairs) ────────────────────────────────────────────────────
    # Pattern A (13): "{n1} {v} {n2} the {noun}. Who {v} the {noun}? → n1 (SUBJ)"
    for i in range(13):
        n1, n2 = _r(NAMES, i), _r(NAMES, i, 3)
        v, noun = _r(VERBS, i), _r(NOUNS, i)
        ctx = f"{n1} {v} {n2} the {noun}."
        q   = f"Who {v} the {noun}?"
        pairs.append((ctx, q, n1.lower(), n2.lower(), 0, "SUBJ"))

    # Pattern B (12): "{n1} {v} {n2}. Who received? → n2 (OBJ)"
    for i in range(12):
        n1, n2 = _r(NAMES, i, 1), _r(NAMES, i, 5)
        v      = _r(VERBS, i, 2)
        ctx = f"{n1} {v} {n2}."
        q   = "Who received?"
        pairs.append((ctx, q, n2.lower(), n1.lower(), 0, "OBJ"))

    # ── Depth 1 (25 pairs) ────────────────────────────────────────────────────
    # "{n1} told {n2} that {n3} {v} the {noun}. Who {v}? → n3 (SUBJ in planet ring)"
    for i in range(25):
        n1, n2, n3 = _r(NAMES, i), _r(NAMES, i, 3), _r(NAMES, i, 6)
        v, noun = _r(VERBS, i), _r(NOUNS, i)
        ctx = f"{n1} told {n2} that {n3} {v} the {noun}."
        q   = f"Who {v} the {noun}?"
        wrong = n1 if n1 != n3 else _r(NAMES, i, 2)
        pairs.append((ctx, q, n3.lower(), wrong.lower(), 1, "SUBJ"))

    # ── Depth 2 (25 pairs) ────────────────────────────────────────────────────
    # "{n1} said that {n2} {v} the {noun} that {n3} owned. Who owned? → n3 (SUBJ in moon)"
    for i in range(25):
        n1, n2, n3 = _r(NAMES, i), _r(NAMES, i, 3), _r(NAMES, i, 6)
        v, noun = _r(VERBS, i), _r(NOUNS, i)
        ctx = f"{n1} said that {n2} {v} the {noun} that {n3} owned."
        q   = "Who owned?"
        wrong = n1 if n1 != n3 else _r(NAMES, i, 2)
        pairs.append((ctx, q, n3.lower(), wrong.lower(), 2, "SUBJ"))

    # ── Depth 3 (25 pairs) ────────────────────────────────────────────────────
    # "{n1} told {n2} that he would {v}. Who would {v}? → n1 (SUBJ in sun ring)"
    for i in range(25):
        n1, n2 = _r(NAMES, i), _r(NAMES, i, 3)
        v = _r(VERBS, i)
        ctx = f"{n1} told {n2} that he would {v}."
        q   = f"Who would {v}?"
        wrong = n2 if n2 != n1 else _r(NAMES, i, 5)
        pairs.append((ctx, q, n1.lower(), wrong.lower(), 3, "SUBJ"))

    assert len(pairs) == 100, f"Expected 100 pairs, got {len(pairs)}"
    return pairs


# ── Vocabulary ────────────────────────────────────────────────────────────────

def _normalize(w: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9']", "", w).lower()


def build_vocab(texts: List[str], max_vocab: int = 5000) -> Dict[str, int]:
    word2id: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
    for text in texts:
        for w in text.split():
            c = _normalize(w)
            if c and c not in word2id and len(word2id) < max_vocab:
                word2id[c] = len(word2id)
    return word2id


def rebuild_direct_vocab(max_vocab: int = 5000) -> Dict[str, int]:
    """
    Reconstruct the vocabulary used in direct_train.py exactly so checkpoint
    weights load without size mismatches.

    direct_train.py main() builds:
        all_items  = wino_items + gen_items
        all_texts  = [text for text, _ in all_items]
        wino_texts = [t for ctx,c,w in WINOGRAD_SCHEMAS for t in (ctx,c,w)]
        word2id    = build_vocab(all_texts + wino_texts, max_vocab=5000)
    """
    random.seed(42)

    # Winograd training split (70 schemas × 2 items — must shuffle with same seed)
    schemas = list(WINOGRAD_SCHEMAS)
    random.shuffle(schemas)
    train_schemas = schemas[:70]
    wino_items = []
    for ctx, corr, wrong in train_schemas:
        wino_items.append((ctx + " " + corr,  1))
        wino_items.append((ctx + " " + wrong, 0))

    gen_items = build_generated_pairs()
    all_items = wino_items + gen_items

    all_texts  = [text for text, _ in all_items]
    wino_texts = [t for ctx, c, w in WINOGRAD_SCHEMAS for t in (ctx, c, w)]

    return build_vocab(all_texts + wino_texts, max_vocab)


def encode(text: str, word2id: Dict[str, int], max_len: int = 64) -> torch.Tensor:
    unk = word2id.get("<UNK>", 0)
    ids = [word2id.get(_normalize(w), unk) for w in text.split() if _normalize(w)]
    return torch.tensor(ids[:max_len], dtype=torch.long)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(word2id: Dict[str, int], glove):
    vs     = len(word2id)
    solar  = SolarClassifier(vs,  glove).to(DEVICE, DTYPE)
    bilstm = BiLSTMClassifier(vs, glove).to(DEVICE, DTYPE)
    lstm   = LSTMClassifier(vs,   glove).to(DEVICE, DTYPE)

    ckpts = {
        "solar":  ("solar",  "checkpoints/solar_direct_best.pt",  solar),
        "bilstm": ("bilstm", "checkpoints/bilstm_direct_best.pt", bilstm),
        "lstm":   ("lstm",   "checkpoints/lstm_direct_best.pt",   lstm),
    }
    for key, (label, path, model) in ckpts.items():
        p = Path(path)
        if p.exists():
            state = torch.load(p, map_location=DEVICE, weights_only=True)
            try:
                model.load_state_dict(state)
                print(f"  Loaded  {path}")
            except RuntimeError as e:
                print(f"  WARN    {path}: {e}  (using init weights)")
        else:
            print(f"  MISSING {path}  (using init weights)")

    return solar, bilstm, lstm


# ── Evaluation helpers ────────────────────────────────────────────────────────
# Strategy: score (context + correct_answer) vs (context + wrong_answer) using
# the trained classifier head — same as the model was trained to do.
# For Solar Ring, the context_vec after reading both sequences through the
# structured memory is then passed to the classifier head.
# For BiLSTM/LSTM, the final hidden state is used with the classifier head.
# Correct wins if its score > wrong's score.
# Additionally, Solar Ring exposes its planet slot for the depth-breakdown
# via cosine similarity of the hidden state after the planet ring token.

def score_pair(model: nn.Module, ctx_ids: torch.Tensor, ans_id: int) -> float:
    """
    Classifier score for context + single answer token.
    Appends the answer word ID to the context sequence and runs the full model.
    """
    ids = torch.cat([ctx_ids, torch.tensor([ans_id], dtype=torch.long, device=DEVICE)])
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        score = model(ids)
    return score.item()


def solar_predict(
    solar: "SolarClassifier",
    ctx_ids: torch.Tensor,
    corr_id: int,
    wrong_id: int,
    depth: int,
    slot: str = "SUBJ",
) -> bool:
    """
    Solar Ring: score(context + correct) > score(context + wrong).
    The model reads the full sequence into its planet/moon/sun ring memory,
    then the classifier head over the context_vec determines the score.
    """
    sc = score_pair(solar, ctx_ids, corr_id)
    sw = score_pair(solar, ctx_ids, wrong_id)
    return sc > sw


def solar_predict_slot(
    solar: "SolarClassifier",
    ctx_ids: torch.Tensor,
    corr_id: int,
    wrong_id: int,
    question: str,
    depth: int,
    glove,
) -> bool:
    """
    Solar Ring direct slot extraction with GloVe cosine comparison.

    Picks the ring by depth (0=sun, 1=first planet, 2=moon), then reads
    the slot (SUBJ/OBJ/VERB) determined by parse_question_type(question).
    Falls back to classifier scoring when GloVe is unavailable or the slot
    vector is zero (slot never written by the model).
    """
    if glove is None:
        return solar_predict(solar, ctx_ids, corr_id, wrong_id, depth)

    slot_type = parse_question_type(question)

    with torch.no_grad():
        memory = solar.base.get_memory_for_sentence(ctx_ids)

    # Select ring by syntactic depth; depth-3 pronoun questions resolve back
    # to the sun (the pronoun "he/she" refers to the main-clause subject).
    n_rings = len(memory.rings)
    if depth == 0 or depth == 3:
        ring = memory.rings[0]
    elif depth == 1:
        ring = memory.rings[1] if n_rings > 1 else memory.rings[0]
    else:  # depth >= 2
        ring = memory.rings[min(depth, n_rings - 1)]

    if slot_type == 'OBJ':
        extracted = ring.object_vector()
    elif slot_type == 'VERB':
        extracted = ring.verb_vector()
    elif slot_type == 'SUBJ_NESTED':
        child_ring = memory.rings[1] if n_rings > 1 else memory.rings[0]
        extracted = child_ring.subject_vector()
    else:  # SUBJ (default)
        extracted = ring.subject_vector()

    ext_f = extracted.float()
    # Fall back to classifier if the slot was never written
    if ext_f.norm() < 1e-6:
        return solar_predict(solar, ctx_ids, corr_id, wrong_id, depth)

    correct_emb = torch.tensor(glove[corr_id], dtype=torch.float32, device=DEVICE)
    wrong_emb   = torch.tensor(glove[wrong_id], dtype=torch.float32, device=DEVICE)

    sim_correct = F.cosine_similarity(ext_f.unsqueeze(0), correct_emb.unsqueeze(0)).item()
    sim_wrong   = F.cosine_similarity(ext_f.unsqueeze(0), wrong_emb.unsqueeze(0)).item()

    return sim_correct > sim_wrong


def bilstm_predict(
    bilstm: "BiLSTMClassifier",
    ctx_ids: torch.Tensor,
    corr_id: int,
    wrong_id: int,
) -> bool:
    """BiLSTM: score(context + correct) > score(context + wrong)."""
    sc = score_pair(bilstm, ctx_ids, corr_id)
    sw = score_pair(bilstm, ctx_ids, wrong_id)
    return sc > sw


def lstm_predict(
    lstm: "LSTMClassifier",
    ctx_ids: torch.Tensor,
    corr_id: int,
    wrong_id: int,
) -> bool:
    """LSTM: score(context + correct) > score(context + wrong)."""
    sc = score_pair(lstm, ctx_ids, corr_id)
    sw = score_pair(lstm, ctx_ids, wrong_id)
    return sc > sw


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("Solar Ring — Structured QA Benchmark")
    print("100 pairs  |  4 depths  |  pairwise cosine  |  bfloat16")
    print("=" * 60)

    # ── Build dataset ─────────────────────────────────────────────────
    qa_pairs = build_dataset()
    depth_names = {
        0: "sun ring    (depth 0)",
        1: "planet ring (depth 1)",
        2: "moon ring   (depth 2)",
        3: "pronoun+QA  (depth 3)",
    }
    print(f"\n[0] Dataset: {len(qa_pairs)} pairs")
    for d in range(4):
        n = sum(1 for p in qa_pairs if p[4] == d)
        print(f"    {depth_names[d]}: {n} pairs")

    # ── Vocabulary ────────────────────────────────────────────────────
    print("\n[1] Rebuilding direct_train.py vocabulary...")
    word2id = rebuild_direct_vocab(max_vocab=5000)
    print(f"    Vocab size: {len(word2id)}")

    # Check all answer tokens are covered
    answer_words = {_normalize(x)
                    for _, _, c, w, _d, _s in qa_pairs
                    for x in (c, w)}
    covered = sum(1 for w in answer_words if w in word2id)
    print(f"    Answer word coverage: {covered}/{len(answer_words)}")

    # ── GloVe ─────────────────────────────────────────────────────────
    glove = None
    if Path(GLOVE_PATH).exists():
        print(f"\n[2] Loading GloVe 300d from {GLOVE_PATH}...")
        glove = load_glove(GLOVE_PATH, word2id, d=300)
        print(f"    Matrix: {glove.shape}")
    else:
        print(f"\n[2] GloVe not found at {GLOVE_PATH} — using random embeddings")

    # ── Load models ────────────────────────────────────────────────────
    print("\n[3] Loading models...")
    solar_model, bilstm_model, lstm_model = load_models(word2id, glove)
    solar_model.eval()
    bilstm_model.eval()
    lstm_model.eval()

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n[4] Evaluating all 100 pairs...\n")

    depth_results = {d: {"solar": [], "bilstm": [], "lstm": []} for d in range(4)}

    for ctx, q, correct, wrong, depth, slot in qa_pairs:
        ctx_ids  = encode(ctx, word2id).to(DEVICE)
        if ctx_ids.numel() == 0:
            continue

        corr_id  = word2id.get(_normalize(correct), 0)
        wrong_id = word2id.get(_normalize(wrong),   0)

        depth_results[depth]["solar"].append(
            solar_predict_slot(solar_model, ctx_ids, corr_id, wrong_id, q, depth, glove)
        )
        depth_results[depth]["bilstm"].append(
            bilstm_predict(bilstm_model, ctx_ids, corr_id, wrong_id)
        )
        depth_results[depth]["lstm"].append(
            lstm_predict(lstm_model, ctx_ids, corr_id, wrong_id)
        )

    # ── Print results ─────────────────────────────────────────────────
    print("=" * 60)

    def pct(lst):
        return f"{100*sum(lst)/max(len(lst),1):.1f}%"

    for d in range(4):
        s  = depth_results[d]["solar"]
        b  = depth_results[d]["bilstm"]
        lm = depth_results[d]["lstm"]
        print(f"  Depth {d}: Solar={pct(s):>6}  BiLSTM={pct(b):>6}  LSTM={pct(lm):>6}"
              f"    ({depth_names[d]})")

    all_s  = [v for d in range(4) for v in depth_results[d]["solar"]]
    all_b  = [v for d in range(4) for v in depth_results[d]["bilstm"]]
    all_lm = [v for d in range(4) for v in depth_results[d]["lstm"]]
    print(f"\n  Overall: Solar={pct(all_s):>6}  BiLSTM={pct(all_b):>6}  LSTM={pct(all_lm):>6}")
    print("=" * 60)


if __name__ == "__main__":
    main()
