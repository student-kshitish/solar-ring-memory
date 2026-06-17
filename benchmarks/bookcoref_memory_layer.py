"""
Cross-chunk entity memory layer for BookCoref.

Pipeline:
  FCoref per 512-token chunk (same as real baseline)
  → EntityMemory: merge clusters across chunks using
      (1) normalized surface-form match    — primary, deterministic
      (2) RoBERTa embedding cosine sim     — fallback, threshold 0.85
  → CoNLL F1 evaluation (MUC + B³ + CEAFe)

Thresholds chosen by inspection, NOT tuned on the sealed test books.
Sealed test set: Siddhartha + Pride & Prejudice. Evaluated ONCE.
"""

import json, re, warnings, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as cosine_dist
import nltk, spacy
from nltk.tokenize import sent_tokenize
from deepdiff import Delta
from deepdiff.serialization import json_loads

# ── silence noise ─────────────────────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

import fastcoref.modeling as _fm
if not hasattr(_fm.FCorefModel, "all_tied_weights_keys"):
    _fm.FCorefModel.all_tied_weights_keys = property(
        lambda self: {k: None for k in (self._tied_weights_keys or [])})
from fastcoref import FCoref

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path(__file__).parent.parent / "results"
CACHE_DIR    = Path.home() / ".cache/huggingface/datasets/downloads"
TEST_JSONL   = CACHE_DIR / "f7f0d411d70d2531eb0935fa3cd69683125c645b94d490183126497304497d97"
DELTA_FILE   = CACHE_DIR / "f0eca9ccd4652f8052f04a8a66c79dd2d5cd21e5f5be43a29713fad599f9e05c"
LENGTHS_FILE = CACHE_DIR / "fcb01fa02443a80b9f6a9b0fa0cc56de9af0b21d8b4a5297ea00cfd2ff30a433"

LOCAL_TEXTS = {
    "2500": "/tmp/pg2500_exact.txt",
    "1342": "/tmp/pg1342_exact.txt",
}
CHUNK_SIZE = 512

# ── CoNLL F1 (self-contained) ─────────────────────────────────────────────────

def _f1(p, r):
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

def muc(pred, gold):
    def _score(key, response):
        total = sum(len(c) for c in key)
        if total == 0: return 0.0
        num = 0
        for kc in key:
            seen, unmatched = set(), 0
            for m in kc:
                matched = next((i for i,rc in enumerate(response) if m in rc), None)
                if matched is not None: seen.add(matched)
                else: unmatched += 1
            num += len(kc) - max(len(seen)+unmatched, 1)
        return num / sum(len(c)-1 for c in key)
    r = _score(gold, pred); p = _score(pred, gold)
    return p, r, _f1(p, r)

def b_cubed(pred, gold):
    gmap = {m:i for i,c in enumerate(gold) for m in c}
    pmap = {m:i for i,c in enumerate(pred) for m in c}
    all_m = set(gmap) | set(pmap)
    if not all_m: return 1.,1.,1.
    pv, rv = [], []
    for m in all_m:
        gc = gold[gmap[m]] if m in gmap else set()
        pc = pred[pmap[m]] if m in pmap else set()
        ov = gc & pc
        pv.append(len(ov)/len(pc) if pc else 0.)
        rv.append(len(ov)/len(gc) if gc else 0.)
    p, r = sum(pv)/len(pv), sum(rv)/len(rv)
    return p, r, _f1(p, r)

def ceafe(pred, gold):
    if not gold or not pred: return 0.,0.,0.
    phi = lambda g,p: 2*len(g&p)/(len(g)+len(p)) if (len(g)+len(p))>0 else 0.
    cost = np.array([[phi(g,p) for p in pred] for g in gold])
    ri, ci = linear_sum_assignment(-cost)
    sc = float(cost[ri,ci].sum())
    return sc/len(pred), sc/len(gold), _f1(sc/len(pred), sc/len(gold))

def conll_f1(pred_cls, gold_cls):
    g = [set(map(tuple,c)) for c in gold_cls if c]
    p = [set(map(tuple,c)) for c in pred_cls if c]
    mp,mr,mf = muc(p,g); bp,br,bf = b_cubed(p,g); ep,er,ef = ceafe(p,g)
    return {
        "muc":   {"p":round(mp*100,2),"r":round(mr*100,2),"f":round(mf*100,2)},
        "b3":    {"p":round(bp*100,2),"r":round(br*100,2),"f":round(bf*100,2)},
        "ceafe": {"p":round(ep*100,2),"r":round(er*100,2),"f":round(ef*100,2)},
        "conll_f1": round((mf+bf+ef)/3*100, 2),
    }

# ── surface-form helpers ──────────────────────────────────────────────────────

PRONOUNS = {
    'he','she','it','they','his','her','its','their','him','them',
    'we','our','us','i','my','me','you','your',
    'who','whom','which','that','this','these','those',
    'himself','herself','itself','themselves','yourself','yourselves',
    'myself','ourselves','nothing','everything','something','anything',
    'the','a','an','one','some','both','each','every','all',
    'here','there','what','whoever','whatever',
}

def normalize_surface(text: str) -> str:
    """Lowercase, strip possessives and leading articles."""
    text = text.lower().strip()
    text = re.sub(r"'s$|s'$", "", text)
    text = re.sub(r"^(the|a|an) +", "", text)
    return text.strip(".,!?;:\"'()[]{}").strip()

def cluster_surface_forms(mentions: list[tuple], chunk_tokens: list[str]) -> set[str]:
    """Normalized non-pronoun strings for all mentions in a chunk cluster."""
    forms = set()
    for (t_s, t_e) in mentions:
        raw = " ".join(chunk_tokens[t_s : t_e+1])
        norm = normalize_surface(raw)
        if len(norm) > 2 and norm not in PRONOUNS:
            forms.add(norm)
    return forms

def representative_text(mentions: list[tuple], chunk_tokens: list[str]) -> Optional[str]:
    """Return the longest non-pronoun mention string, or None if all pronouns."""
    candidates = []
    for (t_s, t_e) in mentions:
        raw = " ".join(chunk_tokens[t_s : t_e+1])
        norm = normalize_surface(raw)
        if len(norm) > 2 and norm not in PRONOUNS:
            candidates.append(raw)
    if not candidates:
        return None
    return max(candidates, key=len)

# ── embedding helper ──────────────────────────────────────────────────────────

def embed_text(text: str, tokenizer, encoder, device: str) -> Optional[np.ndarray]:
    """RoBERTa mean-pool embedding for a short text span (≤64 subwords)."""
    enc = tokenizer(text, return_tensors="pt", max_length=64,
                    truncation=True, add_special_tokens=True).to(device)
    with torch.no_grad():
        out = encoder(**enc)
    hidden = out.last_hidden_state[0, 1:-1]  # skip CLS and SEP
    if hidden.shape[0] == 0:
        return None
    return hidden.mean(0).cpu().float().numpy()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine_dist(a.astype(float), b.astype(float)))

# ── entity memory ─────────────────────────────────────────────────────────────

EMB_THRESHOLD = 0.85   # cosine similarity threshold for embedding match

@dataclass
class Entity:
    entity_id:    int
    mentions:     list          = field(default_factory=list)  # doc-global (t_s, t_e) tuples
    surface_forms: set          = field(default_factory=set)   # normalized strings
    embedding:    Optional[np.ndarray] = None
    _n_mentions:  int           = 0                            # for running mean tracking

class EntityMemory:
    """Incremental entity memory — processes one chunk at a time."""

    def __init__(self, tokenizer, encoder, device: str,
                 emb_threshold: float = EMB_THRESHOLD):
        self.entities: list[Entity] = []
        self.tokenizer  = tokenizer
        self.encoder    = encoder
        self.device     = device
        self.emb_threshold = emb_threshold

    # ── matching ──────────────────────────────────────────────────────────────

    def _find_surface_match(self, forms: set[str]) -> Optional[int]:
        """Return entity index if any surface form matches an existing entity."""
        for i, ent in enumerate(self.entities):
            if forms & ent.surface_forms:
                return i
        return None

    def _find_embedding_match(self, emb: Optional[np.ndarray]) -> Optional[int]:
        """Return entity index of highest cosine-sim entity above threshold, or None."""
        if emb is None:
            return None
        best_idx, best_sim = None, self.emb_threshold
        for i, ent in enumerate(self.entities):
            if ent.embedding is None:
                continue
            sim = cosine_sim(emb, ent.embedding)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx

    # ── update ────────────────────────────────────────────────────────────────

    def _merge(self, entity_idx: int, new_mentions: list,
               new_forms: set, new_emb: Optional[np.ndarray]):
        ent = self.entities[entity_idx]
        ent.mentions.extend(new_mentions)
        ent.surface_forms.update(new_forms)
        # Running mean for embedding
        n_new = len(new_mentions)
        if new_emb is not None:
            if ent.embedding is None:
                ent.embedding = new_emb
            else:
                n_old = ent._n_mentions
                ent.embedding = (ent.embedding * n_old + new_emb * n_new) / (n_old + n_new)
        ent._n_mentions += n_new

    def _create(self, mentions: list, forms: set, emb: Optional[np.ndarray]):
        ent = Entity(
            entity_id=len(self.entities),
            mentions=list(mentions),
            surface_forms=set(forms),
            embedding=emb,
            _n_mentions=len(mentions),
        )
        self.entities.append(ent)

    # ── process one chunk ─────────────────────────────────────────────────────

    def add_cluster(self,
                    global_mentions: list[tuple],
                    forms: set[str],
                    emb: Optional[np.ndarray]) -> int:
        """
        Merge a cluster into an existing entity or create a new one.
        global_mentions: doc-global (t_s, t_e) token spans for storage.
        forms: pre-computed surface forms (use chunk-LOCAL spans to extract them).
        emb:  pre-computed embedding of the representative mention.
        """
        # Priority 1: surface form
        idx = self._find_surface_match(forms)
        if idx is not None:
            self._merge(idx, global_mentions, forms, emb)
            return idx

        # Priority 2: embedding similarity (only for clusters with named mentions)
        if emb is not None:
            idx = self._find_embedding_match(emb)
            if idx is not None:
                self._merge(idx, global_mentions, forms, emb)
                return idx

        # No match → new entity
        self._create(global_mentions, forms, emb)
        return len(self.entities) - 1

    def predicted_clusters(self) -> list[list[tuple]]:
        return [ent.mentions for ent in self.entities if ent.mentions]

# ── text reconstruction helpers ───────────────────────────────────────────────

def reconstruct_book(key: str, all_keys: set, delta, lengths: dict,
                     nlp, verbose: bool = True) -> tuple[list[str], list[str]]:
    text = Path(LOCAL_TEXTS[key]).read_text(encoding="utf-8")
    if verbose:
        print(f"  Tokenising {len(text):,} chars...")
    raw_flat = [t.text for s in nlp.pipe(sent_tokenize(text), n_process=1) for t in s]
    if verbose:
        print(f"  Applying delta...")
    wrapper = {k: [] for k in all_keys}
    wrapper[key] = raw_flat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = wrapper + delta
    flat = patched[key]
    sents = []
    pos = 0
    for slen in lengths[key]:
        sents.append(flat[pos:pos+slen])
        pos += slen
    if verbose:
        print(f"  Post-delta: {len(flat):,} tokens in {len(sents)} sents")
    return flat, sents

def chunk_starts_from_lengths(sent_lens: list[int], chunk_size: int) -> list[int]:
    starts, pos, total = [0], 0, sum(sent_lens)
    for slen in sent_lens:
        pos += slen
        if pos - starts[-1] >= chunk_size and pos < total:
            starts.append(pos)
    return starts

def build_char_to_tok(tokens: list[str]):
    pos, c2t = 0, []
    for i, tok in enumerate(tokens):
        for _ in tok:
            c2t.append(i)
            pos += 1
        c2t.append(None)   # space separator
        pos += 1
    return " ".join(tokens), c2t

def charspan_to_tokspan(c_start: int, c_end_excl: int, c2t: list):
    c_end = c_end_excl - 1
    if c_start >= len(c2t) or c_end >= len(c2t): return None
    t_s = c2t[c_start]; t_e = c2t[c_end]
    if t_s is None or t_e is None: return None
    return (t_s, t_e)

def gold_clusters(book: dict) -> list[list[tuple]]:
    return [[(m[0], m[1]) for m in ch["mentions"]] for ch in book["characters"] if ch["mentions"]]

# ── full pipeline per book ────────────────────────────────────────────────────

def run_book(book: dict, flat: list[str], sent_lens: list[int],
             coref, device: str) -> dict:
    """
    Run FCoref on 512-token chunks and collect:
      (a) chunked-only predictions (baseline)
      (b) memory-merged predictions (after)
    Returns a dict with both sets of results.
    """
    doc_key = book["doc_key"]
    gcls    = gold_clusters(book)
    n_gold_mentions = sum(len(c) for c in gcls)
    total_tokens    = len(flat)

    starts   = chunk_starts_from_lengths(sent_lens, CHUNK_SIZE)
    ends     = starts[1:] + [total_tokens]
    n_chunks = len(starts)

    tokenizer = coref.tokenizer
    encoder   = coref.model.roberta

    # Two parallel output streams
    pred_base   = []       # baseline: per-chunk clusters, never linked
    memory      = EntityMemory(tokenizer, encoder, device)

    print(f"  {n_chunks} chunks × ≤{CHUNK_SIZE} tokens  |  "
          f"gold: {len(gcls)} entities / {n_gold_mentions} mentions")
    print(f"  Processing chunks: ", end="", flush=True)

    for ci, (cs, ce) in enumerate(zip(starts, ends)):
        chunk_toks = flat[cs:ce]
        chunk_text, c2t = build_char_to_tok(chunk_toks)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = coref.predict(texts=[chunk_text])

        char_clusters = preds[0].get_clusters(as_strings=False)

        for char_cl in char_clusters:
            # Convert char spans → chunk-LOCAL token spans first
            local_cl = []
            for (c_s, c_e) in char_cl:
                span = charspan_to_tokspan(c_s, c_e, c2t)
                if span is not None:
                    local_cl.append(span)  # (local_t_s, local_t_e)
            if not local_cl:
                continue

            # Compute surface forms and embedding using LOCAL indices + chunk tokens
            forms = cluster_surface_forms(local_cl, chunk_toks)
            rep   = representative_text(local_cl, chunk_toks)
            emb   = embed_text(rep, tokenizer, encoder, device) if rep is not None else None

            # Shift to doc-global for storage
            global_cl = [(s + cs, e + cs) for (s, e) in local_cl]

            # (a) baseline: chunk-isolated clusters
            pred_base.append(global_cl)

            # (b) memory: merge into entity memory using pre-computed signals
            memory.add_cluster(global_cl, forms, emb)

        if (ci + 1) % 50 == 0 or ci == n_chunks - 1:
            print(f"{ci+1}", end=" ", flush=True)

    print()
    pred_memory = memory.predicted_clusters()

    # Evaluate both
    scores_base   = conll_f1(pred_base,   gcls)
    scores_memory = conll_f1(pred_memory, gcls)

    n_collapsed = len(memory.entities)

    print(f"\n  Clusters  before: {len(pred_base):4d}  →  after: {n_collapsed:4d}  "
          f"(gold: {len(gcls)})")
    print(f"  Mentions  before: {sum(len(c) for c in pred_base):5d}  →  "
          f"after: {sum(len(c) for c in pred_memory):5d}  (gold: {n_gold_mentions})")

    def fmt(s):
        return (f"MUC={s['muc']['f']}  B³={s['b3']['f']}  "
                f"CEAFe={s['ceafe']['f']}  CoNLL={s['conll_f1']}")

    print(f"\n  BEFORE  (chunked, no memory): {fmt(scores_base)}")
    print(f"  AFTER   (memory layer):        {fmt(scores_memory)}")
    delta_f1 = round(scores_memory["conll_f1"] - scores_base["conll_f1"], 2)
    print(f"  DELTA                        : {delta_f1:+.2f}% CoNLL F1")

    return {
        "doc_key":          doc_key,
        "n_chunks":         n_chunks,
        "n_gold_entities":  len(gcls),
        "n_gold_mentions":  n_gold_mentions,
        "n_pred_clusters_base":   len(pred_base),
        "n_pred_clusters_memory": n_collapsed,
        "emb_threshold":    EMB_THRESHOLD,
        "scores_before":    scores_base,
        "scores_after":     scores_memory,
        "delta_conll_f1":   delta_f1,
    }

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading lengths & delta (208 MB)...")
    with open(LENGTHS_FILE) as f:
        lengths = json.load(f)
    with open(DELTA_FILE) as f:
        raw_str = f.read()
    all_keys = set(re.findall(r"root\['(\d+)'\]", raw_str))
    delta = Delta(delta_path=str(DELTA_FILE), deserializer=json_loads)

    print("Loading FCoref model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coref = FCoref(device=device, enable_progress_bar=False)
    print(f"  GPU: {torch.cuda.memory_allocated()/1e6:.0f} MB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB")

    print("Loading sealed gold annotations...")
    with open(TEST_JSONL) as f:
        gold_books = [json.loads(l) for l in f]

    all_results = {}

    for book in gold_books:
        key = book["gutenberg_key"]
        if key == "0" or "animal_farm" in book["doc_key"]:
            print(f"\nSkipping {book['doc_key']} (Animal Farm — no public text)")
            continue

        print(f"\n{'='*64}")
        print(f"Book: {book['doc_key']}  (Gutenberg #{key})")

        flat, sents = reconstruct_book(key, all_keys, delta, lengths, nlp)
        sent_lens   = [len(s) for s in sents]

        result = run_book(book, flat, sent_lens, coref, device)
        all_results[book["doc_key"]] = result

    # ── final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("FINAL SUMMARY — BEFORE vs AFTER")
    print(f"{'='*64}")
    total_before, total_after = [], []
    for doc_key, res in all_results.items():
        sb = res["scores_before"];  sa = res["scores_after"]
        total_before.append(sb["conll_f1"])
        total_after.append(sa["conll_f1"])
        print(f"\n{doc_key}")
        print(f"  Entities: {res['n_pred_clusters_base']} chunks → "
              f"{res['n_pred_clusters_memory']} after memory "
              f"(gold: {res['n_gold_entities']})")
        print(f"  BEFORE  MUC={sb['muc']['f']}  B³={sb['b3']['f']}  "
              f"CEAFe={sb['ceafe']['f']}  CoNLL={sb['conll_f1']}")
        print(f"  AFTER   MUC={sa['muc']['f']}  B³={sa['b3']['f']}  "
              f"CEAFe={sa['ceafe']['f']}  CoNLL={sa['conll_f1']}")
        print(f"  DELTA   {res['delta_conll_f1']:+.2f}% CoNLL F1  "
              f"(B³ {sb['b3']['f']} → {sa['b3']['f']}  "
              f"CEAFe {sb['ceafe']['f']} → {sa['ceafe']['f']})")

    print(f"\nMean CoNLL F1  BEFORE: {sum(total_before)/len(total_before):.1f}%  "
          f"AFTER: {sum(total_after)/len(total_after):.1f}%  "
          f"DELTA: {(sum(total_after)-sum(total_before))/len(total_before):+.1f}%")

    out_path = RESULTS_DIR / "bookcoref_memory_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
