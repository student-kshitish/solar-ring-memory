"""
Context-aware cross-chunk entity memory layer.

Key addition over the simple version (bookcoref_memory_layer.py):
  Every cluster gets a CONTEXT embedding in addition to a text embedding.
  The context embedding encodes a ±window-token window around the mention
  through FCoref's own RoBERTa backbone.

  Three-level merge priority:
    1. Surface-form match          (same as v1, deterministic)
    2. Named-text embedding cosine  (same as v1, threshold text_threshold)
    3. Context embedding cosine     (NEW — fires for pronoun-only clusters too)

Tuning protocol (anti-leakage):
  - Dev book: O Pioneers! (Gutenberg #24) — silver, never in sealed test set.
  - Sweep context_window ∈ {16, 32, 64} and ctx_threshold ∈ {0.78..0.90}.
  - Freeze settings. Evaluate ONCE on sealed Siddhartha + P&P.

Outputs:
  results/bookcoref_context_memory_results.json
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
TRAIN_JSONL  = CACHE_DIR / "b6d7fe96eb07e502f23744e02314cd2db8c520074ee2fd9e0cf3ca45fb726009"
DELTA_FILE   = CACHE_DIR / "f0eca9ccd4652f8052f04a8a66c79dd2d5cd21e5f5be43a29713fad599f9e05c"
LENGTHS_FILE = CACHE_DIR / "fcb01fa02443a80b9f6a9b0fa0cc56de9af0b21d8b4a5297ea00cfd2ff30a433"

# Dev book: O Pioneers! — confirmed NOT in sealed test set (Siddhartha/P&P/Animal Farm)
DEV_BOOK_KEY = "24"
LOCAL_TEXTS  = {
    "2500": "/tmp/pg2500_exact.txt",  # Siddhartha (sealed)
    "1342": "/tmp/pg1342_exact.txt",  # P&P (sealed)
    "24":   "/tmp/pg24_dev.txt",      # O Pioneers! (dev — will be downloaded)
}
WM_URL = "https://web.archive.org/web/20250406105821/https://gutenberg.org/ebooks/{key}.txt.utf-8"
CHUNK_SIZE = 512

# ── CoNLL F1 ─────────────────────────────────────────────────────────────────
def _f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0.0
def muc(pred, gold):
    def _s(key, resp):
        tot = sum(len(c) for c in key)
        if tot == 0: return 0.
        num = 0
        for kc in key:
            seen, un = set(), 0
            for m in kc:
                h = next((i for i,rc in enumerate(resp) if m in rc), None)
                if h is not None: seen.add(h)
                else: un += 1
            num += len(kc) - max(len(seen)+un, 1)
        return num / sum(len(c)-1 for c in key)
    r = _s(gold,pred); p = _s(pred,gold); return p, r, _f1(p,r)
def b_cubed(pred, gold):
    gm={m:i for i,c in enumerate(gold) for m in c}
    pm={m:i for i,c in enumerate(pred) for m in c}
    all_m=set(gm)|set(pm)
    if not all_m: return 1.,1.,1.
    pv,rv=[],[]
    for m in all_m:
        gc=gold[gm[m]] if m in gm else set(); pc=pred[pm[m]] if m in pm else set(); ov=gc&pc
        pv.append(len(ov)/len(pc) if pc else 0.); rv.append(len(ov)/len(gc) if gc else 0.)
    p,r=sum(pv)/len(pv),sum(rv)/len(rv); return p,r,_f1(p,r)
def ceafe(pred, gold):
    if not gold or not pred: return 0.,0.,0.
    phi=lambda g,p: 2*len(g&p)/(len(g)+len(p)) if (len(g)+len(p))>0 else 0.
    cost=np.array([[phi(g,p) for p in pred] for g in gold])
    ri,ci=linear_sum_assignment(-cost); sc=float(cost[ri,ci].sum())
    return sc/len(pred),sc/len(gold),_f1(sc/len(pred),sc/len(gold))
def conll_f1(pred_cls, gold_cls):
    g=[set(map(tuple,c)) for c in gold_cls if c]; p=[set(map(tuple,c)) for c in pred_cls if c]
    mp,mr,mf=muc(p,g); bp,br,bf=b_cubed(p,g); ep,er,ef=ceafe(p,g)
    return {"muc":{"p":round(mp*100,2),"r":round(mr*100,2),"f":round(mf*100,2)},
            "b3": {"p":round(bp*100,2),"r":round(br*100,2),"f":round(bf*100,2)},
            "ceafe":{"p":round(ep*100,2),"r":round(er*100,2),"f":round(ef*100,2)},
            "conll_f1": round((mf+bf+ef)/3*100, 2)}

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
def normalize_surface(text):
    text = text.lower().strip()
    text = re.sub(r"'s$|s'$", "", text)
    text = re.sub(r"^(the|a|an) +", "", text)
    return text.strip(".,!?;:\"'()[]{}").strip()
def cluster_surface_forms(local_cl, chunk_toks):
    forms = set()
    for (t_s, t_e) in local_cl:
        raw = " ".join(chunk_toks[t_s:t_e+1])
        norm = normalize_surface(raw)
        if len(norm) > 2 and norm not in PRONOUNS:
            forms.add(norm)
    return forms
def representative_text(local_cl, chunk_toks):
    cands = []
    for (t_s, t_e) in local_cl:
        raw = " ".join(chunk_toks[t_s:t_e+1])
        norm = normalize_surface(raw)
        if len(norm) > 2 and norm not in PRONOUNS:
            cands.append(raw)
    return max(cands, key=len) if cands else None

# ── embedding helpers ──────────────────────────────────────────────────────────
def cosine_sim(a, b): return float(1.0 - cosine_dist(a.astype(float), b.astype(float)))

def embed_text(text, tokenizer, encoder, device):
    if not text or not text.strip(): return None
    enc = tokenizer(text, return_tensors="pt", max_length=64,
                    truncation=True, add_special_tokens=True).to(device)
    with torch.no_grad():
        out = encoder(**enc)
    h = out.last_hidden_state[0, 1:-1]
    return h.mean(0).cpu().float().numpy() if h.shape[0] > 0 else None

def embed_context_window(t_s, t_e, flat, tokenizer, encoder, device, window):
    """
    Embed the ±window token context around a mention using RoBERTa.
    Returns mean-pooled hidden states (excludes CLS and SEP).
    """
    ctx_start = max(0, t_s - window)
    ctx_end   = min(len(flat), t_e + window + 1)
    ctx_text  = " ".join(flat[ctx_start:ctx_end])
    if not ctx_text.strip():
        return None
    enc = tokenizer(ctx_text, return_tensors="pt", max_length=128,
                    truncation=True, add_special_tokens=True).to(device)
    with torch.no_grad():
        out = encoder(**enc)
    h = out.last_hidden_state[0, 1:-1]
    return h.mean(0).cpu().float().numpy() if h.shape[0] > 0 else None

def cluster_context_emb(global_cl, flat, tokenizer, encoder, device, window, max_mentions=3):
    """Mean of context embeddings over up to max_mentions mentions in the cluster."""
    embs = []
    for (t_s, t_e) in global_cl[:max_mentions]:
        emb = embed_context_window(t_s, t_e, flat, tokenizer, encoder, device, window)
        if emb is not None:
            embs.append(emb)
    return np.mean(embs, axis=0) if embs else None

# ── char → token helpers ─────────────────────────────────────────────────────
def build_char_to_tok(tokens):
    c2t = []
    for i, tok in enumerate(tokens):
        for _ in tok: c2t.append(i)
        c2t.append(None)
    return " ".join(tokens), c2t
def charspan_to_tokspan(c_start, c_end_excl, c2t):
    c_end = c_end_excl - 1
    if c_start >= len(c2t) or c_end >= len(c2t): return None
    t_s = c2t[c_start]; t_e = c2t[c_end]
    if t_s is None or t_e is None: return None
    return (t_s, t_e)
def chunk_starts_from_lengths(sent_lens, chunk_size):
    starts, pos, total = [0], 0, sum(sent_lens)
    for slen in sent_lens:
        pos += slen
        if pos - starts[-1] >= chunk_size and pos < total: starts.append(pos)
    return starts

# ── entity memory v2 ──────────────────────────────────────────────────────────
@dataclass
class EntityV2:
    entity_id:   int
    mentions:    list = field(default_factory=list)
    surface_forms: set = field(default_factory=set)
    text_emb:    Optional[np.ndarray] = None
    ctx_emb:     Optional[np.ndarray] = None
    _n:          int = 0

class EntityMemoryV2:
    def __init__(self):
        self.entities: list[EntityV2] = []

    def _running_mean(self, old_emb, new_emb, n_old, n_new):
        if old_emb is None: return new_emb
        if new_emb is None: return old_emb
        return (old_emb * n_old + new_emb * n_new) / (n_old + n_new)

    def _merge(self, idx, mentions, forms, text_emb, ctx_emb):
        e = self.entities[idx]
        n_new = len(mentions)
        e.text_emb = self._running_mean(e.text_emb, text_emb, e._n, n_new)
        # Only update ctx_emb from named-mention clusters; keeps entity ctx signature stable
        if forms:
            e.ctx_emb = self._running_mean(e.ctx_emb, ctx_emb, e._n, n_new)
        e.mentions.extend(mentions)
        e.surface_forms.update(forms)
        e._n += n_new

    def _best_sim_match(self, emb, attr, threshold):
        if emb is None: return None
        best_idx, best_sim = None, threshold
        for i, e in enumerate(self.entities):
            e_emb = getattr(e, attr)
            if e_emb is None: continue
            s = cosine_sim(emb, e_emb)
            if s > best_sim: best_sim, best_idx = s, i
        return best_idx

    def _best_named_ctx_match(self, emb, threshold):
        """Only match pronoun clusters against entities that have named mentions.
        Prevents pronoun→pronoun chaining collapse."""
        if emb is None: return None
        best_idx, best_sim = None, threshold
        for i, e in enumerate(self.entities):
            if not e.surface_forms: continue  # skip pronoun-only entities
            if e.ctx_emb is None: continue
            s = cosine_sim(emb, e.ctx_emb)
            if s > best_sim: best_sim, best_idx = s, i
        return best_idx

    def add_cluster(self, global_mentions, forms, text_emb, ctx_emb,
                    text_threshold=0.85, ctx_threshold=0.84):
        is_named = bool(forms)
        # 1. Surface form (strong, deterministic)
        for i, e in enumerate(self.entities):
            if forms & e.surface_forms:
                self._merge(i, global_mentions, forms, text_emb, ctx_emb)
                return i
        # 2. Named text embedding (only for named clusters)
        if is_named:
            idx = self._best_sim_match(text_emb, 'text_emb', text_threshold)
            if idx is not None:
                self._merge(idx, global_mentions, forms, text_emb, ctx_emb)
                return idx
        # 3. Context embedding: ONLY pronoun-only → named entities (blocks chaining)
        if not is_named:
            idx = self._best_named_ctx_match(ctx_emb, ctx_threshold)
            if idx is not None:
                self._merge(idx, global_mentions, forms, text_emb, ctx_emb)
                return idx
        # 4. New entity
        e = EntityV2(entity_id=len(self.entities), mentions=list(global_mentions),
                     surface_forms=set(forms), text_emb=text_emb, ctx_emb=ctx_emb,
                     _n=len(global_mentions))
        self.entities.append(e)
        return len(self.entities) - 1

    def predicted_clusters(self):
        return [e.mentions for e in self.entities if e.mentions]

# ── precompute cluster features for a book ────────────────────────────────────

def precompute_clusters(flat, sent_lens, coref, device, ctx_window):
    """
    Run FCoref once on all 512-token chunks; compute features for each cluster.
    Returns list of (global_cl, forms, text_emb, ctx_emb).
    Separates FCoref inference (slow) from ctx_emb recomputation (fast).
    """
    tokenizer = coref.tokenizer
    encoder   = coref.model.roberta
    starts = chunk_starts_from_lengths(sent_lens, CHUNK_SIZE)
    ends   = starts[1:] + [len(flat)]
    records = []
    for cs, ce in zip(starts, ends):
        chunk_toks = flat[cs:ce]
        chunk_text, c2t = build_char_to_tok(chunk_toks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = coref.predict(texts=[chunk_text])
        for char_cl in preds[0].get_clusters(as_strings=False):
            local_cl = [span for span in
                        (charspan_to_tokspan(c_s, c_e, c2t) for (c_s, c_e) in char_cl)
                        if span is not None]
            if not local_cl: continue
            forms    = cluster_surface_forms(local_cl, chunk_toks)
            rep_txt  = representative_text(local_cl, chunk_toks)
            text_emb = embed_text(rep_txt, tokenizer, encoder, device) if rep_txt else None
            global_cl = [(s+cs, e+cs) for (s,e) in local_cl]
            ctx_emb  = cluster_context_emb(global_cl, flat, tokenizer, encoder, device,
                                           window=ctx_window)
            records.append((global_cl, forms, text_emb, ctx_emb))
    return records

def recompute_ctx_embs(records, flat, tokenizer, encoder, device, ctx_window):
    """Recompute only the context embeddings for a new window size (fast, no FCoref)."""
    return [(gcl, forms, te, cluster_context_emb(gcl, flat, tokenizer, encoder,
                                                  device, ctx_window))
            for (gcl, forms, te, _) in records]

# ── apply memory to pre-computed records ─────────────────────────────────────

def apply_memory(records, text_threshold, ctx_threshold):
    """Pure function: apply EntityMemoryV2 to pre-computed records. No FCoref calls."""
    mem = EntityMemoryV2()
    baseline_pred = []
    for (gcl, forms, te, ce) in records:
        baseline_pred.append(gcl)
        mem.add_cluster(gcl, forms, te, ce, text_threshold, ctx_threshold)
    return baseline_pred, mem.predicted_clusters(), len(mem.entities)

# ── text + delta reconstruction ───────────────────────────────────────────────

def download_text(key):
    path = Path(LOCAL_TEXTS[key])
    if path.exists():
        return path.read_text(encoding="utf-8")
    url = WM_URL.format(key=key)
    print(f"  Downloading text for key {key} from Wayback Machine...")
    import subprocess
    subprocess.run(["curl", "-L", "--max-time", "60", "-s", url, "-o", str(path)],
                   check=True, capture_output=True)
    return path.read_text(encoding="utf-8")

def reconstruct_book(key, all_keys, delta, lengths, nlp, label=""):
    text = download_text(key)
    print(f"  [{label}] tokenising {len(text):,} chars...")
    raw_flat = [t.text for s in nlp.pipe(sent_tokenize(text), n_process=1) for t in s]
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
    print(f"  [{label}] {len(flat):,} tokens, {len(sents)} sents")
    return flat, [len(s) for s in sents]

def gold_clusters(book):
    return [[(m[0],m[1]) for m in ch["mentions"]] for ch in book["characters"] if ch["mentions"]]

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading lengths & delta (208 MB)...")
    with open(LENGTHS_FILE) as f: lengths = json.load(f)
    with open(DELTA_FILE) as f: raw_str = f.read()
    all_keys = set(re.findall(r"root\['(\d+)'\]", raw_str))
    delta = Delta(delta_path=str(DELTA_FILE), deserializer=json_loads)

    print("Loading FCoref model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coref = FCoref(device=device, enable_progress_bar=False)
    print(f"  GPU: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    # ── STEP 0: confirm dev book is NOT in sealed test set ────────────────────
    print(f"\n{'='*64}")
    print(f"STEP 0 — Dev book: O Pioneers! (Gutenberg #{DEV_BOOK_KEY})")
    sealed_keys = {"2500", "1342", "0"}   # Siddhartha, P&P, Animal Farm
    assert DEV_BOOK_KEY not in sealed_keys, "ANTI-LEAKAGE: dev book must not be in test set!"
    print(f"  Confirmed: key={DEV_BOOK_KEY} NOT in sealed test set {sealed_keys}")

    with open(TRAIN_JSONL) as f:
        silver_books = [json.loads(l) for l in f]
    dev_book = next(b for b in silver_books if b["gutenberg_key"] == DEV_BOOK_KEY)
    dev_gold = gold_clusters(dev_book)
    print(f"  Dev annotations: {len(dev_gold)} entities, "
          f"{sum(len(c) for c in dev_gold)} mentions")

    # ── STEP 1: precompute dev book cluster features ──────────────────────────
    print(f"\n{'='*64}")
    print("STEP 1 — Precompute FCoref clusters for dev book (O Pioneers!)...")
    dev_flat, dev_sent_lens = reconstruct_book(DEV_BOOK_KEY, all_keys, delta, lengths, nlp,
                                               label="dev")
    starts = chunk_starts_from_lengths(dev_sent_lens, CHUNK_SIZE)
    print(f"  {len(starts)} chunks × ≤{CHUNK_SIZE} tokens")

    # Initial precompute with window=32 (records contain FCoref output + ctx_embs)
    print("  Running FCoref + computing initial ctx_embs (window=32)...")
    dev_records_32 = precompute_clusters(dev_flat, dev_sent_lens, coref, device, ctx_window=32)
    n_cl = len(dev_records_32)
    print(f"  {n_cl} clusters extracted")

    # ── STEP 2: threshold sweep on dev book ──────────────────────────────────
    print(f"\n{'='*64}")
    print("STEP 2 — Threshold sweep on DEV book (O Pioneers!) only")

    TEXT_THRESH_VALS = [0.82, 0.85, 0.88]
    CTX_THRESH_VALS  = [0.80, 0.83, 0.86, 0.88, 0.90, 0.92, 0.94]
    CTX_WINDOWS      = [16, 32, 64]

    # Precompute ctx_embs for the other two window sizes
    print("  Computing ctx_embs for window=16 and window=64 (fast, no FCoref)...")
    tokenizer = coref.tokenizer; encoder = coref.model.roberta
    dev_records_16 = recompute_ctx_embs(dev_records_32, dev_flat, tokenizer, encoder,
                                        device, ctx_window=16)
    dev_records_64 = recompute_ctx_embs(dev_records_32, dev_flat, tokenizer, encoder,
                                        device, ctx_window=64)
    window_records = {16: dev_records_16, 32: dev_records_32, 64: dev_records_64}

    print(f"  Sweeping {len(CTX_WINDOWS)} windows × {len(TEXT_THRESH_VALS)} text_thresh × "
          f"{len(CTX_THRESH_VALS)} ctx_thresh = "
          f"{len(CTX_WINDOWS)*len(TEXT_THRESH_VALS)*len(CTX_THRESH_VALS)} configs...\n")

    best_f1, best_params = -1, None
    sweep_rows = []
    for w in CTX_WINDOWS:
        recs = window_records[w]
        for tt in TEXT_THRESH_VALS:
            for ct in CTX_THRESH_VALS:
                _, pred_mem, n_ents = apply_memory(recs, tt, ct)
                scores = conll_f1(pred_mem, dev_gold)
                f1 = scores["conll_f1"]
                b3 = scores["b3"]["f"]
                ce = scores["ceafe"]["f"]
                sweep_rows.append((w, tt, ct, n_ents, f1, b3, ce))
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (w, tt, ct)
                print(f"  w={w:2d} tt={tt:.2f} ct={ct:.2f}  "
                      f"entities={n_ents:4d}  CoNLL={f1:.2f}  B³={b3:.2f}  CEAFe={ce:.2f}")

    best_window, best_tt, best_ct = best_params
    print(f"\n  *** DEV BEST: window={best_window} text_thresh={best_tt} "
          f"ctx_thresh={best_ct}  CoNLL={best_f1:.2f}% ***")
    print(f"  (These settings are now FROZEN — sealed books evaluated once with them)\n")

    # ── STEP 3: evaluate ONCE on sealed test books ────────────────────────────
    print(f"\n{'='*64}")
    print("STEP 3 — SEALED EVALUATION (one shot, frozen dev settings)")
    print(f"  window={best_window}, text_threshold={best_tt}, ctx_threshold={best_ct}")

    with open(TEST_JSONL) as f:
        test_books = [json.loads(l) for l in f]

    all_results = {}

    for book in test_books:
        key = book["gutenberg_key"]
        if key == "0" or "animal_farm" in book["doc_key"]:
            print(f"\n  Skipping {book['doc_key']} (Animal Farm)")
            continue

        print(f"\n  {'='*56}")
        print(f"  Book: {book['doc_key']}  (Gutenberg #{key})")
        gcls = gold_clusters(book)
        n_gold_mentions = sum(len(c) for c in gcls)

        flat, sent_lens = reconstruct_book(key, all_keys, delta, lengths, nlp, label="sealed")

        # Re-precompute with best_window (if same as 32, dev records already exist)
        print(f"  Precomputing with window={best_window}...")
        test_records = precompute_clusters(flat, sent_lens, coref, device, ctx_window=best_window)

        # Baseline (no memory)
        pred_base, _, _ = apply_memory(test_records, text_threshold=1.1, ctx_threshold=1.1)

        # Simple memory (surface + named text embedding only — replicate v1 logic)
        _, pred_simple, n_ents_simple = apply_memory(test_records, best_tt, 1.1)

        # Context-aware memory (surface + text emb + context emb)
        _, pred_ctx, n_ents_ctx = apply_memory(test_records, best_tt, best_ct)

        s_base   = conll_f1(pred_base,   gcls)
        s_simple = conll_f1(pred_simple, gcls)
        s_ctx    = conll_f1(pred_ctx,    gcls)

        delta_simple = round(s_simple["conll_f1"] - s_base["conll_f1"], 2)
        delta_ctx    = round(s_ctx["conll_f1"]    - s_base["conll_f1"], 2)
        delta_v2_v1  = round(s_ctx["conll_f1"]    - s_simple["conll_f1"], 2)

        print(f"\n  Clusters: {len(pred_base)} → {n_ents_simple} (simple) → "
              f"{n_ents_ctx} (ctx-aware)  [gold: {len(gcls)}]")

        def fmt(s):
            return (f"MUC={s['muc']['f']:5.2f}  B³={s['b3']['f']:5.2f}  "
                    f"CEAFe={s['ceafe']['f']:5.2f}  CoNLL={s['conll_f1']:5.2f}%")

        print(f"  BEFORE (no memory):       {fmt(s_base)}")
        print(f"  SIMPLE memory (v1):       {fmt(s_simple)}  Δ={delta_simple:+.2f}%")
        print(f"  CONTEXT-AWARE memory (v2):{fmt(s_ctx)}  Δ={delta_ctx:+.2f}%  "
              f"(vs simple: {delta_v2_v1:+.2f}%)")

        all_results[book["doc_key"]] = {
            "doc_key":          book["doc_key"],
            "gutenberg_key":    key,
            "n_gold_entities":  len(gcls),
            "n_gold_mentions":  n_gold_mentions,
            "n_clusters_base":  len(pred_base),
            "n_clusters_simple":n_ents_simple,
            "n_clusters_ctx":   n_ents_ctx,
            "dev_settings": {"window": best_window, "text_threshold": best_tt,
                             "ctx_threshold": best_ct},
            "scores_before":  s_base,
            "scores_simple":  s_simple,
            "scores_context": s_ctx,
            "delta_simple_vs_before": delta_simple,
            "delta_ctx_vs_before":    delta_ctx,
            "delta_ctx_vs_simple":    delta_v2_v1,
        }

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("FINAL SUMMARY — Full progression")
    print(f"{'='*64}")
    b_before, b_simple, b_ctx = [], [], []
    for dk, res in all_results.items():
        b_before.append(res["scores_before"]["conll_f1"])
        b_simple.append(res["scores_simple"]["conll_f1"])
        b_ctx.append(res["scores_context"]["conll_f1"])
        sb = res["scores_before"]; ss = res["scores_simple"]; sc = res["scores_context"]
        print(f"\n{dk}")
        print(f"  Clusters: {res['n_clusters_base']} → {res['n_clusters_simple']} → "
              f"{res['n_clusters_ctx']}  (gold: {res['n_gold_entities']})")
        print(f"  BEFORE  : MUC={sb['muc']['f']}  B³={sb['b3']['f']}  "
              f"CEAFe={sb['ceafe']['f']}  CoNLL={sb['conll_f1']}")
        print(f"  SIMPLE  : MUC={ss['muc']['f']}  B³={ss['b3']['f']}  "
              f"CEAFe={ss['ceafe']['f']}  CoNLL={ss['conll_f1']}"
              f"  Δ={res['delta_simple_vs_before']:+.2f}%")
        print(f"  CTX-AWARE: MUC={sc['muc']['f']}  B³={sc['b3']['f']}  "
              f"CEAFe={sc['ceafe']['f']}  CoNLL={sc['conll_f1']}"
              f"  Δ={res['delta_ctx_vs_before']:+.2f}%  (vs simple: "
              f"{res['delta_ctx_vs_simple']:+.2f}%)")

    print(f"\nMean CoNLL F1:")
    print(f"  BEFORE   : {sum(b_before)/len(b_before):.2f}%")
    print(f"  SIMPLE   : {sum(b_simple)/len(b_simple):.2f}%  "
          f"Δ={sum(b_simple)/len(b_simple)-sum(b_before)/len(b_before):+.2f}%")
    print(f"  CTX-AWARE: {sum(b_ctx)/len(b_ctx):.2f}%  "
          f"Δ={sum(b_ctx)/len(b_ctx)-sum(b_before)/len(b_before):+.2f}%  "
          f"(vs simple: {sum(b_ctx)/len(b_ctx)-sum(b_simple)/len(b_simple):+.2f}%)")

    sweep_data = [{"window":w,"text_thresh":tt,"ctx_thresh":ct,
                   "n_entities":n,"conll_f1":f1,"b3":b3,"ceafe":ce}
                  for (w,tt,ct,n,f1,b3,ce) in sweep_rows]
    out = {"dev_book": DEV_BOOK_KEY, "best_dev_params": best_params,
           "best_dev_f1": best_f1, "sweep": sweep_data, "results": all_results}
    out_path = RESULTS_DIR / "bookcoref_context_memory_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
