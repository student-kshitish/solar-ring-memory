"""
BookCoref REAL baseline — FCoref model, own mention detection, 512-token chunks.

Model: biu-nlp/f-coref (RoBERTa-base + coref head, 90.5M params, ~375 MB GPU).
       Bigger LingMessCoref would give higher F1 but same conclusion.
Chunk: 512 tokens, sentence-boundary-aligned, NO cross-chunk linking.
Text:  delta-applied tokenisation (bookcoref opcode + values_changed applied).
       This is the same token space as the gold annotation indices.
Eval:  CoNLL F1 (MUC + B3 + CEAFe) — self-contained implementation.

Run ONCE on sealed books (Siddhartha + P&P). Do NOT push.
"""

import json, re, urllib.request, warnings, logging, sys
from collections import defaultdict
from pathlib import Path

import nltk, spacy, torch
from nltk.tokenize import sent_tokenize
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── silence fastcoref HTTP noise ──────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# ── monkey-patch before import (transformers 5.x vs fastcoref 4.x compat) ────
import fastcoref.modeling as _fm
if not hasattr(_fm.FCorefModel, "all_tied_weights_keys"):
    _fm.FCorefModel.all_tied_weights_keys = property(
        lambda self: {k: None for k in (self._tied_weights_keys or [])}
    )
from fastcoref import FCoref

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path(__file__).parent.parent / "results"
CACHE_DIR    = Path.home() / ".cache/huggingface/datasets/downloads"
SEALED_PATH  = RESULTS_DIR / "bookcoref_gold_sealed.json"
BASELINE_OUT = RESULTS_DIR / "bookcoref_real_baseline_results.json"

TEST_JSONL   = CACHE_DIR / "f7f0d411d70d2531eb0935fa3cd69683125c645b94d490183126497304497d97"
DELTA_FILE   = CACHE_DIR / "f0eca9ccd4652f8052f04a8a66c79dd2d5cd21e5f5be43a29713fad599f9e05c"
LENGTHS_FILE = CACHE_DIR / "fcb01fa02443a80b9f6a9b0fa0cc56de9af0b21d8b4a5297ea00cfd2ff30a433"

# Exact Wayback Machine snapshots used by BookCoref (timestamp 20250406105821).
# The ebooks/{key}.txt.utf-8 URL is a different (UTF-8 clean) version from
# cache/epub/{key}/pg{key}.txt; using the wrong URL causes token misalignment.
GUTENBERG_URLS = {
    "2500": "https://web.archive.org/web/20250406105821/https://gutenberg.org/ebooks/2500.txt.utf-8",
    "1342": "https://web.archive.org/web/20250406/https://gutenberg.org/ebooks/1342.txt.utf-8",
}
# Pre-downloaded cached copies (avoids repeated WM requests)
GUTENBERG_LOCAL = {
    "2500": "/tmp/pg2500_exact.txt",
    "1342": "/tmp/pg1342_exact.txt",
}
CHUNK_SIZE = 512     # tokens per chunk (sentence-boundary-aligned)

# ── CoNLL-F1 (self-contained) ─────────────────────────────────────────────────

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

# ── text reconstruction ───────────────────────────────────────────────────────

def download_gutenberg(key):
    # Prefer pre-downloaded exact WM snapshots; fall back to live download.
    local = Path(GUTENBERG_LOCAL[key])
    if local.exists():
        # Must use encoding="utf-8" (NOT utf-8-sig) to match bookcoref.py line 224.
        # The BOM character is intentionally kept so sent_tokenize sees the same text.
        return local.read_text(encoding="utf-8")
    print(f"  Downloading Wayback Machine snapshot for Gutenberg #{key}...")
    import subprocess
    result = subprocess.run(
        ["curl", "-L", "--max-time", "60", "-s", GUTENBERG_URLS[key], "-o", str(local)],
        capture_output=True
    )
    if result.returncode != 0 or local.stat().st_size < 10000:
        raise RuntimeError(f"Download failed for key {key}")
    return local.read_text(encoding="utf-8")

def tokenize_book(text, nlp):
    sents = sent_tokenize(text)
    return [[t.text for t in doc] for doc in nlp.pipe(sents, n_process=1)]

def apply_delta(raw_flat, key, delta_obj, lengths, all_keys):
    """Apply bookcoref delta using deepdiff with empty-list stubs for other books."""
    wrapper = {k: [] for k in all_keys}
    wrapper[key] = raw_flat
    patched = wrapper + delta_obj
    flat = patched[key]
    sents, pos = [], 0
    for slen in lengths[key]:
        sents.append(flat[pos:pos+slen])
        pos += slen
    return sents, flat

# ── char ↔ token mapping ──────────────────────────────────────────────────────

def build_char_to_tok(tokens):
    """
    Join tokens with single spaces and map each char position → token index.
    Returns (joined_text, char_to_tok list, tok_start_chars list).
    """
    parts, starts = [], []
    pos = 0
    for tok in tokens:
        starts.append(pos)
        parts.append(tok)
        pos += len(tok) + 1           # +1 for the joining space
    text = " ".join(parts)
    char_to_tok = [None] * len(text)
    for i, tok in enumerate(tokens):
        s = starts[i]
        for c in range(s, s + len(tok)):
            if c < len(char_to_tok):
                char_to_tok[c] = i
    return text, char_to_tok, starts

def charspan_to_tokspan(c_start, c_end_excl, char_to_tok):
    """
    Convert a char span [c_start, c_end_excl) to token span [t_start, t_end] inclusive.
    Returns None if the span doesn't fall on a recognised token.
    """
    c_end = c_end_excl - 1           # last char of the mention (inclusive)
    if c_start >= len(char_to_tok) or c_end >= len(char_to_tok):
        return None
    t_start = char_to_tok[c_start]
    t_end   = char_to_tok[c_end]
    if t_start is None or t_end is None:
        return None
    return (t_start, t_end)

# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_starts_from_lengths(sent_lens, chunk_size):
    starts, pos, total = [0], 0, sum(sent_lens)
    for slen in sent_lens:
        pos += slen
        if pos - starts[-1] >= chunk_size and pos < total:
            starts.append(pos)
    return starts

# ── gold helpers ──────────────────────────────────────────────────────────────

def gold_clusters(book):
    return [[(m[0], m[1]) for m in ch["mentions"]] for ch in book["characters"] if ch["mentions"]]

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading lengths...")
    with open(LENGTHS_FILE) as f:
        lengths = json.load(f)

    print("Loading delta (208 MB)...")
    from deepdiff import Delta
    from deepdiff.serialization import json_loads
    with open(DELTA_FILE) as f:
        delta_raw_str = f.read()
    all_keys = set(re.findall(r"root\['(\d+)'\]", delta_raw_str))
    delta = Delta(delta_path=str(DELTA_FILE), deserializer=json_loads)

    print("Loading gold annotations (sealed)...")
    with open(TEST_JSONL) as f:
        gold_books = [json.loads(l) for l in f]
    print(f"  {len(gold_books)} books in sealed test set")

    print(f"\nLoading FCoref model (biu-nlp/f-coref, 90.5M params)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coref = FCoref(device=device, enable_progress_bar=False)
    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e6:.0f} MB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB")

    all_results = {}

    for book in gold_books:
        key  = book["gutenberg_key"]
        dkey = book["doc_key"]

        if key == "0" or "animal_farm" in dkey:
            print(f"\nSkipping {dkey} (no public Gutenberg text)")
            continue

        print(f"\n{'='*64}")
        print(f"Book: {dkey}  (Gutenberg #{key})")

        # ── reconstruct annotated text ────────────────────────────────────
        raw_text = download_gutenberg(key)
        print(f"  Tokenising {len(raw_text):,} chars...")
        raw_sents = tokenize_book(raw_text, nlp)
        raw_flat  = [t for s in raw_sents for t in s]
        print(f"  Raw: {len(raw_flat):,} tokens in {len(raw_sents)} sents")

        print(f"  Applying delta (extracting {len(lengths[key])} annotated sents)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ann_sents, ann_flat = apply_delta(raw_flat, key, delta, lengths, all_keys)
        total_tokens = len(ann_flat)
        print(f"  Post-delta: {total_tokens:,} tokens in {len(ann_sents)} sents")

        gcls = gold_clusters(book)
        n_mentions = sum(len(c) for c in gcls)
        print(f"  Gold: {len(gcls)} entity clusters, {n_mentions} mentions")

        # ── chunk boundaries ──────────────────────────────────────────────
        sent_lens = [len(s) for s in ann_sents]
        chunk_starts = chunk_starts_from_lengths(sent_lens, CHUNK_SIZE)
        n_chunks = len(chunk_starts)
        chunk_ends = chunk_starts[1:] + [total_tokens]
        print(f"  Chunks: {n_chunks} (size ≤{CHUNK_SIZE} tokens, sentence-aligned)")

        # ── run FCoref per chunk, no cross-chunk linking ──────────────────
        pred_clusters_all = []
        total_pred_mentions = 0
        max_gpu_mb = 0

        print(f"  Running FCoref on each chunk... ", end="", flush=True)
        for ci, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
            chunk_tokens = ann_flat[cs:ce]

            # Reconstruct text and build char→tok map
            chunk_text, c2t, tok_starts = build_char_to_tok(chunk_tokens)

            # Run model (suppressing its HTTP log noise)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = coref.predict(texts=[chunk_text])

            gpu_mb = torch.cuda.memory_allocated() / 1e6
            if gpu_mb > max_gpu_mb:
                max_gpu_mb = gpu_mb

            # Convert char spans → doc-global token spans
            char_clusters = preds[0].get_clusters(as_strings=False)
            for char_cl in char_clusters:
                tok_cl = []
                for (c_s, c_e) in char_cl:
                    span = charspan_to_tokspan(c_s, c_e, c2t)
                    if span is not None:
                        # Shift from chunk-local to doc-global
                        tok_cl.append((span[0] + cs, span[1] + cs))
                if tok_cl:
                    pred_clusters_all.append(tok_cl)
                    total_pred_mentions += len(tok_cl)

            if (ci + 1) % 20 == 0 or ci == n_chunks - 1:
                print(f"{ci+1}/{n_chunks}", end=" ", flush=True)

        print()
        print(f"  Predicted: {len(pred_clusters_all)} clusters, "
              f"{total_pred_mentions} mentions")
        print(f"  Peak GPU during inference: {max_gpu_mb:.0f} MB")

        # ── evaluate ──────────────────────────────────────────────────────
        scores = conll_f1(pred_clusters_all, gcls)
        print(f"\n  [REAL BASELINE] FCoref, 512-tok chunks, no cross-chunk linking:")
        print(f"    MUC   P={scores['muc']['p']}  R={scores['muc']['r']}  F={scores['muc']['f']}")
        print(f"    B³    P={scores['b3']['p']}  R={scores['b3']['r']}  F={scores['b3']['f']}")
        print(f"    CEAFe P={scores['ceafe']['p']}  R={scores['ceafe']['r']}  F={scores['ceafe']['f']}")
        print(f"    CoNLL F1 = {scores['conll_f1']:.1f}%")

        all_results[dkey] = {
            "doc_key":         dkey,
            "gutenberg_key":   key,
            "model":           "biu-nlp/f-coref (FCoref, 90.5M params, RoBERTa-base)",
            "chunk_size":      CHUNK_SIZE,
            "total_tokens":    total_tokens,
            "n_chunks":        n_chunks,
            "n_gold_clusters": len(gcls),
            "n_gold_mentions": n_mentions,
            "n_pred_clusters": len(pred_clusters_all),
            "n_pred_mentions": total_pred_mentions,
            "peak_gpu_mb":     round(max_gpu_mb),
            "scores":          scores,
        }

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("STEP 2 — FINAL REAL BASELINE")
    print(f"{'='*64}")
    for dkey, res in all_results.items():
        s = res["scores"]
        print(f"\n{dkey}")
        print(f"  Model : {res['model']}")
        print(f"  Chunks: {res['n_chunks']} × ≤{res['chunk_size']} tokens, no cross-chunk linking")
        print(f"  Gold  : {res['n_gold_clusters']} clusters / {res['n_gold_mentions']} mentions")
        print(f"  Pred  : {res['n_pred_clusters']} clusters / {res['n_pred_mentions']} mentions")
        print(f"  MUC   : F={s['muc']['f']}")
        print(f"  B³    : F={s['b3']['f']}")
        print(f"  CEAFe : F={s['ceafe']['f']}")
        print(f"  *** CoNLL F1 = {s['conll_f1']:.1f}% ***")

    BASELINE_OUT.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {BASELINE_OUT}")

    # ── STEP 3: two-ceiling clarification ────────────────────────────────────
    print(f"\n{'='*64}")
    print("STEP 3 — TWO CEILINGS (do not confuse them)")
    print(f"{'='*64}")
    print("""
CEILING A — Oracle-linking ceiling (~33% CoNLL F1):
  What you get if you have PERFECT gold mentions AND PERFECTLY link them
  across chunks. Already measured. This is a floor on the linking problem,
  NOT the real target. You can never beat it; a real model that detects
  imperfect mentions cannot reach it.

CEILING B — Real achievable target (unknown, to be measured later):
  (same real FCoref model) + cross-chunk memory linking.
  The honest before/after comparison is:
    REAL BASELINE (above)  vs  REAL BASELINE + MEMORY LAYER
  NOT vs the oracle number. Oracle is a separate analysis tool.
""")

    # ── STEP 4: honest assessment ─────────────────────────────────────────────
    print(f"{'='*64}")
    print("STEP 4 — HONEST ASSESSMENT")
    print(f"{'='*64}")
    for dkey, res in all_results.items():
        base_f1  = res["scores"]["conll_f1"]
        oracle_f1 = 33.5 if "pride" in dkey else 35.5   # from prior run
        full_gap = 100.0 - base_f1

        # Realistic target: cross-chunk linking won't be perfect.
        # The paper's 82.2 → 67.0 degradation at 1500-tok windows suggests
        # ~15 pts lost at 1500 tok; at 512 tok the loss is larger.
        # Realistically a good cross-chunk memory layer could recover
        # 30–50% of the gap between baseline and oracle-linking ceiling.
        # Oracle ceiling is ~33-35%.
        oracle_gap = oracle_f1 - base_f1
        low_target  = round(base_f1 + oracle_gap * 0.30, 1)
        high_target = round(base_f1 + oracle_gap * 0.60, 1)

        print(f"\n{dkey}:")
        print(f"  Real baseline (FCoref, chunked, no memory): {base_f1:.1f}%")
        print(f"  Oracle-linking ceiling (gold mentions):     {oracle_f1:.1f}%")
        print(f"  Gap to oracle ceiling:                    +{oracle_gap:.1f}%")
        print(f"  Realistic memory-layer target band:       ~{low_target}–{high_target}%")
        print(f"  Plausible genuine gain on table:          +{round(oracle_gap*0.30,1)}–"
              f"+{round(oracle_gap*0.60,1)}% CoNLL F1")
        print(f"")
        print(f"  The honest comparison is:")
        print(f"    BEFORE: {base_f1:.1f}%  (real model, chunked, no cross-chunk memory)")
        print(f"    AFTER : ???%  (same model + Solar Ring cross-chunk memory)")
        print(f"    GAIN  : anything above 0 is real; > +{round(oracle_gap*0.20,1)}% would be notable")


if __name__ == "__main__":
    main()
