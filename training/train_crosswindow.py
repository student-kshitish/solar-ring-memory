"""
train_crosswindow.py — A100 step 3+4: train the cross-window decider and MEASURE
whether SRM's 300-d memory space can align to the coref 256-d span space.
================================================================================
This is the paste-ready script for tomorrow's slot (runbook steps 14 & 18).

WHAT IT DOES
- Trains SrmAntecedentProjector (300->256) + the pair head jointly on a SILVER DEV
  book, on cross-window (mention <-> SRM-antecedent) pairs.
- MEASURES cross-window F1 on dev. This is the go/no-go (runbook step 16).
- Prints a blunt verdict: ALIGNED (proceed to sealed) or NULL (single Linear can't
  align; upgrade projector or accept the null — do NOT spend the sealed shot).

WHAT IT DOES NOT DO
- It does NOT touch sealed books (Siddhartha / P&P). Hard blocklist, same as
  cross_encoder.py.
- It does NOT run the sealed eval. That's a separate, locked, once-only step.
- It does NOT assume success. A flat dev curve is a valid, recorded outcome.

WIRING (do this first tomorrow — 4 TODO points marked below):
  T1: import your real modules (paths from tonight's commit)
  T2: point --dev-book at a silver (non-sealed) BookCoref json
  T3: confirm the dev book's cross-window gold links exist (antecedent in earlier chunk)
  T4: confirm projector/pair_ffn param names for the optimizer

USAGE (A100):
  python train_crosswindow.py --check
  python train_crosswindow.py --measure-alignment --dev-book <silver>.json --out xw_ckpt/
  # read the verdict. If NULL -> stop, record null. If ALIGNED -> continue runbook.
"""

import os, sys, json, math, random, argparse, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# HARD BLOCKLIST — identical discipline to cross_encoder.py. Sealed books never
# enter training or dev tuning.
# ---------------------------------------------------------------------------
SEALED_BOOKS = {"siddhartha", "pride_and_prejudice", "pride-and-prejudice", "pnp"}
def _norm(n): return os.path.splitext(os.path.basename(n))[0].strip().lower().replace(" ", "_")
def assert_not_sealed(name, ctx):
    if _norm(name) in SEALED_BOOKS:
        raise RuntimeError(f"BLOCKED: '{name}' is SEALED, requested during '{ctx}'. "
                           "Sealed books are evaluated once, never trained/tuned on.")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    srm_dim: int = 300
    span_dim: int = 256          # coref pair space (span_proj_dim)
    dist_feat_dim: int = 20      # matches pair_ffn input 3*256+20 = 788
    lr: float = 2e-4             # projector+head only (small, fast)
    epochs: int = 8
    batch_size: int = 32
    seed: int = 13
    fp16: bool = True
    # ALIGNMENT VERDICT THRESHOLDS (pre-registered — decide BEFORE seeing numbers)
    # dev cross-window pairwise F1 must beat a recency baseline by this margin to
    # count as "aligned". Below margin = null (single Linear insufficient).
    null_margin: float = 0.05    # +5 F1 over recency-only baseline
    upgrade_hint_margin: float = 0.02  # 2-5 F1 = marginal, try MLP projector

# ===========================================================================
# T1 — WIRE YOUR REAL MODULES HERE (paths from tonight's commit).
# Replace these imports. The fallbacks below let the file PARSE without the repo
# so you can eyeball it tonight; they raise if actually run without wiring.
# ===========================================================================
def _wire_modules():
    """Import real repo modules. Called at run time, not import time."""
    try:
        import torch  # noqa
        from coref.srm_bridge import SrmAntecedentProjector, gather_srm_antecedents  # T1
        from coref.wire_srm_into_coref import build_virtual_antecedents, prepend_to_pairwise  # T1
        from coref.model import pairwise_scores  # T1  (the existing asymmetric decider)
        return dict(SrmAntecedentProjector=SrmAntecedentProjector,
                    gather_srm_antecedents=gather_srm_antecedents,
                    build_virtual_antecedents=build_virtual_antecedents,
                    prepend_to_pairwise=prepend_to_pairwise,
                    pairwise_scores=pairwise_scores)
    except ImportError as e:
        raise ImportError(
            f"Wire T1: real modules not found ({e}). This script must run from the "
            "repo root on the A100 with tonight's commit present.") from e

# ---------------------------------------------------------------------------
# DATA: build cross-window training pairs from a dev book's GOLD.
# A positive = (anaphor span in chunk k, its gold entity's representation from an
# EARLIER chunk). Negatives = other entities' earlier reprs. This is exactly the
# cross-window link the memory is supposed to enable.
# ---------------------------------------------------------------------------
def build_xw_pairs(dev_book: str, cfg: Cfg) -> List[dict]:
    """Returns list of {anaphor_repr, srm_repr(300), label, recency_rank, dist}.
    T3: assumes the dev book json exposes per-chunk mentions with gold entity ids
    AND a way to get the anaphor's contextual span repr. Adapt the field names to
    your BookCoref schema. Structure below documents the contract."""
    assert_not_sealed(dev_book, "build_xw_pairs")
    book = json.load(open(dev_book))
    # ---- EXPECTED SCHEMA (adapt to yours) --------------------------------
    # book["chunks"]: list of chunks, each with:
    #     "mentions": [{"span_repr":[256], "gold_entity":int, "is_anaphor":bool,
    #                   "start":int}]
    # book["entity_reprs"]: {gold_entity:int -> [300]}  (SRM-side memory vector)
    # ----------------------------------------------------------------------
    chunks = book["chunks"]
    ent_reprs = {int(k): v for k, v in book["entity_reprs"].items()}
    pairs = []
    seen_entity_chunk = {}  # gold_entity -> earliest chunk index seen
    for ci, ch in enumerate(chunks):
        for m in ch["mentions"]:
            ge = int(m["gold_entity"])
            # cross-window positive only if this entity appeared in an EARLIER chunk
            if m.get("is_anaphor") and ge in seen_entity_chunk and seen_entity_chunk[ge] < ci:
                if ge not in ent_reprs:
                    continue
                pairs.append({
                    "anaphor_repr": m["span_repr"],          # [256]
                    "srm_repr": ent_reprs[ge],                # [300]
                    "label": 1,
                    "recency_rank": ci - seen_entity_chunk[ge],
                    "dist": min(1.0, (ci - seen_entity_chunk[ge]) / 20.0),
                })
                # negatives: other entities seen earlier
                others = [e for e in seen_entity_chunk
                          if e != ge and seen_entity_chunk[e] < ci and e in ent_reprs]
                random.shuffle(others)
                for oe in others[:3]:
                    pairs.append({
                        "anaphor_repr": m["span_repr"],
                        "srm_repr": ent_reprs[oe],
                        "label": 0,
                        "recency_rank": ci - seen_entity_chunk[oe],
                        "dist": min(1.0, (ci - seen_entity_chunk[oe]) / 20.0),
                    })
            seen_entity_chunk.setdefault(ge, ci)
    return pairs

# ---------------------------------------------------------------------------
# RECENCY BASELINE — the honest comparison. If the trained projector can't beat
# "just pick the most recent earlier entity," the alignment added nothing.
# ---------------------------------------------------------------------------
def recency_baseline_f1(pairs: List[dict]) -> float:
    # group by anaphor; predict positive for the smallest recency_rank candidate
    from collections import defaultdict
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[id(p["anaphor_repr"]) if isinstance(p["anaphor_repr"], list)
               else tuple(p["anaphor_repr"][:4])].append(p)
    tp = fp = fn = 0
    for g in groups.values():
        # pick min recency_rank as the prediction
        pred = min(g, key=lambda x: x["recency_rank"])
        for p in g:
            is_pred = (p is pred)
            if is_pred and p["label"] == 1: tp += 1
            elif is_pred and p["label"] == 0: fp += 1
            elif not is_pred and p["label"] == 1: fn += 1
    return tp / (tp + 0.5 * (fp + fn) + 1e-9)

# ---------------------------------------------------------------------------
# TRAIN projector + pair head, then MEASURE dev F1 vs recency baseline.
# ---------------------------------------------------------------------------
def measure_alignment(dev_book: str, out_dir: str, cfg: Cfg):
    import torch, torch.nn as nn
    M = _wire_modules()
    assert_not_sealed(dev_book, "measure_alignment")
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = build_xw_pairs(dev_book, cfg)
    if len(pairs) < 20:
        print(f"[WARN] only {len(pairs)} cross-window pairs in dev — too few to trust "
              "a verdict. Pick a dev book with more cross-chunk entity recurrence.")
    n_pos = sum(p["label"] for p in pairs)
    print(f"[data] {len(pairs)} pairs, {n_pos} pos / {len(pairs)-n_pos} neg")

    base_f1 = recency_baseline_f1(pairs)
    print(f"[baseline] recency-only pairwise F1 = {base_f1:.3f}  (the bar to beat)")

    # T4: SrmAntecedentProjector from your repo; pair head = a thin scorer over
    # [proj_srm, anaphor, proj_srm*anaphor, dist] mirroring pair_ffn's structure.
    projector = M["SrmAntecedentProjector"]().to(dev)   # 300 -> 256
    pair_head = nn.Sequential(
        nn.Linear(3 * cfg.span_dim + cfg.dist_feat_dim, 256), nn.ReLU(),
        nn.Dropout(0.1), nn.Linear(256, 1)).to(dev)

    params = list(projector.parameters()) + list(pair_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    bce = nn.BCEWithLogitsLoss()

    def batch_iter(data, bs):
        random.shuffle(data)
        for i in range(0, len(data), bs):
            yield data[i:i+bs]

    def to_tensors(batch):
        a = torch.tensor([b["anaphor_repr"] for b in batch], dtype=torch.float, device=dev)
        s = torch.tensor([b["srm_repr"] for b in batch], dtype=torch.float, device=dev)
        d = torch.tensor([[b["dist"]] * cfg.dist_feat_dim for b in batch],
                         dtype=torch.float, device=dev)
        y = torch.tensor([b["label"] for b in batch], dtype=torch.float, device=dev)
        return a, s, d, y

    projector.train(); pair_head.train()
    for ep in range(cfg.epochs):
        tot = 0.0
        for batch in batch_iter(pairs, cfg.batch_size):
            a, s, d, y = to_tensors(batch)
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                ps = projector(s)                       # [B,256]
                feat = torch.cat([a, ps, a * ps, d], dim=-1)  # asymmetric interaction
                logit = pair_head(feat).squeeze(-1)
                loss = bce(logit, y)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
        print(f"[train] epoch {ep} loss {tot:.4f}")

    # ---- MEASURE dev F1 (grouped, pick argmax per anaphor) ----
    projector.eval(); pair_head.eval()
    from collections import defaultdict
    groups = defaultdict(list)
    with torch.no_grad():
        for p in pairs:
            a, s, d, _ = to_tensors([p])
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                ps = projector(s)
                feat = torch.cat([a, ps, a * ps, d], dim=-1)
                score = torch.sigmoid(pair_head(feat)).item()
            key = tuple(p["anaphor_repr"][:4])
            groups[key].append((score, p["label"]))
    tp = fp = fn = 0
    for g in groups.values():
        pred_i = max(range(len(g)), key=lambda i: g[i][0])
        for i, (sc, lab) in enumerate(g):
            if i == pred_i and lab == 1: tp += 1
            elif i == pred_i and lab == 0: fp += 1
            elif i != pred_i and lab == 1: fn += 1
    trained_f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"projector": projector.state_dict(),
                "pair_head": pair_head.state_dict()},
               os.path.join(out_dir, "xw_align.pt"))

    # ---- THE VERDICT (pre-registered thresholds) ----
    delta = trained_f1 - base_f1
    print("\n" + "=" * 64)
    print(f"  ALIGNMENT MEASUREMENT (dev book: {os.path.basename(dev_book)})")
    print(f"  recency baseline F1 : {base_f1:.3f}")
    print(f"  trained (proj+head) : {trained_f1:.3f}")
    print(f"  delta               : {delta:+.3f}")
    print("-" * 64)
    if delta >= cfg.null_margin:
        print("  VERDICT: ALIGNED. The 300->256 projection carries real cross-window")
        print("  signal beyond recency. Proceed to sealed eval per runbook step 20.")
    elif delta >= cfg.upgrade_hint_margin:
        print("  VERDICT: MARGINAL. Single Linear is weak but non-zero. Try an MLP or")
        print("  contrastive projector BEFORE deciding. Do NOT spend the sealed shot yet.")
    else:
        print("  VERDICT: NULL. Single Linear does NOT align SRM space to span space")
        print("  beyond recency. This is a CLEAN NULL — a valid, recordable finding.")
        print("  Do NOT run the sealed eval to 'confirm' it. Record it and move on,")
        print("  or upgrade the projector if time remains (step 4 in wiring_notes).")
    print("=" * 64)
    json.dump({"base_f1": base_f1, "trained_f1": trained_f1, "delta": delta,
               "verdict": ("aligned" if delta >= cfg.null_margin else
                           "marginal" if delta >= cfg.upgrade_hint_margin else "null"),
               "time": time.ctime()},
              open(os.path.join(out_dir, "alignment_verdict.json"), "w"), indent=2)
    print(f"[saved] {out_dir}/xw_align.pt  +  alignment_verdict.json  (DOWNLOAD these)")

# ---------------------------------------------------------------------------
def check_env():
    print("Cross-window alignment trainer — A100 step 3+4.")
    print("Before running: wire T1 (imports), T2 (--dev-book silver), T3 (schema),")
    print("T4 (param names). This trains ONLY projector+head (small, minutes).")
    print("It measures dev alignment and prints a go/no-go verdict. It does NOT")
    print("touch sealed books and does NOT run the sealed eval.")
    try:
        import torch
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    except ImportError:
        print("torch not installed here (expected on laptop; present on A100).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--measure-alignment", action="store_true")
    ap.add_argument("--dev-book", default="")
    ap.add_argument("--out", default="xw_ckpt/")
    a = ap.parse_args()
    cfg = Cfg()
    if a.check:
        check_env()
    elif a.measure_alignment:
        if not a.dev_book:
            sys.exit("--dev-book required (silver, NON-sealed)")
        measure_alignment(a.dev_book, a.out, cfg)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
