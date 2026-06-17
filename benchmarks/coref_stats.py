import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset
import numpy as np

ds = load_dataset("coref-data/preco")
train = ds["train"]

n = 3000
tok_lens = []
sent_counts = []
mention_counts = []
span_widths = []
gap_sents = []  # gap in sentence index between consecutive mentions in a multi-mention cluster
gap_sents_all_pairs = []

for i in range(n):
    ex = train[i]
    sents = ex["sentences"]
    sent_counts.append(len(sents))
    tok_lens.append(sum(len(s) for s in sents))
    mc = ex["mention_clusters"]
    mention_counts.append(sum(len(c) for c in mc))
    for c in mc:
        for m in c:
            span_widths.append(m[2] - m[1])
        if len(c) > 1:
            sents_idx = sorted(m[0] for m in c)
            for a, b in zip(sents_idx[:-1], sents_idx[1:]):
                gap_sents.append(b - a)
            for ai in range(len(sents_idx)):
                for bi in range(ai+1, len(sents_idx)):
                    gap_sents_all_pairs.append(sents_idx[bi]-sents_idx[ai])

tok_lens = np.array(tok_lens)
print("=== doc token length (n=%d) ===" % n)
print("mean", tok_lens.mean(), "median", np.median(tok_lens), "p90", np.percentile(tok_lens,90), "max", tok_lens.max())
print("frac docs <= 512 tokens:", (tok_lens<=512).mean())

sc = np.array(sent_counts)
print("=== sentences/doc ===", sc.mean(), np.median(sc), sc.max())

mcnt = np.array(mention_counts)
print("=== mentions/doc ===", mcnt.mean(), np.median(mcnt), mcnt.max())

sw = np.array(span_widths)
print("=== span widths ===")
for w in [1,2,3,4,5,6,8,10,15,20]:
    print(f"  width<= {w}: {(sw<=w).mean():.4f}")
print("max width", sw.max())

gs = np.array(gap_sents_all_pairs)
print("=== pairwise sentence gap among coreferent mentions (all pairs within cluster) ===")
print("n pairs", len(gs))
print("frac gap==0 (same sentence):", (gs==0).mean())
print("frac 1-3:", ((gs>=1)&(gs<=3)).mean())
print("frac 4-9:", ((gs>=4)&(gs<=9)).mean())
print("frac 10+:", (gs>=10).mean())
print("max gap", gs.max() if len(gs) else None)
