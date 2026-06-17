import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from coref.data import precompute_chunks, ChunkDataset, make_collate_fn
from coref.model import CorefModel
from coref.engine import compute_chunk_loss, predict_chunk_clusters, local_span_to_global_id, global_id_for_gold
from coref.scorer import score_document, aggregate

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
ds = load_dataset("coref-data/preco")
train = ds["train"]

for baseline in ["sentence", "window5", "fulldoc"]:
    print(f"\n=== baseline: {baseline} ===")
    chunks = precompute_chunks(train, list(range(3)), baseline, tokenizer)
    print("num chunks:", len(chunks), "avg words/chunk:", sum(len(c.words) for c in chunks) / len(chunks))
    dset = ChunkDataset(chunks)
    collate = make_collate_fn(tokenizer)
    batch = collate([dset[i] for i in range(min(4, len(dset)))])
    print("input_ids shape:", batch["input_ids"].shape)

    model = CorefModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    results = model.forward_batch(input_ids, attn, batch["word_ids"])

    total_loss = 0.0
    for i, (spans, mention_scores, g) in enumerate(results):
        num_words = max(w for w in batch["word_ids"][i] if w is not None) + 1
        loss = compute_chunk_loss(model, spans, mention_scores, g, num_words, batch["gold_clusters_local"][i])
        total_loss = total_loss + loss
    print("initial loss:", float(total_loss))
    opt.zero_grad()
    total_loss.backward()
    opt.step()
    print("backward+step OK")

    # quick overfit check on chunk 0 alone for a few steps
    spans, mention_scores, g = results[0]
    num_words0 = max(w for w in batch["word_ids"][0] if w is not None) + 1
    for step in range(5):
        out = model.forward_batch(input_ids[0:1], attn[0:1], batch["word_ids"][0:1])
        spans, mention_scores, g = out[0]
        loss = compute_chunk_loss(model, spans, mention_scores, g, num_words0, batch["gold_clusters_local"][0])
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  step {step} loss {float(loss):.4f}")

    clusters_local = predict_chunk_clusters(model, spans, mention_scores, g, num_words0)
    print("predicted clusters (local, chunk0):", clusters_local[:5], "... total", len(clusters_local))

    word_doc_id0 = chunks[0].word_doc_id
    pred_global = [[local_span_to_global_id(word_doc_id0, s, e) for (s, e) in cl] for cl in clusters_local]
    gold_global = []
    for cluster in chunks[0].gold_clusters_local:
        gold_global.append([global_id_for_gold(word_doc_id0[s][0], word_doc_id0[s][1], word_doc_id0[e][1] + 1) for (s, e) in cluster])
    doc_score = score_document(gold_global, pred_global)
    print("doc-level CoNLL agg (untrained, 1 chunk, sanity only):", aggregate([doc_score]))

print("\nSMOKE TEST PASSED")
