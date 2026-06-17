import os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from coref.data import precompute_chunks, ChunkDataset, make_collate_fn
from coref.model import CorefModel
from coref.engine import compute_chunk_loss

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
ds = load_dataset("coref-data/preco")
train = ds["train"]

N_DOCS = 200
BATCH = {"sentence": 48, "window5": 16, "fulldoc": 4}

for baseline in ["sentence", "window5", "fulldoc"]:
    chunks = precompute_chunks(train, list(range(N_DOCS)), baseline, tokenizer)
    dset = ChunkDataset(chunks)
    collate = make_collate_fn(tokenizer)
    loader = DataLoader(dset, batch_size=BATCH[baseline], shuffle=True, collate_fn=collate)

    model = CorefModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    n_warmup = 3
    n_timed = 15
    it = iter(loader)
    t0 = None
    n_done = 0
    torch.cuda.synchronize()
    for i in range(n_warmup + n_timed):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if i == n_warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            results = model.forward_batch(input_ids, attn, batch["word_ids"])
            total_loss = 0.0
            for j, (spans, mention_scores, g) in enumerate(results):
                num_words = max(w for w in batch["word_ids"][j] if w is not None) + 1
                total_loss = total_loss + compute_chunk_loss(model, spans, mention_scores, g, num_words, batch["gold_clusters_local"][j])
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if i >= n_warmup:
            n_done += len(batch["doc_idx"])
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    chunks_per_sec = n_done / elapsed
    total_chunks_per_doc = len(chunks) / N_DOCS
    print(f"[{baseline}] chunks/doc={total_chunks_per_doc:.2f} chunks/sec={chunks_per_sec:.2f} "
          f"-> est sec/epoch(32958 docs)={32958*total_chunks_per_doc/chunks_per_sec:.0f} "
          f"peak_mem_GB={torch.cuda.max_memory_allocated()/1e9:.2f}")
    torch.cuda.reset_peak_memory_stats()
    del model, opt
    torch.cuda.empty_cache()
