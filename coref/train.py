import argparse
import json
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from coref.data import precompute_chunks, ChunkDataset, make_collate_fn
from coref.model import CorefModel
from coref.engine import compute_chunk_loss
from coref.dev_eval import evaluate_docs

TRAIN_END = 32958  # train[:32958]
DEV_START = 32958  # last 10% of train = 3,662 docs
# 'sentence' chunk lengths are highly variable (1-99 tokens) which caused
# CUDA OOM via allocator fragmentation at batch_size=48; reduced for safety.
BATCH_SIZE = {"sentence": 24, "window5": 16, "fulldoc": 4}
LOG_PATH_TMPL = "checkpoints/coref_{baseline}_log.jsonl"
CKPT_PATH_TMPL = "checkpoints/coref_{baseline}_best.pt"


def build_optimizer(model, lr_encoder, lr_head):
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW([
        {"params": encoder_params, "lr": lr_encoder},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["sentence", "window5", "fulldoc"])
    ap.add_argument("--max_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--dev_subsample", type=int, default=300)
    ap.add_argument("--evals_per_epoch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_docs", type=int, default=None, help="debug: cap train docs")
    ap.add_argument("--max_minutes", type=float, default=None, help="hard wall-clock budget")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    ds = load_dataset("coref-data/preco")
    train_split = ds["train"]

    n_train = args.max_train_docs or TRAIN_END
    train_doc_ids = list(range(0, n_train))
    dev_doc_ids_full = list(range(DEV_START, len(train_split)))
    rng = random.Random(123)
    dev_subsample_ids = rng.sample(dev_doc_ids_full, min(args.dev_subsample, len(dev_doc_ids_full)))

    print(f"[{args.baseline}] precomputing chunks for {len(train_doc_ids)} train docs...")
    t0 = time.time()
    chunks = precompute_chunks(train_split, train_doc_ids, args.baseline, tokenizer)
    print(f"  -> {len(chunks)} chunks in {time.time()-t0:.1f}s")

    batch_size = args.batch_size or BATCH_SIZE[args.baseline]
    dset = ChunkDataset(chunks)
    collate = make_collate_fn(tokenizer)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)

    model = CorefModel().to(device)
    opt = build_optimizer(model, args.lr_encoder, args.lr_head)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.max_epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    ckpt_path = CKPT_PATH_TMPL.format(baseline=args.baseline)
    log_path = LOG_PATH_TMPL.format(baseline=args.baseline)
    os.makedirs("checkpoints", exist_ok=True)
    log_f = open(log_path, "w")

    best_f1 = -1.0
    evals_without_improve = 0
    step = 0
    start_time = time.time()
    stop = False

    for epoch in range(args.max_epochs):
        if stop:
            break
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    results = model.forward_batch(input_ids, attn, batch["word_ids"])
                    total_loss = 0.0
                    for j, (spans, mention_scores, g) in enumerate(results):
                        num_words = max(w for w in batch["word_ids"][j] if w is not None) + 1
                        total_loss = total_loss + compute_chunk_loss(model, spans, mention_scores, g, num_words, batch["gold_clusters_local"][j])
                    total_loss = total_loss / len(results)
                opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[{args.baseline}] OOM at step {step}, skipping batch (size={len(batch['doc_idx'])})")
                continue
            sched.step()
            step += 1
            if step % 500 == 0:
                torch.cuda.empty_cache()

            if step % 50 == 0:
                elapsed = time.time() - start_time
                rec = {"step": step, "epoch": epoch, "loss": float(total_loss), "elapsed_sec": elapsed}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            if step % eval_every == 0:
                t_eval = time.time()
                result = evaluate_docs(model, train_split, dev_subsample_ids, args.baseline, tokenizer, device)
                f1_now = result["agg"]["conll_f1"]
                rec = {"step": step, "epoch": epoch, "dev_conll_f1": f1_now, "dev_detail": result["agg"],
                       "eval_sec": time.time() - t_eval, "elapsed_sec": time.time() - start_time}
                print(f"[{args.baseline}] step={step} epoch={epoch} dev_conll_f1={f1_now:.4f} "
                      f"(muc={result['agg']['muc']['f1']:.3f} b3={result['agg']['b3']['f1']:.3f} ceafe={result['agg']['ceafe']['f1']:.3f})")
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if f1_now > best_f1:
                    best_f1 = f1_now
                    evals_without_improve = 0
                    torch.save({"model_state": model.state_dict(), "step": step, "dev_conll_f1": f1_now,
                                "baseline": args.baseline, "args": vars(args)}, ckpt_path)
                    print(f"  -> saved new best checkpoint ({ckpt_path}), dev_conll_f1={f1_now:.4f}")
                else:
                    evals_without_improve += 1
                    if evals_without_improve >= args.patience:
                        print(f"[{args.baseline}] early stopping: no dev improvement in {args.patience} evals")
                        stop = True
                        break
                if args.max_minutes and (time.time() - start_time) / 60 > args.max_minutes:
                    print(f"[{args.baseline}] hit wall-clock budget of {args.max_minutes} min, stopping")
                    stop = True
                    break

    log_f.close()
    print(f"[{args.baseline}] DONE. best dev_conll_f1={best_f1:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
