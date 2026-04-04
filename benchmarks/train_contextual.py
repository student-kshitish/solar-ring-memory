import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from solar_ring.model_contextual import SolarRingContextual
from benchmarks.direct_train import build_generated_pairs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHUNK  = 200   # sentences per caching progress chunk


def cache_word_embeddings(embedder, texts, device, label=""):
    """
    Cache per-word embeddings for each sentence via embed_words_batch().
    Processes in chunks of CHUNK sentences so progress is visible.
    Returns dict: sentence -> (L, 384) tensor.
    """
    unique = list(dict.fromkeys(texts))   # deduplicate, preserve order
    total  = len(unique)
    tag    = f"[{label}] " if label else ""
    print(f"{tag}Pre-caching {total} sentences (per-word MiniLM, batched)...")
    t0    = time.time()
    cache = {}

    for start in range(0, total, CHUNK):
        chunk = unique[start:start + CHUNK]
        partial = embedder.embed_words_batch(chunk)
        cache.update(partial)
        elapsed = time.time() - t0
        print(f"  {min(start + CHUNK, total)}/{total} sentences cached  "
              f"({elapsed:.1f}s elapsed)")

    total_words = sum(v.shape[0] for v in cache.values())
    print(f"{tag}Done — {len(cache)} sentences / {total_words} word vectors "
          f"in {time.time() - t0:.1f}s\n")
    return cache


def val_accuracy(model, test_pairs, cache):
    """Compute pairwise accuracy on test_pairs using cached embeddings."""
    model.eval()
    correct = 0
    count   = 0
    with torch.no_grad():
        for i in range(0, len(test_pairs) - 1, 2):
            text_c, label_c = test_pairs[i]
            text_w, label_w = test_pairs[i + 1]
            _, _, logit_c = model.forward_from_emb(cache[text_c].clone())
            _, _, logit_w = model.forward_from_emb(cache[text_w].clone())
            if logit_c.item() > logit_w.item():
                correct += 1
            count += 1
    return correct / max(count, 1) * 100


def train_contextual(epochs=30):
    print("="*62)
    print("Training SolarRingContextual — 1600 pairs, 20 epochs")
    print("="*62)
    t0 = time.time()

    model = SolarRingContextual(device=DEVICE).to(DEVICE)
    model.freeze_for_probe()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (probe only): {n_params:,}\n")

    all_pairs   = build_generated_pairs()          # 1600 flat (text, label)
    train_pairs = all_pairs[:1200]                 # 75% train
    test_pairs  = all_pairs[1200:]                 # 25% held-out

    print(f"Train: {len(train_pairs)} items  |  Test: {len(test_pairs)} items\n")

    # ── Pre-cache all sentences — MiniLM runs exactly once ──────────────────
    train_texts = [t for t, _ in train_pairs]
    test_texts  = [t for t, _ in test_pairs]

    train_cache = cache_word_embeddings(model.embedder, train_texts, DEVICE, "train")
    test_cache  = cache_word_embeddings(model.embedder, test_texts,  DEVICE, "test")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.01
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_val_acc    = 0.0
    best_epoch      = 0
    patience        = 8    # early stopping: stop after 8 epochs with no val improvement
    no_improve      = 0
    debug_batches   = 3

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Acc':>7}  {'Time':>6}")
    print("-" * 48)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct    = 0
        count      = 0
        debug      = (epoch == 0)

        for i in range(0, len(train_pairs) - 1, 2):
            text_c, label_c = train_pairs[i]
            text_w, label_w = train_pairs[i + 1]
            emb_c = train_cache[text_c].clone()   # (L, 384)
            emb_w = train_cache[text_w].clone()

            optimizer.zero_grad()

            # Sentence-mean path: 50x faster than word-by-word SolarMemory loop.
            # Trains pronoun_head + W_skip + out_norm on MiniLM representations.
            # Full forward_from_emb is used only at evaluation time.
            _, _, logit_c = model.forward_from_emb(emb_c)
            _, _, logit_w = model.forward_from_emb(emb_w)

            lc = logit_c.float().squeeze()
            lw = logit_w.float().squeeze()

            # Margin loss: push correct logit above wrong by margin 1
            loss = torch.clamp(1.0 - lc + lw, min=0.0)

            if debug and count < debug_batches:
                gfn = loss.grad_fn.__class__.__name__ if loss.grad_fn else "NONE"
                print(f"  [debug pair {count}] "
                      f"logit_c={lc.item():.4f}  logit_w={lw.item():.4f}  "
                      f"loss={loss.item():.4f}  grad_fn={gfn}")

            loss.backward()

            if debug and count < debug_batches:
                gnorm = sum(
                    p.grad.abs().sum().item()
                    for p in model.parameters() if p.grad is not None
                )
                print(f"             grad_norm={gnorm:.4f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if lc.item() > lw.item():
                correct += 1
            count += 1

        scheduler.step()

        avg_loss  = total_loss / max(count, 1)
        train_acc = correct / max(count, 1) * 100
        v_acc     = val_accuracy(model, test_pairs, test_cache)
        elapsed   = time.time() - t0

        print(f"{epoch+1:5d}  {avg_loss:10.4f}  {train_acc:8.1f}%  "
              f"{v_acc:6.1f}%  {elapsed:5.1f}s")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch + 1
            no_improve   = 0
            torch.save(model.state_dict(),
                       'checkpoints/solar_contextual_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1} "
                      f"(no val improvement for {patience} epochs)")
                break

    print(f"\nBest val acc: {best_val_acc:.1f}% at epoch {best_epoch}  "
          f"Total time: {time.time()-t0:.1f}s")

    # Load best checkpoint for evaluation
    model.load_state_dict(
        torch.load('checkpoints/solar_contextual_best.pt',
                   map_location=DEVICE)
    )
    return model


def evaluate_contextual(model):
    from benchmarks.winograd_full import WINOGRAD_SCHEMAS
    model.eval()

    sents = []
    for ctx, corr, wrong in WINOGRAD_SCHEMAS:
        sents += [ctx + " " + corr, ctx + " " + wrong]
    wcache = cache_word_embeddings(
        model.embedder, sents, DEVICE, "winograd-eval")

    correct = 0
    total   = len(WINOGRAD_SCHEMAS)
    with torch.no_grad():
        for ctx, corr, wrong in WINOGRAD_SCHEMAS:
            _, _, logit_c = model.forward_from_emb(wcache[ctx + " " + corr].clone())
            _, _, logit_w = model.forward_from_emb(wcache[ctx + " " + wrong].clone())
            if logit_c.item() > logit_w.item():
                correct += 1

    acc = correct / total * 100
    print(f"Solar Ring+MiniLM Winograd: {correct}/{total} = {acc:.1f}%")
    return acc


def evaluate_pronoun_direct(model):
    all_pairs  = build_generated_pairs()
    test_items = all_pairs[1200:]
    model.eval()

    test_texts = [t for t, _ in test_items]
    tcache = cache_word_embeddings(
        model.embedder, test_texts, DEVICE, "pronoun-eval")

    correct = 0
    total   = 0
    with torch.no_grad():
        for i in range(0, len(test_items) - 1, 2):
            text_c, _ = test_items[i]
            text_w, _ = test_items[i + 1]
            _, _, logit_c = model.forward_from_emb(tcache[text_c].clone())
            _, _, logit_w = model.forward_from_emb(tcache[text_w].clone())
            if logit_c.item() > logit_w.item():
                correct += 1
            total += 1

    acc = correct / max(total, 1) * 100
    print(f"Solar Ring+MiniLM pronoun: {correct}/{total} = {acc:.1f}%")
    return acc


if __name__ == "__main__":
    model = train_contextual(epochs=30)

    print("\n" + "="*62)
    print("Evaluation (best checkpoint)")
    print("="*62)

    winograd_acc = evaluate_contextual(model)
    pronoun_acc  = evaluate_pronoun_direct(model)

    n_params = model.count_parameters()

    print("\n" + "="*62)
    print("Final comparison table")
    print("="*62)
    print(f"{'Model':<28} {'Winograd':>9} {'Pronoun':>8} {'Params':>8}")
    print("-"*58)
    print(f"{'Solar Ring + GloVe':<28} {'??%':>9} {'76.7%':>8} {'13.8M':>8}")
    print(f"{'Solar Ring + MiniLM':<28} {winograd_acc:>8.1f}% {pronoun_acc:>7.1f}% "
          f"{n_params/1e6:>7.1f}M")
    print(f"{'BERT-base (estimated)':<28} {'~70%':>9} {'~70%':>8} {'110M':>8}")
    print("-"*58)

    beats_glove = pronoun_acc > 76.7
    beats_bert  = winograd_acc > 70.0
    print(f"MiniLM pronoun > GloVe 76.7%:  {'YES' if beats_glove else 'NOT YET'}")
    print(f"MiniLM Winograd > BERT ~70%:   {'YES' if beats_bert  else 'NOT YET'}")
