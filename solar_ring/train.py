"""Training loop for Solar Ring Memory."""

import torch
import torch.optim as optim
from pathlib import Path

from .config import LR, WEIGHT_DECAY, GRAD_CLIP, BATCH_SIZE
from .model import SolarRingModel
from .loss import compute_loss
from .dataset import build_dataloader


SAMPLE_SENTENCES = [
    "John told Mary that the cat chased the dog because it was too big",
    "The scientist who discovered the vaccine won the prize",
    "She said that he believed the earth was round",
    "The book that Mary read was written by the author who lived in Paris",
    "When the rain fell the river flooded because the dam was too small",
    "He told her that the dog which barked at night was not dangerous",
    "The teacher explained that students who study hard usually succeed",
    "Mary saw the man that the dog bit because it was hungry",
    "The company announced that the product which customers loved would be discontinued",
    "She believed that the city where she grew up had changed dramatically",
]


def train_epoch(model, loader, optimizer, device, epoch=0):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        token_ids    = batch["token_ids"].to(device)
        role_labels  = batch["pos_labels"].to(device)
        spawn_labels = batch["spawn_labels"].to(device)
        pronoun_mask = batch["pronoun_mask"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda",
                            dtype=torch.bfloat16):
            logits, aux = model(
                token_ids,
                role_labels=role_labels,
                spawn_labels=spawn_labels,
                pronoun_mask=pronoun_mask,
            )

            losses = compute_loss(
                logits=logits,
                role_logits=aux["role_logits"],
                spawn_logits=aux["spawn_logits"],
                token_ids=token_ids,
                role_labels=role_labels,
                spawn_labels=spawn_labels,
                pronoun_mask=pronoun_mask,
            )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += losses["total"].item()
        steps += 1

        print(
            f"  Epoch {epoch+1} | Step {batch_idx+1}/{len(loader)} | "
            f"total={losses['total'].item():.4f} "
            f"task={losses['L_task'].item():.4f} "
            f"pos={losses['L_pos'].item():.4f} "
            f"spawn={losses['L_spawn'].item():.4f} "
            f"resolve={losses['L_resolve'].item():.4f}"
        )

    return total_loss / max(steps, 1)


def save_checkpoint(model, optimizer, epoch, loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "loss":       loss,
    }, path)
    print(f"  Checkpoint saved → {path}")


def train(
    sentences=None,
    n_epochs=1,
    batch_size=BATCH_SIZE,
    checkpoint_dir="checkpoints",
    device=None,
):
    if sentences is None:
        sentences = SAMPLE_SENTENCES

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    loader, dataset = build_dataloader(sentences, batch_size=batch_size)
    vocab_size = dataset.vocab_size
    print(f"Vocab size: {vocab_size} | Samples: {len(dataset)}")

    model = SolarRingModel(vocab_size=vocab_size).to(device)
    # Cast to bfloat16
    model = model.to(torch.bfloat16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, loader, optimizer, device, epoch=epoch)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        save_checkpoint(
            model, optimizer, epoch, avg_loss,
            f"{checkpoint_dir}/epoch_{epoch+1}.pt"
        )

    return model, dataset
