"""Winograd Schema Challenge benchmark.

Tests pronoun resolution on 20 Winograd sentences for both
SolarRingModel and VanillaLSTM. For each sentence the model is
presented with the full context and asked which of two candidate
completions it assigns higher probability to. The candidate that
correctly resolves the pronoun wins.

Usage (standalone — builds its own vocab from schema text):
    python benchmarks/winograd.py

When called from train_and_benchmark.py, a pre-built word2id dict is
passed so the tokenizer matches the training tokenizer exactly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import List, Tuple, Dict

from solar_ring.model import SolarRingModel
from solar_ring.config import VOCAB_SIZE
from baseline.vanilla_lstm import VanillaLSTM

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Winograd Schema sentences
# Each entry: (context, correct_answer, wrong_answer)
# ---------------------------------------------------------------------------
WINOGRAD_SCHEMAS: List[Tuple[str, str, str]] = [
    (
        "The trophy didn't fit in the suitcase because it was too big.",
        "The trophy was too big.",
        "The suitcase was too big.",
    ),
    (
        "The city councilmen refused the demonstrators a permit because they feared violence.",
        "The city councilmen feared violence.",
        "The demonstrators feared violence.",
    ),
    (
        "The trophy didn't fit in the suitcase because it was too small.",
        "The suitcase was too small.",
        "The trophy was too small.",
    ),
    (
        "The man couldn't lift his son because he was so heavy.",
        "The son was so heavy.",
        "The man was so heavy.",
    ),
    (
        "The lawyers argued with the judge because they were losing the case.",
        "The lawyers were losing the case.",
        "The judge was losing the case.",
    ),
    (
        "The dog bit the man because it was angry.",
        "The dog was angry.",
        "The man was angry.",
    ),
    (
        "Paul tried to call George on the phone but he didn't answer.",
        "George didn't answer.",
        "Paul didn't answer.",
    ),
    (
        "The table won't fit through the door because it is too wide.",
        "The table is too wide.",
        "The door is too wide.",
    ),
    (
        "The table won't fit through the door because it is too narrow.",
        "The door is too narrow.",
        "The table is too narrow.",
    ),
    (
        "Sam's drawing was better than Tom's because he had studied more.",
        "Sam had studied more.",
        "Tom had studied more.",
    ),
    (
        "Joan made sure to thank Susan for all the help she had received.",
        "Joan had received help.",
        "Susan had received help.",
    ),
    (
        "The surgeon sutured the wound, which required great care because it was so delicate.",
        "The wound was so delicate.",
        "The surgeon was so delicate.",
    ),
    (
        "The police arrested the thieves after they hid the stolen goods.",
        "The thieves hid the stolen goods.",
        "The police hid the stolen goods.",
    ),
    (
        "The women stopped the girls from skiing because they were afraid.",
        "The women were afraid.",
        "The girls were afraid.",
    ),
    (
        "Frank was upset with Tom because he broke the window.",
        "Tom broke the window.",
        "Frank broke the window.",
    ),
    (
        "The older students helped the younger ones because they were more experienced.",
        "The older students were more experienced.",
        "The younger ones were more experienced.",
    ),
    (
        "My meeting schedule is already too full and it gives me a headache.",
        "The schedule gives a headache.",
        "The meeting gives a headache.",
    ),
    (
        "The staff ignored the sign because they thought it was wrong.",
        "The staff thought the sign was wrong.",
        "The sign thought the staff was wrong.",
    ),
    (
        "The athlete broke the record and then he celebrated.",
        "The athlete celebrated.",
        "The record celebrated.",
    ),
    (
        "Ann greeted Mary as she entered the room.",
        "Mary entered the room.",
        "Ann entered the room.",
    ),
]


# ---------------------------------------------------------------------------
# Word-level tokenizer
# ---------------------------------------------------------------------------

def _normalize(word: str) -> str:
    """Lowercase and strip boundary punctuation."""
    return word.lower().strip(".,!?;:\"'()-")


def _word_tokenize(text: str, word2id: Dict[str, int]) -> List[int]:
    """
    Split on whitespace, normalize each word, map through word2id.
    Unknown words map to <UNK> (id 0).  Empty tokens (e.g. bare punctuation)
    are dropped so sequence length equals the number of real words.
    """
    unk_id = word2id.get("<UNK>", 0)
    ids = []
    for w in text.split():
        clean = _normalize(w)
        if clean:  # skip empty strings left by punctuation-only tokens
            ids.append(word2id.get(clean, unk_id))
    return ids


def _build_schema_vocab(max_vocab: int = 500) -> Dict[str, int]:
    """
    Build a word2id solely from the Winograd schema text.
    Used when evaluate_winograd is called standalone (no training vocab available).
    """
    word2id: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
    for ctx, correct, wrong in WINOGRAD_SCHEMAS:
        for text in (ctx, correct, wrong):
            for w in text.split():
                clean = _normalize(w)
                if clean and clean not in word2id and len(word2id) < max_vocab:
                    word2id[clean] = len(word2id)
    return word2id


# ---------------------------------------------------------------------------
# Scoring and evaluation
# ---------------------------------------------------------------------------

def _score_continuation(
    model,
    context_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    is_solar: bool,
) -> float:
    """
    Sum of log-probabilities assigned to continuation_ids given context_ids.
    Higher (less negative) is better.
    """
    full_ids = torch.cat([context_ids, continuation_ids], dim=0).unsqueeze(0)  # (1, T)

    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        logits, _ = model(full_ids)  # (1, T, V)

    logits = logits.squeeze(0).float()         # (T, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    ctx_len = context_ids.shape[0]
    score = 0.0
    for i, tok_id in enumerate(continuation_ids.tolist()):
        score += log_probs[ctx_len - 1 + i, tok_id].item()
    return score


def evaluate_winograd(
    model,
    model_name: str,
    is_solar: bool,
    word2id: Dict[str, int],
) -> float:
    """
    Run all 20 schemas, return accuracy (0.0–1.0).

    word2id must be the same vocabulary used when the model was trained so that
    token IDs are consistent between training and evaluation.
    """
    model.eval()
    correct = 0
    total = len(WINOGRAD_SCHEMAS)

    for context, correct_ans, wrong_ans in WINOGRAD_SCHEMAS:
        ctx_ids = torch.tensor(
            _word_tokenize(context, word2id), dtype=torch.long, device=DEVICE
        )
        correct_ids = torch.tensor(
            _word_tokenize(correct_ans, word2id), dtype=torch.long, device=DEVICE
        )
        wrong_ids = torch.tensor(
            _word_tokenize(wrong_ans, word2id), dtype=torch.long, device=DEVICE
        )

        score_correct = _score_continuation(model, ctx_ids, correct_ids, is_solar)
        score_wrong   = _score_continuation(model, ctx_ids, wrong_ids,   is_solar)

        if score_correct > score_wrong:
            correct += 1

    acc = correct / total
    print(f"  {model_name}: {correct}/{total} = {acc:.1%}")
    return acc


def run_winograd_benchmark(
    word2id: Dict[str, int] = None,
    vocab_size: int = None,
) -> dict:
    """
    Instantiate fresh (untrained) models and evaluate on Winograd schemas.

    word2id / vocab_size: if provided, use this vocabulary (must be the same
    vocab used for training so results are comparable).  If omitted, a vocab
    is built from the schema text itself.
    """
    if word2id is None:
        word2id = _build_schema_vocab()
    if vocab_size is None:
        vocab_size = len(word2id)

    print(f"Loading untrained models (vocab_size={vocab_size}, cuda, bfloat16)...")
    solar_model   = SolarRingModel(vocab_size=vocab_size).to(device=DEVICE, dtype=DTYPE)
    vanilla_model = VanillaLSTM(vocab_size=vocab_size).to(device=DEVICE, dtype=DTYPE)

    print("\nWinograd Schema Accuracy (untrained):")
    solar_acc   = evaluate_winograd(solar_model,   "SolarRingModel", is_solar=True,  word2id=word2id)
    vanilla_acc = evaluate_winograd(vanilla_model, "VanillaLSTM",    is_solar=False, word2id=word2id)

    return {
        "solar_winograd":   solar_acc,
        "vanilla_winograd": vanilla_acc,
        "solar_model":      solar_model,
        "vanilla_model":    vanilla_model,
    }


if __name__ == "__main__":
    run_winograd_benchmark()
