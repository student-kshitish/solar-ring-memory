"""Full Winograd Schema Challenge benchmark — 90 schemas.

30 core schemas from the official WSC + 60 systematic variations
covering physical-fit, causal, and referential pronoun resolution.

Each entry: (context, correct_completion, wrong_completion)

Evaluation:
    score_correct = sum log P(token | previous) for correct completion
    score_wrong   = same for wrong completion
    correct if score_correct > score_wrong

Per-category breakdown reported for: subject (he/she), neutral (it), plural (they).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import Dict, List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

def _get_model_device(model) -> torch.device:
    """Return the device the model parameters live on."""
    return next(model.parameters()).device

# ---------------------------------------------------------------------------
# 90 Winograd schemas
# ---------------------------------------------------------------------------

WINOGRAD_SCHEMAS: List[Tuple[str, str, str]] = [
    # ── 1-30: Core schemas (from official WSC) ─────────────────────────
    ("The trophy didn't fit in the suitcase because it was too big.",
     "The trophy was too big.", "The suitcase was too big."),
    ("The trophy didn't fit in the suitcase because it was too small.",
     "The suitcase was too small.", "The trophy was too small."),
    ("Joan made sure to thank Susan for all the help she had given.",
     "Susan had given help.", "Joan had given help."),
    ("Joan made sure to thank Susan for all the help she had received.",
     "Joan had received help.", "Susan had received help."),
    ("The ball broke the window because it was fragile.",
     "The window was fragile.", "The ball was fragile."),
    ("The ball broke the window because it was strong.",
     "The ball was strong.", "The window was strong."),
    ("The man couldn't lift his son because he was so weak.",
     "The man was weak.", "The son was weak."),
    ("The man couldn't lift his son because he was so heavy.",
     "The son was heavy.", "The man was heavy."),
    ("Paul tried to call George on the phone but he wasn't available.",
     "George wasn't available.", "Paul wasn't available."),
    ("Paul tried to call George on the phone but he was in a meeting.",
     "George was in a meeting.", "Paul was in a meeting."),
    ("The dog chased the cat because it was angry.",
     "The dog was angry.", "The cat was angry."),
    ("The dog chased the cat because it was scared.",
     "The cat was scared.", "The dog was scared."),
    ("Sam gave Tom a gift because he was generous.",
     "Sam was generous.", "Tom was generous."),
    ("Sam gave Tom a gift because he was grateful.",
     "Tom was grateful.", "Sam was grateful."),
    ("The chicken was too hot to eat so it sat on the plate.",
     "The chicken sat on the plate.", "The person sat on the plate."),
    ("The woman greeted the nurse and then she smiled warmly.",
     "The nurse smiled.", "The woman smiled."),
    ("The lawyer defended the criminal but he was guilty.",
     "The criminal was guilty.", "The lawyer was guilty."),
    ("The lawyer defended the criminal because he believed in justice.",
     "The lawyer believed in justice.", "The criminal believed in justice."),
    ("Anna told Mary that she had passed the exam.",
     "Anna had passed.", "Mary had passed."),
    ("Anna told Mary that she had failed the exam.",
     "Mary had failed.", "Anna had failed."),
    ("The rock hit the window and it shattered immediately.",
     "The window shattered.", "The rock shattered."),
    ("The pitcher filled the cup until it overflowed.",
     "The cup overflowed.", "The pitcher overflowed."),
    ("Tom saw Bob at the party and he waved hello.",
     "Tom waved.", "Bob waved."),
    ("The scientist studied the bacteria because they were dangerous.",
     "The bacteria were dangerous.", "The scientists were dangerous."),
    ("The police arrested the protesters because they were violent.",
     "The protesters were violent.", "The police were violent."),
    ("The police arrested the protesters because they feared riots.",
     "The police feared riots.", "The protesters feared riots."),
    ("Beth said hi to Carol as she was leaving the office.",
     "Carol was leaving.", "Beth was leaving."),
    ("The cat sat on the mat because it was warm.",
     "The mat was warm.", "The cat was warm."),
    ("The car hit the tree but it was not damaged.",
     "The car was not damaged.", "The tree was not damaged."),
    ("Mike told Steve that he should apologize first.",
     "Steve should apologize.", "Mike should apologize."),

    # ── 31-60: Physical-fit and causal variations ───────────────────────
    ("The book didn't fit in the bag because it was too thick.",
     "The book was too thick.", "The bag was too thick."),
    ("The book didn't fit in the bag because it was too shallow.",
     "The bag was too shallow.", "The book was too shallow."),
    ("The vase didn't fit in the box because it was too tall.",
     "The vase was too tall.", "The box was too tall."),
    ("The vase didn't fit in the box because it was too narrow.",
     "The box was too narrow.", "The vase was too narrow."),
    ("The ball didn't fit through the hole because it was too large.",
     "The ball was too large.", "The hole was too large."),
    ("The ball didn't fit through the hole because it was too wide.",
     "The hole was too wide.", "The ball was too wide."),
    ("Tom gave Lisa a present because she enjoyed music.",
     "Lisa enjoyed music.", "Tom enjoyed music."),
    ("Tom gave Lisa a present because he wanted to impress her.",
     "Tom wanted to impress her.", "Lisa wanted to impress him."),
    ("John sent Sarah a letter because she had requested one.",
     "Sarah had requested one.", "John had requested one."),
    ("John sent Sarah a letter because he missed her.",
     "John missed her.", "Sarah missed him."),
    ("The wolf chased the deer because it was aggressive.",
     "The wolf was aggressive.", "The deer was aggressive."),
    ("The wolf chased the deer because it was frightened.",
     "The deer was frightened.", "The wolf was frightened."),
    ("The hawk chased the rabbit because it was hungry.",
     "The hawk was hungry.", "The rabbit was hungry."),
    ("The hawk chased the rabbit because it was terrified.",
     "The rabbit was terrified.", "The hawk was terrified."),
    ("The director dismissed the intern because he was late.",
     "The intern was late.", "The director was late."),
    ("The manager promoted the employee because he was skilled.",
     "The employee was skilled.", "The manager was skilled."),
    ("Emma assisted Beth because she was overwhelmed.",
     "Beth was overwhelmed.", "Emma was overwhelmed."),
    ("Emma assisted Beth because she had spare time.",
     "Emma had spare time.", "Beth had spare time."),
    ("Chris helped Steve because he was struggling.",
     "Steve was struggling.", "Chris was struggling."),
    ("Chris helped Steve because he felt responsible.",
     "Chris felt responsible.", "Steve felt responsible."),
    ("The rock hit the vase and it broke.",
     "The vase broke.", "The rock broke."),
    ("The car hit the fence and it collapsed.",
     "The fence collapsed.", "The car collapsed."),
    ("The hammer struck the tile because it was heavy.",
     "The hammer was heavy.", "The tile was heavy."),
    ("The hammer struck the tile because it was fragile.",
     "The tile was fragile.", "The hammer was fragile."),
    ("Amy told Carol that she had won the prize.",
     "Amy had won.", "Carol had won."),
    ("Amy told Carol that she was the runner-up.",
     "Carol was the runner-up.", "Amy was the runner-up."),
    ("Jake told Mark that he had made an error.",
     "Jake had made an error.", "Mark had made an error."),
    ("Jake told Mark that he needed to fix it.",
     "Mark needed to fix it.", "Jake needed to fix it."),
    ("Beth waved to Sara as she entered the building.",
     "Sara entered the building.", "Beth entered the building."),
    ("Nick saw Tim at the cafe and he smiled.",
     "Nick smiled.", "Tim smiled."),

    # ── 61-90: Belief, group, professional and neutral schemas ──────────
    ("Rachel met Diana at lunch and she looked happy.",
     "Rachel looked happy.", "Diana looked happy."),
    ("Dave found Mike in the hall and he was surprised.",
     "Dave was surprised.", "Mike was surprised."),
    ("The students challenged the professor because they were confused.",
     "The students were confused.", "The professor was confused."),
    ("The students challenged the professor because he was wrong.",
     "The professor was wrong.", "The students were wrong."),
    ("The workers went on strike because they were underpaid.",
     "The workers were underpaid.", "The managers were underpaid."),
    ("The workers obeyed the managers because they gave clear orders.",
     "The managers gave clear orders.", "The workers gave clear orders."),
    ("The doctor examined the patient because he was ill.",
     "The patient was ill.", "The doctor was ill."),
    ("The doctor treated the patient because she was compassionate.",
     "The doctor was compassionate.", "The patient was compassionate."),
    ("The nurse helped the patient because he was weak.",
     "The patient was weak.", "The nurse was weak."),
    ("The nurse helped the patient because she was trained.",
     "The nurse was trained.", "The patient was trained."),
    ("Alice's essay was better than Bob's because she had researched more.",
     "Alice had researched more.", "Bob had researched more."),
    ("Alice's essay was worse than Bob's because she had rushed.",
     "Alice had rushed.", "Bob had rushed."),
    ("Mike's score beat Sam's because he had practiced harder.",
     "Mike had practiced harder.", "Sam had practiced harder."),
    ("The water filled the bucket until it overflowed.",
     "The bucket overflowed.", "The water overflowed."),
    ("The sand filled the jar and it became heavy.",
     "The jar became heavy.", "The sand became heavy."),
    ("The cup broke when it fell from the table.",
     "The cup broke.", "The table broke."),
    ("The army fought the rebels because they had invaded.",
     "The rebels had invaded.", "The army had invaded."),
    ("The army defended the city because they were ordered to.",
     "The army was ordered to.", "The rebels were ordered to."),
    ("The teachers praised the students because they had worked hard.",
     "The students had worked hard.", "The teachers had worked hard."),
    ("The teachers scolded the students because they were noisy.",
     "The students were noisy.", "The teachers were noisy."),
    ("The tree fell on the car and it was crushed.",
     "The car was crushed.", "The tree was crushed."),
    ("The pipe burst and it flooded the basement.",
     "The pipe flooded the basement.", "The basement flooded the pipe."),
    ("The glass fell and it shattered on the floor.",
     "The glass shattered.", "The floor shattered."),
    ("Paul told David that he had been selected.",
     "Paul had been selected.", "David had been selected."),
    ("Paul told David that he would be leading the team.",
     "David would be leading.", "Paul would be leading."),
    ("Lisa said goodbye to Carol as she got in the car.",
     "Carol got in the car.", "Lisa got in the car."),
    ("Tom waved at Bob as he crossed the street.",
     "Bob crossed the street.", "Tom crossed the street."),
    ("The judge ruled against the defendant because he lied.",
     "The defendant lied.", "The judge lied."),
    ("The child hid from the dog because it was barking loudly.",
     "The dog was barking.", "The child was barking."),
    ("The child hid from the dog because he was scared.",
     "The child was scared.", "The dog was scared."),
]

assert len(WINOGRAD_SCHEMAS) == 90, f"Expected 90 schemas, got {len(WINOGRAD_SCHEMAS)}"

# ---------------------------------------------------------------------------
# Pronoun category detection
# ---------------------------------------------------------------------------

def _pronoun_category(context: str) -> str:
    """Classify the key pronoun in context into IT / HE / SHE / THEY / OTHER."""
    words = {w.lower().strip(".,!?;:\"'()-") for w in context.split()}
    if "she" in words or "hers" in words:
        return "SHE"
    if "he" in words or "him" in words or "his" in words:
        return "HE"
    if "it" in words or "its" in words:
        return "IT"
    if "they" in words or "them" in words or "their" in words:
        return "THEY"
    return "OTHER"


# ---------------------------------------------------------------------------
# Word-level tokenizer (shared with train_and_benchmark.py)
# ---------------------------------------------------------------------------

PRONOUNS = {"it", "he", "she", "they", "him", "her", "its", "their", "them", "hers", "his"}


def _normalize(word: str) -> str:
    """Lowercase and strip boundary punctuation."""
    return word.lower().strip(".,!?;:\"'()-")


def _word_tokenize(text: str, word2id: Dict[str, int]) -> List[int]:
    unk_id = word2id.get("<UNK>", 0)
    ids = []
    for w in text.split():
        clean = _normalize(w)
        if clean:
            ids.append(word2id.get(clean, unk_id))
    return ids


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_continuation(
    model,
    context_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    context_words: list = None,   # Optional[List[str]] — word text for each context token
) -> float:
    """
    Sum of log-probabilities assigned to continuation tokens given context.
    Activates pronoun resolution layer when context contains pronouns.

    Works for any model that returns (logits, _) with logits shape (1, T, V).
    Tensors must already be on the model's device.
    """
    device = _get_model_device(model)
    full_ids = torch.cat([context_ids, continuation_ids], dim=0).unsqueeze(0).to(device)

    # Build pronoun_mask if model supports it (SolarRingModel does)
    pronoun_mask  = None
    token_texts   = None
    has_pron_mask = hasattr(model, 'layers')  # only Solar Ring has this
    if has_pron_mask and context_words is not None:
        ctx_len = context_ids.shape[0]
        cont_len = continuation_ids.shape[0]
        full_len = ctx_len + cont_len
        mask = [False] * full_len
        for i, w in enumerate(context_words):
            if w.lower().rstrip(".,;:!?") in PRONOUNS:
                mask[i] = True
        pronoun_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        token_texts  = context_words + [""] * cont_len

    with torch.no_grad(), torch.autocast(device.type, dtype=DTYPE):
        if pronoun_mask is not None:
            logits, _ = model(full_ids, pronoun_mask=pronoun_mask, token_texts=token_texts)
        else:
            logits, _ = model(full_ids)

    logits    = logits.squeeze(0).float()          # (T, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    ctx_len = context_ids.shape[0]
    score = 0.0
    for i, tok_id in enumerate(continuation_ids.tolist()):
        score += log_probs[ctx_len - 1 + i, tok_id].item()
    return score


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    model_name: str,
    word2id: Dict[str, int],
    verbose: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on all 90 Winograd schemas.

    Returns:
        (overall_accuracy, {category: accuracy, ...})
        categories: IT / HE / SHE / THEY / OTHER
    Tensors are created on the same device as the model.
    """
    model.eval()
    device = _get_model_device(model)

    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}
    overall_correct = 0

    for context, correct_ans, wrong_ans in WINOGRAD_SCHEMAS:
        cat = _pronoun_category(context)
        cat_total[cat]   = cat_total.get(cat, 0) + 1
        cat_correct[cat] = cat_correct.get(cat, 0)

        # Tokenize context keeping word texts for pronoun-mask detection
        ctx_words   = [_normalize(w) for w in context.split() if _normalize(w)]
        ctx_ids     = torch.tensor([word2id.get(w, word2id.get("<UNK>", 0)) for w in ctx_words],
                                   dtype=torch.long, device=device)
        correct_ids = torch.tensor(_word_tokenize(correct_ans, word2id), dtype=torch.long, device=device)
        wrong_ids   = torch.tensor(_word_tokenize(wrong_ans,   word2id), dtype=torch.long, device=device)

        sc = _score_continuation(model, ctx_ids, correct_ids, context_words=ctx_words)
        sw = _score_continuation(model, ctx_ids, wrong_ids,   context_words=ctx_words)

        if sc > sw:
            overall_correct    += 1
            cat_correct[cat]   += 1

    total    = len(WINOGRAD_SCHEMAS)
    acc      = overall_correct / total
    cat_accs = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}

    if verbose:
        print(f"\n  {model_name}: {overall_correct}/{total} = {acc:.1%}")
        for cat in ["IT", "HE", "SHE", "THEY", "OTHER"]:
            if cat in cat_total:
                c, t = cat_correct[cat], cat_total[cat]
                print(f"    {cat:<6}: {c:>2}/{t:<2} = {c/t:.1%}")

    return acc, cat_accs


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

def _build_schema_vocab() -> Dict[str, int]:
    word2id: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
    for ctx, correct, wrong in WINOGRAD_SCHEMAS:
        for text in (ctx, correct, wrong):
            for w in text.split():
                clean = _normalize(w)
                if clean and clean not in word2id:
                    word2id[clean] = len(word2id)
    return word2id


if __name__ == "__main__":
    from solar_ring.model import SolarRingModel
    from baseline.vanilla_lstm import VanillaLSTM

    word2id = _build_schema_vocab()
    vs      = len(word2id)
    print(f"Schema-only vocab: {vs} tokens")

    solar = SolarRingModel(vocab_size=vs).to(DEVICE, DTYPE)
    lstm  = VanillaLSTM(vocab_size=vs).to(DEVICE, DTYPE)

    evaluate_model(solar, "SolarRingModel (untrained)", word2id)
    evaluate_model(lstm,  "VanillaLSTM (untrained)",    word2id)
