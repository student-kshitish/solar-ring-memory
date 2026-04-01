"""Train SolarRingModel, BiLSTM and VanillaLSTM on 3000 balanced pronoun-resolution
sentences, then evaluate all three on 90 Winograd schemas.

Dataset: 3000 sentences — 1000 IT / 1000 HE / 1000 SHE (10 patterns × 100 each)
Training: 15 epochs, StepLR(step_size=5, gamma=0.3): 1e-3 → 3e-4 → 9e-5
GloVe 300d: frozen epochs 1-5, unfrozen epoch 6+
Checkpoints: checkpoints/solar_best.pt  lstm_best.pt  bilstm_best.pt

Usage:
    python benchmarks/train_and_benchmark.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Tuple

# ── Device / CUDA diagnostics ─────────────────────────────────────────────────
print(f"Torch version : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version  : {torch.version.cuda}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {DEVICE}")
if DEVICE.type == "cpu":
    print("WARNING: CUDA not found — training will be very slow")
else:
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from solar_ring.model import SolarRingModel
from solar_ring.loss import compute_loss
from solar_ring.pos_tagger import POSTagger
from solar_ring.glove_loader import load_glove
from solar_ring.config import GRAD_CLIP, ROLE_OTHER
from baseline.vanilla_lstm import VanillaLSTM
from baseline.bilstm import BiLSTM
from benchmarks.winograd_full import (
    WINOGRAD_SCHEMAS,
    _normalize,
    _word_tokenize,
    evaluate_model,
)

DTYPE          = torch.bfloat16
PAD_ID         = 1
GLOVE_PATH     = "data/glove.6B.300d.txt"
TOTAL_EPOCHS   = 15
UNFREEZE_EPOCH = 5       # unfreeze GloVe embeddings at this epoch
EARLY_STOP_PAT = 3       # patience for early stopping
LR_INIT        = 1e-3

PRONOUNS = {"it", "he", "she", "they", "him", "her", "them", "its", "his", "hers"}


# ---------------------------------------------------------------------------
# 1. Balanced pronoun dataset — 3000 sentences (1000 IT / 1000 HE / 1000 SHE)
# ---------------------------------------------------------------------------

def build_pronoun_dataset() -> List[str]:
    """
    3000 pronoun-resolution training sentences:
      1000 IT  pronouns — 10 patterns × 100 variations each
      1000 HE  pronouns — 10 patterns × 100 variations each
      1000 SHE pronouns — 10 patterns × 100 variations each

    Each variation cycles through 10 noun/name tokens AND 10 keyword synonyms
    (10 pairs × 10 keywords = 100 unique sentences per pattern).
    """

    # ── Word lists ───────────────────────────────────────────────────────────
    IT_NOUNS = ["trophy", "suitcase", "ball", "window", "cup",
                "plate",  "box",      "vase", "rock",   "bottle"]

    MALE   = ["John",  "Paul",  "Tom",   "Mike",  "Sam",
              "Bob",   "Steve", "Jake",  "Chris", "Alex"]
    FEMALE = ["Mary",  "Anna",  "Lisa",  "Sarah", "Beth",
              "Emma",  "Diana", "Carol", "Rachel","Nina"]

    # Synonym groups — 10 words each, used to ensure 100 unique sentences
    SYN_FRAGILE = ["fragile","delicate","brittle","breakable","flimsy",
                   "frail",  "weak",    "thin",   "crumbly",  "soft"]
    SYN_SHAT    = ["shattered","cracked","broke","splintered","burst",
                   "fractured","smashed","crumbled","split",  "exploded"]
    SYN_SMALL   = ["small","tiny","compact","narrow","little",
                   "slim", "short","petite","miniature","light"]
    SYN_CRACK   = ["cracked","dented","chipped","scratched","damaged",
                   "bent",  "broke","split",  "fractured","shattered"]
    SYN_ROUND   = ["round","spherical","circular","smooth","curved",
                   "rolling","bouncy","spinning","tumbling","oval"]
    SYN_STRONG  = ["strong","tough","sturdy","solid","hard",
                   "durable","robust","firm","thick","rigid"]
    SYN_FAST    = ["fast","quick","swift","rapid","speedy",
                   "hasty","agile","nimble","brisk","fleet"]
    SYN_HEAVY   = ["heavy","massive","weighty","dense","solid",
                   "bulky","large","thick","enormous","huge"]
    SYN_SHARP   = ["sharp","pointed","jagged","edged","spiky",
                   "thorny","barbed","dangerous","piercing","cutting"]
    SYN_LARGE   = ["large","big","huge","massive","enormous",
                   "giant","wide","broad","tall","bulky"]

    # HE/SHE keyword synonyms
    SYN_LEAVE   = ["leave","go","stop","quit","move",
                   "return","resign","rest","wait","depart"]
    SYN_SKILLED = ["skilled","talented","qualified","competent","experienced",
                   "capable","gifted","diligent","proficient","expert"]
    SYN_WAVE    = ["waved","smiled","nodded","grinned","laughed",
                   "bowed","gestured","beckoned","cheered","beamed"]
    SYN_TIRED   = ["tired","exhausted","sick","hurt","stressed",
                   "upset","weak","unwell","injured","dizzy"]
    SYN_NEED    = ["help","assistance","support","advice","guidance",
                   "information","directions","funds","backup","resources"]
    SYN_DANGER  = ["danger","trouble","risk","harm","threat",
                   "peril","jeopardy","difficulty","crisis","trouble"]
    SYN_BORROW  = ["money","tools","books","supplies","equipment",
                   "materials","cash","resources","funds","items"]
    SYN_WIN     = ["won","succeeded","passed","graduated","qualified",
                   "triumphed","achieved","excelled","completed","finished"]
    SYN_KIND    = ["generous","kind","helpful","thoughtful","giving",
                   "charitable","caring","selfless","gracious","supportive"]
    SYN_REST    = ["rest","relax","sleep","recover","pause",
                   "sit","breathe","calm","nap","unwind"]
    SYN_ADVICE  = ["advice","help","support","guidance","assistance",
                   "opinions","feedback","suggestions","directions","insights"]
    SYN_LATE    = ["late","delayed","behind","overdue","tardy",
                   "slow","off-track","missing","struggling","failing"]
    SYN_HELPA   = ["help","assist","support","guide","advise",
                   "contribute","join","participate","volunteer","cooperate"]
    SYN_PASS    = ["passed","succeeded","won","graduated","qualified",
                   "triumphed","achieved","excelled","completed","finished"]
    SYN_SMILE   = ["smiled","laughed","nodded","waved","grinned",
                   "beamed","cheered","clapped","bowed","glowed"]

    sentences: List[str] = []

    def N(lst, i, off=0): return lst[(i + off) % len(lst)]

    # ── IT PRONOUNS: 1000 sentences (10 patterns × 100) ──────────────────────

    # IT-1: "broke because fragile" → it = noun2
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_FRAGILE, i//10)
        sentences.append(
            f"The {n1} broke the {n2} because it was {kw}. The {n2} was {kw}.")

    # IT-2: "hit and shattered" → it = noun2
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_SHAT, i//10)
        sentences.append(
            f"The {n1} hit the {n2} and it {kw}. The {n2} {kw}.")

    # IT-3: "fit in because small" → it = noun1
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_SMALL, i//10)
        sentences.append(
            f"The {n1} fit in the {n2} because it was {kw}. The {n1} was {kw}.")

    # IT-4: "fell on and cracked" → it = noun2
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_CRACK, i//10)
        sentences.append(
            f"The {n1} fell on the {n2} and it {kw}. The {n2} {kw}.")

    # IT-5: "rolled because round" → it = noun1
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_ROUND, i//10)
        sentences.append(
            f"The {n1} rolled into the {n2} because it was {kw}. The {n1} was {kw}.")

    # IT-6: "broke because strong" → it = noun1
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_STRONG, i//10)
        sentences.append(
            f"The {n1} broke the {n2} because it was {kw}. The {n1} was {kw}.")

    # IT-7: "missed because fast" → it = noun1
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_FAST, i//10)
        sentences.append(
            f"The {n1} missed the {n2} because it was {kw}. The {n1} was {kw}.")

    # IT-8: "damaged because heavy" → it = noun1
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_HEAVY, i//10)
        sentences.append(
            f"The {n1} damaged the {n2} because it was {kw}. The {n1} was {kw}.")

    # IT-9: "avoided because sharp" → it = noun2
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_SHARP, i//10)
        sentences.append(
            f"The {n1} avoided the {n2} because it was {kw}. The {n2} was {kw}.")

    # IT-10: "blocked because large" → it = noun1
    for i in range(100):
        n1, n2, kw = N(IT_NOUNS,i), N(IT_NOUNS,i,3), N(SYN_LARGE, i//10)
        sentences.append(
            f"The {n1} blocked the {n2} because it was {kw}. The {n1} was {kw}.")

    assert len(sentences) == 1000, f"IT: expected 1000, got {len(sentences)}"

    # ── HE PRONOUNS: 1000 sentences (10 patterns × 100) ──────────────────────

    # HE-1: "told he should leave" → he = n2
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_LEAVE, i//10)
        sentences.append(
            f"{n1} told {n2} that he should {kw} first. {n2} should {kw}.")

    # HE-2: "hired because skilled" → he = n2
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_SKILLED, i//10)
        sentences.append(
            f"{n1} hired {n2} because he was {kw}. {n2} was {kw}.")

    # HE-3: "saw and he waved" → he = n1
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_WAVE, i//10)
        sentences.append(
            f"{n1} saw {n2} and he {kw} hello. {n1} {kw}.")

    # HE-4: "helped because tired" → he = n2
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_TIRED, i//10)
        sentences.append(
            f"{n1} helped {n2} because he was {kw}. {n2} was {kw}.")

    # HE-5: "called because needed help" → he = n1
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_NEED, i//10)
        sentences.append(
            f"{n1} called {n2} because he needed {kw}. {n1} needed {kw}.")

    # HE-6: "warned he was in danger" → he = n2
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_DANGER, i//10)
        sentences.append(
            f"{n1} warned {n2} that he was in {kw}. {n2} was in {kw}.")

    # HE-7: "asked if he could borrow" → he = n1
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_BORROW, i//10)
        sentences.append(
            f"{n1} asked {n2} if he could borrow {kw}. {n1} could borrow.")

    # HE-8: "told he had won" → he = n1
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_WIN, i//10)
        sentences.append(
            f"{n1} told {n2} that he had {kw}. {n1} had {kw}.")

    # HE-9: "saw and he smiled" → he = n1
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_SMILE, i//10)
        sentences.append(
            f"{n1} saw {n2} and he {kw} warmly. {n1} {kw}.")

    # HE-10: "thanked because generous" → he = n2
    for i in range(100):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_KIND, i//10)
        sentences.append(
            f"{n1} thanked {n2} because he was {kw}. {n2} was {kw}.")

    assert len(sentences) == 2000, f"HE: expected 2000, got {len(sentences)}"

    # ── SHE PRONOUNS: 1000 sentences (10 patterns × 100) ─────────────────────

    # SHE-1: "told she should rest" → she = n2
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_REST, i//10)
        sentences.append(
            f"{n1} told {n2} that she should {kw}. {n2} should {kw}.")

    # SHE-2: "helped because sick" → she = n2
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_TIRED, i//10)
        sentences.append(
            f"{n1} helped {n2} because she was {kw}. {n2} was {kw}.")

    # SHE-3: "saw and she waved" → she = n1
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_WAVE, i//10)
        sentences.append(
            f"{n1} saw {n2} and she {kw} hello. {n1} {kw}.")

    # SHE-4: "called because needed advice" → she = n1
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_ADVICE, i//10)
        sentences.append(
            f"{n1} called {n2} because she needed {kw}. {n1} needed {kw}.")

    # SHE-5: "thanked because kind" → she = n2
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_KIND, i//10)
        sentences.append(
            f"{n1} thanked {n2} because she was {kw}. {n2} was {kw}.")

    # SHE-6: "warned she was late" → she = n2
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_LATE, i//10)
        sentences.append(
            f"{n1} warned {n2} that she was {kw}. {n2} was {kw}.")

    # SHE-7: "asked if she could help" → she = n1
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_HELPA, i//10)
        sentences.append(
            f"{n1} asked {n2} if she could {kw}. {n1} could {kw}.")

    # SHE-8: "told she had passed" → she = n1
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_PASS, i//10)
        sentences.append(
            f"{n1} told {n2} that she had {kw}. {n1} had {kw}.")

    # SHE-9: "saw and she smiled" → she = n1
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_SMILE, i//10)
        sentences.append(
            f"{n1} saw {n2} and she {kw} warmly. {n1} {kw}.")

    # SHE-10: "helped because tired" → she = n2
    for i in range(100):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_TIRED, i//10)
        sentences.append(
            f"{n1} helped {n2} because she was {kw}. {n2} was {kw}.")

    assert len(sentences) == 3000, f"Expected 3000, got {len(sentences)}"
    return sentences


# ---------------------------------------------------------------------------
# 2. Shared vocabulary
# ---------------------------------------------------------------------------

def build_vocab(sentences: List[str], max_vocab: int = 3000) -> Dict[str, int]:
    word2id: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
    for sent in sentences:
        for w in sent.split():
            clean = _normalize(w)
            if clean and clean not in word2id and len(word2id) < max_vocab:
                word2id[clean] = len(word2id)
    return word2id


# ---------------------------------------------------------------------------
# 3. Word-level dataset
# ---------------------------------------------------------------------------

class PronounDataset(Dataset):
    def __init__(self, sentences: List[str], word2id: Dict[str, int], max_len: int = 64):
        self.word2id = word2id
        self.unk_id  = word2id["<UNK>"]
        self.max_len = max_len
        self.tagger  = POSTagger()
        self.samples = [self._process(s) for s in sentences]

    def _process(self, sentence: str) -> Dict[str, torch.Tensor]:
        from collections import defaultdict
        tags = self.tagger.tag(sentence)
        role_q:  Dict[str, list] = defaultdict(list)
        spawn_q: Dict[str, list] = defaultdict(list)
        for tag in tags:
            w = _normalize(tag["text"])
            if w:
                role_q[w].append(tag["role"])
                spawn_q[w].append(1.0 if tag["spawn"] else 0.0)

        ids, roles, spawns, pmask = [], [], [], []
        for raw in sentence.split():
            c = _normalize(raw)
            if not c: continue
            ids.append(self.word2id.get(c, self.unk_id))
            roles.append(role_q[c].pop(0) if role_q[c] else ROLE_OTHER)
            spawns.append(spawn_q[c].pop(0) if spawn_q[c] else 0.0)
            pmask.append(c in PRONOUNS)

        L = self.max_len
        return {
            "token_ids":    torch.tensor(ids[:L],   dtype=torch.long),
            "pos_labels":   torch.tensor(roles[:L], dtype=torch.long),
            "spawn_labels": torch.tensor(spawns[:L],dtype=torch.float),
            "pronoun_mask": torch.tensor(pmask[:L], dtype=torch.bool),
        }

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def _collate(batch):
    out = {}
    for k in batch[0]:
        seqs = [item[k] for item in batch]
        if seqs[0].dtype == torch.bool:
            out[k] = pad_sequence([s.long() for s in seqs],
                                  batch_first=True, padding_value=0).bool()
        elif seqs[0].dtype == torch.float:
            out[k] = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        else:
            out[k] = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
    return out


# ---------------------------------------------------------------------------
# 4a. SolarRingModel training
# ---------------------------------------------------------------------------

def train_solar(train_sents, val_sents, word2id, glove_matrix) -> SolarRingModel:
    print(f"\n{'='*62}\nTraining SolarRingModel\n{'='*62}")
    vs = len(word2id)
    train_ds = PronounDataset(train_sents, word2id)
    val_ds   = PronounDataset(val_sents,   word2id)
    tl = DataLoader(train_ds, batch_size=1, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=1, shuffle=False)

    model = SolarRingModel(vocab_size=vs, pretrained_embeddings=glove_matrix
                           ).to(DEVICE, DTYPE)
    opt   = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=0.01)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.3)
    print(f"  vocab={vs}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  train={len(train_sents)}  val={len(val_sents)}  epochs={TOTAL_EPOCHS}")

    best_val, pat = float("inf"), 0
    ckpt = Path("checkpoints/solar_best.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(TOTAL_EPOCHS):
        if ep == UNFREEZE_EPOCH and glove_matrix is not None:
            model.embedding.weight.requires_grad_(True)
            print(f"  [epoch {ep+1}] GloVe unfrozen for fine-tuning")

        # train
        model.train(); tl_loss = 0.0
        for batch in tl:
            tok  = batch["token_ids"].to(DEVICE)
            rol  = batch["pos_labels"].to(DEVICE)
            spw  = batch["spawn_labels"].to(DEVICE)
            prn  = batch["pronoun_mask"].to(DEVICE)
            opt.zero_grad()
            with torch.autocast(DEVICE.type, dtype=DTYPE):
                logits, aux = model(tok, role_labels=rol, spawn_labels=spw, pronoun_mask=prn)
                loss = compute_loss(logits, aux["role_logits"], aux["spawn_logits"],
                                    tok, rol, spw, prn)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl_loss += loss.item()

        # validate
        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for batch in vl:
                tok  = batch["token_ids"].to(DEVICE)
                rol  = batch["pos_labels"].to(DEVICE)
                spw  = batch["spawn_labels"].to(DEVICE)
                prn  = batch["pronoun_mask"].to(DEVICE)
                with torch.autocast(DEVICE.type, dtype=DTYPE):
                    logits, aux = model(tok, role_labels=rol, spawn_labels=spw, pronoun_mask=prn)
                    vl_loss += compute_loss(logits, aux["role_logits"], aux["spawn_logits"],
                                            tok, rol, spw, prn)["total"].item()

        ta = tl_loss / len(tl);  va = vl_loss / len(vl)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        print(f"  Epoch {ep+1:02d}/{TOTAL_EPOCHS}  train={ta:.4f}  val={va:.4f}  lr={lr:.0e}", end="")

        if va < best_val:
            best_val = va; pat = 0
            torch.save({"model": model.state_dict(), "word2id": word2id}, ckpt)
            print("  ✓ best")
        else:
            pat += 1
            print(f"  (patience {pat}/{EARLY_STOP_PAT})")
            if pat >= EARLY_STOP_PAT:
                print(f"  Early stopping at epoch {ep+1}")
                break

    print(f"  Loading best checkpoint (val={best_val:.4f})")
    model.load_state_dict(torch.load(ckpt, weights_only=True)["model"])
    return model


# ---------------------------------------------------------------------------
# 4b. Generic LM training (VanillaLSTM and BiLSTM share this)
# ---------------------------------------------------------------------------

def _train_lm_model(
    model: nn.Module,
    model_label: str,
    ckpt_filename: str,
    train_sents, val_sents, word2id, glove_matrix,
) -> nn.Module:
    tl = DataLoader(PronounDataset(train_sents, word2id),
                    batch_size=4, shuffle=True,  collate_fn=_collate)
    vl = DataLoader(PronounDataset(val_sents,   word2id),
                    batch_size=4, shuffle=False, collate_fn=_collate)

    opt   = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=0.01)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.3)
    print(f"  vocab={len(word2id)}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  train={len(train_sents)}  val={len(val_sents)}  epochs={TOTAL_EPOCHS}")

    best_val, pat = float("inf"), 0
    ckpt = Path(f"checkpoints/{ckpt_filename}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(TOTAL_EPOCHS):
        if ep == UNFREEZE_EPOCH and glove_matrix is not None:
            model.embedding.weight.requires_grad_(True)
            print(f"  [epoch {ep+1}] GloVe unfrozen for fine-tuning")

        # train
        model.train(); tl_loss = 0.0
        for batch in tl:
            tok = batch["token_ids"].to(DEVICE)
            opt.zero_grad()
            with torch.autocast(DEVICE.type, dtype=DTYPE):
                logits, _ = model(tok)
                B, T, V   = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V).float(),
                    tok[:, 1:].reshape(-1),
                    ignore_index=PAD_ID,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl_loss += loss.item()

        # validate
        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for batch in vl:
                tok = batch["token_ids"].to(DEVICE)
                with torch.autocast(DEVICE.type, dtype=DTYPE):
                    logits, _ = model(tok)
                    B, T, V   = logits.shape
                    vl_loss += F.cross_entropy(
                        logits[:, :-1].reshape(-1, V).float(),
                        tok[:, 1:].reshape(-1),
                        ignore_index=PAD_ID,
                    ).item()

        ta = tl_loss / len(tl);  va = vl_loss / len(vl)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        print(f"  Epoch {ep+1:02d}/{TOTAL_EPOCHS}  train={ta:.4f}  val={va:.4f}  lr={lr:.0e}", end="")

        if va < best_val:
            best_val = va; pat = 0
            torch.save({"model": model.state_dict(), "word2id": word2id}, ckpt)
            print("  ✓ best")
        else:
            pat += 1
            print(f"  (patience {pat}/{EARLY_STOP_PAT})")
            if pat >= EARLY_STOP_PAT:
                print(f"  Early stopping at epoch {ep+1}")
                break

    print(f"  Loading best checkpoint (val={best_val:.4f})")
    model.load_state_dict(torch.load(ckpt, weights_only=True)["model"])
    return model


def train_lstm(train_sents, val_sents, word2id, glove_matrix) -> VanillaLSTM:
    print(f"\n{'='*62}\nTraining VanillaLSTM\n{'='*62}")
    vs    = len(word2id)
    model = VanillaLSTM(vocab_size=vs, pretrained_embeddings=glove_matrix).to(DEVICE, DTYPE)
    return _train_lm_model(model, "VanillaLSTM", "lstm_best.pt",
                           train_sents, val_sents, word2id, glove_matrix)


def train_bilstm(train_sents, val_sents, word2id, glove_matrix) -> BiLSTM:
    print(f"\n{'='*62}\nTraining BiLSTM\n{'='*62}")
    vs    = len(word2id)
    model = BiLSTM(vocab_size=vs, pretrained_embeddings=glove_matrix).to(DEVICE, DTYPE)
    return _train_lm_model(model, "BiLSTM", "bilstm_best.pt",
                           train_sents, val_sents, word2id, glove_matrix)


# ---------------------------------------------------------------------------
# 5. Table helpers
# ---------------------------------------------------------------------------

def _print_table(rows, headers):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep   = "-+-".join("-" * w for w in col_w)
    fmt   = " | ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def _pct(v): return f"{v:.1%}" if v is not None else "n/a"
def _delta(d): return f"+{d:.1%}" if d >= 0 else f"{d:.1%}"


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 62)
    print("Solar Ring Memory — Train & Benchmark (GloVe 300d, 3 Models)")
    print("=" * 62)

    # ── Step 0: Dataset and vocab ─────────────────────────────────────────
    print("\n[0/6] Building balanced 3000-sentence dataset...")
    all_training  = build_pronoun_dataset()
    winograd_text = [t for ctx, c, w in WINOGRAD_SCHEMAS for t in (ctx, c, w)]

    random.seed(42)
    random.shuffle(all_training)
    split       = int(0.8 * len(all_training))
    train_sents = all_training[:split]   # 2400
    val_sents   = all_training[split:]   # 600

    word2id    = build_vocab(all_training + winograd_text, max_vocab=3000)
    vocab_size = len(word2id)
    print(f"  Vocabulary : {vocab_size} tokens")
    print(f"  Train      : {len(train_sents)} sentences  |  Val : {len(val_sents)}")
    print(f"  Winograd   : {len(WINOGRAD_SCHEMAS)} schemas")

    # ── Step 1: GloVe ─────────────────────────────────────────────────────
    print(f"\n[1/6] Loading GloVe 300d from {GLOVE_PATH}...")
    glove = None
    if Path(GLOVE_PATH).exists():
        glove = load_glove(GLOVE_PATH, word2id, d=300)
        print(f"  Matrix shape: {glove.shape}")
    else:
        print(f"  WARNING: {GLOVE_PATH} not found — random init only")

    # ── Step 2: Untrained baselines ───────────────────────────────────────
    print("\n[2/6] Untrained baselines on Winograd schemas...")
    def _fresh(cls):
        return cls(vocab_size=vocab_size, pretrained_embeddings=glove).to(DEVICE, DTYPE)

    solar_before, solar_cats_before = evaluate_model(
        _fresh(SolarRingModel), "SolarRingModel (untrained)", word2id)
    bilstm_before, bilstm_cats_before = evaluate_model(
        _fresh(BiLSTM), "BiLSTM (untrained)", word2id)
    lstm_before, lstm_cats_before = evaluate_model(
        _fresh(VanillaLSTM), "VanillaLSTM (untrained)", word2id)

    # ── Step 3-5: Train all 3 models ─────────────────────────────────────
    print("\n[3/6] Training SolarRingModel...")
    solar_model = train_solar(train_sents, val_sents, word2id, glove)

    print("\n[4/6] Training BiLSTM...")
    bilstm_model = train_bilstm(train_sents, val_sents, word2id, glove)

    print("\n[5/6] Training VanillaLSTM...")
    lstm_model = train_lstm(train_sents, val_sents, word2id, glove)

    # ── Step 6: Evaluate trained models ───────────────────────────────────
    print("\n[6/6] Final Winograd evaluation (trained)...")
    solar_after, solar_cats = evaluate_model(
        solar_model, "SolarRingModel (trained)", word2id)
    bilstm_after, bilstm_cats = evaluate_model(
        bilstm_model, "BiLSTM (trained)", word2id)
    lstm_after, lstm_cats = evaluate_model(
        lstm_model, "VanillaLSTM (trained)", word2id)

    # ── Final results table ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"Results — 90 Winograd Schemas  ({TOTAL_EPOCHS} epochs, GloVe 300d)")
    print("=" * 62)

    def row(name, before, after, cats, params):
        it  = _pct(cats.get("IT"))
        he  = _pct(cats.get("HE"))
        she = _pct(cats.get("SHE"))
        return [name, _pct(before), _pct(after), _delta(after - before), it, he, she, f"{params:,}"]

    solar_params  = sum(p.numel() for p in solar_model.parameters())
    bilstm_params = sum(p.numel() for p in bilstm_model.parameters())
    lstm_params   = sum(p.numel() for p in lstm_model.parameters())

    headers = ["Model", "Untrained", f"After {TOTAL_EPOCHS}ep", "Gain",
               "IT%", "HE%", "SHE%", "Params"]
    rows = [
        row("Solar Ring",   solar_before,  solar_after,  solar_cats,  solar_params),
        row("BiLSTM",       bilstm_before, bilstm_after, bilstm_cats, bilstm_params),
        row("Vanilla LSTM", lstm_before,   lstm_after,   lstm_cats,   lstm_params),
    ]
    print()
    _print_table(rows, headers)

    print(f"\nTarget: Solar Ring > 68%   →  {'MET ✓' if solar_after > 0.68 else f'gap {0.68-solar_after:.1%}'}")
    print(f"Solar Ring beats BiLSTM?  →  {'YES ✓' if solar_after > bilstm_after else 'NO'}")
    solar_he  = solar_cats.get("HE", 0);  bilstm_he  = bilstm_cats.get("HE", 0)
    solar_she = solar_cats.get("SHE", 0); bilstm_she = bilstm_cats.get("SHE", 0)
    print(f"Solar Ring HE > BiLSTM?   →  {'YES ✓' if solar_he  > bilstm_he  else 'NO'}"
          f"  ({_pct(solar_he)} vs {_pct(bilstm_he)})")
    print(f"Solar Ring SHE > BiLSTM?  →  {'YES ✓' if solar_she > bilstm_she else 'NO'}"
          f"  ({_pct(solar_she)} vs {_pct(bilstm_she)})")
    it_acc = solar_cats.get("IT", 0)
    print(f"Solar Ring IT > 65%?      →  {'YES ✓' if it_acc > 0.65 else f'gap {0.65-it_acc:.1%}'}"
          f"  ({_pct(it_acc)})")


if __name__ == "__main__":
    main()
