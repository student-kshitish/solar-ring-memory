"""Dataset and DataLoader for Solar Ring Memory training."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .config import MAX_SEQ_LEN, ROLE_OTHER
from .pos_tagger import POSTagger


# PRONOUN_TOKENS (lowercased) used for pronoun_mask
PRONOUNS = {"it", "he", "she", "they", "him", "her", "them", "its", "his", "hers"}


class SolarRingDataset(Dataset):
    """
    Tokenizes sentences with a simple word-level vocabulary and
    produces:
        token_ids   (T,)  int64
        pos_labels  (T,)  int64 — role IDs from POSTagger
        spawn_labels(T,)  float — 1.0 if token triggers ring spawn
        pronoun_mask(T,)  bool  — True for pronoun tokens
    """

    def __init__(self, sentences: list[str], max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        self.tagger  = POSTagger()

        # Build vocabulary from all sentences
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for sent in sentences:
            for word in sent.lower().split():
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)

        # Pre-process all sentences
        self.samples = [self._process(s) for s in sentences]

    def _process(self, sentence: str):
        tags = self.tagger.tag(sentence)
        tokens  = [t["text"] for t in tags]
        roles   = [t["role"] for t in tags]
        spawns  = [1.0 if t["spawn"] else 0.0 for t in tags]
        is_pron = [t["text"].lower() in PRONOUNS for t in tags]

        # Add BOS/EOS
        ids = ([self.word2id["<BOS>"]]
               + [self.word2id.get(w.lower(), self.word2id["<UNK>"]) for w in tokens]
               + [self.word2id["<EOS>"]])
        roles   = [ROLE_OTHER] + roles   + [ROLE_OTHER]
        spawns  = [0.0]        + spawns  + [0.0]
        is_pron = [False]      + is_pron + [False]

        # Truncate
        ids     = ids[:self.max_len]
        roles   = roles[:self.max_len]
        spawns  = spawns[:self.max_len]
        is_pron = is_pron[:self.max_len]

        return {
            "token_ids":    torch.tensor(ids,     dtype=torch.long),
            "pos_labels":   torch.tensor(roles,   dtype=torch.long),
            "spawn_labels": torch.tensor(spawns,  dtype=torch.float),
            "pronoun_mask": torch.tensor(is_pron, dtype=torch.bool),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        seqs = [item[k] for item in batch]
        if seqs[0].dtype == torch.bool:
            padded = pad_sequence(
                [s.long() for s in seqs], batch_first=True, padding_value=0
            ).bool()
        elif seqs[0].dtype == torch.float:
            padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        else:
            padded = pad_sequence(seqs, batch_first=True, padding_value=0)
        out[k] = padded
    return out


def build_dataloader(sentences: list[str], batch_size: int = 4, shuffle: bool = True):
    dataset = SolarRingDataset(sentences)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, collate_fn=collate_fn
    )
    return loader, dataset
