from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from coref.chunking import make_chunks, mentions_to_chunk_local


@dataclass
class ChunkExample:
    doc_idx: int
    words: list[str]
    word_doc_id: list[tuple[int, int]]
    gold_clusters_local: list[list[tuple[int, int]]]  # list of clusters, each list of (start,end_incl)


def build_chunks_for_doc(doc_idx: int, sentences: list[list[str]], mention_clusters, baseline: str, tok_len_fn) -> list[ChunkExample]:
    chunks = make_chunks(sentences, baseline, tok_len_fn)
    per_chunk_gold = mentions_to_chunk_local(mention_clusters, chunks)
    out = []
    for c, gold in zip(chunks, per_chunk_gold):
        out.append(ChunkExample(doc_idx=doc_idx, words=c.words, word_doc_id=c.word_doc_id, gold_clusters_local=gold))
    return out


def precompute_chunks(hf_split, doc_indices: list[int], baseline: str, tokenizer) -> list[ChunkExample]:
    def tok_len_fn(words: list[str]) -> int:
        return len(tokenizer(words, is_split_into_words=True, add_special_tokens=False)["input_ids"])

    all_chunks: list[ChunkExample] = []
    for di in doc_indices:
        ex = hf_split[di]
        all_chunks.extend(build_chunks_for_doc(di, ex["sentences"], ex["mention_clusters"], baseline, tok_len_fn))
    return all_chunks


class ChunkDataset(Dataset):
    def __init__(self, chunks: list[ChunkExample]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def make_collate_fn(tokenizer, max_len: int = 512):
    def collate(batch: list[ChunkExample]):
        word_lists = [c.words for c in batch]
        enc = tokenizer(
            word_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        word_ids_batch = [enc.word_ids(i) for i in range(len(batch))]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_ids": word_ids_batch,
            "gold_clusters_local": [c.gold_clusters_local for c in batch],
            "word_doc_id": [c.word_doc_id for c in batch],
            "doc_idx": [c.doc_idx for c in batch],
        }

    return collate
