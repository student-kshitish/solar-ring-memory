"""GloVe pretrained embedding loader for Solar Ring Memory."""

import numpy as np
from typing import Dict


def load_glove(glove_path: str, vocab: Dict[str, int], d: int = 300) -> np.ndarray:
    """
    Load GloVe vectors for words in vocab.

    Returns weight matrix of shape (vocab_size, d) as float32 numpy array.
    Unknown words get random vectors scaled to GloVe norm (~0.4 for 300d).
    """
    vectors: Dict[str, np.ndarray] = {}

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                vectors[word] = np.array(parts[1:], dtype=np.float32)

    vocab_size = len(vocab)
    scale = np.sqrt(1.0 / d)
    matrix = np.random.uniform(-scale, scale, (vocab_size, d)).astype(np.float32)

    found = 0
    for word, idx in vocab.items():
        if word in vectors:
            matrix[idx] = vectors[word]
            found += 1

    print(f"  GloVe: {found}/{vocab_size} vocab words found in embeddings")
    return matrix
