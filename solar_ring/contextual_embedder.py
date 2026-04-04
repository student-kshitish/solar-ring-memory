from sentence_transformers import SentenceTransformer
import torch

MINILM_DIM = 384

class ContextualEmbedder:
    def __init__(self, device):
        self.device = device
        self.d = MINILM_DIM
        print("Loading MiniLM-L6-v2 (frozen)...")
        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=str(device)
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"MiniLM loaded. Output dim: {self.d}")

    def embed_words(self, sentence: str) -> torch.Tensor:
        words = sentence.split()
        embeddings = []
        with torch.no_grad():
            for i, word in enumerate(words):
                context = ' '.join(words[max(0, i-3):i+1])
                emb = self.model.encode(
                    context,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings.append(emb.to(self.device))
        if not embeddings:
            return torch.zeros(1, self.d, device=self.device)
        return torch.stack(embeddings)

    def embed_sentence(self, sentence: str) -> torch.Tensor:
        with torch.no_grad():
            emb = self.model.encode(
                sentence,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return emb.to(self.device)

    def embed_words_batch(self, sentences: list) -> dict:
        """
        Embed multiple sentences efficiently.
        Collects all word contexts across sentences,
        runs one batched MiniLM inference,
        splits results back per sentence.
        Returns dict: sentence -> (L, 384) tensor.
        """
        all_contexts = []
        sentence_lengths = []

        for sentence in sentences:
            words = sentence.split()
            sentence_lengths.append(len(words) if words else 1)
            if not words:
                all_contexts.append("")
            else:
                for i, word in enumerate(words):
                    context = ' '.join(words[max(0, i - 3):i + 1])
                    all_contexts.append(context)

        # One batched MiniLM call for ALL word contexts
        with torch.no_grad():
            all_embs = self.model.encode(
                all_contexts,
                convert_to_tensor=True,
                batch_size=64,
                show_progress_bar=False,
            )

        result = {}
        idx = 0
        for sent, length in zip(sentences, sentence_lengths):
            result[sent] = all_embs[idx:idx + length].to(self.device)
            idx += length
        return result
