"""
Embedding utility for semantic retrieval across memory stores.
"""
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                # Fallback: use a simple hash-based approach for testing
                self._model = _FallbackEmbedder()
                self._dimension = 384
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension (populated after first encode)."""
        if self._dimension is None:
            _ = self.model  # trigger lazy load
        return self._dimension or 384

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True,
                                           show_progress_bar=False)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Fallback: random embeddings
            return np.random.randn(len(texts), self.dimension).astype(np.float32)

    def single_encode(self, text: str) -> np.ndarray:
        """Encode a single text to embedding vector."""
        emb = self.encode([text])
        return emb[0] if len(emb) > 0 else np.zeros(self.dimension, dtype=np.float32)


class _FallbackEmbedder:
    """Fallback embedder using TF-IDF-like hashing for testing without sentence-transformers."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        import hashlib
        dim = 384
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Simple character n-gram hashing
            for n in range(2, 5):
                for j in range(len(text) - n + 1):
                    h = int(hashlib.md5(text[j:j+n].encode()).hexdigest()[:8], 16)
                    result[i, h % dim] += 1
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result

    def get_sentence_embedding_dimension(self):
        return 384


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors a (N×D) and b (M×D) → N×M matrix."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)
