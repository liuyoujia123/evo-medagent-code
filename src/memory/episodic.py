"""
Episodic Memory (E): stores compressed records of prior diagnostic episodes.
Each episode ej = (φj, τj, ŷj, yj, σj, γj) per the paper's Eq. (3).
"""
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from ..utils.embedding import Embedder, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single diagnostic episode record."""
    episode_id: int
    phi: str                          # φ: compact case descriptor (derived from image + question)
    question: str                     # the clinical question
    tool_trace: str                   # τ: tool-interaction trace
    prediction: str                   # ŷ: agent's answer
    ground_truth: str                 # y: ground-truth answer
    is_correct: bool                  # whether prediction matches ground truth
    summary: str = ""                 # σ: one-sentence retrospective summary
    guideline: str = ""              # γ: actionable guideline from reflection

    def to_text(self) -> str:
        """Render episode as a readable context string for injection into agent prompt."""
        outcome = "CORRECT" if self.is_correct else "INCORRECT"
        parts = [
            f"[Prior Case #{self.episode_id}] Outcome: {outcome}",
            f"Question: {self.question}",
            f"Agent answered: {self.prediction}",
            f"Correct answer: {self.ground_truth}",
        ]
        if self.summary:
            parts.append(f"Takeaway: {self.summary}")
        if self.guideline:
            parts.append(f"Guideline: {self.guideline}")
        return "\n".join(parts)


class EpisodicMemory:
    """
    Episodic memory store that maintains compressed records of prior diagnostic episodes.
    Supports insertion, retrieval by semantic similarity, and serialization.
    """

    def __init__(self, embedder: Embedder, max_episodes: int = 200,
                 retrieval_k: int = 3, relevance_threshold: float = 0.3):
        self.embedder = embedder
        self.max_episodes = max_episodes
        self.retrieval_k = retrieval_k
        self.relevance_threshold = relevance_threshold

        self._episodes: List[Episode] = []
        self._embeddings: Optional[np.ndarray] = None  # N×D matrix
        self._next_id: int = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    def add(self, episode: Episode) -> None:
        """Add a new episode and update the embedding matrix."""
        # Assign ID
        episode.episode_id = self._next_id
        self._next_id += 1

        # Compute embedding
        embed_text = self._make_embed_text(episode)
        emb = self.embedder.single_encode(embed_text)

        # Add to store
        self._episodes.append(episode)
        if self._embeddings is None:
            self._embeddings = emb.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])

        # Evict oldest if over capacity
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)
            self._embeddings = self._embeddings[1:]

    def retrieve(self, case_descriptor: str, k: Optional[int] = None) -> List[Episode]:
        """
        Retrieve top-k most relevant episodes for the current case.
        Uses cosine similarity between case descriptor embedding and stored episode embeddings.
        Eq. (4): R_epi_i = Top-K_{e∈E_{i-1}} sepi(x_i, e)
        """
        k = k or self.retrieval_k
        if not self._episodes:
            return []

        # Encode current case
        query_emb = self.embedder.single_encode(case_descriptor)

        # Compute similarities
        sims = cosine_similarity(query_emb.reshape(1, -1), self._embeddings)[0]

        # Sort descending and filter by threshold
        indices = np.argsort(sims)[::-1]
        results = []
        for idx in indices:
            if sims[idx] < self.relevance_threshold:
                continue
            results.append(self._episodes[idx])
            if len(results) >= k:
                break

        return results

    def format_context(self, episodes: List[Episode]) -> str:
        """Format retrieved episodes as a context string for the agent prompt."""
        if not episodes:
            return "No relevant prior cases found."
        return "\n\n".join(e.to_text() for e in episodes)

    def _make_embed_text(self, episode: Episode) -> str:
        """Build a text representation for embedding."""
        return f"Case: {episode.phi}\nQuestion: {episode.question}"

    def save(self, path: str) -> None:
        """Serialize memory state to JSON."""
        data = {
            "next_id": self._next_id,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "phi": ep.phi,
                    "question": ep.question,
                    "tool_trace": ep.tool_trace,
                    "prediction": ep.prediction,
                    "ground_truth": ep.ground_truth,
                    "is_correct": ep.is_correct,
                    "summary": ep.summary,
                    "guideline": ep.guideline,
                }
                for ep in self._episodes
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Episodic memory saved: {len(self._episodes)} episodes → {path}")

    def load(self, path: str) -> None:
        """Deserialize memory state from JSON."""
        import os
        if not os.path.exists(path):
            logger.warning(f"No episodic memory file at {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 0)
        self._episodes.clear()
        self._embeddings = None  # reset to rebuild from loaded episodes

        for ed in data.get("episodes", []):
            ep = Episode(
                episode_id=ed["episode_id"],
                phi=ed["phi"],
                question=ed["question"],
                tool_trace=ed["tool_trace"],
                prediction=ed["prediction"],
                ground_truth=ed["ground_truth"],
                is_correct=ed["is_correct"],
                summary=ed.get("summary", ""),
                guideline=ed.get("guideline", ""),
            )
            # Rebuild embedding
            embed_text = self._make_embed_text(ep)
            emb = self.embedder.single_encode(embed_text)
            self._episodes.append(ep)
            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])

        logger.info(f"Episodic memory loaded: {len(self._episodes)} episodes ← {path}")
