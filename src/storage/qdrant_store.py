"""
Qdrant vector store connector for Evo-MedAgent.

Used for episodic memory (E): stores embeddings of case descriptors
and supports fast semantic similarity search.

Usage:
    store = QdrantStore(config)
    store.upsert(episode_id, embedding_vector, metadata_dict)
    results = store.search(query_vector, top_k=3)
"""
import os
import logging
import uuid
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant vector database connector.

    When enabled=False, acts as a no-op pass-through.
    When enabled=True, connects to Qdrant for persistent vector storage.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "evo_medagent_episodic",
        vector_dim: int = 384,
        enabled: bool = False,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.enabled = enabled

        self._client = None
        self._collection_ready = False

        if enabled:
            self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish connection and ensure collection exists."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if self.api_key:
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self._client = QdrantClient(url=self.url)

            # Check if collection exists, create if not
            collections = self._client.get_collections().collections
            names = [c.name for c in collections]

            if self.collection_name not in names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    f"Qdrant collection '{self.collection_name}' created "
                    f"(dim={self.vector_dim}, distance=COSINE)"
                )
            else:
                logger.info(
                    f"Qdrant collection '{self.collection_name}' ready "
                    f"(already exists)"
                )

            self._collection_ready = True

        except ImportError:
            logger.warning(
                "qdrant-client not installed. "
                "Install with: pip install qdrant-client"
            )
            self.enabled = False

        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Falling back to in-memory mode.")
            self.enabled = False

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(
        self,
        point_id: Optional[str],
        vector: np.ndarray,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """
        Insert or update a vector point.

        Args:
            point_id: unique ID (auto-generated if None)
            vector: embedding vector (numpy array, shape [D])
            payload: metadata dict (serializable to JSON)

        Returns:
            point_id of the inserted/updated point
        """
        if not self.enabled or not self._collection_ready:
            return point_id or str(uuid.uuid4())

        try:
            from qdrant_client.models import PointStruct

            pid = point_id or str(uuid.uuid4())
            vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)

            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=pid,
                        vector=vector_list,
                        payload=payload,
                    )
                ],
            )
            return pid

        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            return point_id or str(uuid.uuid4())

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: embedding of the query
            top_k: number of results
            score_threshold: minimum similarity score (0-1)

        Returns:
            list of dicts with keys: id, score, payload
        """
        if not self.enabled or not self._collection_ready:
            return []

        try:
            vector_list = (
                query_vector.tolist()
                if hasattr(query_vector, "tolist")
                else list(query_vector)
            )

            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=vector_list,
                limit=top_k,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def delete(self, point_id: str) -> bool:
        """Delete a point by ID."""
        if not self.enabled or not self._collection_ready:
            return False

        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            return False

    def count(self) -> int:
        """Return number of stored points."""
        if not self.enabled or not self._collection_ready:
            return 0

        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        if not self._client:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


def create_qdrant_store(config: dict) -> QdrantStore:
    """Factory: create QdrantStore from config dict."""
    qdrant_cfg = config.get("qdrant", {}) if config else {}
    return QdrantStore(
        url=qdrant_cfg.get("url", "http://localhost:6333"),
        api_key=os.getenv(qdrant_cfg.get("api_key_env", "")) or None,
        collection_name=qdrant_cfg.get("collection_name", "evo_medagent_episodic"),
        vector_dim=qdrant_cfg.get("vector_dim", 384),
        enabled=qdrant_cfg.get("enabled", False),
    )
