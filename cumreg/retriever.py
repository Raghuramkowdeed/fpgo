"""Example retrieval for in-context learning with kNN, recency, and diversity strategies.

Supports OLMo embeddings (reusing the generation model) or SentenceTransformer,
and min_reward_threshold filtering for best-of-N cache entries.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from .config import Config
from .datasets.base import Problem


# Lazy imports for embedding dependencies
_SentenceTransformer = None
_cosine_similarity = None
_KMeans = None


def _load_embedding_deps():
    global _SentenceTransformer, _cosine_similarity, _KMeans
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        from sklearn.metrics.pairwise import cosine_similarity as cs
        from sklearn.cluster import KMeans as KM
        _SentenceTransformer, _cosine_similarity, _KMeans = ST, cs, KM


@dataclass
class HistoryEntry:
    problem: Problem
    response: str
    score: float
    trace: str = ""  # Agent Lightning trace (reasoning chain / CoT)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class ExampleRetriever:
    """Retrieves few-shot examples from history with optional correct-only filtering.

    Supports:
    - OLMo embeddings (reuses loaded model, no second model load)
    - SentenceTransformer embeddings (fallback)
    - min_reward_threshold filtering (not just score > 0)
    - best-of-N storage: add() accepts best response + score from N candidates
    """

    def __init__(self, config: Config, model=None, tokenizer=None):
        """
        Args:
            config: Experiment config.
            model: Loaded LM for OLMo embeddings (optional).
            tokenizer: Corresponding tokenizer (optional).
        """
        self.config = config
        self.history: List[HistoryEntry] = []
        self.embedder = None

        # Sliding window
        if config.max_history_batches is not None:
            self.max_window = config.max_history_batches * config.batch_size
        else:
            self.max_window = None

        # Load embedder if needed
        if config.retrieval_strategy in ("knn", "diversity"):
            if config.use_olmo_embeddings and model is not None:
                from .embedder import OLMoEmbedder
                self.embedder = OLMoEmbedder(
                    model, tokenizer,
                    layer=config.embedding_layer,
                    max_length=config.embedding_max_length,
                )
                print("Using OLMo embeddings for retrieval")
            else:
                _load_embedding_deps()
                print(f"Loading embedding model: {config.embedding_model}")
                self.embedder = _SentenceTransformer(config.embedding_model)

    def add(self, problem: Problem, response: str, score: float, trace: str = ""):
        """Add a solved (or attempted) example to history."""
        entry = HistoryEntry(
            problem=problem,
            response=response,
            score=score,
            trace=trace,
        )
        if self.embedder is not None:
            emb = self.embedder.encode(problem.question, convert_to_numpy=True)
            if emb.ndim == 2:
                emb = emb[0]
            entry.embedding = emb
        self.history.append(entry)

    def _get_candidates(self) -> List[HistoryEntry]:
        """Get candidate pool: windowed + filtered by min_reward_threshold."""
        pool = self.history
        # Sliding window
        if self.max_window is not None and len(pool) > self.max_window:
            pool = pool[-self.max_window:]
        # Filter by reward threshold (subsumes retrieve_correct_only when threshold > 0)
        threshold = getattr(self.config, "min_reward_threshold", 0.0)
        if self.config.retrieve_correct_only or threshold > 0:
            min_score = max(threshold, 1e-9 if self.config.retrieve_correct_only else 0.0)
            pool = [e for e in pool if e.score >= min_score]
        return pool

    def get_examples(self, current_question: str, k: Optional[int] = None) -> List[HistoryEntry]:
        """Retrieve k examples for the current question."""
        if k is None:
            k = self.config.k_shots
        pool = self._get_candidates()
        if len(pool) < k:
            return list(pool)  # Return whatever we have (cold start)

        strategy = self.config.retrieval_strategy
        if strategy == "recency":
            return pool[-k:]
        elif strategy == "knn":
            return self._retrieve_knn(pool, current_question, k)
        elif strategy == "diversity":
            return self._retrieve_diversity(pool, k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

    def _retrieve_knn(self, pool: List[HistoryEntry], question: str, k: int) -> List[HistoryEntry]:
        query_emb = self.embedder.encode(question, convert_to_numpy=True)
        if query_emb.ndim == 2:
            query_emb = query_emb[0]

        pool_with_emb = [e for e in pool if e.embedding is not None]
        if len(pool_with_emb) == 0:
            return pool[-k:]

        pool_embs = np.stack([e.embedding for e in pool_with_emb])

        # OLMo embeddings are L2-normalized, so dot product = cosine sim
        sims = pool_embs @ query_emb
        top_k = np.argsort(sims)[-k:][::-1]
        return [pool_with_emb[i] for i in top_k]

    def _retrieve_diversity(self, pool: List[HistoryEntry], k: int) -> List[HistoryEntry]:
        _load_embedding_deps()
        n_clusters = min(k, len(pool))
        pool_embs = np.stack([e.embedding for e in pool])
        kmeans = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pool_embs)
        selected = []
        for cid in range(n_clusters):
            indices = [i for i, l in enumerate(labels) if l == cid]
            if indices:
                selected.append(pool[indices[-1]])
        return selected[:k]

    def num_eligible(self) -> int:
        """Count entries meeting reward threshold."""
        return len(self._get_candidates())

    def __len__(self):
        return len(self.history)
