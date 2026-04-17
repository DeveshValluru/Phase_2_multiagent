"""SentenceBERT retriever.

- SCIMON: builds top-k index over the 114K training corpus, caches to disk.
- IdeaBench: lightweight re-rank of the ≤3 allowed reference abstracts.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)


class SentenceBERTRetriever:
    """Build-or-load wrapper around sentence-transformers.

    Forces CPU device by default because on Sol Gaudi nodes vLLM has already
    claimed all 8 HPUs for the Llama-3.3-70B generator; sentence-transformers
    would otherwise auto-select HPU and fail with "Device acquire failed".
    Override via SBERT_DEVICE env var if needed.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.device = os.environ.get("SBERT_DEVICE", "cpu")

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("loading SentenceBERT: %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self._ensure_model()
        embs = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return embs.astype(np.float32)


class CorpusIndex:
    """Cached top-k semantic neighbor index over a fixed text corpus."""

    def __init__(self, index_path: str | Path, model_name: str):
        self.index_path = Path(index_path)
        self.retriever = SentenceBERTRetriever(model_name)
        self._embs: np.ndarray | None = None
        self._texts: list[str] | None = None

    def build_or_load(self, corpus_texts: list[str]) -> None:
        if self.index_path.exists():
            log.info("loading cached SBERT index from %s", self.index_path)
            data = np.load(self.index_path, allow_pickle=True)
            self._embs = data["embs"]
            loaded_texts = data["texts"].tolist()
            # If corpus changed, rebuild
            if len(loaded_texts) == len(corpus_texts):
                self._texts = loaded_texts
                return
            log.warning("corpus size changed (%d -> %d); rebuilding index",
                        len(loaded_texts), len(corpus_texts))

        log.info("building SBERT index over %d texts -> %s",
                 len(corpus_texts), self.index_path)
        self._embs = self.retriever.encode(corpus_texts)
        self._texts = list(corpus_texts)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.index_path, embs=self._embs, texts=np.array(self._texts, dtype=object)
        )

    def top_k(self, query: str, k: int = 20) -> list[tuple[int, float, str]]:
        """Return list of (index, score, text) sorted by decreasing cosine."""
        if self._embs is None or self._texts is None:
            raise RuntimeError("index not built; call build_or_load first")
        q = self.retriever.encode([query])[0]   # already normalized
        sims = self._embs @ q
        # top-k by partial argsort
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i]), self._texts[i]) for i in idx]


class InlineReRanker:
    """Small-corpus re-rank (e.g., IdeaBench's 3 reference abstracts)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.retriever = SentenceBERTRetriever(model_name)

    def rerank(self, query: str, candidates: list[str], top_k: int | None = None):
        if not candidates:
            return []
        embs = self.retriever.encode([query] + list(candidates))
        q, docs = embs[0], embs[1:]
        sims = docs @ q
        order = np.argsort(-sims)
        out = [(int(i), float(sims[i]), candidates[i]) for i in order]
        if top_k is not None:
            out = out[:top_k]
        return out
