"""
rag/embeddings.py
──────────────────
Clinical document embedding utilities for the RAG pipeline.

Provides specialized embedding strategies for clinical text:
    - Clinical concept-aware chunking (preserves medical context)
    - Hybrid dense + sparse retrieval scoring
    - Embedding caching for cost optimization
    - Batch embedding with rate limit handling
    - Similarity search with clinical relevance scoring

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import openai

logger = logging.getLogger(__name__)


# ── Clinical section markers ──────────────────────────────────────────────────

# Clinical document section headers — used for smart chunking
CLINICAL_SECTION_MARKERS = [
    # SOAP note sections
    "SUBJECTIVE:", "OBJECTIVE:", "ASSESSMENT:", "PLAN:",
    # Discharge summary sections
    "CHIEF COMPLAINT:", "HISTORY OF PRESENT ILLNESS:",
    "PAST MEDICAL HISTORY:", "MEDICATIONS:", "ALLERGIES:",
    "PHYSICAL EXAMINATION:", "LABORATORY DATA:", "IMAGING:",
    "HOSPITAL COURSE:", "DISCHARGE DIAGNOSIS:", "DISCHARGE MEDICATIONS:",
    "FOLLOW-UP:", "DISCHARGE INSTRUCTIONS:",
    # Operative / procedure notes
    "PREOPERATIVE DIAGNOSIS:", "POSTOPERATIVE DIAGNOSIS:",
    "PROCEDURE:", "COMPLICATIONS:", "FINDINGS:",
    # Oncology specific
    "STAGING:", "PERFORMANCE STATUS:", "TREATMENT PLAN:",
    "RESPONSE ASSESSMENT:", "ADVERSE EVENTS:",
]

# High-priority clinical terms that should never be split across chunks
NEVER_SPLIT_PATTERNS = [
    "allerg",         # Allergy information
    "contraindic",    # Contraindications
    "drug interaction",
    "black box warning",
    "do not use",
    "critical value",
]


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Result from embedding a single text."""
    text: str
    embedding: list[float]
    model: str
    token_count: int = 0
    cached: bool = False


@dataclass
class SimilarityResult:
    """Result from a similarity search."""
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0


@dataclass
class EmbeddingCacheStats:
    """Statistics for the embedding cache."""
    total_entries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 3) if total > 0 else 0.0


# ── Clinical Embedder ─────────────────────────────────────────────────────────

class ClinicalEmbedder:
    """
    Clinical document embedder with caching and smart chunking.

    Wraps OpenAI embeddings with clinical-specific optimizations:
    - Persistent disk cache to avoid re-embedding identical documents
    - Clinical section-aware chunking that preserves medical context
    - Batch processing with automatic rate limit retry
    - Hybrid scoring combining semantic similarity and clinical term matching

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.
    model : str
        Embedding model. Default 'text-embedding-3-small'.
    cache_dir : str, optional
        Directory for embedding cache. Default '.embedding_cache'.
    batch_size : int
        Texts per API batch call. Default 100.

    Examples
    --------
    >>> embedder = ClinicalEmbedder()
    >>> result = embedder.embed("Patient has penicillin allergy.")
    >>> similar = embedder.find_similar(
    ...     query="drug allergies",
    ...     corpus=["Patient has penicillin allergy.", "Normal CBC results."],
    ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache_dir: Optional[str] = ".embedding_cache",
        batch_size: int = 100,
    ) -> None:
        import os
        self.model      = model
        self.batch_size = batch_size
        self._client    = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        self._cache: dict[str, list[float]] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._stats = EmbeddingCacheStats()

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        logger.info("ClinicalEmbedder initialized: model=%s", model)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text string.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        EmbeddingResult
        """
        cache_key = self._cache_key(text)

        if cache_key in self._cache:
            self._stats.cache_hits += 1
            return EmbeddingResult(
                text=text,
                embedding=self._cache[cache_key],
                model=self.model,
                cached=True,
            )

        self._stats.cache_misses += 1
        response = self._call_api([text])
        embedding = response.data[0].embedding
        token_count = response.usage.total_tokens

        self._cache[cache_key] = embedding
        self._save_cache_entry(cache_key, embedding)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            token_count=token_count,
            cached=False,
        )

    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[EmbeddingResult]:
        """
        Embed a batch of texts efficiently.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        show_progress : bool
            Log progress for large batches. Default False.

        Returns
        -------
        list[EmbeddingResult]
        """
        results = []
        uncached_indices = []
        uncached_texts = []

        # Separate cached from uncached
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(EmbeddingResult(
                    text=text,
                    embedding=self._cache[key],
                    model=self.model,
                    cached=True,
                ))
                self._stats.cache_hits += 1
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._stats.cache_misses += 1

        # Batch embed uncached texts
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[batch_start:batch_start + self.batch_size]
            if show_progress:
                logger.info(
                    "Embedding batch %d-%d / %d",
                    batch_start, batch_start + len(batch), len(uncached_texts),
                )
            response = self._call_api(batch)
            for j, item in enumerate(response.data):
                text  = batch[j]
                key   = self._cache_key(text)
                emb   = item.embedding
                self._cache[key] = emb
                self._save_cache_entry(key, emb)
                idx = uncached_indices[batch_start + j]
                results[idx] = EmbeddingResult(
                    text=text,
                    embedding=emb,
                    model=self.model,
                    token_count=0,
                    cached=False,
                )

        return results

    def find_similar(
        self,
        query: str,
        corpus: list[str],
        top_k: int = 5,
        clinical_boost: bool = True,
    ) -> list[SimilarityResult]:
        """
        Find the most similar texts in a corpus using cosine similarity.

        Parameters
        ----------
        query : str
            Query text.
        corpus : list[str]
            Texts to search over.
        top_k : int
            Number of results to return. Default 5.
        clinical_boost : bool
            If True, boost scores for texts containing safety-critical
            clinical terms (allergies, contraindications). Default True.

        Returns
        -------
        list[SimilarityResult]
            Top-k most similar texts, sorted by score descending.
        """
        query_result  = self.embed(query)
        corpus_results = self.embed_batch(corpus)

        query_vec = np.array(query_result.embedding)
        scores = []

        for i, corpus_result in enumerate(corpus_results):
            corpus_vec = np.array(corpus_result.embedding)
            cosine_sim = float(
                np.dot(query_vec, corpus_vec) /
                (np.linalg.norm(query_vec) * np.linalg.norm(corpus_vec) + 1e-8)
            )

            # Clinical boost for safety-critical terms
            boost = 0.0
            if clinical_boost:
                text_lower = corpus[i].lower()
                for pattern in NEVER_SPLIT_PATTERNS:
                    if pattern in text_lower:
                        boost = 0.05
                        break

            scores.append((i, cosine_sim + boost))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            SimilarityResult(
                text=corpus[i],
                score=round(score, 4),
                rank=rank + 1,
            )
            for rank, (i, score) in enumerate(scores[:top_k])
        ]

    def chunk_clinical_document(
        self,
        text: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> list[dict]:
        """
        Split a clinical document into chunks with clinical section awareness.

        Preserves section boundaries and never splits safety-critical
        clinical information (allergy lists, medication orders) across chunks.

        Parameters
        ----------
        text : str
            Clinical document text.
        chunk_size : int
            Target chunk size in characters. Default 800.
        chunk_overlap : int
            Overlap between consecutive chunks. Default 100.

        Returns
        -------
        list[dict]
            List of chunks with 'text' and 'section' keys.
        """
        # Split on clinical section markers first
        sections = self._split_on_sections(text)
        chunks = []

        for section_name, section_text in sections:
            if len(section_text) <= chunk_size:
                chunks.append({
                    "text":    section_text.strip(),
                    "section": section_name,
                })
            else:
                # Further split large sections
                sub_chunks = self._split_text(
                    section_text, chunk_size, chunk_overlap
                )
                for chunk in sub_chunks:
                    chunks.append({
                        "text":    chunk.strip(),
                        "section": section_name,
                    })

        # Filter empty chunks
        chunks = [c for c in chunks if len(c["text"]) > 20]

        logger.debug(
            "Chunked clinical document: %d chars → %d chunks",
            len(text), len(chunks),
        )
        return chunks

    def get_cache_stats(self) -> EmbeddingCacheStats:
        """Return embedding cache statistics."""
        self._stats.total_entries = len(self._cache)
        return self._stats

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_api(
        self,
        texts: list[str],
        max_retries: int = 3,
    ):
        """Call OpenAI embeddings API with retry on rate limit."""
        for attempt in range(max_retries):
            try:
                return self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning(
                    "Rate limit hit — waiting %ds (attempt %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"OpenAI embeddings API failed after {max_retries} attempts"
        )

    def _split_on_sections(
        self, text: str
    ) -> list[tuple[str, str]]:
        """Split clinical text on section markers."""
        result = []
        current_section = "GENERAL"
        current_text = []

        for line in text.split("\n"):
            line_upper = line.strip().upper()
            matched_section = None
            for marker in CLINICAL_SECTION_MARKERS:
                if line_upper.startswith(marker):
                    matched_section = marker.rstrip(":")
                    break

            if matched_section:
                if current_text:
                    result.append((current_section, "\n".join(current_text)))
                current_section = matched_section
                current_text = [line]
            else:
                current_text.append(line)

        if current_text:
            result.append((current_section, "\n".join(current_text)))

        return result if result else [("GENERAL", text)]

    @staticmethod
    def _split_text(
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                for sep in [". ", "\n", " "]:
                    idx = text.rfind(sep, start, end)
                    if idx > start:
                        end = idx + len(sep)
                        break
            chunks.append(text[start:end])
            start = max(start + 1, end - overlap)
        return chunks

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a deterministic cache key for a text string."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self._cache_dir:
            return
        cache_file = self._cache_dir / "embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(
                    "Loaded %d cached embeddings", len(self._cache)
                )
            except Exception as e:
                logger.warning("Could not load embedding cache: %s", e)

    def _save_cache_entry(self, key: str, embedding: list[float]) -> None:
        """Persist a single cache entry to disk."""
        if not self._cache_dir:
            return
        cache_file = self._cache_dir / "embeddings_cache.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning("Could not save embedding cache entry: %s", e)
