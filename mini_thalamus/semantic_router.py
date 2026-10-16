"""Stable semantic matcher for Agent capability descriptions."""

import logging
import math
import os
import re


_log = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> set[str]:
    normalized = str(text or "").lower()
    result = set(_TOKEN_PATTERN.findall(normalized))
    compact = "".join(sorted(result))
    for size in (2, 3):
        for index in range(0, max(0, len(compact) - size + 1)):
            result.add(compact[index:index + size])
    return result


class SemanticMatcher:
    """Rank Agent manifests by safe semantic similarity.

    The default backend is dependency-free and does not import torch or
    sentence-transformers. Embedding matching remains available when an encoder
    is injected or ``MT_SEMANTIC_BACKEND=embedding`` is set.
    """

    def __init__(self, encoder=None, model_name: str = None, backend: str = None):
        self._encoder = encoder
        self.model_name = model_name or os.environ.get(
            "MT_SEMANTIC_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.backend = backend or os.environ.get("MT_SEMANTIC_BACKEND", "safe")
        self.last_error = ""

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            # Windows DLL conflict: torch must not load before pyarrow.
            import pyarrow  # noqa: F401
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device="cpu")
            return self._encoder
        except Exception as exc:
            self.last_error = str(exc)
            _log.info("Semantic matcher unavailable; using LLM fallback: %s", exc)
            return None

    def rank(self, query: str, agent_specs: dict, top_k: int = 3) -> list[dict]:
        if self._encoder is not None or self.backend == "embedding":
            return self._embedding_rank(query, agent_specs, top_k=top_k)
        return self._safe_rank(query, agent_specs, top_k=top_k)

    def _safe_rank(self, query: str, agent_specs: dict, top_k: int = 3) -> list[dict]:
        if not query or not agent_specs:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored = []
        for agent_id, spec in agent_specs.items():
            profile = " ".join([
                spec.get("name", agent_id),
                spec.get("description", ""),
                " ".join(spec.get("intents", []) or []),
                " ".join(spec.get("examples", []) or []),
                " ".join(spec.get("keywords", []) or []),
            ])
            profile_tokens = _tokens(profile)
            if not profile_tokens:
                continue
            overlap = query_tokens & profile_tokens
            if not overlap:
                continue
            precision = len(overlap) / max(1, len(query_tokens))
            recall = len(overlap) / max(1, len(profile_tokens))
            score = math.sqrt(precision * recall)
            if score < 0.08:
                continue
            scored.append({
                "agent_id": agent_id,
                "score": max(0.0, min(1.0, score)),
                "evidence": "safe_semantic_similarity",
            })
        return sorted(
            scored,
            key=lambda item: (-item["score"], item["agent_id"]),
        )[:top_k]

    def _embedding_rank(self, query: str, agent_specs: dict, top_k: int = 3) -> list[dict]:
        encoder = self._get_encoder()
        if encoder is None or not agent_specs:
            return []

        agent_ids = list(agent_specs)
        profiles = []
        for agent_id in agent_ids:
            spec = agent_specs[agent_id]
            profiles.append(" ".join([
                spec.get("name", agent_id),
                spec.get("description", ""),
                " ".join(spec.get("intents", []) or []),
                " ".join(spec.get("examples", []) or []),
                " ".join(spec.get("keywords", []) or []),
            ]))

        try:
            vectors = encoder.encode(
                [query, *profiles],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vector = vectors[0]
            scored = []
            for index, agent_id in enumerate(agent_ids):
                score = float(query_vector @ vectors[index + 1])
                scored.append({
                    "agent_id": agent_id,
                    "score": max(0.0, min(1.0, score)),
                    "evidence": "embedding_similarity",
                })
            return sorted(
                scored,
                key=lambda item: item["score"],
                reverse=True,
            )[:top_k]
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("Semantic matching failed: %s", exc)
            return []


__all__ = ["SemanticMatcher"]
