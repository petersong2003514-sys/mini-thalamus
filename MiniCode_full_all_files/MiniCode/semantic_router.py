"""Optional embedding-based matcher for Agent capability descriptions."""

import logging
import os


_log = logging.getLogger(__name__)


class SemanticMatcher:
    """Rank Agent manifests by embedding similarity.

    The model is loaded only on the first ambiguous request. If
    sentence-transformers or the configured model is unavailable, ``rank``
    returns an empty list and the Router proceeds to its LLM stage.
    """

    def __init__(self, encoder=None, model_name: str = None):
        self._encoder = encoder
        self.model_name = model_name or os.environ.get(
            "SHUSHU_SEMANTIC_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.last_error = ""

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device="cpu")
            return self._encoder
        except Exception as exc:
            self.last_error = str(exc)
            _log.info("Semantic matcher unavailable; using LLM fallback: %s", exc)
            return None

    def rank(self, query: str, agent_specs: dict, top_k: int = 3) -> list[dict]:
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
