"""Optional ``sentence-transformers`` backend.

Imported lazily so the core package stays installable with ``numpy`` alone.
Install with ``pip install "ragforge[st]"``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Embedder, l2_normalize, register_embedder


@register_embedder
class SentenceTransformerEmbedder(Embedder):
    """Wraps any `sentence-transformers` checkpoint.

    Args:
        model: Hub id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.
        query_prefix / passage_prefix: Instruction prefixes for asymmetric models
            (E5 wants ``"query: "`` / ``"passage: "``, BGE wants an instruction on
            the query only). Getting these wrong silently costs several points of
            recall, which is precisely the kind of thing the eval harness catches.
    """

    name = "sentence-transformer"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
    ) -> None:
        self.model_id = model
        self.batch_size = int(batch_size)
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self._model = None
        self._dim: int | None = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "sentence-transformers is not installed. "
                    'Run `pip install "ragforge[st]"`, or use the built-in '
                    "`hashing` embedder which needs no extra downloads."
                ) from exc
            self._model = SentenceTransformer(self.model_id, device=self.device)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        self._ensure_model()
        assert self._dim is not None
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"st:{self.model_id}:q{self.query_prefix}:p{self.passage_prefix}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = [self.passage_prefix + text for text in texts]
        vectors = model.encode(
            payload,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(vectors) if self.normalize else vectors

    def encode_query(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        vector = model.encode(
            [self.query_prefix + text], convert_to_numpy=True, show_progress_bar=False
        )
        vector = np.asarray(vector, dtype=np.float32)
        return (l2_normalize(vector) if self.normalize else vector)[0]

    def describe(self) -> str:
        return f"st({self.model_id})"
