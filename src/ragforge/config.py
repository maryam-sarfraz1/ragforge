"""Declarative pipeline configuration.

Every knob that affects retrieval quality lives in one serialisable object. That
is what makes ``ragforge sweep`` possible: a grid search is just a list of these,
and a leaderboard row can name the exact configuration that produced it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


def _load_structured(path: str) -> Any:
    """Read JSON, or YAML when PyYAML happens to be available."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                f"{path} looks like YAML but PyYAML is not installed. "
                "Run `pip install pyyaml`, or use a .json file."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


@dataclass
class PipelineConfig:
    """A complete, reproducible description of one retrieval stack."""

    name: str = "default"

    chunker: str = "sentence"
    chunker_args: dict[str, Any] = field(default_factory=dict)

    embedder: str = "hashing"
    embedder_args: dict[str, Any] = field(default_factory=dict)
    embedder_cache: str | None = None
    fit_embedder: bool = True

    store: str = "memory"
    store_args: dict[str, Any] = field(default_factory=dict)

    retriever: str = "hybrid"
    bm25_args: dict[str, Any] = field(default_factory=dict)
    hybrid_args: dict[str, Any] = field(default_factory=dict)

    rerank: str | None = None
    rerank_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"dense", "bm25", "hybrid"}
        if self.retriever not in valid:
            raise ValueError(f"retriever must be one of {sorted(valid)}, got {self.retriever!r}")
        if self.rerank not in (None, "mmr", "cross-encoder"):
            raise ValueError(f"rerank must be None, 'mmr' or 'cross-encoder', got {self.rerank!r}")

    @property
    def label(self) -> str:
        """Short human-readable identity, used as the leaderboard row key."""
        parts = [self.chunker]
        size = self.chunker_args.get("size") or self.chunker_args.get("max_size")
        if size:
            parts[-1] += f"@{size}"
        parts.append(self.embedder)
        parts.append(self.retriever)
        if self.retriever == "hybrid":
            fusion = self.hybrid_args.get("fusion", "rrf")
            parts[-1] += f"[{fusion}]"
        if self.rerank:
            parts.append(f"+{self.rerank}")
        return " / ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineConfig:
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown config keys: {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}"
            )
        return cls(**payload)

    @classmethod
    def from_file(cls, path: str) -> PipelineConfig:
        return cls.from_dict(_load_structured(path))

    def merged(self, **overrides: Any) -> PipelineConfig:
        """A copy with fields replaced — the building block for sweeps."""
        payload = self.to_dict()
        payload.update(overrides)
        return PipelineConfig.from_dict(payload)


def expand_grid(base: PipelineConfig, grid: dict[str, list[Any]]) -> list[PipelineConfig]:
    """Cartesian product of ``grid`` applied on top of ``base``.

    ``{"chunker": ["sentence", "fixed"], "retriever": ["dense", "hybrid"]}`` yields
    four configs. Names are auto-assigned from each config's label so leaderboard
    rows stay traceable back to the exact settings.
    """
    import itertools

    if not grid:
        return [base]
    keys = list(grid)
    configs: list[PipelineConfig] = []
    for combination in itertools.product(*(grid[key] for key in keys)):
        overrides = dict(zip(keys, combination))
        config = base.merged(**overrides)
        config.name = config.label
        configs.append(config)
    return configs


def load_grid(path: str) -> list[PipelineConfig]:
    """Load a sweep definition: ``{"base": {...}, "grid": {...}}`` or a bare list."""
    payload = _load_structured(path)
    if isinstance(payload, list):
        return [PipelineConfig.from_dict(item) for item in payload]
    base = PipelineConfig.from_dict(payload.get("base", {}))
    return expand_grid(base, payload.get("grid", {}))
