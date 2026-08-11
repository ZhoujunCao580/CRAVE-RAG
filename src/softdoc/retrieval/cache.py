"""Small embedding caches for single-document research retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from softdoc.ids import stable_digest
from softdoc.retrieval.models import EmbeddingCacheKey, EmbeddingCacheRecord


class EmbeddingCacheError(RuntimeError):
    """Raised when a persistent embedding cache record is corrupt."""


class EmbeddingCache(Protocol):
    def get(self, key: EmbeddingCacheKey) -> list[float] | None: ...

    def put(self, record: EmbeddingCacheRecord) -> None: ...


class MemoryEmbeddingCache:
    """Process-local cache used by default and by unit tests."""

    def __init__(self) -> None:
        self._records: dict[str, EmbeddingCacheRecord] = {}

    def get(self, key: EmbeddingCacheKey) -> list[float] | None:
        record = self._records.get(_key_digest(key))
        if record is None or record.key != key:
            return None
        return list(record.vector)

    def put(self, record: EmbeddingCacheRecord) -> None:
        self._records[_key_digest(record.key)] = record.model_copy(deep=True)


class FileEmbeddingCache:
    """JSON-per-vector cache with atomic replacement and strict validation."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def get(self, key: EmbeddingCacheKey) -> list[float] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            record = EmbeddingCacheRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise EmbeddingCacheError(
                f"Invalid embedding cache record {path}: {exc}"
            ) from exc
        if record.key != key:
            raise EmbeddingCacheError(
                f"Embedding cache identity mismatch in {path}"
            )
        return list(record.vector)

    def put(self, record: EmbeddingCacheRecord) -> None:
        path = self._path(record.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(record.model_dump_json(), encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _path(self, key: EmbeddingCacheKey) -> Path:
        digest = _key_digest(key)
        return self.directory / digest[:2] / f"{digest}.json"


def _key_digest(key: EmbeddingCacheKey) -> str:
    return stable_digest(key.model_dump(mode="json"), length=64)
