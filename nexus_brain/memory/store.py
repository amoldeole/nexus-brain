"""Vector store abstraction for long-term memory.

``InMemoryVectorStore`` is a tiny numpy-backed store with optional JSON
persistence so the MVP runs with zero infrastructure. It exposes the same
three operations (``add`` / ``search`` / ``get``) a production adapter for
Chroma, Qdrant, pgvector or Weaviate would expose.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np

from nexus_brain.core.schemas import MemoryItem, MemoryKind


class VectorStore(Protocol):
    def add(self, item: MemoryItem) -> None: ...

    def get(self, item_id: str) -> Optional[MemoryItem]: ...

    def search(self, embedding: list[float], k: int, kinds: Optional[Iterable[MemoryKind]] = None) -> list[tuple[MemoryItem, float]]: ...

    def all(self, kinds: Optional[Iterable[MemoryKind]] = None) -> list[MemoryItem]: ...

    def update(self, item: MemoryItem) -> None: ...

    def delete(self, item_id: str) -> bool: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self, path: Optional[Path | str] = None) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []
        self._dirty = True
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    def add(self, item: MemoryItem) -> None:
        self._items[item.id] = item
        self._dirty = True
        self._persist()

    def update(self, item: MemoryItem) -> None:
        self._items[item.id] = item
        self._dirty = True
        self._persist()

    def delete(self, item_id: str) -> bool:
        existed = self._items.pop(item_id, None) is not None
        self._dirty = True
        self._persist()
        return existed

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def all(self, kinds: Optional[Iterable[MemoryKind]] = None) -> list[MemoryItem]:
        ks = set(kinds) if kinds else None
        return [i for i in self._items.values() if ks is None or i.kind in ks]

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        self._ids = [i for i, it in self._items.items() if it.embedding]
        if self._ids:
            self._matrix = np.asarray([self._items[i].embedding for i in self._ids], dtype=np.float32)
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = self._matrix / norms
        else:
            self._matrix = None
        self._dirty = False

    def search(
        self,
        embedding: list[float],
        k: int,
        kinds: Optional[Iterable[MemoryKind]] = None,
    ) -> list[tuple[MemoryItem, float]]:
        if self._dirty:
            self._rebuild()
        if self._matrix is None:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n == 0:
            return []
        q = q / n
        sims = self._matrix @ q
        ks = set(kinds) if kinds else None
        order = np.argsort(-sims)
        out: list[tuple[MemoryItem, float]] = []
        for idx in order:
            item = self._items[self._ids[int(idx)]]
            if ks is not None and item.kind not in ks:
                continue
            out.append((item, float(sims[int(idx)])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ #
    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [i.model_dump(mode="json") for i in self._items.values()]
        self.path.write_text(json.dumps(data))

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text() or "[]")
        except json.JSONDecodeError:
            data = []
        for raw in data:
            item = MemoryItem.model_validate(raw)
            self._items[item.id] = item
        self._dirty = True
