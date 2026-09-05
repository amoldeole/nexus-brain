"""Embedding providers.

The default ``HashingEmbedder`` is dependency-free and deterministic: it maps
word unigrams + bigrams + character trigrams into a fixed-size vector using the
hashing trick, with TF weighting and L2 normalisation. It is *not* a semantic
model - it is good enough for lexical-overlap retrieval in tests and demos and
keeps the MVP runnable offline. Swap in ``SentenceTransformerEmbedder`` or an
API embedder for production; the interface is identical.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it", "i", "you",
    "me", "my", "we", "be", "at", "as", "by", "with", "that", "this", "can", "do", "please",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512, seed: str = "nexus") -> None:
        self.dim = dim
        self.seed = seed

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(f"{self.seed}|{feature}".encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        if not toks:
            return vec.tolist()
        feats: list[tuple[str, float]] = []
        for t in toks:
            feats.append((f"w:{t}", 1.0))
            padded = f"#{t}#"
            for i in range(len(padded) - 2):
                feats.append((f"c:{padded[i:i+3]}", 0.35))
        for a, b in zip(toks, toks[1:]):
            feats.append((f"b:{a}_{b}", 0.8))
        for f, w in feats:
            idx, sign = self._bucket(f)
            vec[idx] += sign * w
        # sub-linear tf then l2 normalise
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:
    """Optional semantic embedder (requires `pip install sentence-transformers`)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def text_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content tokens - a cheap, explainable fallback signal."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))
