"""记忆存储层：向量索引（ANN）。

默认实现：词袋哈希向量 + 余弦相似度暴力检索（标准库，零依赖）。
接口与后续可替换后端对齐：insert / delete / search。生产环境可将
_search 替换为 FAISS / HNSW (MIT) 或 Qdrant / Milvus (Apache-2.0)，
替换无需改动检索层调用方。
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class BagOfWordsVectorizer:
    """中文友好的词袋哈希向量：按连续词元哈希到固定维度。"""

    _TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")

    def __init__(self, dim: int = 256, seed: str = "emotion-memory-v0"):
        self.dim = dim
        self._seed = seed

    def _token_hash(self, token: str) -> int:
        raw = hashlib.md5((self._seed + "|" + token).encode("utf-8")).digest()
        return int.from_bytes(raw[:4], "big") % self.dim

    def vectorize(self, text: str, entities: Optional[Sequence[str]] = None) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._TOKEN_RE.findall(str(text).lower()):
            self._add_tokens(vec, tok, weight=1.0)
        for ent in entities or []:
            self._add_tokens(vec, str(ent).lower(), weight=2.0)
        return vec

    def _add_tokens(self, vec: List[float], raw: str, weight: float) -> None:
        """CJK 文本按单字 + 相邻二字对（bigram）哈希，英文保持整词。

        中文无空格分词，整句作为一个 token 会与查询词零交集；
        单字 + bigram 保证"爬山"这类查询能与原文中的相邻字对命中。
        """
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in raw)
        if not has_cjk:
            vec[self._token_hash(raw)] += weight
            return
        for ch in raw:
            if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
                vec[self._token_hash(ch)] += weight
        for i in range(len(raw) - 1):
            big = raw[i : i + 2]
            if all("\u4e00" <= c <= "\u9fff" or c.isalnum() for c in big):
                vec[self._token_hash(big)] += 1.5 * weight

    def dimension(self) -> int:
        return self.dim


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；零向量返回 0.0。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


class AnnVectorIndex:
    """内存 ANN 索引（暴力基线）。可替换为 HNSW/FAISS 后端。"""

    def __init__(self, vectorizer: Optional[BagOfWordsVectorizer] = None):
        self.vectorizer = vectorizer or BagOfWordsVectorizer()
        self._vectors: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def insert(self, memory_id: str, text: str, entities: Optional[Sequence[str]] = None) -> List[float]:
        vec = self.vectorizer.vectorize(text, entities)
        with self._lock:
            self._vectors[memory_id] = vec
        return vec

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            return self._vectors.pop(memory_id, None) is not None

    def size(self) -> int:
        with self._lock:
            return len(self._vectors)

    def search(
        self,
        query_vec: Sequence[float],
        ids: Optional[Iterable[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """在 ids 子集（或全量）内按余弦相似度召回 top_k。

        返回 [(memory_id, cosine_score), ...] 降序。
        """
        with self._lock:
            if ids is None:
                items = list(self._vectors.items())
            else:
                items = [(i, self._vectors[i]) for i in ids if i in self._vectors]
        scored = [(mid, cosine(query_vec, vec)) for mid, vec in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def vector_of(self, memory_id: str) -> Optional[List[float]]:
        with self._lock:
            return self._vectors.get(memory_id)
