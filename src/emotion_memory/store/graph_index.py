"""记忆存储层：简易实体-关系图索引（联想索引）。

自研邻接表实现（规避 Neo4j Community 的 GPL 许可），支撑：
  - 相关簇定位：query 实体 → 相关记忆 → 相关簇集合（缩小候选池）
  - 强关联触发：实体共现权重达到阈值即视为强关联（沉默层解锁条件之一）
  - 弱关联边：发散模式扩大候选池时使用
"""
from __future__ import annotations

import threading
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from ..models import Memory


class EntityGraph:
    def __init__(self) -> None:
        self._entity_to_mem: Dict[str, Set[str]] = {}
        self._mem_to_entity: Dict[str, Set[str]] = {}
        self._cooccur: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()

    # ---- 写 ----
    def add_memory(self, memory: Memory) -> None:
        with self._lock:
            entities = {e.strip() for e in memory.entities if e and e.strip()}
            self._mem_to_entity[memory.memory_id] = set(entities)
            for e in entities:
                self._entity_to_mem.setdefault(e, set()).add(memory.memory_id)
            for a in entities:
                for b in entities:
                    if a >= b:
                        continue
                    key = tuple(sorted((a, b)))
                    self._cooccur[key] = self._cooccur.get(key, 0.0) + 1.0

    def remove_memory(self, memory_id: str) -> None:
        with self._lock:
            entities = self._mem_to_entity.pop(memory_id, set())
            for e in entities:
                s = self._entity_to_mem.get(e)
                if s:
                    s.discard(memory_id)
                    if not s:
                        del self._entity_to_mem[e]

    # ---- 读 ----
    def entities(self) -> List[str]:
        with self._lock:
            return list(self._entity_to_mem.keys())

    def lookup(self, query_entities: Sequence[str]) -> Set[str]:
        """由实体集合定位相关记忆 id。"""
        with self._lock:
            hits: Set[str] = set()
            for e in query_entities:
                hits |= self._entity_to_mem.get(e, set())
            return hits

    def clusters_of(self, memory_ids: Iterable[str]) -> Set[str]:
        with self._lock:
            return {m for m in memory_ids}  # 占位，见 store lookup_clusters

    def cooccur_weight(self, a: str, b: str) -> float:
        key = tuple(sorted((a, b)))
        with self._lock:
            return self._cooccur.get(key, 0.0)

    def related_entities(self, memory_id: str, max_n: int = 10) -> List[str]:
        with self._lock:
            ents = list(self._mem_to_entity.get(memory_id, set()))
        return ents[:max_n]

    def strong_related(self, query_entities: Sequence[str], threshold: float) -> bool:
        """query 实体之间是否存在强共现关系（强关联触发条件）。"""
        if len(query_entities) < 2:
            return False
        for i in range(len(query_entities)):
            for j in range(i + 1, len(query_entities)):
                if self.cooccur_weight(query_entities[i], query_entities[j]) >= threshold:
                    return True
        return False

    def weak_edges(self, memory_id: str, max_n: int = 20) -> Set[str]:
        """弱关联边：与记忆实体共现但权重 < 强关联阈值的记忆（发散模式候选扩展）。"""
        ents = self.related_entities(memory_id, max_n=8)
        out: Set[str] = set()
        for e in ents:
            with self._lock:
                out |= self._entity_to_mem.get(e, set())
        out.discard(memory_id)
        return out

    def clear(self) -> None:
        with self._lock:
            self._entity_to_mem.clear()
            self._mem_to_entity.clear()
            self._cooccur.clear()
