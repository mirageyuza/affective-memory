"""检索层：相关簇定位 → ANN 召回 → 情感增益/dB 权衡排序 → top-k 注入。

流程（规格 §3.7 记忆膨胀对策）：
  1. 图索引先行：query 实体 → 相关记忆 → 相关簇（缩小候选到相关簇）；
  2. ANN 召回：在候选簇内按余弦 top_m（默认暴力 ANN，可换 HNSW/FAISS）；
  3. 命中调制：遗忘可及性 × 震荡命中 H(t) × 情感增益；
  4. 沉默层过滤（见 silence.py）；
  5. dB 权衡排序取 top-k，受 token 预算约束。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Set

from ..config import Dial
from ..models import Memory, RankedHit
from ..store.storage import MemoryStore
from ..store.vector_index import AnnVectorIndex, BagOfWordsVectorizer, cosine
from ..store.graph_index import EntityGraph
from ..algorithms.hit_function import OscillationBank
from ..algorithms.affect import rank_hits
from .context import ContextInjector


class RecallEngine:
    def __init__(
        self,
        store: MemoryStore,
        vector_index: AnnVectorIndex,
        graph: EntityGraph,
        bank: OscillationBank,
        dial: Optional[Dial] = None,
    ):
        self.store = store
        self.vec = vector_index
        self.graph = graph
        self.bank = bank
        self.dial = dial or Dial()
        self.injector = ContextInjector(dial=self.dial)

    # ---- 外部注入（发散模式 / 并发联想可替换）----
    def _ann_recall(
        self,
        query_vec,
        candidate_ids: Optional[Set[str]],
        top_m: int,
    ) -> List[tuple]:
        return self.vec.search(query_vec, ids=candidate_ids, top_k=top_m)

    # ---- 主流程 ----
    def recall(
        self,
        query: str,
        query_entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        now: Optional[float] = None,
        expand_candidates: Optional[Set[str]] = None,
        direct_query: bool = False,
        view: str = "polluted",
    ) -> List[RankedHit]:
        dial = self.dial
        now = now if now is not None else time.time()
        k = top_k or dial.top_k

        # 1) 图索引先行：相关簇定位
        q_entities = [e for e in (query_entities or []) if e]
        graph_ids = self.graph.lookup(q_entities) if q_entities else set()
        candidate_ids: Set[str] = set(graph_ids)
        if expand_candidates:
            candidate_ids |= set(expand_candidates)
        # 候选为空时回退全库（防冷启动漏检）
        if not candidate_ids:
            candidate_ids = {m.memory_id for m in self.store.all()}

        # 2) ANN 召回（top_m 放大系数，供发散/降权使用）
        top_m = max(k * 3, 20)
        query_vec = self.vec.vectorizer.vectorize(query, q_entities)
        ann_hits = self._ann_recall(query_vec, candidate_ids, top_m)
        id2mem = {m.memory_id: m for m in self.store.all()}

        # 3) 命中调制 + 低频保护噪声（噪声温度保证低频记忆也有被想起概率）
        raw_hits: List[tuple] = []
        for mid, cossim in ann_hits:
            mem = id2mem.get(mid)
            if mem is None:
                continue
            gain = self.bank.gain_of(mem.affect, dial)
            hit_val = self.bank.hit(mem.cluster, now - mem.created_at, gain, dial)
            raw = cossim * (0.4 + 0.6 * hit_val)
            raw_hits.append((mem, raw))

        # 4) 沉默层过滤（在排序前过滤，节约排序开销）
        from ..silence import SilenceLayer
        strong = self.graph.strong_related(q_entities, dial.strong_relation_threshold) if q_entities else False
        silencer = SilenceLayer()
        raw_hits = silencer.filter_raw(
            raw_hits, direct_query=direct_query, strong_related=strong
        )

        # 5) 情感增益 + dB 权衡排序
        ranked = rank_hits(raw_hits, dial)[: k]

        # 5b) 双层视图绑定：view=original 回溯原文，否则输出污染视图（默认）
        for hit in ranked:
            if view == "original":
                hit.display_text = hit.memory.original_text or hit.memory.text
            else:
                hit.display_text = hit.memory.text or hit.memory.original_text

        # 6) top-k 上下文注入（token 预算）
        self.injector.attach_context(ranked)
        return ranked
