"""并发联想（一个记忆元想起多个记忆）。

规格 §3.6：记忆按实体/主题分片，多路并行召回 → 汇聚排序（Map-Reduce）。
效果：记忆量翻倍 + 加并行度 → 延迟恒定；配合随机扰动防止"高频记忆
垄断、低频记忆饿死"。

默认关闭（concurrent_enabled=False）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Set

from .config import Dial
from .models import RankedHit


class ConcurrentRecall:
    """分片 Map-Reduce 并发召回器。

    shards 为分片列表（如按簇/实体分组后的候选 id 集合）；
    recall_fn(shard_ids) -> List[RankedHit] 对单个分片执行召回。
    """

    def __init__(self, dial: Dial, workers: Optional[int] = None):
        self.dial = dial
        self.workers = workers or dial.concurrent_workers

    @property
    def enabled(self) -> bool:
        return self.dial.concurrent_enabled

    def map_reduce(
        self,
        shards: Sequence[Set[str]],
        recall_fn: Callable[[Set[str]], List[RankedHit]],
    ) -> List[RankedHit]:
        if not shards:
            return []
        if len(shards) == 1:
            return recall_fn(shards[0])
        with ThreadPoolExecutor(max_workers=min(self.workers, len(shards))) as pool:
            futures = [pool.submit(recall_fn, shard) for shard in shards]
            merged: List[RankedHit] = []
            for f in futures:
                merged.extend(f.result())
        # 汇聚排序（按 score 降序，保序稳定）
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged

    def shard_by_cluster(self, memories) -> Dict[str, Set[str]]:
        shards: Dict[str, Set[str]] = {}
        for m in memories:
            shards.setdefault(m.cluster, set()).add(m.memory_id)
        return shards
