"""沉默层：可提及 / 沉默两级可见性 + 解锁条件 + 永不沉默。

- 沉默记忆（silence=True）默认不注入，仅在"用户主动问起"（direct_query）
  或"强关联触发"（图共现权重 ≥ 阈值）时解锁；
- 永不沉默（never_silent）：高危、时间敏感类记忆（到期提醒、待办、
  异常告警）无论何时都必须注入；设置了到期时间 due_at 的记忆同样视为
  时间敏感，到期前即强制可见；
- 实现为可见性权限层：注入前过滤，只影响召回结果，不触碰原文。
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from .models import Memory, RankedHit


class SilenceLayer:
    def filter(
        self,
        ranked: Iterable[RankedHit],
        direct_query: bool = False,
        strong_related: bool = False,
    ) -> List[RankedHit]:
        out = []
        for hit in ranked:
            if self._visible(hit.memory, direct_query=direct_query, strong_related=strong_related):
                out.append(hit)
        return out

    def filter_raw(
        self,
        hits: Iterable[Tuple[Memory, float]],
        direct_query: bool = False,
        strong_related: bool = False,
    ) -> List[Tuple[Memory, float]]:
        out = []
        for mem, raw in hits:
            if self._visible(mem, direct_query=direct_query, strong_related=strong_related):
                out.append((mem, raw))
        return out

    def _visible(
        self,
        mem: Memory,
        direct_query: bool = False,
        strong_related: bool = False,
    ) -> bool:
        # 高危 / 时间敏感：永不沉默
        if mem.never_silent:
            return True
        if not mem.silence:
            return True
        # 沉默记忆：仅当用户主动问起或强关联触发时解锁
        return direct_query or strong_related
