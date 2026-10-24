"""发散模式（过激时的杂相想法）。

规格 §3.5：当 g(情感)·A > 阈值 → 进入发散模式：
  - 噪声温度拉高；
  - 跨簇随机跳转概率拉高；
  - 检索候选池扩大（含弱关联边）。

默认关闭（divergent_enabled=False）。"解除全部限制"= 拉满刻度盘，
不是拆掉安全底线。
"""
from __future__ import annotations

import random
from typing import Optional, Set

from .config import Dial
from .models import Memory
from .store.graph_index import EntityGraph


class DivergenceController:
    def __init__(self, dial: Dial, rng: Optional[random.Random] = None):
        self.dial = dial
        self._rng = rng or random.Random()

    @property
    def enabled(self) -> bool:
        return self.dial.divergent_enabled

    def should_engage(self, gain: float, amplitude: float) -> bool:
        """触发条件：g · A > threshold（且开关开启）。"""
        return self.enabled and (gain * amplitude) > self.dial.divergent_threshold

    def expand_candidates(
        self,
        candidates: Set[str],
        memories: "list[Memory]",
        graph: EntityGraph,
        jump_prob: Optional[float] = None,
    ) -> Set[str]:
        """扩大候选池：加入弱关联边记忆 + 随机跨簇跳转。

        jump_prob 缺省时用刻度盘 jump_probability（发散时内部拉高）。
        """
        prob = self.dial.jump_probability if jump_prob is None else jump_prob
        prob = min(1.0, max(0.0, prob))
        out = set(candidates)
        # 弱关联边扩展
        for mem in memories:
            if mem.memory_id in candidates:
                out |= graph.weak_edges(mem.memory_id, max_n=20)
        # 跨簇随机跳转
        extra = [m.memory_id for m in memories if m.memory_id not in out]
        take = max(0, int(prob * len(extra)))
        if take > 0:
            out |= set(self._rng.sample(extra, min(take, len(extra))))
        return out

    def hot_noise(self) -> float:
        """发散模式下噪声温度拉高（上限 1.0）。"""
        return min(1.0, self.dial.noise_temperature * 2.0 + 0.2)
