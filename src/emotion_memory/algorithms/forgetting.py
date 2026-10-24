"""遗忘曲线：可及性分数衰减与强化。

遵循艾宾浩斯式规律：

    score(t) = 基础强度 · 衰减(ΔT) · 强化(访问次数)

- 衰减项：exp(-decay_base · sqrt(ΔT_hours))，长期不触达则可及性下降；
- 强化项：1 + reinforce · log(1 + access_count)，高频使用被强化；
- 衰减只降"被想起的概率"，不删内容——永久不忘但降低提及率。
"""
from __future__ import annotations

import math
import time
from typing import Optional

DEFAULT_DECAY_BASE = 0.05
DEFAULT_REINFORCE = 0.18
FLOOR = 0.001  # 可及性下限 > 0，保证"永久不忘"


def compute_accessibility(
    base_strength: float,
    dt_hours: float,
    access_count: int,
    decay_base: float = DEFAULT_DECAY_BASE,
    reinforce: float = DEFAULT_REINFORCE,
) -> float:
    dt = max(0.0, float(dt_hours))
    decay = math.exp(-decay_base * math.sqrt(dt))
    boost = 1.0 + reinforce * math.log1p(max(0, int(access_count)))
    score = float(base_strength) * decay * boost
    return min(1.0, max(FLOOR, score))


class ForgettingCurve:
    """可及性分数管理器：批量推进时间 / 单条触达强化。"""

    def __init__(
        self,
        decay_base: float = DEFAULT_DECAY_BASE,
        reinforce: float = DEFAULT_REINFORCE,
    ):
        self.decay_base = decay_base
        self.reinforce = reinforce

    def advance(self, memory, now: Optional[float] = None) -> float:
        """把一条记忆推进到当前时刻，更新可及性分数。"""
        now = now if now is not None else time.time()
        dt_hours = max(0.0, (now - memory.created_at) / 3600.0)
        memory.accessibility = compute_accessibility(
            memory.base_strength, dt_hours, memory.access_count,
            self.decay_base, self.reinforce,
        )
        return memory.accessibility

    def advance_many(self, memories, now: Optional[float] = None):
        now = now if now is not None else time.time()
        for m in memories:
            self.advance(m, now=now)

    def touch(self, memory, now: Optional[float] = None) -> float:
        """触达强化：访问计数 +1 并立即重算可及性。"""
        memory.touch(now)
        return self.advance(memory, now=now)
