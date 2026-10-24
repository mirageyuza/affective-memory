"""情感增益与 dB 权衡排序（韦伯-费希纳对数律）。

- 记忆附加情感标签 (valence, arousal)，情感增益 g 只在检索排序层生效；
- 强度统一换算 dB 刻度参与排序：

      强度_dB = 20 · log10(强度 / 基准)

  解决神经反应强度跨数量级时的数值精度问题；
- 核心红线：排序永不改写记忆原文，任何输出都不触碰 memory.text。
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from ..config import Dial
from ..models import AffectLabel, Memory, RankedHit

REFERENCE = 1e-6          # dB 基准强度
DB_FLOOR = -120.0         # 低于基准视为静默


def to_db(strength: float, reference: float = REFERENCE) -> float:
    if strength <= 0.0 or strength < reference:
        return DB_FLOOR
    return 20.0 * math.log10(strength / reference)


def affect_gain(affect: AffectLabel, dial: Dial) -> float:
    magnitude = 0.5 * abs(affect.valence) + 0.5 * affect.arousal
    return 0.5 + dial.emotion_gain * min(1.0, magnitude)


def rank_hits(
    hits: Iterable[Tuple[Memory, float]],
    dial: Dial,
) -> List[RankedHit]:
    """对 (memory, raw_score) 集合做情感增益 + dB 权衡排序。

    返回 RankedHit 列表（score 降序），只读排序，绝不修改 Memory 原文。
    """
    ranked: List[RankedHit] = []
    for memory, raw in hits:
        gain = affect_gain(memory.affect, dial)
        score = max(0.0, float(raw)) * gain * memory.accessibility
        db = to_db(score)
        ranked.append(
            RankedHit(
                memory=memory,
                score=score,
                db=db,
                cosine=float(raw),
                gain=gain,
                accessibility=memory.accessibility,
            )
        )
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def emotional_centroid(memories: Iterable[Memory]) -> Tuple[float, float]:
    """记忆集合的情感质心 (valence, arousal)，用于 benchmark 情感偏差。"""
    items = list(memories)
    if not items:
        return (0.0, 0.0)
    n = len(items)
    return (
        sum(m.affect.valence for m in items) / n,
        sum(m.affect.arousal for m in items) / n,
    )
