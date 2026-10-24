"""核心算法包：遗忘曲线 / 震荡命中函数 / 情感增益与 dB 权衡。"""
from .forgetting import ForgettingCurve, compute_accessibility
from .hit_function import OscillationBank
from .affect import (
    REFERENCE,
    DB_FLOOR,
    to_db,
    affect_gain,
    rank_hits,
    emotional_centroid,
)

__all__ = [
    "ForgettingCurve",
    "compute_accessibility",
    "OscillationBank",
    "REFERENCE",
    "DB_FLOOR",
    "to_db",
    "affect_gain",
    "rank_hits",
    "emotional_centroid",
]
