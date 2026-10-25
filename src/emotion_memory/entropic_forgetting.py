"""熵增遗忘 (v0.5)：遗忘 = 熵增不是删除。

物理四定律第三定律：
  ① 在现有遗忘曲线基础上增加熵模型：记忆有序 → 无序，
     细节先模糊（细节通道衰减快）、情绪残留后淡（情感通道衰减慢）；
  ② 熵越高检索置信度越低（recall 时按 1 - entropy 降权）；
  ③ 原文永不清除（不破原文存档红线），高熵记忆仅降权；
  ④ 与情感污染双层配套：污染版（情绪残余）比原文细节活得久——
     detail_fidelity 衰减快于 emotional_residue，而污染视图承载情绪残余。

实现约束（纯标准库、确定性）：
  - advance() 按 dt 推进 detail_fidelity / emotional_residue / entropy，
    只更新元数据，绝不触碰 original_text；
  - confidence() = 1 - entropy，供检索层降权；
  - 高危/时间敏感（due_at 非空、reminded 标记）不因此沉默：
    本模块只降权不删除，沉默层仍保证永不沉默。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .models import Memory


class EntropicForgetting:
    """熵增遗忘引擎：细节与情绪双通道衰变。"""

    # 细节通道相对情感通道的衰减倍数（细节先模糊，情绪残留后淡）
    DETAIL_DECAY_MULTIPLIER = 2.0

    def __init__(self, rate: float = 0.05):
        self.rate = min(1.0, max(0.001, float(rate)))
        self.log: List[Dict[str, Any]] = []  # 熵增审计

    # ------------------------------------------------------------ 熵推进
    def advance(
        self,
        memory: Memory,
        dt_hours: float,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """推进熵增：细节先模糊、情绪残留后淡。

        指数衰变：detail = detail0 * exp(-2*rate*sqrt(dt))
                 residue = residue0 * exp(-rate*sqrt(dt))
        熵 = 1 - (0.5*detail + 0.5*residue) ∈ [0,1]。
        原文（original_text）永不修改，只更新元数据字段。
        """
        dt = max(0.0, float(dt_hours))
        decay_detail = math.exp(-self.DETAIL_DECAY_MULTIPLIER * self.rate * math.sqrt(dt))
        decay_residue = math.exp(-self.rate * math.sqrt(dt))
        memory.detail_fidelity = self._clamp01(memory.detail_fidelity * decay_detail)
        memory.emotional_residue = self._clamp01(memory.emotional_residue * decay_residue)
        memory.entropy = self._clamp01(
            1.0 - 0.5 * memory.detail_fidelity - 0.5 * memory.emotional_residue
        )
        rec = {
            "memory_id": memory.memory_id,
            "dt_hours": round(dt, 4),
            "detail_fidelity": round(memory.detail_fidelity, 4),
            "emotional_residue": round(memory.emotional_residue, 4),
            "entropy": round(memory.entropy, 4),
            "protected": bool(memory.due_at is not None),
        }
        self.log.append(rec)
        return rec

    # ------------------------------------------------------------ 置信度
    @staticmethod
    def confidence(memory: Memory) -> float:
        """检索置信度 = 1 - 熵（熵越高置信度越低，仅降权不删除）。"""
        return max(0.0, min(1.0, 1.0 - memory.entropy))

    def decay_factor(self, memory: Memory) -> float:
        """检索降权因子：默认 1.0（未开熵模型不降权）；开熵模型用置信度。"""
        return self.confidence(memory)

    # ------------------------------------------------------------ 批量
    def decay_all(self, memories, dt_hours: float) -> Dict[str, Any]:
        """全库熵增推进，返回统计。"""
        results = [self.advance(m, dt_hours) for m in memories]
        return {
            "advanced": len(results),
            "average_entropy": round(
                (sum(r["entropy"] for r in results) / len(results)) if results else 0.0,
                4,
            ),
            "details": results,
        }

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    # ------------------------------------------------------------ 报告
    def report(self) -> Dict[str, Any]:
        """熵增审计报告。"""
        return {"log_count": len(self.log), "log": list(self.log), "rate": self.rate}
