"""情感守恒律 (v0.5)：情感不凭空产生。

物理四定律第一定律：
  ① 每次情感标注必须携带事件来源（来源事件/交互的原文），无源情感注入一律拒绝；
  ② 情感污染版的情感值必须从原文情感迁移而来（污染 = 情感重分配，禁止无中生有）；
  ③ 系统情感总量预算：设置总情感能量上限，来源不足时情感强度自然衰减，
     防止"情绪通货膨胀"。

实现约束（纯标准库、确定性、可审计）：
  - emotion_energy() 把 (valence, arousal) 折算为 0..1 的情感能量；
  - check_annotation() 在开启守恒时拒绝无源情感；
  - cap_energy() 超预算时按比例衰减新注入强度（自然衰减而非删数据）；
  - validate_pollution() 限制污染强度不得超过原文情感能量（重分配上限）。
原文永不修改：本模块只读取情感标签与元数据，不触碰 original_text。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .models import Memory

# 污染重分配允许的弹性系数：污染版情感能量允许略高于原文（重分配放大上限）
POLLUTION_REALLOC_SLACK = 1.2


class ConservationError(ValueError):
    """情感守恒违规：无源注入 / 超出预算 / 污染无中生有。"""


def emotion_energy(affect: Any) -> float:
    """情感标签折算为 0..1 情感能量（效价强度 60% + 唤醒度 40%）。"""
    if affect is None:
        return 0.0
    if isinstance(affect, dict):
        valence = float(affect.get("valence", 0.0))
        arousal = float(affect.get("arousal", 0.0))
    elif isinstance(affect, (tuple, list)):
        valence = float(affect[0]) if len(affect) > 0 else 0.0
        arousal = float(affect[1]) if len(affect) > 1 else 0.0
    else:
        try:
            valence = float(affect.valence)
            arousal = float(affect.arousal)
        except Exception:
            return 0.0
    energy = min(1.0, abs(valence) * 0.6 + min(1.0, max(0.0, arousal)) * 0.4)
    return max(0.0, energy)


class EmotionConservation:
    """情感守恒审计器：来源校验 + 预算控制 + 污染重分配限制。"""

    def __init__(self, budget: float = 100.0):
        self.budget = max(1.0, float(budget))
        self.audit: list = []  # 守恒审计流水（确定性可追溯）

    # ------------------------------------------------------------ 来源校验
    @staticmethod
    def validate_source(source_text: Optional[str]) -> bool:
        """来源有效：必须是非空字符串（事件/交互原文或可追溯引用）。"""
        return bool(source_text is not None and str(source_text).strip())

    def check_annotation(
        self,
        text: str,
        affect: Any,
        source_text: Optional[str],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """校验一次情感标注：无源情感注入一律拒绝（strict=True）。

        返回审计记录；strict=False 时无源仅记 audit 并打 reason 标记，
        供关闭守恒开关的兼容模式使用（不抛错）。
        """
        energy = emotion_energy(affect)
        record = {
            "text": text,
            "energy": round(energy, 4),
            "source": source_text,
            "accepted": True,
            "reason": "ok",
        }
        if not self.validate_source(source_text):
            if strict:
                self.audit.append({**record, "accepted": False, "reason": "no_source"})
                raise ConservationError(
                    "无源情感注入一律拒绝：情感标注必须携带事件来源原文（source_text）"
                )
            record["accepted"] = False
            record["reason"] = "no_source_but_lenient"
        self.audit.append(record)
        return record

    # ------------------------------------------------------------ 总量预算
    def total_energy(self, memories) -> float:
        """当前系统情感能量总量（sum over 原文情感能量）。"""
        return sum(emotion_energy(m.affect) for m in memories)

    def cap_energy(self, energy: float, total: float) -> Dict[str, Any]:
        """来源不足时情感强度自然衰减：总能量超预算则按比例衰减新注入强度。

        返回 {"energy": 衰减后能量, "scale": 衰减系数, "over": bool}。
        衰减是连续比例（防止情绪通货膨胀），不删除任何记忆。
        """
        energy = max(0.0, float(energy))
        if total + energy <= self.budget:
            scale = 1.0
            capped = energy
        else:
            room = max(0.0, self.budget - total)
            scale = min(1.0, room / max(1e-9, energy))
            capped = energy * scale
        return {
            "energy": round(capped, 6),
            "scale": round(scale, 6),
            "over": scale < 1.0,
            "budget": self.budget,
            "total_before": round(total, 6),
        }

    def adjust_memory_intensity(
        self,
        memory: Memory,
        all_memories,
        affect: Any = None,
    ) -> Dict[str, Any]:
        """引擎集成入口：注入记忆前按预算衰减其情感强度（自然衰减）。

        衰减以情感能量折算系数 scale 作用到 affect 的 arousal（唤醒度）上，
        保持效价方向不变；返回审计记录。
        """
        energy = emotion_energy(affect if affect is not None else memory.affect)
        total = self.total_energy(all_memories)
        res = self.cap_energy(energy, total)
        self.audit.append(
            {
                "memory_id": memory.memory_id,
                "energy": round(energy, 4),
                "scale": res["scale"],
                "over": res["over"],
                "reason": "budget_attenuated" if res["over"] else "ok",
            }
        )
        return res

    # ------------------------------------------------------------ 污染重分配
    def validate_pollution(
        self,
        memory: Memory,
        intensity: float,
        slack: float = POLLUTION_REALLOC_SLACK,
    ) -> Dict[str, Any]:
        """污染 = 情感重分配：污染强度不得超过原文情感能量 × 弹性系数。

        返回 {"max_intensity": 允许上限, "allowed": bool, "clamped": float}。
        超过上限的污染强度被钳制到上限（禁止无中生有），并记录审计。
        """
        original_energy = emotion_energy(memory.affect)
        max_intensity = max(0.0, original_energy * slack)
        raw = max(0.0, float(intensity))
        allowed = raw <= max_intensity
        clamped = min(raw, max_intensity)
        self.audit.append(
            {
                "memory_id": memory.memory_id,
                "original_energy": round(original_energy, 4),
                "requested_intensity": round(raw, 4),
                "max_intensity": round(max_intensity, 4),
                "allowed": allowed,
                "clamped": round(clamped, 4),
            }
        )
        return {
            "max_intensity": round(max_intensity, 4),
            "allowed": bool(allowed),
            "clamped": round(clamped, 4),
            "original_energy": round(original_energy, 4),
        }

    # ------------------------------------------------------------ 审计导出
    def audit_report(self) -> Dict[str, Any]:
        """导出守恒审计：总注入 / 拒绝 / 衰减 / 钳制统计。"""
        total = len(self.audit)
        rejected = sum(1 for r in self.audit if not r.get("accepted", True) or r.get("reason") == "no_source")
        attenuated = sum(1 for r in self.audit if r.get("reason") == "budget_attenuated")
        clamped = sum(1 for r in self.audit if "allowed" in r and not r["allowed"])
        return {
            "audit_count": total,
            "rejected_no_source": rejected,
            "budget_attenuated": attenuated,
            "pollution_clamped": clamped,
            "budget": self.budget,
        }
