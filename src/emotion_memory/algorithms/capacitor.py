"""电容记忆模型（charge / capacitance / 放电阈值 + standby 备选池）。

RC 电路类比：
  - 交互触达 = 充电：charge += Δq（上限 capacitance，Δq 随 base_strength 加权）；
  - 被想起   = 放电：charge -= discharge_rate（不低于 0，低于放电阈值则不再优先）；
  - 时间流逝 = 漏电：charge *= exp(-Δt/τ)（与遗忘曲线衰减协同，双通道遗忘）；
  - 常驻备选记忆点池（standby）：每次预留 top-k 备选记忆点——这些记忆
    "已充满电但尚未被唤起"，是主动回忆 / 旋转磁场的高潜候选。

与遗忘曲线协同：总可及电位 = accessibility · (0.5 + 0.5 · charge/capacitance)。
衰减只降"被想起的概率"，不删内容——永久不忘，但电位会慢慢漏掉。
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

from ..models import Memory

DEFAULT_TAU_HOURS = 48.0          # RC 时间常数（漏电半衰约 33 小时）
INITIAL_CHARGE_FRACTION = 0.5     # 新记忆初始半充
CHARGE_PER_TOUCH = 0.25           # 每次触达充电量
DISCHARGE_PER_RECALL = 0.1        # 每次被想起放电量


class CapacitorModel:
    """电容管理器：充电 / 放电 / 漏电 / 电位 / standby 备选池。"""

    def __init__(
        self,
        tau_hours: float = DEFAULT_TAU_HOURS,
        charge_per_touch: float = CHARGE_PER_TOUCH,
        discharge_per_recall: float = DISCHARGE_PER_RECALL,
        standby_pool_size: int = 3,
    ):
        self.tau_hours = max(1.0, float(tau_hours))
        self.charge_per_touch = charge_per_touch
        self.discharge_per_recall = discharge_per_recall
        self.standby_pool_size = max(1, int(standby_pool_size))

    # ---------------------------------------------------------- 充电
    def init_memory(self, memory: Memory) -> None:
        """新记忆电容初始化：容量归一 + 初始半充 + 放电阈值默认。"""
        if memory.capacitance <= 0:
            memory.capacitance = 1.0
        if memory.charge <= 0:
            memory.charge = memory.capacitance * INITIAL_CHARGE_FRACTION
        if memory.discharge_threshold <= 0:
            memory.discharge_threshold = 0.2
        memory.charge = min(memory.charge, memory.capacitance)

    def touch(self, memory: Memory) -> float:
        """触达充电：Δq = charge_per_touch · max(0.1, base_strength)，上限 capacitance。"""
        delta = self.charge_per_touch * max(0.1, memory.base_strength)
        memory.charge = min(memory.capacitance, memory.charge + delta)
        return memory.charge

    # ---------------------------------------------------------- 放电
    def recall_discharge(self, memory: Memory) -> float:
        """被想起放电：charge -= discharge_per_recall（不低于 0）。"""
        memory.charge = max(0.0, memory.charge - self.discharge_per_recall)
        return memory.charge

    def is_charged(self, memory: Memory) -> bool:
        """是否处于可被想起的充电态（charge 不低于放电阈值）。"""
        return memory.charge >= memory.capacitance * max(0.0, memory.discharge_threshold)

    # ---------------------------------------------------------- 漏电
    def leak(self, memory: Memory, dt_hours: float) -> float:
        """时间漏电：charge *= exp(-dt/τ)，与遗忘曲线衰减协同。"""
        dt = max(0.0, float(dt_hours))
        memory.charge = max(0.0, memory.charge * math.exp(-dt / self.tau_hours))
        return memory.charge

    # ---------------------------------------------------------- 协同
    def potential(self, memory: Memory) -> float:
        """总可及电位 = 遗忘可及性 × 电容因子（遗忘曲线与电容模型协同）。"""
        ratio = memory.charge / memory.capacitance if memory.capacitance > 0 else 0.0
        return memory.accessibility * (0.5 + 0.5 * ratio)

    # ---------------------------------------------------------- standby 备选池
    def standby_pool(
        self,
        memories: List[Memory],
        top_k: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[Memory]:
        """常驻备选记忆点池：每次预留 top-k 备选记忆点。

        选择逻辑：高电位（充电足 + 可及性高）× 最近未被想起（久未触达加分）→ top-k。
        沉默记忆（非永不沉默）不进入备选池，避免默认提及泄露。
        """
        k = top_k or self.standby_pool_size
        now = now if now is not None else time.time()
        scored = []
        for m in memories:
            if m.silence and not m.never_silent:
                continue
            last = m.last_access_at or m.created_at
            recency_penalty = min(1.0, max(0.0, (now - last) / 86400.0))
            score = self.potential(m) * (1.0 + recency_penalty)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]
