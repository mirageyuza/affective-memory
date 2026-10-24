"""旋转磁场唤起调度（三相顺序启动）。

规格延伸 §3.1：H(t) 原接口保持不变，本模块在其之上新增三相旋转磁场视图：

- 记忆簇按稳定哈希分配到 A/B/C 三个通道，通道相位互差 120°（2π/3）；
- 主导通道随时间顺序轮转（0 → 1 → 2 → 0 ...），像三相异步电动机定子
  绕组顺序通电一样，形成"连续联想流"：当前主导通道内的记忆簇优先唤起，
  其余通道异步跟随；
- 每簇激活强度 = g · A · [leader_boost · sin(ωt + φ + θ_ch) + follower·弱叠加]；
- 输出联想流（associative stream）供主动回忆 / 上下文注入使用。

不破坏现有 H(t) 接口：本模块只消费 OscillationBank 的簇元数据做组合，
OscillationBank.hit() / score_memory() 原语义保持不变。
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from ..config import Dial
from ..models import Memory
from .hit_function import OscillationBank

PHASE_OFFSETS = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]  # 0° / 120° / 240°
CHANNEL_NAMES = ["A", "B", "C"]
LEADER_PERIOD_HOURS = 6.0   # 主导通道轮转周期（模拟小时）
LEADER_BOOST = 1.0          # 主导通道增益系数
FOLLOWER_FACTOR = 0.4       # 从属通道衰减因子（异步跟随）


class ThreePhaseField:
    """三相旋转磁场调度器：通道分配 / 相位激活 / 联想流排序。"""

    def __init__(self, bank: OscillationBank, seed: int = 11):
        self.bank = bank
        self._rng = random.Random(seed)
        self._channel_of: Dict[str, int] = {}

    # ---------------------------------------------------------- 通道分配
    def channel_of(self, cluster: str) -> int:
        """簇 → 稳定通道（0/1/2，确定性哈希分配，跨会话不变）。"""
        if cluster not in self._channel_of:
            self._channel_of[cluster] = self._rng.randrange(3)
        return self._channel_of[cluster]

    def channel_name(self, cluster: str) -> str:
        return CHANNEL_NAMES[self.channel_of(cluster)]

    def leader_channel(self, t: float) -> int:
        """主导通道：随时间顺序轮转 0→1→2→0...（三相顺序启动）。"""
        return int(math.floor(max(0.0, t) / LEADER_PERIOD_HOURS)) % 3

    def field_state(self, t: float) -> Dict[str, Any]:
        """当前场状态（可观测 / 审计）：主导通道 + 三相顺序 + 相位差。"""
        leader = self.leader_channel(t)
        return {
            "t": t,
            "leader_channel": CHANNEL_NAMES[leader],
            "sequence": CHANNEL_NAMES[leader:] + CHANNEL_NAMES[:leader],
            "phase_offsets_deg": [0, 120, 240],
            "channel_count": 3,
        }

    # ---------------------------------------------------------- 相位激活
    def phase_activation(
        self,
        cluster: str,
        t: float,
        gain: float,
        dial: Optional[Dial] = None,
    ) -> float:
        """单簇三相场激活强度（旋转磁场版命中值，替代 H(t) 参与排序时使用）。"""
        dial = dial or self.bank.dial
        meta = self.bank.register(cluster)
        ch = self.channel_of(cluster)
        leader = self.leader_channel(t)
        a_eff = meta.amplitude * dial.amplitude
        carrier = math.sin(meta.frequency * t + meta.phase + PHASE_OFFSETS[ch])
        weight = LEADER_BOOST if ch == leader else FOLLOWER_FACTOR
        base = gain * a_eff * weight * carrier
        noise = (2.0 * self._rng.random() - 1.0) * dial.noise_temperature
        return max(0.0, base + noise)

    # ---------------------------------------------------------- 联想流
    def stream(
        self,
        memories: List[Memory],
        t: float,
        dial: Optional[Dial] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """联想流：按三相场激活强度排序，附带通道标签，形成连续联想序列。

        返回 [{memory, activation, channel, leader}]，top-k 截断。
        """
        dial = dial or self.bank.dial
        leader = self.leader_channel(t)
        rows: List[Dict[str, Any]] = []
        for m in memories:
            gain = self.bank.gain_of(m.affect, dial)
            act = self.phase_activation(m.cluster, t, gain, dial)
            rows.append({
                "memory": m,
                "activation": round(act, 6),
                "channel": CHANNEL_NAMES[self.channel_of(m.cluster)],
                "leader": self.channel_of(m.cluster) == leader,
            })
        rows.sort(key=lambda r: r["activation"], reverse=True)
        if k is not None:
            rows = rows[: k]
        return rows
