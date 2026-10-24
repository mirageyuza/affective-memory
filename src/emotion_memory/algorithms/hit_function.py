"""震荡命中函数（回忆追溯函数）。

    H(t) = g(情感) · Σᵢ Aᵢ·sin(ωᵢt + φᵢ) + 噪声(T)

- 每个记忆簇分配独立频率 ω 与相位 φ（三相交流电式多通道激活）；
- g 为情感增益，由情感标签（效价/唤醒度）调制振幅；
- 噪声(T) 由温度参数控制（随机共振，增强弱信号可及性）；
- 结果钳制在 [0, ∞)：负向震荡视为不唤起。
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from ..config import Dial
from ..models import AffectLabel, ClusterMeta, Memory


class OscillationBank:
    """为记忆簇分配频率/相位/振幅，并计算 H(t)。确定性可复现（seed）。"""

    def __init__(self, dial: Optional[Dial] = None, seed: int = 7):
        self.dial = dial or Dial()
        self._rng = random.Random(seed)
        self._bank: Dict[str, ClusterMeta] = {}

    def register(self, cluster: str, amplitude: float = 1.0) -> ClusterMeta:
        if cluster not in self._bank:
            omega = 0.2 + self._rng.random() * 0.8          # rad / (模拟小时)
            phi = self._rng.random() * 2.0 * math.pi
            self._bank[cluster] = ClusterMeta(
                cluster=cluster, frequency=omega, phase=phi, amplitude=amplitude
            )
        return self._bank[cluster]

    def hit(
        self,
        cluster: str,
        t: float,
        gain: float,
        dial: Optional[Dial] = None,
    ) -> float:
        """单簇命中值：g · A · sin(ωt + φ) + 噪声(T)。"""
        dial = dial or self.dial
        meta = self.register(cluster)
        a_eff = meta.amplitude * dial.amplitude
        base = gain * a_eff * math.sin(meta.frequency * t + meta.phase)
        noise = (2.0 * self._rng.random() - 1.0) * dial.noise_temperature
        return max(0.0, base + noise)

    def gain_of(self, affect: AffectLabel, dial: Optional[Dial] = None) -> float:
        """情感增益 g：效价绝对强度与唤醒度共同调制，0.5 ~ 1.5 区间。"""
        dial = dial or self.dial
        magnitude = 0.5 * abs(affect.valence) + 0.5 * affect.arousal
        return 0.5 + dial.emotion_gain * min(1.0, magnitude)

    def score_memory(
        self,
        memory: Memory,
        t: float,
        dial: Optional[Dial] = None,
    ) -> float:
        """单条记忆的命中分 = g · H(t)（再叠加可及性由检索层处理）。"""
        dial = dial or self.dial
        g = self.gain_of(memory.affect, dial)
        return g * self.hit(memory.cluster, t, 1.0, dial)
