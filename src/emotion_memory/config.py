"""自由度刻度盘（受控疯狂）。

所有"不可预测性"集中为可调参数，默认温和（低噪声高稳定）。
安全底线永不解除：记忆内容不可篡改、敏感数据不可外泄、告别/迁移流程确定性执行。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Dial:
    """联想不可预测 / 情绪波动 / 发散频率 / 反应灵敏度的统一刻度盘。

    参数说明（对应规格 §5 自由度刻度盘）：
      noise_temperature : 噪声温度 T，0（精准）→ 1.0（狂想）
      emotion_gain      : 情感增益幅度 g，温和 → 剧烈震荡
      jump_probability  : 跨簇跳转概率，偶尔灵光 → 高频跑题
      amplitude         : 命中函数振幅 A，迟钝 → 过激（触发杂相模式）
      divergent_enabled : 发散模式开关（默认关闭）
      concurrent_enabled: 并发联想开关（默认关闭）
    """

    noise_temperature: float = 0.1
    emotion_gain: float = 0.3
    jump_probability: float = 0.05
    amplitude: float = 1.0

    divergent_enabled: bool = False
    divergent_threshold: float = 1.5   # g·A 超过该值进入发散模式
    concurrent_enabled: bool = False
    concurrent_workers: int = 4

    top_k: int = 5
    token_budget: int = 2048
    strong_relation_threshold: float = 2.0  # 图共现权重达到该值视为强关联

    # ---- v0.3 新功能刻度盘（默认关闭 / 温和，保持既有行为） ----
    three_phase_enabled: bool = False       # 旋转磁场（三相顺序启动）开关
    three_phase_speed: float = 1.0          # 主导通道轮转速度乘数
    proactive_enabled: bool = False         # 主动回忆引擎开关
    proactive_interval_hours: float = 6.0   # 主动开口最小间隔（小时）
    capacitor_enabled: bool = False         # 电容记忆模型开关
    capacitor_tau_hours: float = 48.0       # RC 漏电时间常数（小时）
    capacitor_charge_per_touch: float = 0.25  # 每次触达充电量
    capacitor_discharge_per_recall: float = 0.1  # 每次被想起放电量
    standby_pool_size: int = 3              # standby 备选池容量 top-k

    def validate(self) -> "Dial":
        """参数合法性收拢：任何值都不得破坏安全底线。"""
        self.noise_temperature = min(1.0, max(0.0, self.noise_temperature))
        self.emotion_gain = min(2.0, max(0.0, self.emotion_gain))
        self.jump_probability = min(1.0, max(0.0, self.jump_probability))
        self.amplitude = min(5.0, max(0.1, self.amplitude))
        self.top_k = max(1, int(self.top_k))
        self.three_phase_speed = min(10.0, max(0.1, self.three_phase_speed))
        self.proactive_interval_hours = min(720.0, max(0.5, self.proactive_interval_hours))
        self.capacitor_tau_hours = min(720.0, max(1.0, self.capacitor_tau_hours))
        self.capacitor_charge_per_touch = min(2.0, max(0.0, self.capacitor_charge_per_touch))
        self.capacitor_discharge_per_recall = min(2.0, max(0.0, self.capacitor_discharge_per_recall))
        self.standby_pool_size = max(1, int(self.standby_pool_size))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dial":
        known = {k: v for k, v in cls().__dict__.items() if not k.startswith("_")}
        data = {k: d.get(k, v) for k, v in known.items()}
        return cls(**data).validate()
