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

    # ---- v0.5 物理四定律 + 人格流形刻度盘（默认关闭 / 温和，保持既有行为） ----
    conservation_enabled: bool = False      # 情感守恒律开关（无源注入拒绝 + 能量预算）
    emotion_budget: float = 100.0          # 系统情感能量总预算（上限）
    potential_enabled: bool = False        # 势差驱动迁移开关（高势能沉淀长期区）
    potential_threshold: float = 0.6       # 高势能沉淀阈值
    importance_weight: float = 0.4         # 势能公式中重复次数加权系数
    repeat_weight: float = 0.3             # 势能公式中重要性加权系数
    entropy_enabled: bool = False          # 熵增遗忘开关（熵降权 + 细节先模糊）
    entropy_rate: float = 0.05             # 熵增速率（细节通道衰减快于情感通道）
    turbulent_enabled: bool = False        # 湍流联想开关（涡旋扩散）
    turbulence_decay: float = 0.6          # 涡旋能量随层距衰减系数（0..1）
    turbulence_max_depth: int = 3          # 涡流链最大扩散层数
    manifold_enabled: bool = False         # 人格流形开关（全谱系性格空间）
    manifold_bifurcation_r: float = 3.6    # 非线性分叉参数 r（确定性混沌区 2.8~4.0）

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
        # ---- v0.5 约束 ----
        self.emotion_budget = max(1.0, float(self.emotion_budget))
        self.potential_threshold = min(5.0, max(0.05, float(self.potential_threshold)))
        self.importance_weight = min(2.0, max(0.0, float(self.importance_weight)))
        self.repeat_weight = min(2.0, max(0.0, float(self.repeat_weight)))
        self.entropy_rate = min(1.0, max(0.001, float(self.entropy_rate)))
        self.turbulence_decay = min(1.0, max(0.05, float(self.turbulence_decay)))
        self.turbulence_max_depth = max(1, min(10, int(self.turbulence_max_depth)))
        self.manifold_bifurcation_r = min(4.0, max(2.8, float(self.manifold_bifurcation_r)))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dial":
        known = {k: v for k, v in cls().__dict__.items() if not k.startswith("_")}
        data = {k: d.get(k, v) for k, v in known.items()}
        return cls(**data).validate()
