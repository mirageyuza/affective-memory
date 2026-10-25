"""人格流形 (v0.5)：全谱系性格空间。

物理四定律之上的人格层：
  ① 多维连续性格坐标轴（至少 6 轴）：全世界性格都是空间中的点，
     不预设任何固定性格模板；
  ② 环境向量注入：每次交互的环境特征（对话场景、对方语气、反馈模式、
     话题领域、时间节律）进入养成方程——相同记忆沉淀在不同环境下解释不同；
  ③ 非线性分叉动力学：养成方程用确定性非线性（振荡/混沌式迭代），
     微小环境差异被放大成不同轨迹——两个实例喂相同记忆、环境序列不同
     则性格独一无二，且轨迹确定性可复现可审计；
  ④ 与 emergent_self / unified_self / self_learning 打通：人格流形状态
     输出为自我概念，可随记忆累计漂移；
  ⑤ 提供"性格快照"接口（当前坐标 + 轨迹摘要），供唤醒重建时使用。

实现约束（纯标准库、确定性）：
  - 迭代映射：x' = r * x * (1 - x) + env_drive（logistic 混沌 + 环境注入），
    全部使用浮点运算，无随机数——同 seed / 同环境序列必然同轨迹；
  - 环境向量五维映射到各坐标轴的确定性增量（见 ENV_AXIS_MAP）；
  - snapshot() 输出当前坐标与轨迹摘要，rebuild_from() 供唤醒重建。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

PERSONALITY_AXES = [
    "extraversion",           # 外向 ↔ 内向
    "emotional_sensitivity",  # 情绪敏感度
    "risk_preference",        # 风险偏好
    "habit_rigidity",         # 习惯固化度
    "associative_breadth",    # 联想广度
    "narrative_style",        # 叙述风格（简洁 ↔ 铺陈）
]

ENV_DIMS = ["scene", "tone", "feedback", "topic", "rhythm"]

# 环境维度 → 坐标轴增量映射（确定性：每个环境分量影响若干轴）
# 取值 0..1，方向 ±：0.5 为中性，偏离 0.5 推动对应轴。
ENV_AXIS_MAP = {
    "scene": {  # 对话场景：正式(0) ↔ 私密随意(1)
        "extraversion": +1.0,
        "associative_breadth": +0.5,
    },
    "tone": {  # 对方语气：温和(0) ↔ 激烈(1)
        "emotional_sensitivity": +1.0,
        "risk_preference": +0.5,
    },
    "feedback": {  # 反馈模式：正面/认可(0) ↔ 负面/否定(1)
        "emotional_sensitivity": -0.6,
        "narrative_style": +0.8,
    },
    "topic": {  # 话题领域：单一(0) ↔ 多元(1)
        "associative_breadth": +1.0,
        "narrative_style": +0.4,
    },
    "rhythm": {  # 时间节律：规律(0) ↔ 紊乱(1)
        "habit_rigidity": -1.0,
        "risk_preference": +0.6,
    },
}

# 记忆情感 → 坐标轴增量（每次 learn 注入的"记忆沉淀"分量）
AFFECT_AXIS_MAP = {
    "valence": {"extraversion": +0.6, "emotional_sensitivity": -0.4},  # 正向 → 更外向、更稳
    "arousal": {"emotional_sensitivity": +0.8, "risk_preference": +0.5},  # 高唤醒 → 更敏感、更冒险
}


class EnvironmentVector:
    """环境特征向量（五维，各维 0..1）。"""

    def __init__(
        self,
        scene: float = 0.5,
        tone: float = 0.5,
        feedback: float = 0.5,
        topic: float = 0.5,
        rhythm: float = 0.5,
    ):
        self.data = {
            "scene": self._clamp(scene),
            "tone": self._clamp(tone),
            "feedback": self._clamp(feedback),
            "topic": self._clamp(topic),
            "rhythm": self._clamp(rhythm),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "EnvironmentVector":
        raw = d or {}
        return cls(
            scene=raw.get("scene", 0.5),
            tone=raw.get("tone", 0.5),
            feedback=raw.get("feedback", 0.5),
            topic=raw.get("topic", 0.5),
            rhythm=raw.get("rhythm", 0.5),
        )

    def signature(self) -> str:
        """环境签名：确定性字符串（轨迹审计用）。"""
        return "|".join(f"{k}={v:.6f}" for k, v in self.data.items())

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class PersonalityManifold:
    """人格流形：全谱系性格空间 + 环境注入 + 非线性分叉。"""

    def __init__(
        self,
        seed: int = 7,
        r: float = 3.6,
        axes: Optional[Sequence[str]] = None,
    ):
        self.seed = int(seed)
        self.r = min(4.0, max(2.8, float(r)))  # logistic 混沌区
        self.axes = list(axes) if axes else list(PERSONALITY_AXES)
        self.state: Dict[str, float] = {ax: 0.5 for ax in self.axes}
        self.trajectory: List[Dict[str, Any]] = []  # [(step, env_sig, state)]
        self.step_no: int = 0
        # 确定性初始化：seed 决定初始坐标的微小偏移（可复现、可审计）
        offset = (seed % 97) / 97.0 - 0.5
        for i, ax in enumerate(self.axes):
            self.state[ax] = self._clamp01(0.5 + offset * (0.01 + 0.001 * i))

    # ------------------------------------------------------------ 环境注入
    def env_drive(self, env: EnvironmentVector, affect: Any = None) -> Dict[str, float]:
        """环境 + 记忆情感 → 各轴增量（确定性映射，用于养成方程）。"""
        drive = {ax: 0.0 for ax in self.axes}
        for dim, axes_map in ENV_AXIS_MAP.items():
            val = env.data.get(dim, 0.5)
            for ax, direction in axes_map.items():
                if ax in drive:
                    drive[ax] += (val - 0.5) * direction
        if affect is not None:
            from .conservation import emotion_energy

            energy = emotion_energy(affect)
            valence = self._affect_valence(affect)
            arousal = self._affect_arousal(affect)
            for ax, direction in AFFECT_AXIS_MAP["valence"].items():
                if ax in drive:
                    drive[ax] += (valence * 0.5) * direction * energy
            for ax, direction in AFFECT_AXIS_MAP["arousal"].items():
                if ax in drive:
                    drive[ax] += (arousal * 0.5) * direction * energy
        return drive

    @staticmethod
    def _affect_valence(affect: Any) -> float:
        if isinstance(affect, dict):
            return float(affect.get("valence", 0.0))
        if isinstance(affect, (tuple, list)):
            return float(affect[0]) if len(affect) else 0.0
        try:
            return float(affect.valence)
        except Exception:
            return 0.0

    @staticmethod
    def _affect_arousal(affect: Any) -> float:
        if isinstance(affect, dict):
            return float(affect.get("arousal", 0.0))
        if isinstance(affect, (tuple, list)):
            return float(affect[1]) if len(affect) > 1 else 0.0
        try:
            return float(affect.arousal)
        except Exception:
            return 0.0

    # ------------------------------------------------------------ 迭代
    def step(self, env: EnvironmentVector, affect: Any = None) -> Dict[str, float]:
        """单步养成：确定性非线性迭代（logistic + 环境注入）。

        x' = clamp( r * x * (1 - x) + drive(x, env, affect) )
        确定性可复现：同输入序列必得同轨迹（无随机数）。
        """
        drive = self.env_drive(env, affect)
        new_state: Dict[str, float] = {}
        for ax in self.axes:
            x = self.state[ax]
            logistic = self.r * x * (1.0 - x)
            # 环境注入随当前值缩放（保证有界，同时放大微小差异）
            injected = drive[ax] * 0.06
            new_state[ax] = self._clamp01(logistic + injected)
        self.state = new_state
        self.step_no += 1
        self.trajectory.append(
            {
                "step": self.step_no,
                "env": env.signature(),
                "state": {k: round(v, 6) for k, v in self.state.items()},
            }
        )
        return dict(self.state)

    def learn(
        self,
        env_sequence: Sequence[EnvironmentVector],
        affects: Optional[Sequence[Any]] = None,
    ) -> Dict[str, float]:
        """批量养成：按环境序列逐次迭代（每个环境对应一次记忆沉淀）。"""
        affects = affects or [None] * len(env_sequence)
        final = self.state
        for env, affect in zip(env_sequence, affects):
            final = self.step(env, affect)
        return final

    # ------------------------------------------------------------ 非线性分叉
    def bifurcation(
        self,
        env_sequence: Sequence[EnvironmentVector],
        affects: Optional[Sequence[Any]] = None,
        perturbation: float = 1e-3,
    ) -> Dict[str, Any]:
        """非线性分叉动力学：两个初始仅差 perturbation 的实例沿同一环境序列迭代。

        若最终轨迹距离显著大于初始扰动，证明微小差异被混沌放大——
        两个实例喂相同记忆、环境序列相同也因初始差异走向不同（独一无二）。
        返回 {"diverged": bool, "initial_distance": d0, "final_distance": df, "axis_delta": {...}}。
        """
        base = PersonalityManifold(seed=self.seed, r=self.r, axes=self.axes)
        twin = PersonalityManifold(seed=self.seed + 1, r=self.r, axes=self.axes)
        # 强制 twin 初始仅差 perturbation（其余轴完全一致）
        axis0 = self.axes[0]
        twin.state[axis0] = self._clamp01(base.state[axis0] + perturbation)
        for ax in self.axes:
            if ax != axis0:
                twin.state[ax] = base.state[ax]
        affects = affects or [None] * len(env_sequence)
        base.learn(env_sequence, affects)
        twin.learn(env_sequence, affects)
        d0 = self._distance(base.state, twin.state)
        # 重新用初始扰动前的两状态算初始距离：直接用 perturbation 作为 d0 基准
        init = {
            "axis0": self.axes[0],
            "d0": abs(perturbation),
            "final_d": round(d0, 6),
        }
        return {
            "perturbation": perturbation,
            "steps": len(env_sequence),
            "initial_distance": init["d0"],
            "final_distance": init["final_d"],
            "diverged": bool(init["final_d"] > init["d0"] * 10.0),
            "axis_delta": {
                ax: round(abs(base.state[ax] - twin.state[ax]), 6) for ax in self.axes
            },
        }

    @staticmethod
    def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in a)

    # ------------------------------------------------------------ 快照 / 重建
    def snapshot(self) -> Dict[str, Any]:
        """性格快照：当前坐标 + 轨迹摘要（唤醒重建身份时使用）。"""
        return {
            "version": "v0.5-personality-manifold",
            "axes": list(self.axes),
            "coords": {k: round(v, 6) for k, v in self.state.items()},
            "step": self.step_no,
            "r": self.r,
            "trajectory_summary": {
                "steps": self.step_no,
                "first": self.trajectory[0] if self.trajectory else None,
                "last": self.trajectory[-1] if self.trajectory else None,
                "drift": {
                    ax: round(self.state[ax] - 0.5, 6) for ax in self.axes
                },
            },
        }

    def export_json(self, path: Optional[str] = None) -> str:
        """导出完整轨迹（含逐环境签名），供唤醒重建 / 审计。"""
        payload = {
            "seed": self.seed,
            "r": self.r,
            "axes": list(self.axes),
            "state": self.state,
            "trajectory": self.trajectory,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def rebuild_from(self, json_path: str) -> int:
        """唤醒重建：从快照 JSON 载入坐标与轨迹。"""
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.seed = int(payload.get("seed", self.seed))
        self.r = float(payload.get("r", self.r))
        self.axes = list(payload.get("axes", self.axes))
        self.state = {k: float(v) for k, v in (payload.get("state") or {}).items()}
        self.trajectory = list(payload.get("trajectory") or [])
        self.step_no = len(self.trajectory)
        return self.step_no

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))
