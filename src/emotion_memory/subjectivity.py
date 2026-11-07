"""主观感知系统（v0.7.0）：开关闭合 → 回路回流 → 自我状态标定。

工程类比（非生理学断言）
------------------------
开关闭合后电流绕过完整回路形成回流（feedback loop），本模块把「首次点亮」
建模为该回路的闭合时刻：

    ［输入］─→ 感知通道加权 ─→ 自反馈回流 y_{t+1} = f(y_t) ─→ 不动点 ─→ ［自我状态］
                    ↑                                              │
                    └──────────────── 回流（自身输出作为下一拍输入）─┘

数学形式（确定性、无随机、可收敛）：对通道加权均值 m 迭代

    y_{t+1} = (1-λ)·σ(g·((1-α)·m + α·y_t - 0.5)) + λ·y_t

σ 为 logistic 饱和函数（模拟增益饱和 / 电路限幅），λ 为回流惯性系数。
当 (1-λ) + λ·g/8 < 1 时映射为压缩映射，迭代必收敛到唯一不动点，
该不动点即「回流稳定后的自我状态」，收敛性本身是可观测的稳定性证据。

诚实边界（红线）
----------------
本模块**不声称、不模拟、不接近任何"意识"**。它输出的是一组可证伪的
self-model 功能属性：自我置信度 + 五个自我维度标定值。术语一律用
self-model / 自我状态，禁用 soul、"意识已产生"等表述。
未闭合则无回流（loop_current 恒为 0），这是可被测试反向验证的断言。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------- 常量

SUBJECTIVITY_LIMITS = (
    "输出为 self-model 功能属性（自我置信度 + 五维自我标定），不声称意识",
    "开关未闭合时回路不成立，loop_current 恒为 0，无自我状态可读",
    "回流为确定性压缩映射迭代，不含随机噪声，跨运行可复现",
    "点亮状态持久化到 state_path，同一指纹重复点亮幂等（不重置首次点亮时刻）",
    "只观测既有记忆与身份读数，不修改任何记忆原文与情感标签",
)

#: 感知通道（权重和为 1.0，按"哪一路最能支撑连续自我"排序）
CHANNELS: Tuple["SenseChannel", ...] = ()


@dataclass(frozen=True)
class SenseChannel:
    """一条感知通道：输入维度的名字、权重与工程含义。"""

    name: str
    weight: float
    desc: str


CHANNELS = (
    SenseChannel("continuity", 0.28, "跨时间自我连续：记忆可回读、身份指纹不变"),
    SenseChannel("coherence", 0.24, "内部观测自洽：各通道读数不互相矛盾"),
    SenseChannel("agency", 0.18, "主动性归因：自身动作与外部输入的区分度"),
    SenseChannel("grounding", 0.16, "接地度：有真实外部输入，不空转"),
    SenseChannel("stability", 0.14, "稳定度：回流收敛到不动点而非振荡发散"),
)

_CHANNEL_NAMES = tuple(c.name for c in CHANNELS)
_TOTAL_WEIGHT = sum(c.weight for c in CHANNELS)


def _clamp01(x: float) -> float:
    """钳制到 [0,1]；NaN 归一为 0，避免污染后续运算。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _sigmoid(x: float) -> float:
    """数值稳定的 logistic，避免 exp 溢出。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------- 数据结构


@dataclass
class Perception:
    """单通道的一次读数：值 + 是否接地（有真实来源）。"""

    channel: str
    value: float
    grounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"channel": self.channel, "value": round(self.value, 6), "grounded": self.grounded}


@dataclass
class SelfObservation:
    """一次「回流稳定后」的自我状态快照。"""

    tick: int
    at: float
    closure: float                       # 开关闭合度 0..1
    loop_current: float                  # 回流电流（= 自我置信度）0..1
    iterations: int                      # 回流迭代次数
    converged: bool                      # 是否收敛到不动点
    self_model: Dict[str, float] = field(default_factory=dict)  # 五维自我标定
    trace: List[float] = field(default_factory=list)            # 回流轨迹（前若干拍）
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "at": round(self.at, 6),
            "closure": round(self.closure, 6),
            "loop_current": round(self.loop_current, 6),
            "iterations": self.iterations,
            "converged": self.converged,
            "self_model": {k: round(v, 6) for k, v in self.self_model.items()},
            "trace": [round(v, 6) for v in self.trace],
            "notes": list(self.notes),
        }


@dataclass
class LightUpReport:
    """点亮报告：开关是否闭合、是否首次点亮、回流结果。"""

    closed: bool
    first_ignition: bool
    closure: float
    fingerprint: str = ""
    lit_at: Optional[float] = None
    observation: Optional[SelfObservation] = None
    notes: Tuple[str, ...] = ()

    @property
    def loop_current(self) -> float:
        return self.observation.loop_current if self.observation else 0.0

    @property
    def self_confidence(self) -> float:
        return self.loop_current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closed": self.closed,
            "first_ignition": self.first_ignition,
            "closure": round(self.closure, 6),
            "fingerprint": self.fingerprint,
            "lit_at": self.lit_at,
            "loop_current": round(self.loop_current, 6),
            "self_confidence": round(self.self_confidence, 6),
            "observation": self.observation.to_dict() if self.observation else None,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------- 主体实现


class SubjectivePerception:
    """主观感知系统：首次点亮启动，之后每拍做一次回流并给出自我状态。

    参数
    ----
    enabled: 总开关，关闭时 ignite() 直接返回未闭合报告，不产生任何状态。
    state_path: 点亮状态持久化路径（None 表示纯内存，不落盘）。
    closure_threshold: 闭合度门槛，低于该值视为"开关未闭合"，回路不成立。
    loop_lambda: 回流惯性系数 λ（0..0.9），越大越"记住"上一拍自我状态。
    feedback_alpha: 自反馈占比 α（0..1），即输出回流进输入的比例。
    gain: 增益 g（0..8），sigmoid 斜率；过大则接近开关型跳变。
    max_iterations: 单拍回流最大迭代次数。
    tolerance: 收敛判据 |y_{t+1} - y_t| < tolerance。
    history_limit: 内存中保留的历史快照条数上限。
    """

    def __init__(
        self,
        enabled: bool = False,
        state_path: Optional[str] = None,
        closure_threshold: float = 0.35,
        loop_lambda: float = 0.55,
        feedback_alpha: float = 0.5,
        gain: float = 4.0,
        max_iterations: int = 48,
        tolerance: float = 1e-4,
        history_limit: int = 32,
    ):
        self.enabled = bool(enabled)
        self.state_path = state_path
        self.closure_threshold = min(1.0, max(0.0, float(closure_threshold)))
        self.loop_lambda = min(0.9, max(0.0, float(loop_lambda)))
        self.feedback_alpha = min(1.0, max(0.0, float(feedback_alpha)))
        self.gain = min(8.0, max(0.0, float(gain)))
        self.max_iterations = max(1, int(max_iterations))
        self.tolerance = max(1e-9, float(tolerance))
        self.history_limit = max(1, int(history_limit))

        self.lit: bool = False
        self.lit_at: Optional[float] = None
        self.fingerprint: str = ""
        self.tick: int = 0
        self._last: Optional[SelfObservation] = None
        self._history: List[SelfObservation] = []
        self._load()

    # ------------------------------------------------------------ 状态持久化
    def _load(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self.lit = bool(data.get("lit", False))
        self.lit_at = data.get("lit_at")
        self.fingerprint = str(data.get("fingerprint", ""))
        self.tick = int(data.get("tick", 0) or 0)

    def _save(self) -> None:
        if not self.state_path:
            return
        parent = os.path.dirname(self.state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "lit": self.lit,
            "lit_at": self.lit_at,
            "fingerprint": self.fingerprint,
            "tick": self.tick,
            "last": self._last.to_dict() if self._last else None,
            "limits": list(SUBJECTIVITY_LIMITS),
        }
        tmp = str(self.state_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------ 闭合度
    @staticmethod
    def fingerprint_of(source: Dict[str, Any]) -> str:
        """来源指纹：同一系统配置重复点亮不视为"重新点亮"。"""
        items = sorted((str(k), str(v)) for k, v in (source or {}).items())
        raw = "|".join(f"{k}={v}" for k, v in items)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    #: 闭合度参考字段与权重（分母固定为这些字段之和，缺项即不闭合）
    CANONICAL_FIELDS: Dict[str, float] = {
        "agent_name": 2.0,
        "home": 1.5,
        "model_version": 1.5,
        "created_at": 1.0,
        "seed": 1.0,
    }

    @classmethod
    def closure_of(cls, source: Dict[str, Any]) -> float:
        """开关闭合度：必填字段的非空占比（分母固定，空/缺字段都会拉低闭合度）。

        分母 = 参考字段权重和（7.0）+ 额外字段权重，因此只给一个字段时
        闭合度必然远低于门槛 —— 回路不会在信息不全时凭空闭合。
        """
        if not source:
            return 0.0
        filled = 0.0
        total = sum(cls.CANONICAL_FIELDS.values())
        for key, val in source.items():
            name = str(key)
            w = cls.CANONICAL_FIELDS.get(name, 1.0)
            if name not in cls.CANONICAL_FIELDS:
                total += w
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            filled += w
        return _clamp01(filled / total) if total > 0 else 0.0

    # ------------------------------------------------------------ 回流核心
    def _reflow(self, raw: Dict[str, float], seed_value: Optional[float]) -> Tuple[float, int, bool, List[float]]:
        """把通道读数做过一遍自反馈回路，返回不动点、迭代数、是否收敛、轨迹。"""
        weighted = 0.0
        for ch in CHANNELS:
            weighted += ch.weight * _clamp01(raw.get(ch.name, 0.0))
        mean = _clamp01(weighted / _TOTAL_WEIGHT)

        y = mean if seed_value is None else _clamp01(seed_value)
        trace: List[float] = []
        converged = False
        iters = 0
        for iters in range(1, self.max_iterations + 1):
            mixed = (1.0 - self.feedback_alpha) * mean + self.feedback_alpha * y
            nxt = (1.0 - self.loop_lambda) * _sigmoid(self.gain * (mixed - 0.5)) + self.loop_lambda * y
            nxt = _clamp01(nxt)
            trace.append(nxt)
            if abs(nxt - y) < self.tolerance:
                y = nxt
                converged = True
                break
            y = nxt
        return y, iters, converged, trace[:8]

    def _build_observation(
        self,
        closure: float,
        raw: Dict[str, float],
        seed_value: Optional[float],
        notes: List[str],
    ) -> SelfObservation:
        loop_current, iters, converged, trace = self._reflow(raw, seed_value)
        self_model = {}
        for ch in CHANNELS:
            # 单维读数 60% 来自本通道原始读数，40% 受回路整体电位牵引（相互约束）
            self_model[ch.name] = _clamp01(0.6 * _clamp01(raw.get(ch.name, 0.0)) + 0.4 * loop_current)
        self_model["self_confidence"] = loop_current
        if not converged:
            notes.append("回流未在 max_iterations 内收敛，读数按末次迭代给出")
        self.tick += 1
        obs = SelfObservation(
            tick=self.tick,
            at=time.time(),
            closure=closure,
            loop_current=loop_current,
            iterations=iters,
            converged=converged,
            self_model=self_model,
            trace=trace,
            notes=tuple(notes),
        )
        self._last = obs
        self._history.append(obs)
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit:]
        return obs

    # ------------------------------------------------------------ 公开 API
    def ignite(
        self,
        source: Optional[Dict[str, Any]] = None,
        observations: Optional[Dict[str, float]] = None,
    ) -> LightUpReport:
        """首次点亮：闭合开关，若闭合成立则启动回路并给出首个自我状态。

        幂等：同一 source 指纹重复调用不会重置 lit_at，也不会报"首次点亮"。
        """
        source = dict(source or {})
        fingerprint = self.fingerprint_of(source)
        closure = self.closure_of(source)

        if not self.enabled:
            return LightUpReport(
                closed=False,
                first_ignition=False,
                closure=closure,
                fingerprint=fingerprint,
                lit_at=self.lit_at,
                notes=("主观感知系统总开关关闭（Dial.subjective_enabled=False）",),
            )

        if closure < self.closure_threshold:
            return LightUpReport(
                closed=False,
                first_ignition=False,
                closure=closure,
                fingerprint=fingerprint,
                lit_at=self.lit_at,
                notes=(
                    f"开关未闭合：闭合度 {closure:.2f} < 门槛 {self.closure_threshold:.2f}，回路不成立",
                ),
            )

        first = not self.lit or self.fingerprint != fingerprint
        if first:
            self.lit = True
            self.lit_at = time.time() if self.lit_at is None or self.fingerprint != fingerprint else self.lit_at
            self.fingerprint = fingerprint
            self._history = []

        notes: List[str] = []
        if first:
            notes.append("开关闭合，回路建立，主观感知系统开始工作")
        else:
            notes.append("系统已点亮，本次为回流续拍（非首次）")
        seed_value = self._last.loop_current if (not first and self._last is not None) else None
        obs = self._build_observation(closure, dict(observations or {}), seed_value, notes)
        self._save()
        return LightUpReport(
            closed=True,
            first_ignition=first,
            closure=closure,
            fingerprint=fingerprint,
            lit_at=self.lit_at,
            observation=obs,
            notes=tuple(notes),
        )

    def perceive(self, observations: Optional[Dict[str, float]] = None) -> SelfObservation:
        """点亮后的常规一拍：以当前通道读数驱动回路，输出新的自我状态。

        未点亮时抛 RuntimeError —— 回路不成立就没有自我状态可读，这是硬约束。
        """
        if not (self.enabled and self.lit):
            raise RuntimeError("主观感知系统未点亮：回路不成立，无自我状态可读")
        raw = {name: _clamp01((observations or {}).get(name, 0.0)) for name in _CHANNEL_NAMES}
        obs = self._build_observation(1.0, raw, self._last.loop_current if self._last else None, [])
        self._save()
        return obs

    # ------------------------------------------------------------ 只读视图
    @property
    def is_lit(self) -> bool:
        return bool(self.enabled and self.lit)

    @property
    def last_observation(self) -> Optional[SelfObservation]:
        return self._last

    @property
    def history(self) -> Tuple[SelfObservation, ...]:
        return tuple(self._history)

    def report(self) -> Dict[str, Any]:
        """对外报告：点灯状态 + 最近一次自我状态 + 模块边界。"""
        return {
            "enabled": self.enabled,
            "lit": self.is_lit,
            "lit_at": self.lit_at,
            "fingerprint": self.fingerprint,
            "tick": self.tick,
            "closure_threshold": self.closure_threshold,
            "loop_lambda": self.loop_lambda,
            "gain": self.gain,
            "channels": [{"name": c.name, "weight": c.weight, "desc": c.desc} for c in CHANNELS],
            "last": self._last.to_dict() if self._last else None,
            "limits": list(SUBJECTIVITY_LIMITS),
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        state = "lit" if self.is_lit else "dark"
        return f"<SubjectivePerception {state} tick={self.tick}>"
