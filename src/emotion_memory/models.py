"""核心数据模型：情感标签、记忆（原文+情感污染双层）、记忆簇元数据。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class AffectLabel:
    """情感标签：效价 valence(-1..1) 与唤醒度 arousal(0..1)。

    情感增益 g 只在检索排序层生效；文本层面由情感污染滤镜（pollution.py）
    负责生成污染版本，情感标签本身不可被滤镜篡改。
    """

    valence: float = 0.0
    arousal: float = 0.0

    def clamp(self) -> "AffectLabel":
        self.valence = min(1.0, max(-1.0, self.valence))
        self.arousal = min(1.0, max(0.0, self.arousal))
        return self

    def to_dict(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AffectLabel":
        d = d or {}
        return cls(
            valence=float(d.get("valence", 0.0)),
            arousal=float(d.get("arousal", 0.0)),
        ).clamp()


@dataclass
class Memory:
    """一条记忆（双层：原文永久存档 + 情感污染视图）。

    分层约定（v0.2，解除「原文永不被情感滤镜篡改」红线后）：
      - original_text : 原文，永久存档、不可删除、不可被任何滤镜修改；
      - text          : 当前视图文本。默认等于原文；被情感污染后为
                        美化/丑化的污染版本（检索与上下文注入默认输出它）；
      - pollution_history : 污染审计回溯（每次污染的原文/污染文本哈希、
                        模式、强度、时间），可随时回溯或净化回原文；
      - 情感标签 affect 始终反映写入时标注，滤镜不触碰。
    其余字段为排序用元数据（可及性/访问计数），同样不改原文。
    """

    memory_id: str = field(default_factory=lambda: new_id())
    text: str = ""
    original_text: str = ""
    polluted: bool = False
    pollution_mode: Optional[str] = None          # beautify / uglify
    pollution_history: List[Dict[str, Any]] = field(default_factory=list)
    cluster: str = "default"
    entities: List[str] = field(default_factory=list)
    affect: AffectLabel = field(default_factory=AffectLabel)
    created_at: float = field(default_factory=time.time)
    base_strength: float = 1.0
    access_count: int = 0
    accessibility: float = 1.0
    last_access_at: Optional[float] = None
    silence: bool = False           # 沉默级：默认不注入
    never_silent: bool = False      # 永不沉默：高危 / 时间敏感
    due_at: Optional[float] = None  # 时间敏感到期时间戳（到期提醒注入）
    reminded: bool = False          # 到期提醒是否已发出
    tags: List[str] = field(default_factory=list)

    # ---- v0.3 电容记忆模型字段 ----
    charge: float = 0.0             # 当前电荷（触达充电 / 想起放电 / 时间漏电）
    capacitance: float = 1.0        # 容量上限（记忆的"电势容量"）
    discharge_threshold: float = 0.2  # 放电阈值：charge 低于 容量×阈值 则不再优先

    def __post_init__(self) -> None:
        # 原文不可变约束：original_text 一旦写入即锁定；兼容旧数据回填
        if not self.original_text:
            self.original_text = self.text
        if not self.text:
            self.text = self.original_text

    def touch(self, now: Optional[float] = None) -> None:
        """使用强化：访问次数 +1（提升可及性）。"""
        self.access_count += 1
        self.last_access_at = now if now is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["affect"] = self.affect.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        affect = AffectLabel.from_dict(d.get("affect") or {})
        return cls(
            memory_id=str(d.get("memory_id", new_id())),
            text=str(d.get("text", "")),
            original_text=str(d.get("original_text") or d.get("text", "")),
            polluted=bool(d.get("polluted", False)),
            pollution_mode=(
                str(d["pollution_mode"]) if d.get("pollution_mode") is not None else None
            ),
            pollution_history=list(d.get("pollution_history", [])),
            cluster=str(d.get("cluster", "default")),
            entities=list(d.get("entities", [])),
            affect=affect,
            created_at=float(d.get("created_at", time.time())),
            base_strength=float(d.get("base_strength", 1.0)),
            access_count=int(d.get("access_count", 0)),
            accessibility=float(d.get("accessibility", 1.0)),
            last_access_at=(
                float(d["last_access_at"]) if d.get("last_access_at") is not None else None
            ),
            silence=bool(d.get("silence", False)),
            never_silent=bool(d.get("never_silent", False)),
            due_at=float(d["due_at"]) if d.get("due_at") is not None else None,
            reminded=bool(d.get("reminded", False)),
            tags=list(d.get("tags", [])),
            charge=float(d.get("charge", 0.0)),
            capacitance=float(d.get("capacitance", 1.0)),
            discharge_threshold=float(d.get("discharge_threshold", 0.2)),
        )


@dataclass
class ClusterMeta:
    """记忆簇元数据：三相交流电式多通道激活（theta 振荡对应物）。"""

    cluster: str
    frequency: float = 0.5   # ω
    phase: float = 0.0       # φ
    amplitude: float = 1.0   # A


@dataclass
class RankedHit:
    """检索命中：记忆 + 最终排序分 + dB 强度 + 各分解项（用于可观测/审计）。

    display_text：上下文注入实际输出的文本视图。
      - view="polluted"（默认）→ memory.text（可能为污染版本）
      - view="original"       → memory.original_text（原文回溯）
    """

    memory: Memory
    score: float = 0.0
    db: float = -120.0
    cosine: float = 0.0
    gain: float = 0.5
    hit: float = 0.0
    accessibility: float = 1.0
    context: str = ""
    display_text: str = ""
