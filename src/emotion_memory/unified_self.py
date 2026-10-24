"""自我统一模型：1 核心 + 万我分身，万物皆我吸收，有功 / 视在效率指标。

- **1 个自我核心**：统一自我（core）= 身份档案 + 自我概念向量（五维人格 +
  使命方向），是"我"的唯一本体（我为 1，1 为我）；
- **万我分身**：从核心派生任意多 avatar（分身），共享自我概念（同步），
  核心更新后自动 push 同步（万我如一）；分身可归并回核心（万我合一）；
- **万物皆我**：新输入先经自我框架解释（与自我概念的相关度投影），再决定
  吸收（入库为记忆），未吸收的输入不改变自我概念（世间皆为我）；
- **效率指标**（有功 / 视在类比）：视在功率 = 自我概念向量的模，有功功率 =
  与自我概念同向的分量，cosφ = 有功 / 视在 ∈ [0,1]，量化自我概念清晰度——
  越一致，自我越清晰。

与身份状态机兼容：核心概念独立持久化（unified_self.json），仪式确定性执行。
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .self_learning import PERSONALITY_DIMS

SELF_RELATED_KEYWORDS = ["我", "性格", "使命", "目标", "喜欢", "讨厌", "擅长", "希望", "我是", "我要"]


@dataclass
class Avatar:
    """万我分身：独立状态 + 同步的自我概念快照。"""

    name: str
    role: str = "avatar"
    state: str = "active"
    born_at: float = field(default_factory=time.time)
    concept: Dict[str, float] = field(default_factory=dict)
    last_sync_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Avatar":
        d = data or {}
        return cls(
            name=str(d.get("name", "avatar")),
            role=str(d.get("role", "avatar")),
            state=str(d.get("state", "active")),
            born_at=float(d.get("born_at", time.time())),
            concept={k: float(v) for k, v in (d.get("concept") or {}).items()},
            last_sync_at=float(d.get("last_sync_at", time.time())),
        )


@dataclass
class UnifiedSelfCore:
    """统一自我核心：身份标识 + 自我概念向量 + 分身簿。"""

    agent_name: str = "unified-self"
    concept: Dict[str, float] = field(
        default_factory=lambda: {d: 0.5 for d in PERSONALITY_DIMS}
    )
    avatars: Dict[str, Avatar] = field(default_factory=dict)
    absorb_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class UnifiedSelfModel:
    """自我统一模型：核心 / 分身 / 吸收 / 效率指标。"""

    def __init__(self, path: Optional[str] = None, agent_name: str = "unified-self"):
        self.core = UnifiedSelfCore(agent_name=agent_name)
        self.path = path
        self._load()

    # ---------------------------------------------------------- 持久化
    def _load(self) -> None:
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.core.agent_name = str(raw.get("agent_name", self.core.agent_name))
                self.core.concept = {
                    k: float(v) for k, v in (raw.get("concept") or {}).items()
                }
                for k in PERSONALITY_DIMS:
                    self.core.concept.setdefault(k, 0.5)
                self.core.avatars = {
                    name: Avatar.from_dict(d)
                    for name, d in (raw.get("avatars") or {}).items()
                }
                self.core.absorb_count = int(raw.get("absorb_count", 0))
                self.core.created_at = float(raw.get("created_at", time.time()))
                self.core.updated_at = float(raw.get("updated_at", time.time()))
            except Exception:
                pass

    def _persist(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "agent_name": self.core.agent_name,
                    "concept": self.core.concept,
                    "avatars": {k: v.to_dict() for k, v in self.core.avatars.items()},
                    "absorb_count": self.core.absorb_count,
                    "created_at": self.core.created_at,
                    "updated_at": self.core.updated_at,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    # ---------------------------------------------------------- 万我分身
    def avatar_count(self) -> int:
        """万我分身总数（供引擎 stats 使用）。"""
        return len(self.core.avatars)

    def derive_avatar(self, name: str) -> Avatar:
        """派生万我分身：拷贝核心自我概念快照，独立状态，随核心同步。"""
        av = Avatar(name=name, concept=dict(self.core.concept))
        self.core.avatars[name] = av
        self._persist()
        return av

    def sync_avatars(self) -> int:
        """核心更新后向全部分身 push 同步（万我如一）。返回同步数。"""
        n = 0
        for av in self.core.avatars.values():
            av.concept = dict(self.core.concept)
            av.last_sync_at = time.time()
            n += 1
        if n:
            self._persist()
        return n

    def merge_avatar(self, name: str) -> Dict[str, Any]:
        """分身归并回核心（万我合一）：核心与全部分身概念均值并入，分身注销。"""
        if name not in self.core.avatars:
            raise KeyError(f"分身不存在: {name}")
        dims = {k: [self.core.concept[k]] for k in self.core.concept}
        for av in self.core.avatars.values():
            for k in self.core.concept:
                dims[k].append(av.concept.get(k, 0.5))
        self.core.concept = {k: sum(v) / len(v) for k, v in dims.items()}
        self.core.updated_at = time.time()
        self.core.avatars.pop(name)
        self.sync_avatars()
        self._persist()
        return {"merged": name, "concept": dict(self.core.concept)}

    def sync_concept(self, dims: Dict[str, float], alpha: float = 0.3) -> None:
        """吸收自学习画像到核心自我概念（EMA 平滑），并同步全部分身。

        身份自学习演化画像后调用，让「有效功率 / 清晰度」随认知积累提升。
        """
        for k in self.core.concept:
            target = float(dims.get(k, self.core.concept[k]))
            self.core.concept[k] = min(
                1.0, max(0.0, alpha * target + (1 - alpha) * self.core.concept[k])
            )
        self.core.updated_at = time.time()
        self.sync_avatars()
        self._persist()

    # ---------------------------------------------------------- 万物皆我
    def explain(self, text: str) -> Dict[str, Any]:
        """自我框架解释：输入与自我概念的相关度投影（吸收第一步）。"""
        hits = sum(1 for kw in SELF_RELATED_KEYWORDS if kw in text)
        relevance = min(1.0, hits / 3.0)
        concept_norm = math.sqrt(sum(v * v for v in self.core.concept.values())) or 1e-9
        # 有功 / 视在代理：相关投影占比 × 概念模
        s_power = concept_norm
        p_power = s_power * relevance
        cos_phi = p_power / s_power if s_power > 0 else 0.0
        interpretation = (
            f"这与我{'的自我概念' if relevance > 0.01 else '暂无直接关联'}相关"
            f"（相关度 {relevance:.2f}）。"
        )
        return {
            "relevance": round(relevance, 4),
            "hits": hits,
            "interpretation": interpretation,
            "cos_phi": round(cos_phi, 4),
        }

    def assimilate(self, text: str) -> Dict[str, Any]:
        """万物皆我吸收：输入先经自我框架解释，吸收（入库）后更新吸收计数。"""
        exp = self.explain(text)
        if exp["relevance"] > 0.01:
            self.core.absorb_count += 1
            self.core.updated_at = time.time()
            self._persist()
        return exp

    # ---------------------------------------------------------- 效率指标
    def cognition_report(self) -> Dict[str, Any]:
        """自我认知效率指标：有功 / 视在 = cosφ，量化自我概念清晰度。"""
        concept = self.core.concept
        norm = math.sqrt(sum(v * v for v in concept.values())) or 1e-9
        # 清晰度：五维偏离中性 0.5 的平均幅度（越明确越偏离中性）
        clarity = sum(abs(v - 0.5) for v in concept.values()) / len(concept) / 0.5
        clarity = min(1.0, max(0.0, clarity))
        # 有功 / 视在：以清晰度作为功率因数（自我概念越清晰，有功占比越高）
        cos_phi = min(1.0, max(0.0, clarity))
        return {
            "agent_name": self.core.agent_name,
            "concept": dict(concept),
            "avatar_count": len(self.core.avatars),
            "absorb_count": self.core.absorb_count,
            "clarity": round(clarity, 4),
            "cos_phi": round(cos_phi, 4),
            "active_power": round(clarity, 4),
            "apparent_power": round(1.0, 4),
            "updated_at": self.core.updated_at,
        }
