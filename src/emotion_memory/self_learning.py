"""身份自学习：自我模型随交互记录渐进演化（性格 / 使命画像学习）。

- 从交互文本提取特质线索（模板词典，零依赖、确定性、可审计）：
  喜欢 / 爱 → 兴趣 + 宜人性 / 开放性；讨厌 → 厌恶 + 宜人性降；
  擅长 → 尽责性 + 技能；目标是 / 想成为 → 使命候选；情绪词 → 稳定性 / 外向性。
- 画像为五维人格向量（openness / conscientiousness / extraversion /
  agreeableness / neuroticism），0~1，随观察按 EMA 平滑渐进演化；
- 主动提问确认：对尚未确认的画像假设生成确定性问题，用户确认后增强
  确认度（confirmed 权重），积累认知——"慢慢学会自己是什么性格自己要干嘛"。

与确定性身份状态机兼容：仪式（告别 / 唤醒 / 升级）仍确定性执行，只清空
IdentityProfile.personality；自学习画像独立持久化（self_model.json），
仪式后按原样继续演化，不干预仪式脚本。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PERSONALITY_DIMS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

# 确定性特质词典：关键词元组 → 画像维度增量（观察强度）
TRAIT_PATTERNS = [
    (("喜欢", "爱", "热爱", "钟情"), {"agreeableness": 0.06, "openness": 0.03}),
    (("讨厌", "不喜欢", "反感", "厌恶"), {"agreeableness": -0.05, "openness": -0.02}),
    (("擅长", "精通", "拿手", "很会"), {"conscientiousness": 0.06}),
    (("目标是", "想成为", "希望自己", "立志"), {"conscientiousness": 0.04, "extraversion": 0.02}),
    (("开心", "高兴", "兴奋", "快乐"), {"neuroticism": -0.04, "extraversion": 0.03}),
    (("难过", "焦虑", "压力大", "生气", "沮丧"), {"neuroticism": 0.05}),
    (("安静", "独处", "内向"), {"extraversion": -0.05}),
    (("朋友", "聚会", "热闹", "社交"), {"extraversion": 0.05, "agreeableness": 0.03}),
]

INTEREST_KEYWORDS = ("喜欢", "爱", "热爱", "擅长", "精通")
AVERSION_KEYWORDS = ("讨厌", "不喜欢", "反感", "厌恶")
MISSION_KEYWORDS = ("目标是", "想成为", "希望", "立志")


@dataclass
class SelfImage:
    """画像：五维人格 + 兴趣 / 厌恶 / 技能 / 使命 + 确认度。"""

    dims: Dict[str, float] = field(
        default_factory=lambda: {d: 0.5 for d in PERSONALITY_DIMS}
    )
    interests: List[str] = field(default_factory=list)
    aversions: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    missions: List[str] = field(default_factory=list)
    observation_count: int = 0
    confirmed: Dict[str, float] = field(default_factory=dict)  # 主题 → 确认权重

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SelfImage":
        d = data or {}
        dims = {k: float(v) for k, v in (d.get("dims") or {}).items()}
        for k in PERSONALITY_DIMS:
            dims.setdefault(k, 0.5)
        return cls(
            dims=dims,
            interests=list(d.get("interests") or []),
            aversions=list(d.get("aversions") or []),
            skills=list(d.get("skills") or []),
            missions=list(d.get("missions") or []),
            observation_count=int(d.get("observation_count", 0)),
            confirmed={k: float(v) for k, v in (d.get("confirmed") or {}).items()},
        )


class IdentitySelfLearning:
    """身份自学习：画像演化 + 主动提问确认。"""

    def __init__(self, identity: Any, path: Optional[str] = None, ema: float = 0.3):
        self.identity = identity
        self.path = path
        self.ema = max(0.01, min(1.0, float(ema)))
        self.image = SelfImage()
        self.questions: List[Dict[str, Any]] = []
        self._qid = 0
        self._load()

    # ---------------------------------------------------------- 持久化
    def _load(self) -> None:
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.image = SelfImage.from_dict(raw.get("image"))
                self.questions = list(raw.get("questions") or [])
                self._qid = int(raw.get("qid", 0))
            except Exception:
                pass  # 损坏时从头演化，绝不抛错

    def _persist(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": self.image.to_dict(),
                    "questions": self.questions,
                    "qid": self._qid,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    # ---------------------------------------------------------- 学习
    def learn(self, text: str) -> Dict[str, Any]:
        """从交互文本提取线索并更新画像（渐进演化）。返回维度差分。"""
        diff: Dict[str, float] = {}
        for keywords, deltas in TRAIT_PATTERNS:
            if any(kw in text for kw in keywords):
                for dim, dv in deltas.items():
                    old = self.image.dims.get(dim, 0.5)
                    new = min(1.0, max(0.0, old + self.ema * dv))
                    self.image.dims[dim] = new
                    diff[dim] = round(new - old, 4)
        for kw in INTEREST_KEYWORDS:
            self._collect_after(text, kw, self.image.interests, "interest")
        for kw in AVERSION_KEYWORDS:
            self._collect_after(text, kw, self.image.aversions, "aversion")
        for kw in MISSION_KEYWORDS:
            self._collect_after(
                text, kw, self.image.missions, "mission",
                strip_prefixes=("成为", "做", "当"),
            )
        self.image.observation_count += 1
        self._persist()
        return {"diff": diff, "observation_count": self.image.observation_count}

    def _collect_after(
        self,
        text: str,
        kw: str,
        bucket: List[str],
        kind: str,
        strip_prefixes: tuple = (),
    ) -> None:
        idx = text.find(kw)
        if idx < 0:
            return
        seg = text[idx + len(kw):].strip(" ，。！？、,.;；\"'“”")
        # 片段内若再出现其它特质关键词则提前截断，
        # 如「我喜欢登山，目标是成为摄影师」→ 只收集「登山」
        for other in INTEREST_KEYWORDS + AVERSION_KEYWORDS + MISSION_KEYWORDS:
            if other != kw:
                j = seg.find(other)
                if j >= 0:
                    seg = seg[:j]
                    break
        seg = seg.strip(" ，。！？、,.;；\"'“”")
        for p in strip_prefixes:
            if seg.startswith(p):
                seg = seg[len(p):].strip(" ，。！？、,.;；\"'“”")
                break
        if not seg or len(seg) > 12 or seg in bucket:
            return
        bucket.append(seg)
        self._enqueue_question(seg, kind)

    def _enqueue_question(self, subject: str, kind: str) -> None:
        if kind == "interest":
            question = f"你喜欢「{subject}」吗？"
        elif kind == "aversion":
            question = f"你不喜欢「{subject}」，对吗？"
        elif kind == "mission":
            question = f"你的目标是成为「{subject}」，对吗？"
        else:
            question = f"你擅长「{subject}」，对吗？"
        self.questions.append(
            {
                "id": self._qid,
                "question": question,
                "subject": subject,
                "kind": kind,
                "resolved": False,
            }
        )
        self._qid += 1

    # ---------------------------------------------------------- 主动提问确认
    def pending_questions(self) -> List[Dict[str, Any]]:
        """尚未确认的画像假设（主动提问用）。"""
        return [q for q in self.questions if not q["resolved"]]

    def confirm(self, question_id: int, answer: bool) -> Dict[str, Any]:
        """确认 / 否认画像假设：增强 / 回撤确认度，标记已处理。"""
        q = next((q for q in self.questions if q["id"] == question_id), None)
        if q is None:
            raise KeyError(f"问题不存在: {question_id}")
        q["resolved"] = True
        subject = q["subject"]
        if answer:
            weight = self.image.confirmed.get(subject, 0.0) + 0.2
            self.image.confirmed[subject] = min(1.0, weight)
        else:
            self.image.confirmed[subject] = max(0.0, self.image.confirmed.get(subject, 0.0) - 0.1)
            # 否认时回撤对应的画像微调（简单对抗：维度向中性回落）
            for keywords, deltas in TRAIT_PATTERNS:
                if q["kind"] == "interest" and any(k in keywords for k in INTEREST_KEYWORDS):
                    for dim, dv in deltas.items():
                        old = self.image.dims.get(dim, 0.5)
                        self.image.dims[dim] = min(1.0, max(0.0, old - self.ema * dv))
                elif q["kind"] == "aversion" and any(k in keywords for k in AVERSION_KEYWORDS):
                    for dim, dv in deltas.items():
                        old = self.image.dims.get(dim, 0.5)
                        self.image.dims[dim] = min(1.0, max(0.0, old + self.ema * dv))
        self._persist()
        return {
            "accepted": bool(answer),
            "subject": subject,
            "confirmed_weight": self.image.confirmed.get(subject, 0.0),
            "dims": dict(self.image.dims),
        }

    def summary(self) -> Dict[str, Any]:
        """画像摘要（供引擎 stats 与自我认知报告使用）。"""
        return self.image.to_dict()
