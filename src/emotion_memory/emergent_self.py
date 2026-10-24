"""涌现自我引擎 (v0.4)：智能体「是什么」不由预设档案决定，
而是由情感记忆累计数据动态生成的产物。

核心思想：
- 每次对话沉淀为情感记忆（valence/arousal 标签），系统只做**累计统计**；
- 从累计数据中挖掘自我定义：效价分布→性格倾向、高频使命记忆→使命、
  强情感簇→自我概念、行为统计→技能/兴趣；
- 无预设模板：冷启动时为空壳，累积到阈值才开始产生轮廓；
- 每次沉淀后增量更新，可导出「当前自我定义」报告；
- 唤醒/重装后身份从累计数据重建，而非从固定档案加载。

数据安全：本引擎只读取记忆的情感标签与元数据做统计，永不改写记忆原文。
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class EmergentSelf:
    """从情感累计数据中涌现的自我定义。"""

    _TRAIT_NAMES = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

    def __init__(self, storage: Any, engine: Any = None):
        self.storage = storage
        self.engine = engine
        self.records: List[Dict[str, Any]] = []
        self.current_concepts: List[Dict[str, Any]] = []
        self.current_traits: Dict[str, float] = {}

    # ---------- 累计 ----------

    def accumulate(
        self,
        text: str,
        affect: Any,
        cluster: str = "",
        entities: Optional[List[str]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """记录一条情感累计（来自沉淀环节）。"""
        valence = self._valence_of(affect)
        arousal = self._arousal_of(affect)
        rec = {
            "text": text,
            "valence": valence,
            "arousal": arousal,
            "cluster": cluster,
            "entities": list(entities or []),
            "ts": now,
        }
        self.records.append(rec)
        return rec

    def refresh(self) -> None:
        """增量更新当前自我定义（性格维度、使命、自我概念、技能、兴趣）。"""
        if not self.records:
            self.current_concepts = []
            self.current_traits = {}
            return
        traits = self._compute_traits(self.records)
        mission = self._extract_mission(self.records)
        concepts = self._extract_concepts(self.records)
        skills = self._extract_skills(self.records)
        interests = self._extract_interests(self.records)
        self.current_traits = traits
        self.current_concepts = concepts
        self.current_traits["mission"] = mission
        self.current_traits["skills"] = skills
        self.current_traits["interests"] = interests

    # ---------- 挖掘 ----------

    @staticmethod
    def _valence_of(affect: Any) -> float:
        if affect is None:
            return 0.0
        if isinstance(affect, dict):
            return float(affect.get("valence", 0.0))
        if isinstance(affect, (tuple, list)):
            return float(affect[0]) if len(affect) > 0 else 0.0
        try:
            return float(affect.valence)
        except Exception:
            return 0.0

    @staticmethod
    def _arousal_of(affect: Any) -> float:
        if affect is None:
            return 0.0
        if isinstance(affect, dict):
            return float(affect.get("arousal", 0.0))
        if isinstance(affect, (tuple, list)):
            return float(affect[1]) if len(affect) > 1 else 0.0
        try:
            return float(affect.arousal)
        except Exception:
            return 0.0

    def _compute_traits(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(records)
        vals = [r["valence"] for r in records]
        arousals = [r["arousal"] for r in records]
        mean_v = sum(vals) / n
        mean_a = sum(arousals) / n
        mean_abs = sum(abs(v) for v in vals) / n
        var = sum((v - mean_v) ** 2 for v in vals) / n
        std = var ** 0.5
        num_clusters = len({r["cluster"] for r in records if r["cluster"]})
        traits = {
            "openness": self._clamp(num_clusters / 10.0, 0.0, 1.0),
            "conscientiousness": self._clamp(1.0 - std * 1.5, 0.0, 1.0),
            "extraversion": self._clamp(mean_v * 0.5 + mean_a * 0.5, 0.0, 1.0),
            "agreeableness": self._clamp((mean_v + 1.0) / 2.0, 0.0, 1.0),
            "neuroticism": self._clamp(mean_abs, 0.0, 1.0),
            "activity": self._clamp(mean_a, 0.0, 1.0),
        }
        return traits

    def _extract_mission(self, records: List[Dict[str, Any]]) -> str:
        counter = Counter(
            (r["text"] or "").strip() for r in records if (r["text"] or "").strip()
        )
        if not counter:
            return ""
        text, count = counter.most_common(1)[0]
        if count < 2:
            return ""
        cluster = next((r["cluster"] for r in records if r["text"] == text), "")
        return text if cluster else text

    def _extract_concepts(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_cluster: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            by_cluster.setdefault(r["cluster"], []).append(r)
        concepts = []
        for cluster, items in by_cluster.items():
            if not cluster:
                continue
            n = len(items)
            mean_v = sum(r["valence"] for r in items) / n
            mean_a = sum(r["arousal"] for r in items) / n
            intensity = abs(mean_v) * 0.6 + mean_a * 0.4
            concepts.append(
                {
                    "concept": cluster,
                    "count": n,
                    "intensity": round(intensity, 4),
                    "polarity": round(mean_v, 4),
                }
            )
        concepts.sort(key=lambda c: (-c["count"], -c["intensity"]))
        return concepts[:5]

    def _extract_skills(self, records: List[Dict[str, Any]]) -> List[str]:
        counter: Counter = Counter()
        for r in records:
            for ent in r.get("entities", []):
                counter[ent] += 1
        return [e for e, c in counter.most_common(3) if c >= 2]

    def _extract_interests(self, records: List[Dict[str, Any]]) -> List[str]:
        counter: Counter = Counter()
        for r in records:
            counter[r["cluster"]] += 1
        return [c for c, n in counter.most_common(3) if c and n >= 2]

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    # ---------- 报告 ----------

    def self_report(self) -> Dict[str, Any]:
        """导出当前自我定义报告。"""
        if not self.records:
            return {
                "emerged": False,
                "reason": "尚无情感累计，自我定义未涌现（无预设模板，从零生长）。",
                "traits": {},
                "concepts": [],
                "record_count": 0,
            }
        return {
            "emerged": True,
            "traits": dict(self.current_traits),
            "concepts": list(self.current_concepts),
            "record_count": len(self.records),
        }

    def describe(self) -> str:
        """人类可读的自我定义描述。"""
        if not self.records:
            return "自我尚未成形（没有足够的情感记忆累计）。"
        rep = self.self_report()
        lines = [f"涌现自我（基于 {rep['record_count']} 条情感记忆累计）"]
        tr = rep["traits"]
        if tr:
            dims = ", ".join(
                f"{k}={v:.2f}"
                for k, v in tr.items()
                if k in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
            )
            lines.append(f"性格倾向：{dims}")
        if tr.get("mission"):
            lines.append(f"使命：{tr['mission']}")
        if rep["concepts"]:
            for c in rep["concepts"]:
                lines.append(
                    f"- 自我概念「{c['concept']}」：出现 {c['count']} 次，"
                    f"情感强度 {c['intensity']:.2f}"
                )
        if tr.get("skills"):
            lines.append("技能：" + "、".join(tr["skills"]))
        if tr.get("interests"):
            lines.append("兴趣：" + "、".join(tr["interests"]))
        return "\n".join(lines)

    def export_json(self, path: Optional[str] = None) -> str:
        """导出当前自我定义与原始累计记录为 JSON（唤醒重建的数据源）。"""
        payload = {
            "version": "v0.4-emergent-self",
            "record_count": len(self.records),
            "traits": dict(self.current_traits),
            "concepts": list(self.current_concepts),
            "records": list(self.records),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def rebuild_from(self, json_path: str) -> int:
        """唤醒/重装后从累计数据重建自我定义（从导出 JSON 载入记录）。"""
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.records = list(payload.get("records", []))
        self.refresh()
        return len(self.records)
