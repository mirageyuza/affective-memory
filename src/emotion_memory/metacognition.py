"""元认知层：置信度估计 + 拒答策略 + 校准度量（ECE / Brier）。

为什么需要这一层
----------------
记忆系统的失败模式不是"检索不到"，而是"检索到一点点就理直气壮地说"。
元认知层把「我有多确定」变成可计算、可分解、可审计的数字，并给出
三档输出决定：answer（照答）/ hedge（答但标注不确定）/ abstain（拒答）。

三条不可协商的设计约束
--------------------
1. **置信度不得被情感抬高**：估计只用 `RankedHit.cosine`（纯语义相关度），
   不使用含情感增益的 `score` / `db`。让"喜欢"抬高"确信"就是自欺。
2. **拒答必须真的降低错误率**：拒答不是礼貌，是错误率控制手段。校准
   报告（`CalibrationReport`）必须能量化「说确定时的准确率」是否等于
   「声明的置信度」，偏差即 ECE。
3. **保守优先**：没有候选、语义低于地板、样本不足，一律按最保守档处理；
   唯一例外是带 `never_silent` 标记的高危/时间敏感记忆——宁可说"我不确定"
   也不装作它不存在。

与 creed 的关系
--------------
本模块实现 `creed.OATH_CLAUSES` 中「温暖不可伪造」的工程侧：
`abstain_text()` / `hedge_text()` 都是固定模板，**不生成任何情感断言**，
不允许用"我觉得没问题"替代"我没有足够依据"。

局限（诚实记录）
--------------
- 置信度是对"检索是否命中"的估计，不等于"记忆内容在现实世界为真"；
- 校准度量需要真实标注样本，冷启动阶段报告一定是 `insufficient`；
- 权重（0.45/0.25/0.15/0.15）是工程先验，未经大规模数据拟合。

运行时零第三方依赖（仅标准库）。
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

EPS = 1e-9

# ---- 三档输出 ----
LEVEL_ANSWER = "answer"
LEVEL_HEDGE = "hedge"
LEVEL_ABSTAIN = "abstain"

# ---- 固定话术：不含任何情感断言，不得被上层改写为温暖承诺 ----
ABSTAIN_TEXT = "我没有足够依据回答这个问题。（元认知：候选缺失或相关度过低，拒绝编造）"
HEDGE_TEXT = "以下内容依据较弱，请自行核对后再采信。"

METACOGNITION_LIMITS: Tuple[str, ...] = (
    "置信度估计的是「检索命中」而非「事实为真」",
    "权重为工程先验，未经大规模数据拟合",
    "冷启动无标注样本时，校准报告为 insufficient，不得据此声称已校准",
    "拒答阈值需按业务错误代价调整，默认值不构成通用最优",
)


# ====================================================================== 置信度
@dataclass
class ConfidenceBreakdown:
    """置信度分解（每一项都是实测值，可审计，不可凭感觉赋值）。"""

    confidence: float = 0.0        # 合成置信度 0..1
    semantic_top: float = 0.0      # top1 纯语义相关度（不经情感增益）
    margin: float = 0.0            # top1 与 top2 的相对分差 0..1
    dispersion: float = 0.0        # 分数集中度（1=全压 top1，0=完全发散）
    coverage: float = 0.0          # 命中覆盖度 len(hits)/top_k
    support: int = 0               # 命中的候选条数
    protected: bool = False        # 是否含 never_silent 高危/时间敏感记忆
    weights: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 6),
            "semantic_top": round(self.semantic_top, 6),
            "margin": round(self.margin, 6),
            "dispersion": round(self.dispersion, 6),
            "coverage": round(self.coverage, 6),
            "support": self.support,
            "protected": self.protected,
            "weights": dict(self.weights),
        }


class ConfidenceEstimator:
    """把一组检索命中折算成置信度。

    置信度 = 0.45·语义相关度 + 0.25·分差 + 0.15·集中度 + 0.15·覆盖度

    刻意**不**读取 `score`：score 含情感增益与可及性调制，
    若参与置信度计算，就等于让系统"越喜欢越确信"。
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "semantic_top": 0.45,
        "margin": 0.25,
        "dispersion": 0.15,
        "coverage": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        w = dict(self.DEFAULT_WEIGHTS)
        if weights:
            w.update({k: float(v) for k, v in weights.items() if k in w})
        total = sum(abs(v) for v in w.values()) or 1.0
        self.weights = {k: v / total for k, v in w.items()}

    # ---- 单项 ----
    @staticmethod
    def _cosines(hits: Sequence[Any]) -> List[float]:
        out: List[float] = []
        for h in hits:
            c = float(getattr(h, "cosine", 0.0) or 0.0)
            out.append(0.0 if c < 0.0 else min(1.0, c))
        return out

    @staticmethod
    def _margin(cos: Sequence[float]) -> float:
        if not cos:
            return 0.0
        if len(cos) == 1:
            return 1.0
        return max(0.0, (cos[0] - cos[1]) / max(EPS, cos[0]))

    @staticmethod
    def _dispersion(cos: Sequence[float]) -> float:
        if len(cos) <= 1:
            return 1.0 if cos else 0.0
        total = sum(cos)
        if total <= EPS:
            return 0.0
        p = [c / total for c in cos if c > 0.0]
        if len(p) <= 1:
            return 1.0
        ent = -sum(pi * math.log(pi) for pi in p)
        h_max = math.log(len(p))
        return max(0.0, min(1.0, 1.0 - ent / max(EPS, h_max)))

    def estimate(
        self,
        hits: Sequence[Any],
        top_k: Optional[int] = None,
    ) -> ConfidenceBreakdown:
        cos = self._cosines(hits)
        support = len(cos)
        semantic_top = cos[0] if cos else 0.0
        margin = self._margin(cos)
        dispersion = self._dispersion(cos)
        k = max(1, int(top_k or support or 1))
        coverage = min(1.0, support / k) if support else 0.0
        protected = any(bool(getattr(getattr(h, "memory", None), "never_silent", False)) for h in hits)

        confidence = (
            self.weights["semantic_top"] * semantic_top
            + self.weights["margin"] * margin
            + self.weights["dispersion"] * dispersion
            + self.weights["coverage"] * coverage
        )
        return ConfidenceBreakdown(
            confidence=max(0.0, min(1.0, confidence)),
            semantic_top=semantic_top,
            margin=margin,
            dispersion=dispersion,
            coverage=coverage,
            support=support,
            protected=protected,
            weights=dict(self.weights),
        )


# ====================================================================== 裁决
@dataclass
class RecallVerdict:
    """一次检索的元认知裁决。"""

    level: str = LEVEL_ABSTAIN
    confidence: float = 0.0
    reason: str = ""
    text: str = ABSTAIN_TEXT
    breakdown: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)

    @property
    def should_abstain(self) -> bool:
        return self.level == LEVEL_ABSTAIN

    @property
    def is_grounded(self) -> bool:
        return self.level == LEVEL_ANSWER

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "confidence": round(self.confidence, 6),
            "reason": self.reason,
            "text": self.text,
            "should_abstain": self.should_abstain,
            "breakdown": self.breakdown.as_dict(),
        }


class AbstentionPolicy:
    """按置信度分档，并叠加两条硬规则。

    硬规则（优先于阈值）：
      1. 无候选 → 一律拒答；
      2. 语义相关度低于 `semantic_floor` → 一律拒答（分数再高也不算命中）；
      3. 含 never_silent 记忆时**最低降到 hedge**，不许拒答——
         高危/时间敏感信息宁可标注不确定也要说出来。
    """

    def __init__(
        self,
        answer_threshold: float = 0.55,
        hedge_threshold: float = 0.30,
        semantic_floor: float = 0.15,
        estimator: Optional[ConfidenceEstimator] = None,
    ):
        self.hedge_threshold = max(0.0, min(1.0, float(hedge_threshold)))
        self.answer_threshold = max(self.hedge_threshold, min(0.99, float(answer_threshold)))
        self.semantic_floor = max(0.0, min(0.9, float(semantic_floor)))
        self.estimator = estimator or ConfidenceEstimator()

    def decide(
        self,
        hits: Sequence[Any],
        top_k: Optional[int] = None,
    ) -> RecallVerdict:
        bd = self.estimator.estimate(hits, top_k=top_k)
        if bd.support == 0:
            return RecallVerdict(
                level=LEVEL_ABSTAIN, confidence=0.0, reason="no_candidate",
                text=ABSTAIN_TEXT, breakdown=bd,
            )
        if bd.semantic_top < self.semantic_floor:
            return RecallVerdict(
                level=LEVEL_ABSTAIN, confidence=bd.confidence, reason="below_semantic_floor",
                text=ABSTAIN_TEXT, breakdown=bd,
            )
        if bd.protected and bd.confidence < self.hedge_threshold:
            return RecallVerdict(
                level=LEVEL_HEDGE, confidence=bd.confidence, reason="protected_memory_low_confidence",
                text=HEDGE_TEXT, breakdown=bd,
            )
        if bd.confidence >= self.answer_threshold:
            return RecallVerdict(
                level=LEVEL_ANSWER, confidence=bd.confidence, reason="confidence_above_answer_threshold",
                text="", breakdown=bd,
            )
        if bd.confidence >= self.hedge_threshold:
            return RecallVerdict(
                level=LEVEL_HEDGE, confidence=bd.confidence, reason="confidence_in_hedge_band",
                text=HEDGE_TEXT, breakdown=bd,
            )
        return RecallVerdict(
            level=LEVEL_ABSTAIN, confidence=bd.confidence, reason="confidence_below_hedge_threshold",
            text=ABSTAIN_TEXT, breakdown=bd,
        )


# ====================================================================== 校准
@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float
    gap: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "range": [round(self.lo, 4), round(self.hi, 4)],
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 6),
            "accuracy": round(self.accuracy, 6),
            "gap": round(self.gap, 6),
        }


@dataclass
class CalibrationReport:
    """校准度量结果。空样本不返回"0 误差"，而是显式 insufficient。"""

    n: int = 0
    ece: Optional[float] = None
    brier: Optional[float] = None
    accuracy: Optional[float] = None
    mean_confidence: Optional[float] = None
    bins: List[CalibrationBin] = field(default_factory=list)
    tolerance: float = 0.08
    min_samples: int = 20
    insufficient: bool = True

    @property
    def calibrated(self) -> bool:
        return (not self.insufficient) and self.ece is not None and self.ece <= self.tolerance

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "ece": None if self.ece is None else round(self.ece, 6),
            "brier": None if self.brier is None else round(self.brier, 6),
            "accuracy": None if self.accuracy is None else round(self.accuracy, 6),
            "mean_confidence": None if self.mean_confidence is None else round(self.mean_confidence, 6),
            "calibrated": self.calibrated,
            "insufficient": self.insufficient,
            "tolerance": self.tolerance,
            "min_samples": self.min_samples,
            "bins": [b.as_dict() for b in self.bins],
        }

    def summary(self) -> str:
        if self.insufficient:
            return f"样本不足（n={self.n} < {self.min_samples}），不得声称已校准"
        verdict = "已校准" if self.calibrated else "未校准"
        return (
            f"n={self.n} ECE={self.ece:.4f} Brier={self.brier:.4f} "
            f"准确率={self.accuracy:.4f} 平均置信度={self.mean_confidence:.4f} → {verdict}"
        )


def calibrate(
    samples: Sequence[Tuple[float, bool]],
    n_bins: int = 10,
    tolerance: float = 0.08,
    min_samples: int = 20,
) -> CalibrationReport:
    """计算 ECE / Brier。

    samples: [(置信度, 是否答对), ...]，置信度会被钳制到 [0,1]。
    ECE = Σ (箱样本占比 × |箱内平均置信度 − 箱内准确率|)。
    """
    n_bins = max(1, int(n_bins))
    clean: List[Tuple[float, float]] = []
    for conf, correct in samples:
        c = float(conf)
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        clean.append((c, 1.0 if correct else 0.0))

    n = len(clean)
    if n == 0:
        return CalibrationReport(n=0, tolerance=tolerance, min_samples=min_samples, insufficient=True)

    brier = sum((c - y) ** 2 for c, y in clean) / n
    accuracy = sum(y for _, y in clean) / n
    mean_conf = sum(c for c, _ in clean) / n

    buckets: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for c, y in clean:
        idx = min(n_bins - 1, int(c * n_bins))
        buckets[idx].append((c, y))

    bins: List[CalibrationBin] = []
    ece = 0.0
    width = 1.0 / n_bins
    for i, bucket in enumerate(buckets):
        lo, hi = i * width, (i + 1) * width
        if not bucket:
            bins.append(CalibrationBin(lo=lo, hi=hi, count=0,
                                       mean_confidence=0.0, accuracy=0.0, gap=0.0))
            continue
        bc = sum(c for c, _ in bucket) / len(bucket)
        ba = sum(y for _, y in bucket) / len(bucket)
        gap = abs(bc - ba)
        ece += (len(bucket) / n) * gap
        bins.append(CalibrationBin(lo=lo, hi=hi, count=len(bucket),
                                   mean_confidence=bc, accuracy=ba, gap=gap))

    return CalibrationReport(
        n=n,
        ece=ece,
        brier=brier,
        accuracy=accuracy,
        mean_confidence=mean_conf,
        bins=bins,
        tolerance=tolerance,
        min_samples=min_samples,
        insufficient=n < min_samples,
    )


# ====================================================================== 监视器
class MetacognitiveMonitor:
    """把估计器 + 拒答策略组装成可复用的监视器，并留存审计与校准样本。

    - `verdict(hits)`       → RecallVerdict，不落盘、无副作用以外的东西；
    - `record_decision(v)`  → 追加一条审计（内存 + 可选 jsonl）；
    - `record(conf, ok)`    → 追加一条校准样本；
    - `report()`            → 返回 CalibrationReport。
    """

    def __init__(
        self,
        dialect: Any = None,
        policy: Optional[AbstentionPolicy] = None,
        estimator: Optional[ConfidenceEstimator] = None,
        audit_path: Optional[str] = None,
    ):
        dial = dialect
        self.estimator = estimator or ConfidenceEstimator()
        if policy is None:
            policy = AbstentionPolicy(
                answer_threshold=float(getattr(dial, "answer_threshold", 0.55)),
                hedge_threshold=float(getattr(dial, "hedge_threshold", 0.30)),
                semantic_floor=float(getattr(dial, "semantic_floor", 0.15)),
                estimator=self.estimator,
            )
        self.policy = policy
        self.audit_path = audit_path
        self.decisions: List[Dict[str, Any]] = []
        self.samples: List[Tuple[float, bool]] = []

    # ---- 裁决 ----
    def verdict(self, hits: Sequence[Any], top_k: Optional[int] = None) -> RecallVerdict:
        return self.policy.decide(hits, top_k=top_k)

    def record_decision(self, verdict: RecallVerdict, query: str = "") -> Dict[str, Any]:
        entry = {
            "ts": time.time(),
            "query": query,
            **verdict.as_dict(),
        }
        self.decisions.append(entry)
        if self.audit_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.audit_path)), exist_ok=True)
                with open(self.audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                # 审计写失败不得影响主流程，但也不得假装写过
                entry["audit_write_failed"] = True
        return entry

    # ---- 校准 ----
    def record(self, confidence: float, correct: bool) -> None:
        self.samples.append((float(confidence), bool(correct)))

    def report(self, n_bins: int = 10, tolerance: float = 0.08, min_samples: int = 20) -> CalibrationReport:
        return calibrate(self.samples, n_bins=n_bins, tolerance=tolerance, min_samples=min_samples)

    # ---- 观测 ----
    def stats(self) -> Dict[str, Any]:
        total = len(self.decisions)
        levels = {LEVEL_ANSWER: 0, LEVEL_HEDGE: 0, LEVEL_ABSTAIN: 0}
        for d in self.decisions:
            lv = d.get("level")
            if lv in levels:
                levels[lv] += 1
        return {
            "decisions": total,
            "levels": levels,
            "abstain_rate": (levels[LEVEL_ABSTAIN] / total) if total else 0.0,
            "hedge_rate": (levels[LEVEL_HEDGE] / total) if total else 0.0,
            "answer_rate": (levels[LEVEL_ANSWER] / total) if total else 0.0,
            "calibration_samples": len(self.samples),
        }
