"""势差驱动迁移 (v0.5)：记忆不随意搬迁。

物理四定律第二定律：
  ① 工作区 → 长期区沉淀由"情感势能"决定，
     势能 = 情感强度 × 重复次数 × 重要性加权；
  ② 高势能沉淀为长期记忆，低势能自然流散（保持工作区并随熵增衰减）；
  ③ 电流桥迁移按势差分配迁移量，非均匀搬运（高势能优先全量，低势能跳过/减量）；
  ④ 提供势差计算与迁移决策接口（potential() / decide() / allocate_transfer_quota()）。

安全约束：迁移只改变 region 元数据（workspace / longterm），绝不删除原文；
高势能沉淀不影响原文锁定与 SHA-256 校验协议（由 bridge 负责）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .conservation import emotion_energy
from .models import Memory

REGION_WORKSPACE = "workspace"
REGION_LONGTERM = "longterm"


class PotentialTransfer:
    """情感势能引擎：势差计算 + 沉淀决策 + 迁移配额分配。"""

    def __init__(
        self,
        threshold: float = 0.6,
        importance_weight: float = 0.4,
        repeat_weight: float = 0.3,
    ):
        self.threshold = max(0.05, float(threshold))
        self.importance_weight = max(0.0, float(importance_weight))
        self.repeat_weight = max(0.0, float(repeat_weight))
        self.decisions: List[Dict[str, Any]] = []  # 决策审计

    # ------------------------------------------------------------ 势能计算
    def potential(self, memory: Memory) -> float:
        """情感势能 = 情感强度 × 重复次数强化 × 重要性加权。

        - 情感强度：emotion_energy(memory.affect) ∈ [0,1]；
        - 重复次数强化：1 + repeat_weight * log1p(access_count)（触达越多势能越高）；
        - 重要性加权：importance（用户可标定，默认 1.0，可 >1）。
        """
        energy = emotion_energy(memory.affect)
        repeat = 1.0 + self.repeat_weight * math.log1p(max(0, memory.access_count))
        importance = max(0.0, float(memory.importance))
        return energy * repeat * importance

    # ------------------------------------------------------------ 决策
    def decide(self, memory: Memory) -> Tuple[str, float]:
        """迁移决策：势能 >= 阈值 → longterm（沉淀），否则 workspace（流散）。"""
        p = self.potential(memory)
        region = REGION_LONGTERM if p >= self.threshold else REGION_WORKSPACE
        self.decisions.append(
            {
                "memory_id": memory.memory_id,
                "potential": round(p, 4),
                "region": region,
                "threshold": self.threshold,
            }
        )
        return region, p

    def promote(self, memory: Memory) -> Dict[str, Any]:
        """沉淀单条记忆：高势能置为长期区，低势能保持工作区（自然流散）。"""
        region, p = self.decide(memory)
        moved = memory.region != region
        memory.region = region
        return {
            "memory_id": memory.memory_id,
            "potential": round(p, 4),
            "region": region,
            "promoted": region == REGION_LONGTERM and moved,
            "flowed_away": region == REGION_WORKSPACE,
        }

    def promote_many(self, memories: Sequence[Memory]) -> Dict[str, Any]:
        """批量沉淀：返回统计 + 明细（高势能沉淀 / 低势能流散）。"""
        results = [self.promote(m) for m in memories]
        promoted = [r for r in results if r["promoted"]]
        flowed = [r for r in results if r["flowed_away"]]
        return {
            "scanned": len(results),
            "promoted_count": len(promoted),
            "flowed_away_count": len(flowed),
            "promoted": promoted,
            "flowed_away": flowed,
        }

    # ------------------------------------------------------------ 迁移配额
    def allocate_transfer_quota(
        self,
        memories: Sequence[Memory],
        max_count: Optional[int] = None,
        skip_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """电流桥按势差分配迁移量：非均匀搬运。

        势能高者优先全量迁移；低势能记忆按 skip_ratio 跳过（默认不跳）。
        max_count 给定时只搬势能最高的前 N 条（其余不搬）。
        返回 {"selected": [...], "skipped": [...], "total": n}。
        """
        ranked = sorted(
            ((self.potential(m), m) for m in memories),
            key=lambda t: t[0],
            reverse=True,
        )
        cap = max_count if max_count is not None else len(ranked)
        cap = max(0, min(len(ranked), int(cap)))
        selected = ranked[:cap]
        skipped = ranked[cap:]
        skip_ratio = min(1.0, max(0.0, float(skip_ratio)))
        # 低势能部分按比例再跳过（非均匀搬运：势差小 → 迁移量少）
        if skip_ratio > 0.0 and skipped:
            keep = max(0, int(len(skipped) * (1.0 - skip_ratio)))
            moved_back = skipped[keep:]
            skipped = skipped[:keep]
            selected = selected + moved_back
        return {
            "total": len(memories),
            "selected": [m for _, m in selected],
            "skipped": [m for _, m in skipped],
            "selected_count": len(selected),
            "skipped_count": len(skipped),
        }

    # ------------------------------------------------------------ 报告
    def report(self) -> Dict[str, Any]:
        """决策审计报告（确定性可追溯）。"""
        promoted = sum(1 for d in self.decisions if d["region"] == REGION_LONGTERM)
        return {
            "decisions": list(self.decisions),
            "promoted_count": promoted,
            "threshold": self.threshold,
        }
