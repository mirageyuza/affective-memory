"""湍流联想 (v0.5)：发散联想 = 涡旋扩散不是随机。

物理四定律第四定律：
  ① 三心二意 / 发散模式联想围绕话题中心涡旋式扩散，能量随距离递减传递；
  ② 相关记忆形成涡流链（中心 → 次级 → 外围逐层扩散）；
  ③ 与现有 divergent（发散模式）协同：发散开启时扩散更深、衰减更缓；
     与 concurrent（并发联想）协同：涡流候选可交给并发分片重排。

实现约束（纯标准库、确定性）：
  - vortex_rings() 用实体图构建同心涡流链，返回每层记忆与层距；
  - associate() 沿涡流链按层距衰减系数赋能量分，输出带层距的联想结果；
  - 中心记忆能量最高，外围按 turbulence_decay^depth 递减；
  - 不修改任何记忆：本模块只读图与记忆元数据，输出候选排序。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .store.graph_index import EntityGraph

# 涡流链层级标签
RING_CENTER = 0  # 中心（话题直接命中）
RING_SECONDARY = 1  # 次级（与中心共现实体）
RING_PERIPHERAL = 2  # 外围（次级再扩散）


class TurbulenceAssociator:
    """涡旋联想引擎：围绕话题中心逐层扩散。"""

    def __init__(self, decay: float = 0.6, max_depth: int = 3):
        self.decay = min(1.0, max(0.05, float(decay)))
        self.max_depth = max(1, min(10, int(max_depth)))
        self.last_rings: Dict[int, List[str]] = {}

    # ------------------------------------------------------------ 涡流链
    def vortex_rings(
        self,
        graph: EntityGraph,
        center_ids: Sequence[str],
        max_depth: Optional[int] = None,
    ) -> Dict[int, List[str]]:
        """构建同心涡流链：层 0=中心，层 1=与中心共现实体记忆，层 2=次级扩散……。

        扩散规则（涡旋不随机）：每层 = 上一层记忆的 weak_edges 并集，
        减去已出现过的记忆（防回旋），保证层级单调扩散。
        """
        depth = max_depth if max_depth is not None else self.max_depth
        depth = max(1, min(10, int(depth)))
        rings: Dict[int, List[str]] = {}
        seen: Set[str] = set()
        ring = [str(mid) for mid in center_ids]
        rings[RING_CENTER] = list(ring)
        seen.update(ring)
        for layer in range(1, depth + 1):
            frontier: Set[str] = set()
            for mid in ring:
                frontier |= graph.weak_edges(mid)
            frontier -= seen
            ordered = sorted(frontier)
            if not ordered:
                break
            rings[layer] = ordered
            seen.update(ordered)
            ring = ordered
        self.last_rings = dict(rings)
        return rings

    # ------------------------------------------------------------ 层距能量
    def ring_energy(self, depth: int) -> float:
        """层距能量：中心=1.0，第 depth 层 = decay^depth（能量随距离递减传递）。"""
        return self.decay ** max(0, int(depth))

    # ------------------------------------------------------------ 联想
    def associate(
        self,
        graph: EntityGraph,
        id2mem: Dict[str, Any],
        center_ids: Sequence[str],
        top_k: int = 8,
        divergent: bool = False,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """沿涡流链联想：中心能量最高，外围按衰减系数递减。

        divergent=True（发散模式协同）：扩散深度 +1、衰减放宽（扩散更远更散）。
        返回 {"hits": [(memory, score, depth)], "rings": {...}}。
        """
        depth = max_depth
        if depth is None:
            depth = self.max_depth
            if divergent:
                depth = min(10, depth + 1)
        decay = self.decay
        if divergent:
            decay = min(1.0, (1.0 + decay) / 2.0)  # 发散时衰减放宽 → 外围能量更高
        rings = self.vortex_rings(graph, center_ids, max_depth=depth)
        hits: List[Tuple[Any, float, int]] = []
        for layer, mids in rings.items():
            energy = self.ring_energy(layer)
            for mid in mids:
                mem = id2mem.get(mid)
                if mem is None:
                    continue
                hits.append((mem, energy, layer))
        hits.sort(key=lambda t: (-t[1], t[2], t[0].memory_id))
        top = hits[: max(1, int(top_k))]
        return {
            "hits": top,
            "rings": {str(k): v for k, v in rings.items()},
            "decay": round(decay, 4),
            "max_depth": depth,
            "divergent": bool(divergent),
        }

    # ------------------------------------------------------------ 报告
    def describe_rings(self) -> str:
        """人类可读的涡流链摘要（demo 展示用）。"""
        lines = []
        for depth, mids in sorted(self.last_rings.items()):
            label = {RING_CENTER: "中心", RING_SECONDARY: "次级", RING_PERIPHERAL: "外围"}.get(
                depth, f"第{depth}层"
            )
            lines.append(f"{label}（层距 {depth}，能量 {self.ring_energy(depth):.2f}）：{len(mids)} 条")
        return "\n".join(lines)
