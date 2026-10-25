"""单元测试：湍流联想（v0.5 物理四定律第四定律）。

覆盖：涡旋式扩散（中心 → 次级 → 外围逐层扩散，能量随层距递减）；
涡流链防回旋（每层去重）；发散模式协同（扩散更深、衰减放宽）；
联想只读不改记忆；人类可读摘要。
"""
from __future__ import annotations

import unittest

from emotion_memory.models import Memory
from emotion_memory.store.graph_index import EntityGraph
from emotion_memory.turbulent_association import (
    RING_CENTER,
    RING_SECONDARY,
    TurbulenceAssociator,
)

# 实体链：a1(alpha) → a2(alpha,beta) → b(beta,gamma) → c(gamma)
# weak_edges 按"相关实体下其他记忆"扩散，天然形成涡流链。
CHAIN = [
    ("a1", ["alpha"]),
    ("a2", ["alpha", "beta"]),
    ("b", ["beta", "gamma"]),
    ("c", ["gamma"]),
]


def build_graph(chain):
    graph = EntityGraph()
    memories = {}
    for mid, ents in chain:
        m = Memory(memory_id=mid, text=f"记忆{mid}", original_text=f"记忆{mid}", entities=ents)
        memories[mid] = m
        graph.add_memory(m)
    return graph, memories


class TestVortexRings(unittest.TestCase):
    def setUp(self):
        self.graph, self.memories = build_graph(CHAIN)

    def test_rings_layered(self):
        """涡流链分层：中心 → 次级 → 外围逐层扩散。"""
        ta = TurbulenceAssociator(decay=0.6, max_depth=3)
        rings = ta.vortex_rings(self.graph, center_ids=["a1"])
        self.assertIn(RING_CENTER, rings)
        self.assertEqual(rings[RING_CENTER], ["a1"])
        # 第 1 层 = 与中心共现实体 alpha 的 a2
        self.assertIn("a2", rings[1])
        # 第 2 层 = a2 扩散出的 b；第 3 层 = b 扩散出的 c
        self.assertIn("b", rings[2])
        self.assertIn("c", rings[3])

    def test_no_self_recurrence(self):
        """防回旋：任一记忆不在多个不同层重复出现。"""
        ta = TurbulenceAssociator(decay=0.6, max_depth=4)
        rings = ta.vortex_rings(self.graph, center_ids=["a1"])
        seen = []
        for depth, mids in rings.items():
            for mid in mids:
                self.assertNotIn(mid, seen)
                seen.append(mid)

    def test_ring_energy_decays_with_depth(self):
        """能量随层距递减：中心 1.0 > 次级 decay > 外围 decay^2。"""
        ta = TurbulenceAssociator(decay=0.6)
        self.assertEqual(ta.ring_energy(0), 1.0)
        self.assertGreater(ta.ring_energy(1), ta.ring_energy(2))
        self.assertAlmostEqual(ta.ring_energy(2), 0.36, places=6)


class TestAssociate(unittest.TestCase):
    def setUp(self):
        self.graph, self.memories = build_graph(CHAIN[:3])  # a1/a2/b

    def test_center_energy_highest(self):
        """中心记忆能量最高，外围递减。"""
        ta = TurbulenceAssociator(decay=0.6, max_depth=2)
        res = ta.associate(self.graph, self.memories, center_ids=["a1"], top_k=10)
        center_hits = [h for h in res["hits"] if h[2] == RING_CENTER]
        outer_hits = [h for h in res["hits"] if h[2] != RING_CENTER]
        self.assertTrue(center_hits)
        self.assertTrue(outer_hits)
        self.assertGreaterEqual(center_hits[0][1], outer_hits[0][1])

    def test_divergent_deeper_and_softer_decay(self):
        """发散模式协同：扩散更深、衰减放宽（外围能量更高）。"""
        ta = TurbulenceAssociator(decay=0.6, max_depth=1)
        normal = ta.associate(self.graph, self.memories, center_ids=["a1"], top_k=10, divergent=False)
        div = ta.associate(self.graph, self.memories, center_ids=["a1"], top_k=10, divergent=True)
        self.assertGreaterEqual(div["max_depth"], normal["max_depth"])
        self.assertGreaterEqual(div["decay"], normal["decay"])

    def test_associate_read_only(self):
        """联想只读：不修改任何记忆内容。"""
        ta = TurbulenceAssociator(decay=0.6)
        snapshot = {mid: (m.text, m.original_text, tuple(m.entities)) for mid, m in self.memories.items()}
        ta.associate(self.graph, self.memories, center_ids=["a1"], top_k=5)
        after = {mid: (m.text, m.original_text, tuple(m.entities)) for mid, m in self.memories.items()}
        self.assertEqual(snapshot, after)

    def test_empty_center_returns_empty(self):
        """空中心返回空联想（不抛错）。"""
        ta = TurbulenceAssociator(decay=0.6)
        res = ta.associate(self.graph, self.memories, center_ids=[], top_k=5)
        self.assertEqual(res["hits"], [])


class TestDescribe(unittest.TestCase):
    def setUp(self):
        self.graph, self.memories = build_graph(CHAIN)

    def test_describe_rings_readable(self):
        """人类可读涡流链摘要（demo 展示用）。"""
        ta = TurbulenceAssociator(decay=0.6, max_depth=1)
        ta.vortex_rings(self.graph, center_ids=["a1"])
        text = ta.describe_rings()
        self.assertIn("中心", text)
        self.assertIn("能量", text)


if __name__ == "__main__":
    unittest.main()
