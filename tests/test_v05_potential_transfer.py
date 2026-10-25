"""单元测试：势差驱动迁移（v0.5 物理四定律第二定律）。

覆盖：势能公式（情感强度 × 重复次数 × 重要性加权）；高势能沉淀长期区、
低势能自然流散；触达次数与重要性提升势能；电流桥按势差非均匀分配迁移量
（max_count / skip_ratio）；决策审计。
"""
from __future__ import annotations

import unittest

from emotion_memory.models import AffectLabel, Memory
from emotion_memory.potential_transfer import (
    REGION_LONGTERM,
    REGION_WORKSPACE,
    PotentialTransfer,
)


def make_memory(
    memory_id: str,
    affect: AffectLabel = AffectLabel(0.8, 0.6),
    access_count: int = 0,
    importance: float = 1.0,
    region: str = REGION_WORKSPACE,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        text="原文",
        original_text="原文",
        affect=affect,
        access_count=access_count,
        importance=importance,
        region=region,
    )


class TestPotentialFormula(unittest.TestCase):
    def test_potential_base(self):
        """基础势能 = 情感能量 × 重复强化(1) × 重要性(1)。"""
        pt = PotentialTransfer(threshold=0.6, importance_weight=0.4, repeat_weight=0.3)
        mem = make_memory("a", AffectLabel(0.8, 0.6))  # energy=0.72
        self.assertAlmostEqual(pt.potential(mem), 0.72, places=6)

    def test_potential_increases_with_touches(self):
        """触达次数越多势能越高（重复强化）。"""
        pt = PotentialTransfer(threshold=0.6, importance_weight=0.4, repeat_weight=0.3)
        low = pt.potential(make_memory("a", AffectLabel(0.5, 0.5), access_count=0))
        high = pt.potential(make_memory("b", AffectLabel(0.5, 0.5), access_count=20))
        self.assertGreater(high, low)

    def test_potential_increases_with_importance(self):
        """重要性加权提升势能。"""
        pt = PotentialTransfer(threshold=0.6, importance_weight=0.4, repeat_weight=0.3)
        low = pt.potential(make_memory("a", AffectLabel(0.5, 0.5), importance=1.0))
        high = pt.potential(make_memory("b", AffectLabel(0.5, 0.5), importance=3.0))
        self.assertGreater(high, low)

    def test_zero_energy_never_promotes(self):
        """零情感能量记忆势能为 0，永不沉淀。"""
        pt = PotentialTransfer(threshold=0.1)
        self.assertEqual(pt.potential(make_memory("a", AffectLabel(0.0, 0.0))), 0.0)


class TestPromotionDecision(unittest.TestCase):
    def test_high_potential_promotes_longterm(self):
        """高势能 → longterm（沉淀）。"""
        pt = PotentialTransfer(threshold=0.6)
        mem = make_memory("a", AffectLabel(1.0, 1.0), importance=2.0)  # energy=1.0*2=2.0
        region, p = pt.decide(mem)
        self.assertEqual(region, REGION_LONGTERM)
        self.assertGreaterEqual(p, 0.6)

    def test_low_potential_stays_workspace(self):
        """低势能 → workspace（自然流散）。"""
        pt = PotentialTransfer(threshold=0.6)
        mem = make_memory("a", AffectLabel(0.1, 0.1))  # energy=0.1
        region, p = pt.decide(mem)
        self.assertEqual(region, REGION_WORKSPACE)
        self.assertLess(p, 0.6)

    def test_promote_moves_region(self):
        """promote 只改 region 元数据，不改原文。"""
        pt = PotentialTransfer(threshold=0.5)
        mem = make_memory("a", AffectLabel(0.9, 0.9), importance=1.0)
        res = pt.promote(mem)
        self.assertTrue(res["promoted"])
        self.assertEqual(mem.region, REGION_LONGTERM)
        self.assertEqual(mem.original_text, "原文")

    def test_promote_many_statistics(self):
        """批量沉淀统计：高势能沉淀数 / 低势能流散数。"""
        pt = PotentialTransfer(threshold=0.5)
        mems = [
            make_memory("hi", AffectLabel(1.0, 1.0), importance=1.0),
            make_memory("lo", AffectLabel(0.1, 0.0)),
        ]
        res = pt.promote_many(mems)
        self.assertEqual(res["scanned"], 2)
        self.assertEqual(res["promoted_count"], 1)
        self.assertEqual(res["flowed_away_count"], 1)


class TestTransferQuota(unittest.TestCase):
    def test_quota_selects_high_potential(self):
        """max_count 限定时只搬势能最高的前 N 条（非均匀搬运）。"""
        pt = PotentialTransfer(threshold=0.0)
        mems = [
            make_memory("weak", AffectLabel(0.1, 0.0)),
            make_memory("strong", AffectLabel(1.0, 1.0)),
            make_memory("mid", AffectLabel(0.5, 0.5)),
        ]
        quota = pt.allocate_transfer_quota(mems, max_count=2)
        self.assertEqual(quota["selected_count"], 2)
        self.assertEqual(quota["skipped_count"], 1)
        self.assertIn("strong", [m.memory_id for m in quota["selected"]])
        self.assertNotIn("weak", [m.memory_id for m in quota["selected"]])

    def test_quota_skip_ratio(self):
        """skip_ratio>0 时低势能部分按比例跳过（迁移量随势差缩小）。"""
        pt = PotentialTransfer(threshold=0.0)
        mems = [
            make_memory(f"m{i}", AffectLabel(0.3 + 0.05 * i, 0.0)) for i in range(6)
        ]
        quota = pt.allocate_transfer_quota(mems, max_count=3, skip_ratio=0.5)
        self.assertEqual(quota["selected_count"] + quota["skipped_count"], 6)
        self.assertGreater(quota["skipped_count"], 0)

    def test_quota_default_all_selected(self):
        """无限制时全部入选（默认全量迁移）。"""
        pt = PotentialTransfer(threshold=0.0)
        mems = [make_memory(f"m{i}") for i in range(4)]
        quota = pt.allocate_transfer_quota(mems)
        self.assertEqual(quota["selected_count"], 4)
        self.assertEqual(quota["skipped_count"], 0)


class TestAudit(unittest.TestCase):
    def test_report_tracks_decisions(self):
        """决策审计可追溯（确定性）。"""
        pt = PotentialTransfer(threshold=0.5)
        pt.decide(make_memory("hi", AffectLabel(1.0, 1.0)))
        pt.decide(make_memory("lo", AffectLabel(0.1, 0.0)))
        rep = pt.report()
        self.assertEqual(rep["promoted_count"], 1)
        self.assertEqual(len(rep["decisions"]), 2)


if __name__ == "__main__":
    unittest.main()
