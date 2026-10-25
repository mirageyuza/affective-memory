"""单元测试：情感守恒律（v0.5 物理四定律第一定律）。

覆盖：无源情感注入一律拒绝（strict 抛错 / lenient 标记）；来源有效接受；
情感总量预算（超预算自然衰减，防情绪通货膨胀）；污染 = 情感重分配
（强度超原文能量被钳制，禁止无中生有）；审计统计。
"""
from __future__ import annotations

import unittest

from emotion_memory.conservation import (
    ConservationError,
    EmotionConservation,
    emotion_energy,
)
from emotion_memory.models import AffectLabel, Memory


def make_memory(
    memory_id: str = "m1",
    affect: AffectLabel = AffectLabel(0.8, 0.6),
) -> Memory:
    return Memory(
        memory_id=memory_id,
        text="原文",
        original_text="原文",
        affect=affect,
    )


class TestEmotionEnergy(unittest.TestCase):
    def test_energy_formula(self):
        """情感能量 = 效价强度×0.6 + 唤醒度×0.4。"""
        self.assertAlmostEqual(emotion_energy((1.0, 0.5)), 0.8, places=6)
        self.assertAlmostEqual(emotion_energy((-1.0, 0.0)), 0.6, places=6)
        self.assertEqual(emotion_energy(None), 0.0)

    def test_energy_clamped(self):
        """能量折算被钳制到 [0,1]，永不越界。"""
        self.assertAlmostEqual(emotion_energy((2.0, 1.0)), 1.0, places=6)
        self.assertAlmostEqual(emotion_energy((0.0, 0.0)), 0.0, places=6)

    def test_energy_accepts_affect_label(self):
        """Accept AffectLabel 对象。"""
        self.assertAlmostEqual(emotion_energy(AffectLabel(0.5, 1.0)), 0.7, places=6)


class TestConservationSource(unittest.TestCase):
    def test_no_source_strict_raises(self):
        """无源情感注入一律拒绝：strict=True 抛 ConservationError。"""
        cons = EmotionConservation(budget=100.0)
        with self.assertRaises(ConservationError):
            cons.check_annotation("新记忆", (0.8, 0.6), source_text=None, strict=True)
        with self.assertRaises(ConservationError):
            cons.check_annotation("新记忆", (0.8, 0.6), source_text="  ", strict=True)

    def test_with_source_accepted(self):
        """携带事件来源原文的标注被接受，并写入审计。"""
        cons = EmotionConservation(budget=100.0)
        rec = cons.check_annotation("新记忆", (0.8, 0.6), source_text="用户说：我很难过", strict=True)
        self.assertTrue(rec["accepted"])
        self.assertEqual(cons.audit[0]["reason"], "ok")

    def test_no_source_lenient_marks(self):
        """strict=False 时无源不抛错但标记拒绝原因（兼容关闭守恒模式）。"""
        cons = EmotionConservation(budget=100.0)
        rec = cons.check_annotation("新记忆", (0.8, 0.6), source_text=None, strict=False)
        self.assertFalse(rec["accepted"])
        self.assertEqual(rec["reason"], "no_source_but_lenient")


class TestConservationBudget(unittest.TestCase):
    def test_within_budget_no_attenuation(self):
        """预算内注入：不衰减（scale=1）。"""
        cons = EmotionConservation(budget=10.0)
        res = cons.cap_energy(energy=2.0, total=3.0)
        self.assertEqual(res["scale"], 1.0)
        self.assertFalse(res["over"])

    def test_over_budget_attenuates(self):
        """超预算：按比例自然衰减（scale<1），不删除记忆。"""
        cons = EmotionConservation(budget=10.0)
        res = cons.cap_energy(energy=8.0, total=9.0)
        self.assertTrue(res["over"])
        self.assertLess(res["scale"], 1.0)
        self.assertAlmostEqual(res["energy"] + 9.0, 10.0, places=6)

    def test_adjust_memory_intensity_audits(self):
        """引擎入口：adjust_memory_intensity 返回 scale 并记录预算审计。"""
        cons = EmotionConservation(budget=1.0)
        probe = make_memory("probe", AffectLabel(1.0, 1.0))
        existing = [make_memory("old", AffectLabel(0.8, 0.8))]
        res = cons.adjust_memory_intensity(probe, existing, affect=(1.0, 1.0))
        self.assertTrue(res["over"])
        self.assertLess(res["scale"], 1.0)
        self.assertTrue(any(r["reason"] == "budget_attenuated" for r in cons.audit))

    def test_total_energy(self):
        """total_energy = 全库情感能量之和。"""
        cons = EmotionConservation(budget=100.0)
        mems = [
            make_memory("a", AffectLabel(1.0, 0.5)),
            make_memory("b", AffectLabel(-0.5, 0.0)),
        ]
        self.assertAlmostEqual(cons.total_energy(mems), 0.8 + 0.3, places=6)


class TestConservationPollution(unittest.TestCase):
    def test_pollution_within_limit_allowed(self):
        """污染强度不超原文能量上限：允许（不钳制）。"""
        cons = EmotionConservation(budget=100.0)
        mem = make_memory(affect=AffectLabel(0.8, 0.6))  # energy=0.72
        res = cons.validate_pollution(mem, intensity=0.5)
        self.assertTrue(res["allowed"])
        self.assertAlmostEqual(res["clamped"], 0.5, places=6)

    def test_pollution_over_limit_clamped(self):
        """污染强度超原文情感能量：钳制到上限（禁止无中生有）。"""
        cons = EmotionConservation(budget=100.0)
        mem = make_memory(affect=AffectLabel(0.2, 0.1))  # energy=0.16
        res = cons.validate_pollution(mem, intensity=0.9)
        self.assertFalse(res["allowed"])
        self.assertLess(res["clamped"], 0.9)
        self.assertAlmostEqual(res["clamped"], 0.16 * 1.2, places=6)

    def test_audit_report_counts(self):
        """审计报告统计：拒绝 / 衰减 / 钳制计数。"""
        cons = EmotionConservation(budget=1.0)
        with self.assertRaises(ConservationError):
            cons.check_annotation("a", (0.5, 0.5), None, strict=True)
        cons.check_annotation("b", (0.5, 0.5), "src", strict=True)
        probe = make_memory("p", AffectLabel(1.0, 1.0))
        cons.adjust_memory_intensity(probe, [make_memory("o", AffectLabel(1.0, 1.0))], affect=(1.0, 1.0))
        cons.validate_pollution(make_memory(affect=AffectLabel(0.1, 0.0)), intensity=1.0)
        rep = cons.audit_report()
        self.assertGreaterEqual(rep["rejected_no_source"], 1)
        self.assertGreaterEqual(rep["budget_attenuated"], 1)
        self.assertGreaterEqual(rep["pollution_clamped"], 1)


if __name__ == "__main__":
    unittest.main()
