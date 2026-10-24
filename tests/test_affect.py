"""单元测试：情感增益与 dB 权衡排序（只影响排序，不改原文）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory.config import Dial
from emotion_memory.algorithms.affect import to_db, rank_hits, affect_gain, REFERENCE
from emotion_memory.models import AffectLabel, Memory


class TestAffectAndDb(unittest.TestCase):
    def test_to_db_monotonic(self):
        """dB 刻度单调：强度越大 dB 越大。"""
        self.assertLess(to_db(1e-5), to_db(1e-3))
        self.assertLess(to_db(1e-3), to_db(1.0))
        self.assertEqual(to_db(0.0), -120.0)

    def test_to_db_logarithmic(self):
        """韦伯-费希纳对数律：每 10 倍强度 ≈ +20dB。"""
        self.assertAlmostEqual(to_db(10 * REFERENCE), to_db(REFERENCE) + 20.0, places=6)

    def test_affect_gain_range(self):
        """情感增益受刻度盘 emotion_gain 调制，且落在温和区间。"""
        dial = Dial(emotion_gain=0.3)
        g_flat = affect_gain(AffectLabel(0.0, 0.0), dial)
        g_intense = affect_gain(AffectLabel(1.0, 1.0), dial)
        self.assertGreaterEqual(g_flat, 0.5)
        self.assertGreater(g_intense, g_flat)
        self.assertLessEqual(g_intense, 2.0)

    def test_rank_never_modifies_text(self):
        """排序永不修改记忆原文（核心红线）。"""
        dial = Dial(emotion_gain=1.5)
        m1 = Memory(text="A 原文", affect=AffectLabel(0.1, 0.1))
        m2 = Memory(text="B 原文", affect=AffectLabel(1.0, 1.0))
        text_before = (m1.text, m2.text)
        ranked = rank_hits([(m1, 0.5), (m2, 0.5)], dial)
        self.assertEqual((m1.text, m2.text), text_before)
        self.assertEqual(len(ranked), 2)
        # 高情感记忆应排前
        self.assertEqual(ranked[0].memory.memory_id, m2.memory_id)

    def test_rank_sorts_by_score(self):
        """排序结果按 score 降序。"""
        dial = Dial()
        ms = [Memory(text=str(i), affect=AffectLabel(0.5, 0.5), accessibility=(i + 1) / 10) for i in range(5)]
        ranked = rank_hits([(m, 1.0) for m in ms], dial)
        scores = [r.score for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
