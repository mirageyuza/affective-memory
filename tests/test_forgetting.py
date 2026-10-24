"""单元测试：遗忘曲线（衰减与强化）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory.algorithms.forgetting import compute_accessibility, ForgettingCurve
from emotion_memory.models import Memory


class TestForgettingCurve(unittest.TestCase):
    def test_decay_monotonic(self):
        """时间越长，可及性越低（单调递减）。"""
        s0 = compute_accessibility(1.0, dt_hours=0, access_count=0)
        s1 = compute_accessibility(1.0, dt_hours=24, access_count=0)
        s2 = compute_accessibility(1.0, dt_hours=720, access_count=0)
        self.assertGreater(s0, s1)
        self.assertGreater(s1, s2)

    def test_never_zero(self):
        """永久不忘：可及性永远大于 0。"""
        s = compute_accessibility(1.0, dt_hours=1e6, access_count=0)
        self.assertGreater(s, 0.0)

    def test_reinforcement(self):
        """高频访问被强化：同时间下访问次数越多可及性越高。"""
        s0 = compute_accessibility(1.0, dt_hours=168, access_count=0)
        s5 = compute_accessibility(1.0, dt_hours=168, access_count=5)
        s20 = compute_accessibility(1.0, dt_hours=168, access_count=20)
        self.assertGreater(s5, s0)
        self.assertGreater(s20, s5)

    def test_touch_raises_accessibility(self):
        """触达强化立即提升可及性。"""
        curve = ForgettingCurve()
        m = Memory(text="x", created_at=0.0, base_strength=1.0)
        curve.advance(m, now=1e5)          # 已老化
        before = m.accessibility
        curve.touch(m, now=1e5 + 1)        # 触达
        self.assertGreater(m.accessibility, before)
        self.assertEqual(m.access_count, 1)


if __name__ == "__main__":
    unittest.main()
