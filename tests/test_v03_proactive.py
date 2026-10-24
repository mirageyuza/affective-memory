"""单元测试：主动回忆引擎（定时 / 触发器驱动的自主想起搭话）。"""
from __future__ import annotations

import tempfile
import time
import unittest

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.models import Memory
from emotion_memory.retrieval.proactive import ProactiveRecallEngine


class TestProactiveRecall(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_proactive_")
        self.now = 1_700_000_000.0  # 固定时间锚点，保证确定性
        self.pro = ProactiveRecallEngine(interval_hours=6.0, top_k=3)

    def _mem(self, text):
        m = Memory(text=text)
        return m

    def test_first_time_should_initiate(self):
        """首次触发允许开口（是否开口由上层候选决定）。"""
        self.assertTrue(self.pro.should_initiate(now=self.now))

    def test_interval_respect(self):
        """间隔未满不开口。"""
        self.pro.last_proactive_at = self.now
        self.assertFalse(self.pro.should_initiate(now=self.now + 3600))
        self.assertTrue(self.pro.should_initiate(now=self.now + 6 * 3600 + 1))

    def test_build_opening_contains_greeting_and_memory(self):
        """开场话题 = 时间上下文（早中晚 + 星期）+ 记忆内容。"""
        m = self._mem("上周和老同学去了郊外露营")
        opening = self.pro.build_opening([m], now=self.now, trigger="proactive")
        self.assertIn("星期", opening)
        self.assertIn("露营", opening)

    def test_proactive_tick_no_speak_inside_interval(self):
        """间隔内即便有候选也不开口。"""
        self.pro.last_proactive_at = self.now
        res = self.pro.proactive_tick(
            [self._mem("候选记忆")], now=self.now + 60
        )
        self.assertFalse(res["should_speak"])
        self.assertEqual(res["opening"], "")

    def test_proactive_tick_speaks_with_candidates(self):
        """间隔满足 + 有候选 → 开口并计数。"""
        m = self._mem("我喜欢的城市是成都")
        res = self.pro.proactive_tick([m], now=self.now)
        self.assertTrue(res["should_speak"])
        self.assertEqual(res["trigger"], "proactive")
        self.assertIn("成都", res["opening"])
        self.assertEqual(res["count"], 1)
        # 开口后记录时间：短时间内不再开口
        again = self.pro.proactive_tick([m], now=self.now + 1)
        self.assertFalse(again["should_speak"])

    def test_reminder_trigger_priority(self):
        """到期提醒优先于普通备选池（永不沉默规则）。"""
        due = self._mem("护照明天到期，请尽快处理")
        normal = self._mem("普通闲聊记忆")
        res = self.pro.proactive_tick([normal], due_memories=[due], now=self.now)
        self.assertEqual(res["trigger"], "reminder")
        self.assertIn("护照", res["opening"])

    def test_engine_proactive_tick(self):
        """引擎集成：proactive_tick 返回可审计的开口决策。"""
        sys = EmotionalMemorySystem(
            base_dir=self.dir,
            dial=Dial(proactive_enabled=True, capacitor_enabled=True),
        )
        sys.remember("我喜欢周末去图书馆")
        res = sys.proactive_tick(now=1_700_000_000.0)
        self.assertIn("should_speak", res)
        self.assertIn("opening", res)
        self.assertIn("candidates", res)


if __name__ == "__main__":
    unittest.main()
