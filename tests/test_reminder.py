"""时间敏感提醒单测：到期注入 / 确认去重 / 高危永不沉默。"""
from __future__ import annotations

import tempfile
import time
import unittest

from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.models import RankedHit


def _mk_hit(memory):
    h = RankedHit(memory=memory, display_text=memory.text)
    return h


class TestDueReminders(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_due_")
        self.sys = EmotionalMemorySystem(base_dir=self.dir)
        self.past = time.time() - 10
        self.future = time.time() + 3600

    def test_due_reminders_only_return_expired_unacked(self):
        self.sys.remember("明天上午十点交房租，别忘了", due_at=self.past)
        self.sys.remember("三个月后要体检复查", due_at=self.future)
        due = self.sys.due_reminders()
        self.assertEqual(len(due), 1)
        self.assertIn("交房租", due[0].original_text)

    def test_ack_stops_repeat(self):
        m = self.sys.remember("药品需在晚上八点前服用", due_at=self.past)
        self.assertEqual(len(self.sys.due_reminders()), 1)
        self.sys.ack_reminder(m.memory_id)
        self.assertEqual(len(self.sys.due_reminders()), 0)

    def test_context_injects_due_reminders(self):
        self.sys.remember("周三之前提交报销单", due_at=self.past)
        ctx = self.sys.context_with_reminders([], now=time.time())
        self.assertIn("到期提醒", ctx)
        self.assertIn("报销单", ctx)

    def test_silent_due_still_injected(self):
        # 沉默级 + 到期：时间敏感记忆即使沉默也注入（永不沉默规则）
        self.sys.remember("服务器证书明天过期，务必处理", due_at=self.past, silence=True)
        due = self.sys.due_reminders()
        self.assertEqual(len(due), 1)
        self.assertIn("证书", due[0].original_text)

    def test_stats_counts_due(self):
        self.sys.remember("把护照复印件发给签证中心", due_at=self.past)
        stats = self.sys.stats()
        self.assertGreaterEqual(stats["due_count"], 1)

    def test_never_silent_flag_preserved(self):
        m = self.sys.remember("高危：账密轮换任务必须完成", never_silent=True)
        self.assertTrue(m.never_silent)


if __name__ == "__main__":
    unittest.main()
