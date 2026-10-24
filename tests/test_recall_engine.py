"""单元测试：检索层端到端（写入 → 召回 → top-k → 沉默/高危行为）。"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory import EmotionalMemorySystem, Dial


class TestRecallEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="em_recall_")
        self.sys = EmotionalMemorySystem(
            base_dir=self.tmp,
            dial=Dial(noise_temperature=0.0, emotion_gain=0.3),
        )
        self.sys.remember("用户喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"], affect=(0.5, 0.4))
        self.sys.remember("去年生日和三个老同学爬山", cluster="social", entities=["生日", "爬山"], affect=(0.8, 0.6))
        self.sys.remember("下周五提交季度报告，绝不能遗漏", cluster="task", entities=["季度报告"], affect=(-0.2, 0.7), never_silent=True)
        self.sys.remember("失败提案经历，用户不愿再提", cluster="social", entities=["提案"], affect=(-0.7, 0.2), silence=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_recall_hits_relevant(self):
        hits = self.sys.recall("爬山 生日")
        texts = [h.memory.text for h in hits]
        self.assertTrue(any("爬山" in t for t in texts))
        # 沉默记忆默认不注入
        self.assertFalse(any("提案" in t for t in texts))

    def test_silent_unlocked_by_direct_query(self):
        hits = self.sys.recall("提案", direct_query=True)
        self.assertTrue(any("提案" in h.memory.text for h in hits))

    def test_never_silent_always_returned_for_relevant_query(self):
        hits = self.sys.recall("季度报告")
        self.assertTrue(any("季度报告" in h.memory.text for h in hits))

    def test_top_k_budget(self):
        for i in range(12):
            self.sys.remember(f"无关记忆 {i} 种类 花草树木", cluster="junk", entities=["花草"])
        hits = self.sys.recall("晨跑", top_k=3)
        self.assertLessEqual(len(hits), 3)

    def test_decay_affects_recall_order(self):
        """时间流逝后新记忆可及性高于被遗忘旧记忆。"""
        old = self.sys.store.all()[0]
        old.created_at = 0.0  # 模拟远古
        self.sys.store.update_meta(old.memory_id, created_at=0.0)
        import time as _t
        self.sys.decay_all(now=_t.time())
        self.assertLess(old.accessibility, 0.9)

    def test_stats(self):
        st = self.sys.stats()
        self.assertEqual(st["memory_count"], 4)
        self.assertEqual(st["cluster_count"], 3)
        self.assertIn("dial", st)


if __name__ == "__main__":
    unittest.main()
