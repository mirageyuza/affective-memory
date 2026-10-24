"""单元测试：发散模式与并发联想（默认关闭 + 开启行为）。"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory import EmotionalMemorySystem, Dial


class TestDivergentConcurrent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="em_dc_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fill(self, sys_, n=8):
        for i in range(n):
            sys_.remember(
                f"测试记忆{i} 内容涵盖主题{i % 2} 标签{i}",
                cluster=f"cluster{i % 2}",
                entities=[f"主题{i % 2}", f"标签{i}"],
                affect=(0.3, 0.3),
            )

    def test_divergence_disabled_by_default(self):
        """默认关闭：recall_with_divergence 不触发发散。"""
        sys_ = EmotionalMemorySystem(base_dir=self.tmp, dial=Dial(divergent_enabled=False))
        self._fill(sys_)
        res = sys_.recall_with_divergence("主题0 标签3")
        self.assertFalse(res["engaged"])

    def test_divergence_engages_when_enabled_and_over_threshold(self):
        dial = Dial(
            divergent_enabled=True,
            divergent_threshold=0.5,
            amplitude=2.0,
            emotion_gain=1.5,
        )
        sys_ = EmotionalMemorySystem(base_dir=self.tmp, dial=dial)
        self._fill(sys_)
        res = sys_.recall_with_divergence("主题0 标签3")
        self.assertTrue(res["engaged"])
        self.assertGreaterEqual(len(res["hits"]), 1)

    def test_concurrent_disabled_by_default(self):
        sys_ = EmotionalMemorySystem(base_dir=self.tmp, dial=Dial(concurrent_enabled=False))
        self._fill(sys_)
        hits = sys_.recall_concurrent("主题1")
        self.assertEqual(len(hits), sys_.dial.top_k)  # 单路退化

    def test_concurrent_map_reduce_merges(self):
        dial = Dial(concurrent_enabled=True, concurrent_workers=2)
        sys_ = EmotionalMemorySystem(base_dir=self.tmp, dial=dial)
        self._fill(sys_)
        hits = sys_.recall_concurrent("主题1 标签5", top_k=4)
        self.assertLessEqual(len(hits), 4)
        self.assertGreaterEqual(len(hits), 1)
        # 汇聚结果按 score 降序
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
