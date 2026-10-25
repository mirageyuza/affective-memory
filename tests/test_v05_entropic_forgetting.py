"""单元测试：熵增遗忘（v0.5 物理四定律第三定律）。

覆盖：细节先模糊、情绪残留后淡（detail 衰减快于 residue）；熵随推进增大；
熵越高检索置信度越低（仅降权）；原文永不清除（原文红线）；高危/时间敏感
记忆不沉默（降权不删除）；批量推进与审计。
"""
from __future__ import annotations

import unittest

from emotion_memory.entropic_forgetting import EntropicForgetting
from emotion_memory.models import Memory


def make_memory(memory_id: str = "m1", due_at=None) -> Memory:
    return Memory(
        memory_id=memory_id,
        text="原文内容",
        original_text="原文内容",
        due_at=due_at,
    )


class TestEntropyAdvance(unittest.TestCase):
    def test_initial_state(self):
        """初始：细节保真与情绪残余均为 1，熵为 0。"""
        mem = make_memory()
        self.assertEqual(mem.detail_fidelity, 1.0)
        self.assertEqual(mem.emotional_residue, 1.0)
        self.assertEqual(mem.entropy, 0.0)

    def test_detail_decays_faster_than_residue(self):
        """细节先模糊：detail 衰减快于 emotional_residue。"""
        ef = EntropicForgetting(rate=0.05)
        mem = make_memory()
        ef.advance(mem, dt_hours=48.0)
        self.assertLess(mem.detail_fidelity, mem.emotional_residue)
        self.assertLess(mem.detail_fidelity, 1.0)
        self.assertLess(mem.emotional_residue, 1.0)

    def test_entropy_increases(self):
        """熵随时间增大（记忆有序 → 无序）。"""
        ef = EntropicForgetting(rate=0.05)
        mem = make_memory()
        ef.advance(mem, dt_hours=10.0)
        e1 = mem.entropy
        ef.advance(mem, dt_hours=10.0)
        e2 = mem.entropy
        self.assertGreater(e2, e1)

    def test_original_text_never_touched(self):
        """原文永不清除：advance 只改元数据，original_text 不变。"""
        ef = EntropicForgetting(rate=0.05)
        mem = make_memory()
        ef.advance(mem, dt_hours=1000.0)
        self.assertEqual(mem.original_text, "原文内容")
        self.assertEqual(mem.text, "原文内容")

    def test_entropy_bounded(self):
        """熵永远被钳制在 [0,1]。"""
        ef = EntropicForgetting(rate=0.05)
        mem = make_memory()
        for _ in range(10):
            ef.advance(mem, dt_hours=1000.0)
        self.assertGreaterEqual(mem.entropy, 0.0)
        self.assertLessEqual(mem.entropy, 1.0)


class TestConfidence(unittest.TestCase):
    def test_confidence_inverse_of_entropy(self):
        """检索置信度 = 1 - 熵（熵越高置信度越低）。"""
        low = make_memory("low")
        low.entropy = 0.2
        high = make_memory("high")
        high.entropy = 0.8
        self.assertGreater(EntropicForgetting.confidence(low), EntropicForgetting.confidence(high))
        self.assertAlmostEqual(EntropicForgetting.confidence(high), 0.2, places=6)

    def test_confidence_clamped(self):
        """置信度钳制在 [0,1]。"""
        mem = make_memory()
        mem.entropy = 1.5
        self.assertEqual(EntropicForgetting.confidence(mem), 0.0)


class TestSafety(unittest.TestCase):
    def test_due_at_memory_not_silenced(self):
        """高危/时间敏感记忆（due_at 非空）：推进熵但仅降权，不删除、不沉默。"""
        ef = EntropicForgetting(rate=0.05)
        mem = make_memory("urgent", due_at=1234567890.0)
        rec = ef.advance(mem, dt_hours=200.0)
        self.assertTrue(rec["protected"])
        self.assertGreater(mem.entropy, 0.0)  # 熵仍推进
        self.assertEqual(mem.original_text, "原文内容")  # 内容仍在
        self.assertGreater(ef.confidence(mem), 0.0)  # 只是降权

    def test_decay_all_statistics(self):
        """全库批量推进返回统计与明细。"""
        ef = EntropicForgetting(rate=0.05)
        mems = [make_memory(f"m{i}") for i in range(3)]
        res = ef.decay_all(mems, dt_hours=24.0)
        self.assertEqual(res["advanced"], 3)
        self.assertEqual(len(res["details"]), 3)
        self.assertAlmostEqual(res["average_entropy"], sum(m.entropy for m in mems) / 3, places=4)


class TestAudit(unittest.TestCase):
    def test_advance_logged(self):
        """每次熵推进写入确定性审计日志。"""
        ef = EntropicForgetting(rate=0.05)
        mem = make_memory("logme")
        ef.advance(mem, dt_hours=12.0)
        self.assertEqual(len(ef.log), 1)
        self.assertEqual(ef.log[0]["memory_id"], "logme")
        rep = ef.report()
        self.assertEqual(rep["log_count"], 1)


if __name__ == "__main__":
    unittest.main()
