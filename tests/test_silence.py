"""单元测试：沉默层（可提及/沉默两级 + 解锁条件 + 永不沉默）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory.silence import SilenceLayer
from emotion_memory.models import AffectLabel, Memory, RankedHit


def make_mem(mid, silence=False, never_silent=False):
    return Memory(
        memory_id=mid, text=f"文本{mid}",
        cluster="c", entities=[], affect=AffectLabel(),
        silence=silence, never_silent=never_silent,
    )


class TestSilenceLayer(unittest.TestCase):
    def setUp(self):
        self.layer = SilenceLayer()

    def test_plain_memories_always_visible(self):
        m = make_mem("p1")
        hits = self.layer.filter([RankedHit(memory=m)])
        self.assertEqual(len(hits), 1)

    def test_silent_hidden_by_default(self):
        m = make_mem("s1", silence=True)
        hits = self.layer.filter([RankedHit(memory=m)])
        self.assertEqual(len(hits), 0)

    def test_silent_unlocked_by_direct_query(self):
        m = make_mem("s2", silence=True)
        hits = self.layer.filter([RankedHit(memory=m)], direct_query=True)
        self.assertEqual(len(hits), 1)

    def test_silent_unlocked_by_strong_relation(self):
        m = make_mem("s3", silence=True)
        hits = self.layer.filter([RankedHit(memory=m)], strong_related=True)
        self.assertEqual(len(hits), 1)

    def test_never_silent_always_injected(self):
        """高危/时间敏感记忆永不沉默。"""
        m = make_mem("n1", silence=True, never_silent=True)
        hits = self.layer.filter([RankedHit(memory=m)])
        self.assertEqual(len(hits), 1)

    def test_mixed_batch(self):
        mems = [make_mem("a"), make_mem("b", silence=True), make_mem("c", never_silent=True)]
        hits = self.layer.filter([RankedHit(memory=m) for m in mems])
        ids = {h.memory.memory_id for h in hits}
        self.assertEqual(ids, {"a", "c"})


if __name__ == "__main__":
    unittest.main()
