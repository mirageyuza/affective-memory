"""情感污染双层机制单测：原文永久存档 + 污染视图 + 审计回溯。"""
from __future__ import annotations

import tempfile
import unittest

from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.models import Memory
from emotion_memory.algorithms.pollution import EmotionPollutionFilter


class TestPollutionFilter(unittest.TestCase):
    def setUp(self):
        self.f = EmotionPollutionFilter()
        self.mem = Memory(text="项目失败了，我很失望", original_text="项目失败了，我很失望")

    def test_beautify_replaces_negative(self):
        out, audit = self.f.pollute(self.mem, mode="beautify")
        self.assertIn("宝贵的教训", out)
        self.assertNotIn("失败", out.replace("失败过", ""))
        self.assertEqual(audit["mode"], "beautify")
        self.assertEqual(audit["original_hash"], audit["original_hash"])
        self.assertNotEqual(audit["original_hash"], audit["polluted_hash"])

    def test_uglify_replaces_positive(self):
        m2 = Memory(text="今天真开心，太棒了", original_text="今天真开心，太棒了")
        out, _ = self.f.pollute(m2, mode="uglify")
        self.assertTrue(("没什么特别感觉" in out) or ("普通" in out))

    def test_original_never_modified_by_filter(self):
        original = self.mem.original_text
        self.f.pollute(self.mem, mode="uglify")
        self.assertEqual(self.mem.original_text, original)
        # clean 只改视图，原文不变
        self.f.clean(self.mem)
        self.assertEqual(self.mem.original_text, original)

    def test_clean_restores_and_keeps_history(self):
        self.mem.pollution_history.append({"at": 1, "mode": "beautify"})
        self.f.clean(self.mem)
        self.assertEqual(self.mem.text, self.mem.original_text)
        self.assertFalse(self.mem.polluted)
        self.assertIsNone(self.mem.pollution_mode)
        self.assertEqual(len(self.mem.pollution_history), 1)

    def test_zero_intensity_no_table_change(self):
        out, _ = self.f.pollute(self.mem, mode="beautify", intensity=0.0)
        self.assertEqual(out, self.mem.original_text)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.f.pollute(self.mem, mode="evil")


class TestEngineDualLayer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_poll_")
        self.sys = EmotionalMemorySystem(base_dir=self.dir, seed=1)

    def test_remember_auto_annotates_and_stores_original(self):
        m = self.sys.remember("周末加班到凌晨，很崩溃")
        self.assertEqual(m.text, m.original_text)
        self.assertFalse(m.polluted)
        self.assertIsInstance(m.affect.valence, float)
        self.assertLess(m.affect.valence, 0)  # 消极记忆 valence < 0

    def test_pollution_keeps_original_audit_clean(self):
        m = self.sys.remember("提交的方案被驳回，很失望")
        mid = m.memory_id
        r = self.sys.apply_pollution(mid, mode="beautify", intensity=1.0)
        m2 = self.sys.store.get(mid)
        self.assertNotEqual(r["polluted_text"], m2.original_text)
        self.assertEqual(m2.original_text, m.original_text)  # 原文不变
        self.assertTrue(m2.polluted)
        self.assertEqual(m2.pollution_mode, "beautify")
        self.assertEqual(len(m2.pollution_history), 1)
        # 净化回原文，审计保留
        c = self.sys.clear_pollution(mid)
        m3 = self.sys.store.get(mid)
        self.assertEqual(m3.text, m3.original_text)
        self.assertFalse(m3.polluted)
        self.assertIsNone(m3.pollution_mode)
        self.assertEqual(len(m3.pollution_history), 1)
        self.assertEqual(c["restored_text"], m.original_text)

    def test_recall_view_switch(self):
        m = self.sys.remember("和搭档因为分工吵了一架，心里堵得慌")
        mid = m.memory_id
        self.sys.apply_pollution(mid, mode="beautify", intensity=1.0)
        polluted_hits = self.sys.recall("吵架 分工", top_k=5, view="polluted")
        original_hits = self.sys.recall("吵架 分工", top_k=5, view="original")
        self.assertTrue(any(h.memory.memory_id == mid for h in polluted_hits))
        self.assertTrue(any(h.memory.memory_id == mid for h in original_hits))
        # 默认输出污染版本；view=original 输出原文
        self.assertEqual(
            next(h for h in polluted_hits if h.memory.memory_id == mid).display_text,
            self.sys.store.get(mid).text,
        )
        self.assertEqual(
            next(h for h in original_hits if h.memory.memory_id == mid).display_text,
            m.original_text,
        )

    def test_pollution_does_not_break_recall(self):
        # 索引基于原文：污染前后同查询都应命中
        m = self.sys.remember("去年生日和三个老同学爬山，很开心")
        mid = m.memory_id
        before = self.sys.recall("爬山 生日", top_k=5)
        self.sys.apply_pollution(mid, mode="uglify", intensity=1.0)
        after = self.sys.recall("爬山 生日", top_k=5)
        self.assertTrue(any(h.memory.memory_id == mid for h in before))
        self.assertTrue(any(h.memory.memory_id == mid for h in after))

    def test_view_memory_report(self):
        m = self.sys.remember("房租到期被要求立刻搬走，很狼狈")
        self.sys.apply_pollution(m.memory_id, mode="beautify")
        view = self.sys.view_memory(m.memory_id)
        self.assertEqual(view["original_text"], m.original_text)
        self.assertTrue(view["polluted"])
        self.assertEqual(len(view["pollution_history"]), 1)


class TestStorageGuard(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_guard_")
        self.sys = EmotionalMemorySystem(base_dir=self.dir)

    def test_update_meta_forbids_original(self):
        m = self.sys.remember("硬盘损坏，毕业论文没备份，很绝望")
        with self.assertRaises(ValueError):
            self.sys.store.update_meta(m.memory_id, original_text="被篡改")

    def test_set_pollution_view_never_touches_original(self):
        m = self.sys.remember("航班延误六小时，在机场干等")
        self.sys.store.set_pollution_view(
            m.memory_id, text="航班延误六小时，在机场干等", polluted=True,
            pollution_mode="beautify", pollution_history=[],
        )
        m2 = self.sys.store.get(m.memory_id)
        self.assertEqual(m2.original_text, m.original_text)


if __name__ == "__main__":
    unittest.main()
