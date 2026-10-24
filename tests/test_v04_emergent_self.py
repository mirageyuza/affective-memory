"""单元测试：涌现自我引擎（v0.4）。

覆盖：空累计未涌现（无预设模板）；情感累计统计生成性格维度；
使命从高频重复文本提取；自我概念按强情感簇生成；技能/兴趣；
报告导出与唤醒重建（从累计数据重建身份，非固定档案）。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from emotion_memory.emergent_self import EmergentSelf


class TestEmergentSelf(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_v04_self_")
        self.self_m = EmergentSelf(storage=None)

    def test_empty_no_template(self):
        """无累计数据时自我未涌现，无任何预设定义。"""
        rep = self.self_m.self_report()
        self.assertFalse(rep["emerged"])
        self.assertEqual(rep["record_count"], 0)
        self.assertIn("无预设模板", rep["reason"])

    def test_accumulate_and_traits(self):
        """累计后生成五维性格倾向（纯统计产物）。"""
        self.self_m.accumulate("开心的聚会", affect={"valence": 0.8, "arousal": 0.8}, cluster="社交")
        self.self_m.accumulate("愉快的旅行", affect={"valence": 0.6, "arousal": 0.6}, cluster="旅行")
        self.self_m.refresh()
        rep = self.self_m.self_report()
        self.assertTrue(rep["emerged"])
        self.assertEqual(rep["record_count"], 2)
        for dim in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            self.assertIn(dim, rep["traits"])
        self.assertGreater(rep["traits"]["agreeableness"], 0.5)  # 正效价 → 亲和度高

    def test_mission_from_repeated_text(self):
        """使命从高频重复的使命记忆涌现（无模板预设）。"""
        for _ in range(3):
            self.self_m.accumulate(
                "我想要帮助用户记住重要的事情",
                affect={"valence": 0.5, "arousal": 0.4},
                cluster="使命",
            )
        self.self_m.accumulate("随便聊聊天", affect={"valence": 0.0, "arousal": 0.1}, cluster="闲聊")
        self.self_m.refresh()
        self.assertIn("帮助用户记住重要的事情", self.self_m.current_traits["mission"])

    def test_self_concepts_from_strong_cluster(self):
        """强情感簇 → 自我概念（计数 + 情感强度排序）。"""
        for _ in range(3):
            self.self_m.accumulate("废寝忘食地写代码", affect={"valence": 0.7, "arousal": 0.9}, cluster="编程")
        self.self_m.refresh()
        concepts = self.self_m.current_concepts
        self.assertTrue(any(c["concept"] == "编程" for c in concepts))
        pro = next(c for c in concepts if c["concept"] == "编程")
        self.assertEqual(pro["count"], 3)
        self.assertGreater(pro["intensity"], 0.5)

    def test_skills_interests(self):
        """技能来自实体共现，兴趣来自高频簇。"""
        self.self_m.accumulate("用 Python 写爬虫", affect={"valence": 0.5, "arousal": 0.6}, cluster="编程", entities=["Python", "爬虫"])
        self.self_m.accumulate("用 Python 写爬虫", affect={"valence": 0.5, "arousal": 0.6}, cluster="编程", entities=["Python", "爬虫"])
        self.self_m.refresh()
        self.assertIn("Python", self.self_m.current_traits["skills"])
        self.assertIn("编程", self.self_m.current_traits["interests"])

    def test_export_and_rebuild(self):
        """唤醒重建：身份从累计数据重建，而非固定档案。"""
        for i in range(3):
            self.self_m.accumulate("持续热爱钻研", affect={"valence": 0.6, "arousal": 0.5}, cluster="钻研")
        self.self_m.refresh()
        path = os.path.join(self.dir, "emergent_self.json")
        self.self_m.export_json(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["version"], "v0.4-emergent-self")

        fresh = EmergentSelf(storage=None)
        n = fresh.rebuild_from(path)
        self.assertEqual(n, 3)
        self.assertTrue(fresh.self_report()["emerged"])
        self.assertEqual(fresh.current_concepts, self.self_m.current_concepts)

    def test_accumulate_accepts_affect_object(self):
        """兼容 AffectLabel 对象/字典/tuple 三种情感输入。"""
        from emotion_memory.models import AffectLabel

        self.self_m.accumulate("你好", affect=AffectLabel(0.4, 0.3))
        self.self_m.accumulate("再见", affect={"valence": -0.2, "arousal": 0.1})
        self.assertEqual(len(self.self_m.records), 2)
        self.assertAlmostEqual(self.self_m.records[0]["valence"], 0.4, places=5)
        self.assertAlmostEqual(self.self_m.records[1]["valence"], -0.2, places=5)


if __name__ == "__main__":
    unittest.main()
