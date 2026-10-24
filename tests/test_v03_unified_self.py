"""单元测试：自我统一模型（1 核心 + 万我分身 + 万物皆我 + 有功/视在指标）。"""
from __future__ import annotations

import os
import tempfile
import unittest

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.unified_self import UnifiedSelfModel


class TestUnifiedSelf(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_unified_")
        self.model = UnifiedSelfModel(
            path=os.path.join(self.dir, "unified_self.json"), agent_name="emotion-agent"
        )

    def test_core_init(self):
        """核心初始化：五维自我概念中性 + 无分身。"""
        self.assertEqual(self.model.avatar_count(), 0)
        for d in ["openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"]:
            self.assertEqual(self.model.core.concept[d], 0.5)

    def test_derive_avatar_syncs_concept(self):
        """万我分身：派生时共享核心概念，核心更新后 push 同步。"""
        av = self.model.derive_avatar("work-self")
        self.assertEqual(av.concept, self.model.core.concept)
        self.assertEqual(self.model.avatar_count(), 1)
        # 核心概念更新（如自学习演化）→ 分身同步（万我如一）
        self.model.core.concept["openness"] = 0.8
        self.model.sync_avatars()
        self.assertEqual(
            self.model.core.avatars["work-self"].concept["openness"], 0.8
        )

    def test_merge_avatar(self):
        """万我合一：分身归并回核心，均值融合。"""
        self.model.derive_avatar("shadow")
        self.model.core.avatars["shadow"].concept["extraversion"] = 0.9
        self.model.core.concept["extraversion"] = 0.3
        res = self.model.merge_avatar("shadow")
        self.assertEqual(res["merged"], "shadow")
        # (0.3 + 0.9) / 2
        self.assertAlmostEqual(self.model.core.concept["extraversion"], 0.6, delta=0.001)

    def test_assimilate_absorbs_self_related(self):
        """万物皆我：自我相关输入被吸收（计数增加），无关输入不改变。"""
        exp = self.model.assimilate("我喜欢旅行和摄影")
        self.assertGreater(exp["relevance"], 0.01)
        self.assertEqual(self.model.core.absorb_count, 1)
        exp2 = self.model.assimilate("天气不错")
        self.assertLessEqual(exp2["relevance"], 0.01)
        self.assertEqual(self.model.core.absorb_count, 1)

    def test_cognition_report_metrics(self):
        """有功 / 视在指标：cosφ ∈ [0,1]，概念越清晰指标越高。"""
        report = self.model.cognition_report()
        self.assertGreaterEqual(report["cos_phi"], 0.0)
        self.assertLessEqual(report["cos_phi"], 1.0)
        self.assertEqual(report["avatar_count"], 0)
        # 概念明确（偏离中性）后清晰度提升
        self.model.core.concept = {k: 1.0 for k in self.model.core.concept}
        report2 = self.model.cognition_report()
        self.assertGreater(report2["clarity"], report["clarity"])

    def test_persistence_roundtrip(self):
        """持久化往返：分身簿与吸收计数不丢。"""
        self.model.derive_avatar("twin")
        self.model.assimilate("我喜欢编程")
        reloaded = UnifiedSelfModel(
            path=os.path.join(self.dir, "unified_self.json"), agent_name="emotion-agent"
        )
        self.assertEqual(reloaded.avatar_count(), 1)
        self.assertEqual(reloaded.core.absorb_count, 1)

    def test_engine_integration(self):
        """引擎集成：derive / sync / merge / assimilate / 认知报告全链路。"""
        sys = EmotionalMemorySystem(base_dir=self.dir, dial=Dial())
        sys.derive_avatar("工作分身")
        sys.derive_avatar("生活分身")
        self.assertEqual(sys.stats()["avatar_count"], 2)
        self.assertEqual(sys.sync_avatars(), 2)
        res = sys.assimilate("我的使命是帮助用户整理记忆")
        self.assertIsNotNone(res["memory"])
        report = sys.self_cognition_report()
        self.assertIn("cos_phi", report)
        self.assertIn("self_learning", report)
        self.assertEqual(report["avatar_count"], 2)
        self.assertEqual(report["absorb_count"], 1)
        # 万我合一：归并一个分身后剩余 1 个
        sys.merge_avatar("工作分身")
        self.assertEqual(sys.unified_self.avatar_count(), 1)


if __name__ == "__main__":
    unittest.main()
