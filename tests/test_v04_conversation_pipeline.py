"""单元测试：对话管线（v0.4）。

覆盖：检索注入（预置记忆被注入消息体）、模型回复、自动沉淀
（记忆入库 + 自动情感标注 + 涌现累计 + 电容充电）、情感标签透传。
"""
from __future__ import annotations

import tempfile
import unittest

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.model_adapter import MockModelAdapter
from emotion_memory.conversation_pipeline import ConversationPipeline


class TestConversationPipeline(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_v04_pipe_")
        self.sys = EmotionalMemorySystem(
            base_dir=self.dir,
            dial=Dial(capacitor_enabled=True, noise_temperature=0.0),
            seed=7,
        )
        self.adapter = MockModelAdapter()
        self.pipe = ConversationPipeline(self.sys, self.adapter, top_k=3)

    def test_reply_injects_related_memory(self):
        """预置记忆被检索注入到消息体（polluted 视图）。"""
        self.sys.remember("用户最近在学 Python 递归", cluster="学习", entities=["Python"])
        out = self.pipe.reply("Python 递归好难懂")
        joined = " ".join(m.get("content", "") for m in out["messages"])
        self.assertIn("Python", joined)
        self.assertIn("相关记忆片段", joined)
        self.assertIn("(mock回复)", out["reply"])

    def test_reply_settles_memory_and_emotion(self):
        """回复后自动沉淀：记忆入库 + 情感标注 + 涌现累计。"""
        before = len(self.sys.store.all())
        out = self.pipe.reply("今天心情很好，完成了项目")
        self.assertEqual(len(self.sys.store.all()), before + 1)
        self.assertIsNotNone(out["settle"])
        self.assertGreater(len(self.sys.emergent_self.records), 0)
        mem = self.sys.store.get(out["settle"]["memory_id"])
        self.assertIsNotNone(mem)
        self.assertIsNotNone(mem.affect)

    def test_settle_with_explicit_affect(self):
        """显式情感标签透传并累计。"""
        self.pipe.settle("我很兴奋地期待旅行", reply_text="好的", affect=(0.8, 0.9))
        rec = self.sys.emergent_self.records[-1]
        self.assertAlmostEqual(rec["valence"], 0.8, places=5)
        self.assertAlmostEqual(rec["arousal"], 0.9, places=5)
        self.assertGreater(len(self.sys.emergent_self.current_concepts), 0)

    def test_retrieve_context_respects_top_k(self):
        """检索注入条数受 top_k 约束。"""
        for i in range(5):
            self.sys.remember(f"关于记忆主题的讨论 {i}", cluster="讨论", entities=["记忆"])
        rows = self.pipe.retrieve_context("记忆主题")
        self.assertLessEqual(len(rows), 3)

    def test_history_in_message(self):
        """历史轮次进入 messages，形成多轮上下文。"""
        self.pipe.reply("第一轮问候")
        messages = self.pipe.build_messages("第二轮内容")
        roles = [m["role"] for m in messages]
        self.assertIn("user", roles)
        contents = " ".join(m["content"] for m in messages)
        self.assertIn("第一轮问候", contents)

    def test_reply_no_settle_when_disabled(self):
        """settle=False 时不写库（纯生成路径）。"""
        before = len(self.sys.store.all())
        out = self.pipe.reply("只聊天不沉淀", settle=False)
        self.assertEqual(len(self.sys.store.all()), before)
        self.assertIsNone(out["settle"])


if __name__ == "__main__":
    unittest.main()
