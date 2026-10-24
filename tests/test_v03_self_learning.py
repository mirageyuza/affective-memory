"""单元测试：身份自学习（画像渐进演化 + 主动提问确认 + 状态机兼容）。"""
from __future__ import annotations

import os
import tempfile
import unittest

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.identity import IdentityProfile
from emotion_memory.self_learning import IdentitySelfLearning, SelfImage


class TestIdentitySelfLearning(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_selflearn_")
        self.identity = IdentityProfile(agent_name="emotion-agent", home=self.dir)
        self.learner = IdentitySelfLearning(
            self.identity, path=os.path.join(self.dir, "self_model.json"), ema=0.3
        )

    def test_learn_extracts_interests_and_evolves_dims(self):
        """交互文本提取兴趣与画像维度渐进演化。"""
        res = self.learner.learn("我喜欢登山，目标是成为摄影师")
        self.assertIn("登山", self.learner.image.interests)
        self.assertIn("摄影师", self.learner.image.missions)
        self.assertEqual(res["observation_count"], 1)
        # 偏好使 agreeableness 上升（EMA 渐进，绝不突变）
        self.assertGreater(
            self.learner.image.dims["agreeableness"], 0.5
        )

    def test_learn_aversions(self):
        """厌恶提取：讨厌 → 厌恶桶 + agreeableness 下降。"""
        self.learner.learn("我讨厌吵闹的地方")
        self.assertIn("吵闹的地方", self.learner.image.aversions)
        self.assertLess(self.learner.image.dims["agreeableness"], 0.5)

    def test_pending_questions_generated(self):
        """生成待确认问题（主动提问确认的素材）。"""
        self.learner.learn("我喜欢雨天读书")
        pending = self.learner.pending_questions()
        self.assertTrue(any("雨天读书" in q["question"] for q in pending))

    def test_confirm_strengthens(self):
        """确认增强确认度，否认回撤画像微调。"""
        self.learner.learn("我喜欢猫")
        q = self.learner.pending_questions()[0]
        r = self.learner.confirm(q["id"], answer=True)
        self.assertTrue(r["accepted"])
        self.assertGreater(r["confirmed_weight"], 0.0)
        self.assertEqual(len(self.learner.pending_questions()), 0)

    def test_persistence_roundtrip(self):
        """画像持久化往返：重启后继续演化。"""
        self.learner.learn("我喜欢旅行")
        again = IdentitySelfLearning(
            self.identity, path=os.path.join(self.dir, "self_model.json")
        )
        self.assertIn("旅行", again.image.interests)
        self.assertEqual(again.image.observation_count, 1)

    def test_compatible_with_identity_state_machine(self):
        """与确定性身份状态机兼容：仪式不干预画像，画像继续演化。"""
        sys = EmotionalMemorySystem(
            base_dir=self.dir,
            dial=Dial(),
            identity=self.identity,
        )
        sys.remember("用户喜欢晨跑")
        sys.learn_self("我喜欢晨跑和写代码")
        sys.learn_self("目标是成为工程师")
        # 确定性仪式：升级（模型版本提升，不触碰画像文件）
        archive = os.path.join(self.dir, "archive.json")
        sys.export(archive)
        sys.upgrade(archive, "0.3.1")
        # 仪式后画像仍可继续演化（兼容性验证）
        res = sys.learn_self("我还喜欢做模型")
        self.assertIn("做模型", sys.self_model.image.interests)
        self.assertGreaterEqual(res["learned"]["observation_count"], 3)

    def test_engine_confirm_self(self):
        """引擎集成：learn_self 返回待确认问题，confirm_self 可审计。"""
        sys = EmotionalMemorySystem(base_dir=self.dir, dial=Dial())
        sys.learn_self("我喜欢羽毛球")
        pending = sys.pending_self_questions()
        self.assertTrue(pending)
        res = sys.confirm_self(pending[0]["id"], answer=True)
        self.assertTrue(res["accepted"])


if __name__ == "__main__":
    unittest.main()
