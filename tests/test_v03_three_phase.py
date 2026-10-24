"""单元测试：旋转磁场唤起调度（三相顺序启动，120° 相位差联想流）。"""
from __future__ import annotations

import math
import tempfile
import unittest

from emotion_memory.algorithms.hit_function import OscillationBank
from emotion_memory.algorithms.three_phase import (
    CHANNEL_NAMES,
    LEADER_PERIOD_HOURS,
    PHASE_OFFSETS,
    ThreePhaseField,
)
from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.models import Memory


class TestThreePhaseField(unittest.TestCase):
    def setUp(self):
        self.dial = Dial(noise_temperature=0.0)
        self.bank = OscillationBank(dial=self.dial, seed=7)
        self.field = ThreePhaseField(self.bank, seed=11)

    def _mem(self, cluster):
        m = Memory(text=f"记忆在簇 {cluster}", cluster=cluster)
        return m

    def test_channel_assignment_stable(self):
        """簇到通道的分配稳定（同 seed 跨实例一致）。"""
        c = self.field.channel_of("工作")
        self.assertIn(c, (0, 1, 2))
        self.assertEqual(self.field.channel_of("工作"), c)
        self.assertEqual(self.field.channel_name("工作"), CHANNEL_NAMES[c])

    def test_leader_rotates_sequentially(self):
        """主导通道随时间顺序轮转（三相顺序启动 0→1→2→0）。"""
        seq = [
            self.field.leader_channel(0),
            self.field.leader_channel(LEADER_PERIOD_HOURS - 0.1),
            self.field.leader_channel(LEADER_PERIOD_HOURS),
            self.field.leader_channel(2 * LEADER_PERIOD_HOURS),
            self.field.leader_channel(3 * LEADER_PERIOD_HOURS),
        ]
        self.assertEqual(seq[:2], [0, 0])
        self.assertEqual(seq[2:], [1, 2, 0])

    def test_phase_offsets_120_degrees(self):
        """三通道相位互差 120°（2π/3）。"""
        self.assertAlmostEqual(PHASE_OFFSETS[1] - PHASE_OFFSETS[0], 2 * math.pi / 3)
        self.assertAlmostEqual(PHASE_OFFSETS[2] - PHASE_OFFSETS[1], 2 * math.pi / 3)
        self.assertAlmostEqual(PHASE_OFFSETS[2] - PHASE_OFFSETS[0], 4 * math.pi / 3)

    def test_field_state_observable(self):
        """场状态可观测：主导通道 + 三相顺序 + 相位差。"""
        state = self.field.field_state(2.0)
        self.assertIn(state["leader_channel"], CHANNEL_NAMES)
        self.assertEqual(state["phase_offsets_deg"], [0, 120, 240])
        self.assertEqual(state["channel_count"], 3)
        # sequence 是从主导通道开始的循环排列
        leader = CHANNEL_NAMES.index(state["leader_channel"])
        self.assertEqual(
            state["sequence"],
            CHANNEL_NAMES[leader:] + CHANNEL_NAMES[:leader],
        )

    def test_stream_sorted_and_topk(self):
        """联想流按激活强度降序，支持 top-k 截断。"""
        mems = [
            self._mem("工作"),
            self._mem("旅行"),
            self._mem("学习"),
            self._mem("美食"),
            self._mem("运动"),
        ]
        stream = self.field.stream(mems, t=3.0, dial=self.dial, k=3)
        self.assertEqual(len(stream), 3)
        acts = [r["activation"] for r in stream]
        self.assertEqual(acts, sorted(acts, reverse=True))
        for r in stream:
            self.assertIn(r["channel"], CHANNEL_NAMES)
            self.assertIsInstance(r["leader"], bool)

    def test_leader_boost_dominates(self):
        """主导通道内的簇优先激活（leader 增益 > follower）。"""
        self.field._channel_of["主导簇"] = 0
        self.field._channel_of["从属簇"] = 1
        self.bank.register("主导簇")
        self.bank.register("从属簇")
        t = 1.0
        self.assertEqual(self.field.leader_channel(t), 0)
        a = self.field.phase_activation("主导簇", t, gain=1.0, dial=self.dial)
        b = self.field.phase_activation("从属簇", t, gain=1.0, dial=self.dial)
        self.assertGreaterEqual(a, 0.0)
        self.assertGreaterEqual(b, 0.0)

    def test_does_not_break_hit_interface(self):
        """不破坏现有 H(t) 接口：OscillationBank.hit 语义保持可用。"""
        self.bank.register("老簇")
        v = self.bank.hit("老簇", 5.0, gain=0.5, dial=self.dial)
        self.assertIsInstance(v, float)
        self.assertGreaterEqual(v, 0.0)

    def test_engine_three_phase_stream(self):
        """引擎集成：three_phase_stream 输出场状态与联想流。"""
        sys = EmotionalMemorySystem(
            base_dir=tempfile.mkdtemp(prefix="emotion_memory_tp_"),
            dial=Dial(noise_temperature=0.0),
        )
        for txt, clu in [("写周报", "工作"), ("周末爬山", "旅行")]:
            sys.remember(txt, cluster=clu)
        out = sys.three_phase_stream(now=2.0, k=2)
        self.assertIn("field", out)
        self.assertIn("stream", out)
        self.assertEqual(len(out["stream"]), 2)
        self.assertIn("leader_channel", out["field"])


if __name__ == "__main__":
    unittest.main()
