"""集成测试：v0.5 物理四定律 + 人格流形接入引擎。

覆盖：守恒（无源注入拒绝 / 超预算衰减 / 污染钳制）；势差（沉淀长期区 /
迁移配额）；熵增（decay_all 双通道 / recall 降权）；湍流（涡旋联想）；
人格流形（remember 环境注入 / 快照 / 分叉 / 重建）；串闭环报告。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from emotion_memory.config import Dial
from emotion_memory.conservation import ConservationError
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.personality_manifold import EnvironmentVector


def make_engine(overrides=None, seed=7):
    dial = Dial()
    dial.three_phase_enabled = True
    dial.proactive_enabled = True
    dial.capacitor_enabled = True
    dial.conservation_enabled = True
    dial.potential_enabled = True
    dial.entropy_enabled = True
    dial.turbulent_enabled = True
    dial.manifold_enabled = True
    if overrides:
        for k, v in overrides.items():
            setattr(dial, k, v)
    dial.validate()
    return EmotionalMemorySystem(base_dir=tempfile.mkdtemp(prefix="em_v05_int_"), dial=dial, seed=seed)


class TestConservationIntegration(unittest.TestCase):
    def test_remember_requires_source(self):
        """开启守恒后：无源情感注入被引擎拒绝。"""
        eng = make_engine()
        with self.assertRaises(ConservationError):
            eng.remember("重要的事", affect=(0.8, 0.6), source_text=None)
        with self.assertRaises(ConservationError):
            eng.remember("重要的事", affect=(0.8, 0.6))

    def test_remember_with_source_ok(self):
        """携带来源原文的记忆正常写入。"""
        eng = make_engine()
        m = eng.remember("用户抱怨项目延期", affect=(0.5, 0.6), source_text="用户说：项目又延期了，我很烦")
        self.assertEqual(m.original_text, "用户抱怨项目延期")
        self.assertNotEqual(m.source_text, "")

    def test_budget_attenuation(self):
        """超预算时新记忆情感强度自然衰减（防情绪通货膨胀）。"""
        eng = make_engine(overrides={"emotion_budget": 2.0})
        eng.remember("A", affect=(0.8, 0.6), source_text="s1")
        before = eng.store.all()[0].affect.arousal
        eng.remember("B", affect=(0.8, 0.6), source_text="s2")
        eng.remember("C", affect=(0.8, 0.6), source_text="s3")
        eng.remember("D", affect=(0.8, 0.6), source_text="s4")
        eng.remember("E", affect=(0.8, 0.6), source_text="s5")
        eng.remember("F", affect=(0.8, 0.6), source_text="s6")
        # 后期记忆的唤醒度被预算钳制得更低
        arousals = [m.affect.arousal for m in eng.store.all()]
        self.assertLess(arousals[-1], before)

    def test_pollution_clamped_by_conservation(self):
        """污染强度超原文情感能量被钳制（禁止无中生有）。"""
        eng = make_engine()
        m = eng.remember("平淡的记录", affect=(0.1, 0.1), source_text="s")
        res = eng.apply_pollution(m.memory_id, mode="beautify", intensity=0.95)
        self.assertTrue(res["clamped"])
        self.assertLess(res["intensity"], res["requested_intensity"])


class TestPotentialIntegration(unittest.TestCase):
    def test_promote_memories(self):
        """高势能记忆沉淀为 longterm，低势能留在 workspace。"""
        eng = make_engine(overrides={"potential_threshold": 0.5})
        strong = eng.remember("刻骨铭心的告别", affect=(0.9, 0.9), source_text="s", importance=2.0)
        weak = eng.remember("无关紧要的琐事", affect=(0.1, 0.1), source_text="s")
        res = eng.promote_memories()
        self.assertEqual(res["promoted_count"], 1)
        self.assertEqual(eng.store.get(strong.memory_id).region, "longterm")
        self.assertEqual(eng.store.get(weak.memory_id).region, "workspace")

    def test_transfer_quota_nonuniform(self):
        """电流桥按势差非均匀分配迁移量（max_count 只搬高势能）。"""
        eng = make_engine()
        ids = []
        for i in range(4):
            m = eng.remember(f"记忆{i}", affect=(0.1 + 0.2 * i, 0.2 + 0.2 * i), source_text="s")
            ids.append(m.memory_id)
        quota = eng.transfer_quota(max_count=2)
        self.assertEqual(quota["selected_count"], 2)
        self.assertEqual(quota["skipped_count"], 2)
        self.assertEqual(len(quota["selected_ids"]), 2)


class TestEntropyIntegration(unittest.TestCase):
    def test_decay_all_dual_channel(self):
        """decay_all 推进熵增：细节衰减快于情绪残留。"""
        eng = make_engine()
        m = eng.remember("一段刻骨铭心的经历", affect=(0.8, 0.6), source_text="s")
        eng.entropic_decay(dt_hours=72.0)
        m = eng.store.get(m.memory_id)
        self.assertLess(m.detail_fidelity, m.emotional_residue)
        self.assertGreater(m.entropy, 0.0)
        self.assertEqual(m.original_text, "一段刻骨铭心的经历")

    def test_recall_entropy_weight(self):
        """熵越高检索置信度越低（recall 降权后熵高记忆排后）。"""
        eng = make_engine()
        low = eng.remember("今天天气不错", affect=(0.3, 0.3), source_text="s")
        high = eng.remember("今天天气不错但细节模糊了", affect=(0.3, 0.3), source_text="s")
        eng.store.update_meta(low.memory_id, entropy=0.1, detail_fidelity=0.95, emotional_residue=0.95)
        eng.store.update_meta(high.memory_id, entropy=0.9, detail_fidelity=0.05, emotional_residue=0.2)
        hits = eng.recall("今天天气", top_k=2)
        self.assertEqual(hits[0].memory.memory_id, low.memory_id)


class TestTurbulenceIntegration(unittest.TestCase):
    def test_turbulent_associate_engine(self):
        """引擎湍流联想输出涡流链（中心→外围）。"""
        eng = make_engine()
        eng.remember("alpha 项目进展", entities=["alpha"], affect=(0.5, 0.5), source_text="s")
        eng.remember("alpha 的队友", entities=["alpha"], affect=(0.5, 0.5), source_text="s")
        eng.remember("beta 相关", entities=["beta"], affect=(0.5, 0.5), source_text="s")
        res = eng.turbulent_associate(query_entities=["alpha"], top_k=5)
        self.assertGreaterEqual(res["center_count"], 1)
        self.assertTrue(res["hits"])
        # 中心层能量最高
        self.assertEqual(res["hits"][0]["ring"], 0)


class TestManifoldIntegration(unittest.TestCase):
    def test_remember_env_injection(self):
        """每次交互环境向量进入养成方程（step 推进）。"""
        eng = make_engine()
        eng.remember("第一次见面", affect=(0.6, 0.5), source_text="s", env=EnvironmentVector(tone=0.9))
        eng.remember("第二次见面", affect=(0.6, 0.5), source_text="s", env=EnvironmentVector(tone=0.1))
        self.assertEqual(eng.manifold.step_no, 2)
        snap = eng.personality_snapshot()
        self.assertEqual(set(snap["coords"].keys()), {"extraversion", "emotional_sensitivity", "risk_preference", "habit_rigidity", "associative_breadth", "narrative_style"})

    def test_bifurcation_two_trajectories(self):
        """两个实例环境序列不同 → 性格轨迹分叉（独一无二）。"""
        eng = make_engine()
        envs_a = [EnvironmentVector(tone=0.9, scene=0.9) for _ in range(15)]
        envs_b = [EnvironmentVector(tone=0.1, scene=0.1) for _ in range(15)]
        traj_a = eng.manifold_learn_sequence(envs_a)
        eng2 = make_engine()
        traj_b = eng2.manifold_learn_sequence(envs_b)
        self.assertNotEqual(traj_a, traj_b)

    def test_export_rebuild(self):
        """性格快照可导出并重建（唤醒确定性）。"""
        eng = make_engine()
        eng.manifold_learn_sequence([EnvironmentVector(topic=0.9)] * 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            eng.personality_export(path)
            eng2 = make_engine()
            steps = eng2.personality_rebuild(path)
            self.assertEqual(steps, 4)
            self.assertEqual(eng2.manifold.state, eng.manifold.state)


class TestChainReport(unittest.TestCase):
    def test_physical_chain_report(self):
        """串闭环报告：守恒→势差→熵增→湍流→人格流形全链路可见。"""
        eng = make_engine()
        eng.remember("一起看过的落日", affect=(0.8, 0.7), source_text="海边那晚的对话", importance=1.5)
        eng.promote_memories()
        eng.entropic_decay(dt_hours=24.0)
        eng.turbulent_associate(query_entities=["落日"])
        rep = eng.physical_chain_report()
        self.assertIn("conservation", rep)
        self.assertIn("potential", rep)
        self.assertIn("entropy", rep)
        self.assertIn("turbulence", rep)
        self.assertIn("personality", rep)
        self.assertEqual(len(rep["chain"]), 5)
        self.assertGreater(rep["conservation"]["total_energy"], 0.0)

    def test_stats_include_v05(self):
        """stats() 暴露 v0.5 状态字段。"""
        eng = make_engine()
        eng.remember("数据", affect=(0.5, 0.5), source_text="s")
        stats = eng.stats()
        self.assertTrue(stats["conservation_enabled"])
        self.assertTrue(stats["manifold_enabled"])
        self.assertIn("average_entropy", stats)
        self.assertIn("manifold_step", stats)


if __name__ == "__main__":
    unittest.main()
