"""单元测试：人格流形（v0.5 全谱系性格空间）。

覆盖：至少 6 轴连续性格坐标（无预设模板）；初始坐标确定性且在 [0,1]；
环境向量注入（同记忆不同环境 → 不同轨迹）；非线性迭代确定性可复现；
非线性分叉（微小差异放大为不同轨迹）；性格快照（坐标 + 轨迹摘要）；
导出/唤醒重建（可审计）；批量养成。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from emotion_memory.personality_manifold import (
    ENV_DIMS,
    PERSONALITY_AXES,
    EnvironmentVector,
    PersonalityManifold,
)


def neutral_envs(n: int):
    return [EnvironmentVector() for _ in range(n)]


class TestAxesAndState(unittest.TestCase):
    def test_at_least_six_axes(self):
        """至少 6 轴连续性格坐标轴。"""
        self.assertGreaterEqual(len(PERSONALITY_AXES), 6)
        self.assertIn("extraversion", PERSONALITY_AXES)

    def test_no_fixed_templates(self):
        """初始坐标是连续空间中的点（0.5 ± 微小偏移），而非预设模板。"""
        pm = PersonalityManifold(seed=7)
        self.assertEqual(set(pm.state.keys()), set(PERSONALITY_AXES))
        for v in pm.state.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_seed_changes_initial_state(self):
        """不同 seed → 不同初始坐标（独一无二起点）。"""
        pm1 = PersonalityManifold(seed=1)
        pm2 = PersonalityManifold(seed=99)
        self.assertNotEqual(pm1.state, pm2.state)

    def test_env_dimensions(self):
        """环境向量五维：对话场景/语气/反馈/话题/节律。"""
        self.assertEqual(ENV_DIMS, ["scene", "tone", "feedback", "topic", "rhythm"])


class TestEnvironmentInjection(unittest.TestCase):
    def test_same_memories_different_env_diverge(self):
        """相同记忆沉淀在不同环境下解释不同（轨迹分叉）。"""
        pm_a = PersonalityManifold(seed=7)
        pm_b = PersonalityManifold(seed=7)
        envs_a = [
            EnvironmentVector(scene=0.9, tone=0.9, feedback=0.9, topic=0.9, rhythm=0.9)
            for _ in range(20)
        ]
        envs_b = [
            EnvironmentVector(scene=0.1, tone=0.1, feedback=0.1, topic=0.1, rhythm=0.1)
            for _ in range(20)
        ]
        pm_a.learn(envs_a)
        pm_b.learn(envs_b)
        self.assertNotEqual(pm_a.state, pm_b.state)

    def test_env_drive_direction(self):
        """环境映射方向确定：私密随意场景推动外向轴上升。"""
        pm = PersonalityManifold(seed=7)
        private = EnvironmentVector(scene=1.0)
        drive = pm.env_drive(private)
        self.assertGreater(drive["extraversion"], 0.0)

    def test_affect_injection(self):
        """记忆情感进入养成方程（正向情感 → 外向轴增量）。"""
        pm = PersonalityManifold(seed=7)
        neutral = pm.env_drive(EnvironmentVector())
        emotional = pm.env_drive(EnvironmentVector(), affect=(1.0, 0.5))
        self.assertGreater(emotional["extraversion"], neutral["extraversion"])


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_sequence_reproducible(self):
        """确定性可复现：同 seed 同环境序列 → 同轨迹（无随机数）。"""
        envs = [EnvironmentVector(tone=0.7 + 0.01 * i) for i in range(15)]
        a = PersonalityManifold(seed=7)
        b = PersonalityManifold(seed=7)
        a.learn(envs)
        b.learn(envs)
        self.assertEqual(a.state, b.state)
        self.assertEqual(len(a.trajectory), len(b.trajectory))
        self.assertEqual(a.trajectory[0]["env"], b.trajectory[0]["env"])

    def test_chaotic_iteration_bounded(self):
        """非线性迭代输出始终有界 [0,1]。"""
        pm = PersonalityManifold(seed=7, r=3.9)
        envs = [EnvironmentVector(topic=0.2 + 0.05 * i) for i in range(30)]
        final = pm.learn(envs)
        for v in final.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestBifurcation(unittest.TestCase):
    def test_micro_perturbation_amplified(self):
        """非线性分叉：微小初始差异被放大成显著不同轨迹。"""
        pm = PersonalityManifold(seed=7, r=3.8)
        envs = [EnvironmentVector(scene=0.3 + 0.02 * i) for i in range(25)]
        res = pm.bifurcation(envs, perturbation=1e-3)
        self.assertTrue(res["diverged"])
        self.assertGreater(res["final_distance"], res["initial_distance"] * 10.0)

    def test_bifurcation_deterministic(self):
        """分叉结果可复现（确定性混沌，可审计）。"""
        pm1 = PersonalityManifold(seed=7, r=3.8)
        pm2 = PersonalityManifold(seed=7, r=3.8)
        envs = [EnvironmentVector(tone=0.4 + 0.03 * i) for i in range(20)]
        r1 = pm1.bifurcation(envs)
        r2 = pm2.bifurcation(envs)
        self.assertEqual(r1["final_distance"], r2["final_distance"])
        self.assertEqual(r1["axis_delta"], r2["axis_delta"])


class TestSnapshotRebuild(unittest.TestCase):
    def test_snapshot_coords_and_summary(self):
        """性格快照：当前坐标 + 轨迹摘要（供唤醒重建使用）。"""
        pm = PersonalityManifold(seed=7)
        pm.learn([EnvironmentVector(tone=0.8), EnvironmentVector(tone=0.2)])
        snap = pm.snapshot()
        self.assertEqual(set(snap["coords"].keys()), set(PERSONALITY_AXES))
        self.assertEqual(snap["step"], 2)
        self.assertEqual(snap["trajectory_summary"]["steps"], 2)
        self.assertIsNotNone(snap["trajectory_summary"]["last"])

    def test_export_rebuild_roundtrip(self):
        """导出 JSON → 唤醒重建：坐标与轨迹完整恢复。"""
        pm = PersonalityManifold(seed=7)
        pm.learn([EnvironmentVector(topic=0.9)] * 5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "manifold.json")
            pm.export_json(path)
            restored = PersonalityManifold(seed=0)
            steps = restored.rebuild_from(path)
            self.assertEqual(steps, 5)
            self.assertEqual(restored.state, pm.state)
            self.assertEqual(restored.axes, pm.axes)


if __name__ == "__main__":
    unittest.main()
