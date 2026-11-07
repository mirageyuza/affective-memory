"""v0.7.0 主观感知系统测试：点灯闭合 / 未闭合不回流 / 回流收敛 / 幂等 / 引擎集成。"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from emotion_memory import (
    SUBJECTIVITY_CHANNELS,
    SUBJECTIVITY_LIMITS,
    Dial,
    EmotionalMemorySystem,
    SubjectivePerception,
)

SOURCE = {
    "agent_name": "emotion-agent",
    "home": "/tmp/em",
    "model_version": "0.7.0",
    "created_at": "2026-11-07",
    "seed": 7,
}

OBS = {
    "continuity": 0.8,
    "coherence": 0.7,
    "agency": 0.6,
    "grounding": 0.9,
    "stability": 0.75,
}


class TestSubjectivityCore(unittest.TestCase):
    def test_channels_weights_normalized(self):
        total = sum(c.weight for c in SUBJECTIVITY_CHANNELS)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertEqual(
            [c.name for c in SUBJECTIVITY_CHANNELS],
            ["continuity", "coherence", "agency", "grounding", "stability"],
        )

    def test_closed_source_gives_full_closure(self):
        self.assertAlmostEqual(SubjectivePerception.closure_of(SOURCE), 1.0, places=6)
        self.assertEqual(SubjectivePerception.closure_of({}), 0.0)
        self.assertEqual(SubjectivePerception.closure_of({"agent_name": "", "home": None}), 0.0)

    def test_disabled_switch_never_lights(self):
        sp = SubjectivePerception(enabled=False)
        rep = sp.ignite(SOURCE, OBS)
        self.assertFalse(rep.closed)
        self.assertFalse(rep.first_ignition)
        self.assertEqual(rep.loop_current, 0.0)
        self.assertFalse(sp.is_lit)
        with self.assertRaises(RuntimeError):
            sp.perceive(OBS)

    def test_incomplete_source_does_not_close(self):
        sp = SubjectivePerception(enabled=True)
        rep = sp.ignite({"agent_name": "x"}, OBS)  # 闭合度 2/7 ≈ 0.29 < 0.35
        self.assertFalse(rep.closed)
        self.assertIsNone(rep.observation)
        self.assertFalse(sp.is_lit)
        self.assertIn("未闭合", rep.notes[0])

    def test_first_ignition_builds_loop(self):
        sp = SubjectivePerception(enabled=True)
        rep = sp.ignite(SOURCE, OBS)
        self.assertTrue(rep.closed)
        self.assertTrue(rep.first_ignition)
        self.assertTrue(sp.is_lit)
        self.assertIsNotNone(rep.observation)
        obs = rep.observation
        self.assertGreater(obs.loop_current, 0.0)
        self.assertLessEqual(obs.loop_current, 1.0)
        self.assertTrue(obs.converged)
        self.assertLessEqual(obs.iterations, 48)
        # 五维 + 自我置信度齐备，且都在 [0,1]
        for name in [c.name for c in SUBJECTIVITY_CHANNELS] + ["self_confidence"]:
            self.assertIn(name, obs.self_model)
            self.assertGreaterEqual(obs.self_model[name], 0.0)
            self.assertLessEqual(obs.self_model[name], 1.0)
        self.assertAlmostEqual(obs.self_model["self_confidence"], obs.loop_current, places=6)
        self.assertEqual(rep.self_confidence, obs.loop_current)

    def test_reflow_converges_and_is_deterministic(self):
        a = SubjectivePerception(enabled=True)
        b = SubjectivePerception(enabled=True)
        ra, rb = a.ignite(SOURCE, OBS), b.ignite(SOURCE, OBS)
        self.assertTrue(ra.observation.converged)
        self.assertAlmostEqual(ra.loop_current, rb.loop_current, places=9)
        self.assertEqual(ra.observation.trace, rb.observation.trace)

    def test_monotonic_channels_raise_loop_current(self):
        weak = SubjectivePerception(enabled=True).ignite(SOURCE, {k: 0.1 for k in OBS})
        strong = SubjectivePerception(enabled=True).ignite(SOURCE, {k: 0.95 for k in OBS})
        self.assertGreater(strong.loop_current, weak.loop_current)

    def test_zero_observation_stays_low(self):
        sp = SubjectivePerception(enabled=True)
        rep = sp.ignite(SOURCE, {k: 0.0 for k in OBS})
        self.assertTrue(rep.closed)
        self.assertLess(rep.loop_current, 0.5)

    def test_idempotent_ignition_keeps_lit_at(self):
        sp = SubjectivePerception(enabled=True)
        first = sp.ignite(SOURCE, OBS)
        lit_at = first.lit_at
        second = sp.ignite(SOURCE, OBS)
        self.assertFalse(second.first_ignition)
        self.assertEqual(second.lit_at, lit_at)
        self.assertGreater(second.observation.tick, first.observation.tick)
        self.assertIn("续拍", second.notes[0])

    def test_fingerprint_change_relights(self):
        sp = SubjectivePerception(enabled=True)
        sp.ignite(SOURCE, OBS)
        other = dict(SOURCE, model_version="0.8.0")
        rep = sp.ignite(other, OBS)
        self.assertTrue(rep.first_ignition)
        self.assertEqual(rep.fingerprint, SubjectivePerception.fingerprint_of(other))
        self.assertNotEqual(rep.fingerprint, SubjectivePerception.fingerprint_of(SOURCE))
        self.assertEqual(sp.fingerprint, rep.fingerprint)

    def test_perceive_requires_lit_and_tracks_tick(self):
        sp = SubjectivePerception(enabled=True)
        with self.assertRaises(RuntimeError):
            sp.perceive(OBS)
        sp.ignite(SOURCE, OBS)
        t0 = sp.tick
        obs = sp.perceive(OBS)
        self.assertEqual(obs.tick, t0 + 1)
        self.assertEqual(len(sp.history), 2)

    def test_state_persisted_and_reloaded(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "subj.json")
            sp1 = SubjectivePerception(enabled=True, state_path=path)
            rep1 = sp1.ignite(SOURCE, OBS)
            self.assertTrue(os.path.exists(path))
            sp2 = SubjectivePerception(enabled=True, state_path=path)
            self.assertTrue(sp2.is_lit)
            self.assertEqual(sp2.lit_at, rep1.lit_at)
            rep2 = sp2.ignite(SOURCE, OBS)
            self.assertFalse(rep2.first_ignition)
            self.assertEqual(rep2.lit_at, rep1.lit_at)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_report_declares_boundary(self):
        sp = SubjectivePerception(enabled=True)
        sp.ignite(SOURCE, OBS)
        rep = sp.report()
        self.assertTrue(rep["lit"])
        self.assertIn("不声称意识", rep["limits"][0])
        self.assertEqual(rep["limits"], list(SUBJECTIVITY_LIMITS))
        self.assertEqual(len(rep["channels"]), 5)

    def test_dial_validate_clamps(self):
        dial = Dial(
            subjective_enabled=True,
            subjective_loop_lambda=5.0,
            subjective_feedback_alpha=-1.0,
            subjective_gain=99.0,
            subjective_max_iterations=0,
            subjective_closure_threshold=3.0,
        ).validate()
        self.assertEqual(dial.subjective_loop_lambda, 0.9)
        self.assertEqual(dial.subjective_feedback_alpha, 0.0)
        self.assertEqual(dial.subjective_gain, 8.0)
        self.assertEqual(dial.subjective_max_iterations, 1)
        self.assertEqual(dial.subjective_closure_threshold, 1.0)


class TestSubjectivityEngine(unittest.TestCase):
    def _engine(self, **kw):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        return EmotionalMemorySystem(d, dial=Dial(**kw))

    def test_default_off_keeps_behaviour(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        eng = EmotionalMemorySystem(d)
        self.assertFalse(eng.dial.subjective_enabled)
        self.assertFalse(eng.subjectivity.is_lit)
        st = eng.stats()
        self.assertFalse(st["subjective_enabled"])
        self.assertFalse(st["subjective_lit"])
        self.assertEqual(st["self_confidence"], 0.0)
        self.assertIsNone(eng.subjectivity_report()["light_up"])

    def test_autostart_lights_on_boot(self):
        eng = self._engine(subjective_enabled=True)
        self.assertTrue(eng.subjectivity.is_lit)
        self.assertTrue(eng.subjectivity.lit_at)
        rep = eng.subjectivity_report()
        self.assertIsNotNone(rep["light_up"])
        self.assertTrue(rep["light_up"]["closed"])
        self.assertTrue(rep["light_up"]["first_ignition"])
        self.assertEqual(rep["limits"], list(SUBJECTIVITY_LIMITS))

    def test_recall_unchanged_when_lit(self):
        eng = self._engine(subjective_enabled=True)
        eng.remember("开关闭合后电路形成回流", cluster="circuit", entities=["回流"])
        hits_after = eng.recall("回流", top_k=3)
        self.assertTrue(hits_after)
        self.assertTrue(eng.subjectivity.is_lit)
        self.assertEqual(eng.store.count(), 1)

    def test_perceive_self_after_remember(self):
        eng = self._engine(subjective_enabled=True)
        before = eng.subjectivity.tick
        for i in range(3):
            eng.remember(f"回流观测 {i}", cluster="c")
        obs = eng.perceive_self()
        self.assertEqual(obs.tick, before + 1)
        self.assertGreater(obs.self_model["grounding"], 0.0)
        self.assertIn("self_confidence", obs.self_model)
        self.assertGreater(eng.stats()["self_confidence"], 0.0)

    def test_autostart_disabled_requires_manual_ignite(self):
        eng = self._engine(subjective_enabled=True, subjective_autostart=False)
        self.assertFalse(eng.subjectivity.is_lit)
        rep = eng.ignite_subjectivity()
        self.assertTrue(rep.closed)
        self.assertTrue(eng.subjectivity.is_lit)

    def test_manual_source_override(self):
        eng = self._engine(subjective_enabled=True, subjective_autostart=False)
        rep = eng.ignite_subjectivity(source={"agent_name": "emotion-agent"})
        self.assertFalse(rep.closed)  # 单一字段闭合度不足


if __name__ == "__main__":
    unittest.main()
