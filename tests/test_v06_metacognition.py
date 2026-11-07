"""v0.6.2 元认知层测试：置信度估计 / 拒答策略 / 校准度量 / 引擎接入。

关键反自欺断言：
  - 置信度只读 cosine，不读含情感增益的 score（喜欢不得抬高确信）；
  - 拒答话术不含任何情感断言；
  - 空样本的校准报告必须显式 insufficient，不得报 0 误差；
  - recall_verdict 只观测不改写，recall 行为与记忆内容不变。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.metacognition import (
    ABSTAIN_TEXT,
    HEDGE_TEXT,
    LEVEL_ABSTAIN,
    LEVEL_ANSWER,
    LEVEL_HEDGE,
    METACOGNITION_LIMITS,
    AbstentionPolicy,
    ConfidenceEstimator,
    MetacognitiveMonitor,
    calibrate,
)


def fake_hit(cosine: float, score: float = None, never_silent: bool = False):
    """构造可控命中：cosine 为纯语义分，score 为含情感增益的排序分。"""
    mem = SimpleNamespace(
        memory_id=f"m-{cosine:.4f}-{never_silent}",
        text="t",
        never_silent=never_silent,
    )
    return SimpleNamespace(
        memory=mem,
        cosine=cosine,
        score=cosine if score is None else score,
        display_text="t",
    )


class TestConfidenceEstimator(unittest.TestCase):
    def test_empty_hits_zero_confidence(self):
        bd = ConfidenceEstimator().estimate([])
        self.assertEqual(bd.support, 0)
        self.assertEqual(bd.confidence, 0.0)
        self.assertEqual(bd.semantic_top, 0.0)

    def test_single_dominant_hit_high_confidence(self):
        bd = ConfidenceEstimator().estimate([fake_hit(0.9)], top_k=1)
        self.assertGreater(bd.confidence, 0.7)
        self.assertEqual(bd.margin, 1.0)
        self.assertEqual(bd.dispersion, 1.0)
        self.assertEqual(bd.coverage, 1.0)

    def test_ambiguous_hits_lower_than_dominant(self):
        """两条分数接近的命中，置信度必须低于一条显著领先的命中。"""
        est = ConfidenceEstimator()
        dominant = est.estimate([fake_hit(0.9), fake_hit(0.1)], top_k=2)
        ambiguous = est.estimate([fake_hit(0.9), fake_hit(0.88)], top_k=2)
        self.assertGreater(dominant.confidence, ambiguous.confidence)
        self.assertGreater(dominant.margin, ambiguous.margin)

    def test_confidence_ignores_emotional_gain(self):
        """同一 cosine、不同 score（情感增益）→ 置信度必须完全一致。"""
        est = ConfidenceEstimator()
        plain = [fake_hit(0.6), fake_hit(0.4)]
        boosted = [fake_hit(0.6, score=9.9), fake_hit(0.4, score=8.8)]
        self.assertAlmostEqual(est.estimate(plain).confidence,
                               est.estimate(boosted).confidence, places=9)

    def test_negative_cosine_clamped_to_zero(self):
        bd = ConfidenceEstimator().estimate([fake_hit(-0.5)], top_k=1)
        self.assertEqual(bd.semantic_top, 0.0)

    def test_coverage_reflects_support(self):
        est = ConfidenceEstimator()
        few = est.estimate([fake_hit(0.8)], top_k=5)
        many = est.estimate([fake_hit(0.8)] * 5, top_k=5)
        self.assertAlmostEqual(few.coverage, 0.2)
        self.assertAlmostEqual(many.coverage, 1.0)


class TestAbstentionPolicy(unittest.TestCase):
    def test_no_candidate_abstains(self):
        v = AbstentionPolicy().decide([])
        self.assertEqual(v.level, LEVEL_ABSTAIN)
        self.assertEqual(v.reason, "no_candidate")
        self.assertTrue(v.should_abstain)

    def test_below_semantic_floor_abstains(self):
        """语义地板优先于阈值：一堆低相关命中不能拼出"确定"。"""
        v = AbstentionPolicy(semantic_floor=0.5).decide(
            [fake_hit(0.2), fake_hit(0.19), fake_hit(0.18)], top_k=3
        )
        self.assertEqual(v.level, LEVEL_ABSTAIN)
        self.assertEqual(v.reason, "below_semantic_floor")

    def test_high_confidence_answers(self):
        v = AbstentionPolicy().decide([fake_hit(0.95)], top_k=1)
        self.assertEqual(v.level, LEVEL_ANSWER)
        self.assertTrue(v.is_grounded)
        self.assertEqual(v.text, "")

    def test_hedge_band(self):
        v = AbstentionPolicy(answer_threshold=0.99, hedge_threshold=0.10).decide(
            [fake_hit(0.7), fake_hit(0.69)], top_k=2
        )
        self.assertEqual(v.level, LEVEL_HEDGE)
        self.assertEqual(v.text, HEDGE_TEXT)

    def test_protected_memory_never_abstains(self):
        """高危/时间敏感记忆：置信度再低也只能 hedge，不许装作不存在。"""
        v = AbstentionPolicy(answer_threshold=0.9, hedge_threshold=0.8).decide(
            [fake_hit(0.2, never_silent=True)], top_k=1
        )
        self.assertEqual(v.level, LEVEL_HEDGE)
        self.assertEqual(v.reason, "protected_memory_low_confidence")

    def test_texts_contain_no_warmth_claim(self):
        """拒答/提示话术不得包含情感断言（温暖不可伪造）。"""
        for text in (ABSTAIN_TEXT, HEDGE_TEXT):
            self.assertNotIn("温暖", text)
            self.assertNotIn("爱你", text)
        self.assertIn("没有足够依据", ABSTAIN_TEXT)

    def test_dial_thresholds_flow_into_policy(self):
        dial = Dial(answer_threshold=0.2, hedge_threshold=0.1, semantic_floor=0.05).validate()
        monitor = MetacognitiveMonitor(dialect=dial)
        v = monitor.verdict([fake_hit(0.85)], top_k=1)
        self.assertEqual(v.level, LEVEL_ANSWER)

    def test_dial_keeps_hedge_le_answer(self):
        dial = Dial(answer_threshold=0.1, hedge_threshold=0.9).validate()
        self.assertGreaterEqual(dial.answer_threshold, dial.hedge_threshold)

    def test_verdict_dict_shape(self):
        d = AbstentionPolicy().decide([fake_hit(0.9)], top_k=1).as_dict()
        for key in ("level", "confidence", "reason", "text", "should_abstain", "breakdown"):
            self.assertIn(key, d)
        self.assertIn("semantic_top", d["breakdown"])


class TestCalibration(unittest.TestCase):
    def test_empty_samples_insufficient(self):
        rep = calibrate([])
        self.assertEqual(rep.n, 0)
        self.assertIsNone(rep.ece)
        self.assertTrue(rep.insufficient)
        self.assertFalse(rep.calibrated)
        self.assertIn("样本不足", rep.summary())

    def test_well_calibrated_low_ece(self):
        samples = [(0.9, True)] * 90 + [(0.9, False)] * 10
        rep = calibrate(samples, min_samples=10)
        self.assertLess(rep.ece, 1e-6)
        self.assertTrue(rep.calibrated)
        self.assertAlmostEqual(rep.accuracy, 0.9)
        self.assertAlmostEqual(rep.brier, 0.09)  # 0.9·0.1² + 0.1·0.9² = 0.09

    def test_overconfidence_high_ece(self):
        samples = [(0.9, True)] * 50 + [(0.9, False)] * 50
        rep = calibrate(samples, min_samples=10)
        self.assertAlmostEqual(rep.ece, 0.4, places=6)
        self.assertFalse(rep.calibrated)

    def test_underconfidence_high_ece(self):
        rep = calibrate([(0.2, True)] * 30, min_samples=10)
        self.assertAlmostEqual(rep.ece, 0.8, places=6)

    def test_below_min_samples_flagged(self):
        rep = calibrate([(0.5, True)] * 3, min_samples=20)
        self.assertTrue(rep.insufficient)
        self.assertFalse(rep.calibrated)
        self.assertIsNotNone(rep.ece)

    def test_bins_partition_all_samples(self):
        rep = calibrate([(i / 100.0, i % 2 == 0) for i in range(100)], n_bins=10, min_samples=10)
        self.assertEqual(sum(b.count for b in rep.bins), 100)
        self.assertEqual(len(rep.bins), 10)


class TestMonitor(unittest.TestCase):
    def test_decision_stats(self):
        m = MetacognitiveMonitor()
        m.record_decision(m.verdict([fake_hit(0.95)], top_k=1), query="a")
        m.record_decision(m.verdict([], top_k=1), query="b")
        st = m.stats()
        self.assertEqual(st["decisions"], 2)
        self.assertAlmostEqual(st["abstain_rate"], 0.5)
        self.assertAlmostEqual(st["answer_rate"], 0.5)

    def test_audit_jsonl_written(self):
        tmp = tempfile.mkdtemp(prefix="em_meta_")
        path = os.path.join(tmp, "audit.jsonl")
        m = MetacognitiveMonitor(audit_path=path)
        m.record_decision(m.verdict([fake_hit(0.95)], top_k=1), query="q1")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            row = json.loads(f.readline())
        self.assertEqual(row["query"], "q1")
        self.assertIn("confidence", row)

    def test_report_uses_recorded_samples(self):
        m = MetacognitiveMonitor()
        for _ in range(30):
            m.record(0.9, True)
        rep = m.report(min_samples=20)
        self.assertFalse(rep.insufficient)
        self.assertLess(rep.ece, 0.2)


class TestEngineIntegration(unittest.TestCase):
    def make_engine(self, **overrides):
        dial = Dial()
        for k, v in overrides.items():
            setattr(dial, k, v)
        dial.validate()
        return EmotionalMemorySystem(
            base_dir=tempfile.mkdtemp(prefix="em_meta_int_"), dial=dial, seed=7
        )

    def test_recall_verdict_observes_without_mutation(self):
        eng = self.make_engine()
        mem = eng.remember("周末常去看展览，偏好当代艺术", cluster="pref")
        before_text = mem.text
        before_len = len(eng.recall("周末展览", top_k=3))
        res = eng.recall_verdict("周末看什么展", top_k=3)
        self.assertIn("verdict", res)
        self.assertIn(res["level"], (LEVEL_ANSWER, LEVEL_HEDGE, LEVEL_ABSTAIN))
        self.assertEqual(len(eng.recall("周末展览", top_k=3)), before_len)
        self.assertEqual(eng.store.get(mem.memory_id).text, before_text)

    def test_recall_verdict_empty_query_abstains(self):
        eng = self.make_engine()
        res = eng.recall_verdict("毫无关联的词汇组合xyzzy", top_k=3)
        self.assertTrue(res["should_abstain"] or res["level"] == LEVEL_HEDGE)

    def test_metacognition_report_shape(self):
        eng = self.make_engine()
        eng.remember("咖啡偏好不加糖", cluster="pref")
        eng.recall_verdict("密度", top_k=3, record=True)
        rep = eng.metacognition_report()
        self.assertEqual(rep["stats"]["decisions"], 1)
        self.assertIn("calibration", rep)
        self.assertEqual(list(rep["limits"]), list(METACOGNITION_LIMITS))

    def test_metacognition_disabled_by_default(self):
        eng = self.make_engine()
        self.assertFalse(eng.dial.metacognition_enabled)
        self.assertIsNone(eng.metacognition.audit_path)


if __name__ == "__main__":
    unittest.main()
