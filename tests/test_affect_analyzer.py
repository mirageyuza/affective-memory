"""自动情感标注器单测：零依赖 valence/arousal + 外部接口注入。"""
from __future__ import annotations

import tempfile
import unittest

from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.algorithms.affect_analyzer import (
    DefaultAffectAnalyzer,
    ExternalAffectAnnotator,
    analyze,
)


class TestDefaultAffectAnalyzer(unittest.TestCase):
    def setUp(self):
        self.an = DefaultAffectAnalyzer()

    def test_positive_valence(self):
        v, a = self.an.annotate("今天真开心，太棒了")
        self.assertGreater(v, 0)
        self.assertGreaterEqual(a, 0)
        self.assertLessEqual(a, 1)

    def test_negative_valence(self):
        v, _ = self.an.annotate("项目失败了，很崩溃，很失望")
        self.assertLess(v, 0)

    def test_neutral_near_zero(self):
        v, _ = self.an.annotate("把文件放到桌面上")
        self.assertEqual(v, 0.0)

    def test_negation_flips_polarity(self):
        pos = self.an.annotate("今天很开心")[0]
        neg = self.an.annotate("今天不开心")[0]
        self.assertGreaterEqual(pos, -neg)  # 否定后极性反转（至少不更正向）
        self.assertLess(neg, 0) if pos > 0 else None

    def test_english_support(self):
        v, _ = self.an.annotate("I feel happy and great")
        self.assertGreater(v, 0)

    def test_arousal_bounds(self):
        for text in ["普通的一天", "太棒了！！！", "难受"]:
            _v, a = self.an.annotate(text)
            self.assertGreaterEqual(a, 0)
            self.assertLessEqual(a, 1)

    def test_convenience_function(self):
        v, a = analyze("这是一段没有明显情感的中性文字")
        self.assertIsInstance(v, float)
        self.assertIsInstance(a, float)


class _FakeExternal(ExternalAffectAnnotator):
    """外部模型接口实现：固定返回 (0.9, 0.8)。"""

    def annotate(self, text: str):
        return 0.9, 0.8


class TestExternalInterface(unittest.TestCase):
    def test_injected_annotator_used_by_engine(self):
        d = tempfile.mkdtemp(prefix="emotion_memory_ext_")
        sys = EmotionalMemorySystem(base_dir=d, affect_annotator=_FakeExternal())
        m = sys.remember("无论写什么都用外部模型打标")
        self.assertAlmostEqual(m.affect.valence, 0.9)
        self.assertAlmostEqual(m.affect.arousal, 0.8)

    def test_default_annotator_used_when_none(self):
        d = tempfile.mkdtemp(prefix="emotion_memory_def_")
        sys = EmotionalMemorySystem(base_dir=d)
        m = sys.remember("周末加班到凌晨，项目又要返工")
        self.assertLess(m.affect.valence, 0)  # 内置词典识别出消极

    def test_abstract_method_must_implement(self):
        class _Blank(ExternalAffectAnnotator):
            pass

        with self.assertRaises(NotImplementedError):
            _Blank().annotate("x")


if __name__ == "__main__":
    unittest.main()
