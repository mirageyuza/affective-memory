"""单元测试：震荡命中函数 H(t)。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory.config import Dial
from emotion_memory.algorithms.hit_function import OscillationBank
from emotion_memory.models import AffectLabel, Memory


class TestHitFunction(unittest.TestCase):
    def test_non_negative(self):
        """H(t) 恒为非负（负向震荡视为不唤起）。"""
        bank = OscillationBank(dial=Dial(noise_temperature=0.0), seed=1)
        for t in range(0, 100):
            v = bank.hit("c", t, gain=1.0)
            self.assertGreaterEqual(v, 0.0)

    def test_zero_noise_deterministic(self):
        """T=0 时 H(t) 完全确定（同 seed 可复现）。"""
        a = OscillationBank(dial=Dial(noise_temperature=0.0), seed=42)
        b = OscillationBank(dial=Dial(noise_temperature=0.0), seed=42)
        vals_a = [a.hit("c", t, gain=1.0) for t in range(20)]
        vals_b = [b.hit("c", t, gain=1.0) for t in range(20)]
        self.assertEqual(vals_a, vals_b)

    def test_noise_increases_randomness(self):
        """T>0 时两次采样通常不一致（随机共振）。"""
        bank = OscillationBank(dial=Dial(noise_temperature=0.8), seed=1)
        v1 = [bank.hit("c", 3.0, gain=1.0) for _ in range(5)]
        v2 = [bank.hit("c", 3.0, gain=1.0) for _ in range(5)]
        self.assertNotEqual(v1, v2)

    def test_gain_modulates_amplitude(self):
        """情感增益越大，命中幅值越大。"""
        low = Dial(noise_temperature=0.0, emotion_gain=0.1)
        high = Dial(noise_temperature=0.0, emotion_gain=1.8)
        bank = OscillationBank(dial=low, seed=1)
        m_low = bank.gain_of(AffectLabel(0.9, 0.8), low)
        m_high = bank.gain_of(AffectLabel(0.9, 0.8), high)
        self.assertGreater(m_high, m_low)

    def test_cluster_gets_own_phase_frequency(self):
        """不同簇获得不同频率/相位。"""
        bank = OscillationBank(seed=5)
        m1 = bank.register("c1")
        m2 = bank.register("c2")
        self.assertNotEqual(m1.frequency, m2.frequency)


if __name__ == "__main__":
    unittest.main()
