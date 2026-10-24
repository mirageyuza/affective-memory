"""单元测试：电容记忆模型（充电 / 放电 / 漏电 / standby 备选池 / 协同）。"""
from __future__ import annotations

import tempfile
import time
import unittest

from emotion_memory.algorithms.capacitor import CapacitorModel
from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.models import Memory


class TestCapacitorModel(unittest.TestCase):
    def setUp(self):
        self.cap = CapacitorModel(
            tau_hours=48.0,
            charge_per_touch=0.25,
            discharge_per_recall=0.1,
            standby_pool_size=2,
        )
        self.dir = tempfile.mkdtemp(prefix="emotion_memory_cap_")

    def _mem(self, text="默认记忆", **kw):
        m = Memory(text=text)
        for k, v in kw.items():
            setattr(m, k, v)
        self.cap.init_memory(m)
        return m

    def test_init_half_charged(self):
        """新记忆初始半充且容量归一。"""
        m = self._mem()
        self.assertEqual(m.capacitance, 1.0)
        self.assertAlmostEqual(m.charge, 0.5)
        self.assertEqual(m.discharge_threshold, 0.2)

    def test_touch_charges_and_caps(self):
        """触达充电：charge 增加且不超过容量。"""
        m = self._mem()
        self.cap.touch(m)
        self.assertGreater(m.charge, 0.5)
        m.charge = m.capacitance - 0.01
        self.cap.touch(m)
        self.assertLessEqual(m.charge, m.capacitance)

    def test_recall_discharges_never_negative(self):
        """被想起放电：charge 减少且不低于 0。"""
        m = self._mem()
        before = m.charge
        self.cap.recall_discharge(m)
        self.assertLess(m.charge, before)
        m.charge = 0.0
        self.cap.recall_discharge(m)
        self.assertEqual(m.charge, 0.0)

    def test_leak_exponential(self):
        """时间漏电：指数衰减，与遗忘曲线协同（不删内容）。"""
        m = self._mem()
        t = m.charge
        self.cap.leak(m, 48.0)  # 一个 τ 后降到约 0.368 倍
        self.assertLess(m.charge, t)
        self.assertAlmostEqual(m.charge, t * 0.3678794, delta=0.01)
        self.assertTrue(m.original_text)

    def test_potential_synergy(self):
        """总可及电位 = 遗忘可及性 × 电容因子。"""
        m = self._mem()
        m.accessibility = 0.8
        m.charge = 1.0
        self.assertAlmostEqual(self.cap.potential(m), 0.8)
        m.charge = 0.0
        self.assertAlmostEqual(self.cap.potential(m), 0.4)

    def test_standby_pool_excludes_silent_and_topk(self):
        """备选池：排除沉默记忆（非永不沉默），返回 top-k 高电位候选。"""
        a = self._mem("高频充电的记忆", base_strength=1.0)
        a.charge = 1.0
        b = self._mem("沉默的记忆", silence=True)
        b.charge = 1.0
        c = self._mem("普通记忆", base_strength=0.2)
        c.charge = 0.1
        pool = self.cap.standby_pool([a, b, c], top_k=2, now=time.time())
        self.assertNotIn(b, pool)
        self.assertEqual(len(pool), 2)
        self.assertIn(a, pool)

    def test_engine_charge_persisted_through_touch(self):
        """引擎集成：开启电容开关后 touch 充电并落盘。"""
        sys = EmotionalMemorySystem(
            base_dir=self.dir,
            dial=Dial(capacitor_enabled=True),
        )
        m = sys.remember("去年生日和三个老同学爬山")
        self.assertGreater(m.charge, 0.0)  # 初始化半充
        before = m.charge
        sys.touch(m.memory_id)
        self.assertGreater(m.charge, before)
        # 持久化验证：从存储重新读回 charge 已更新
        reloaded = sys.store.get(m.memory_id)
        self.assertEqual(reloaded.charge, m.charge)

    def test_engine_recall_discharges_and_persists(self):
        """引擎集成：recall 被想起 = 放电并落盘。"""
        sys = EmotionalMemorySystem(
            base_dir=self.dir,
            dial=Dial(capacitor_enabled=True, noise_temperature=0.0),
        )
        m = sys.remember("我喜欢在雨天读书写字")
        before = m.charge
        hits = sys.recall("雨天 读书")
        self.assertTrue(hits)
        self.assertLess(m.charge, before)
        reloaded = sys.store.get(m.memory_id)
        self.assertEqual(reloaded.charge, m.charge)

    def test_serialize_roundtrip_keeps_charge(self):
        """from_dict 保留电容字段（电流桥导入不丢电荷）。"""
        m = self._mem("往返测试")
        m.charge = 0.77
        m.capacitance = 2.0
        m.discharge_threshold = 0.35
        back = Memory.from_dict(m.to_dict())
        self.assertAlmostEqual(back.charge, 0.77)
        self.assertAlmostEqual(back.capacitance, 2.0)
        self.assertAlmostEqual(back.discharge_threshold, 0.35)


if __name__ == "__main__":
    unittest.main()
