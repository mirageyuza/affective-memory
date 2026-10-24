"""单元测试：身份档案与生命周期状态机（告别/唤醒/升级搬家确定性）。"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory import bridge
from emotion_memory.identity import IdentityProfile, IdentityLifecycle, LifecycleState
from emotion_memory.models import AffectLabel, Memory
from emotion_memory.store.storage import MemoryStore


class TestIdentityLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="em_identity_")
        self.store = MemoryStore(os.path.join(self.tmp, "mem.jsonl"))
        self.store.add(Memory(text="重要记忆", cluster="c", affect=AffectLabel(0.6, 0.5)))
        self.store.add(Memory(text="普通记忆", cluster="c", affect=AffectLabel(0.1, 0.1)))
        self.identity = IdentityProfile(agent_name="agent", model_version="1.0", home=self.tmp)
        self.life = IdentityLifecycle(self.store, self.identity, self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_farewell_is_deterministic(self):
        """告别语由固定模板生成，确定性可复现。"""
        r1 = self.life.farewell(os.path.join(self.tmp, "f1.json"))
        r2 = IdentityLifecycle(
            MemoryStore(os.path.join(self.tmp, "mem.jsonl")),
            IdentityProfile(agent_name="agent", model_version="1.0", home=self.tmp),
            self.tmp,
        ).farewell(os.path.join(self.tmp, "f2.json"))
        self.assertEqual(r1["farewell_speech"], r2["farewell_speech"])
        self.assertIn("最终告别", r1["farewell_speech"])
        self.assertIn("2", r1["farewell_speech"])  # 记忆数
        self.assertIsNotNone(r1["export_path"])

    def test_revive_restores_and_compensates(self):
        path = os.path.join(self.tmp, "out.json")
        bridge.export_bundle(self.store, self.identity, path)
        # 伪装上次活跃在 3 天前
        import time as _t
        self.identity.last_active_at = _t.time() - 3 * 86400
        bridge.export_bundle(self.store, self.identity, path)

        fresh = MemoryStore(os.path.join(self.tmp, "mem2.jsonl"))
        fresh_identity = IdentityProfile(agent_name="new", model_version="2.0", home=self.tmp)
        life2 = IdentityLifecycle(fresh, fresh_identity, self.tmp)
        res = life2.revive(path)
        self.assertIn("距上次对话已 3 天", res["compensation"])
        self.assertEqual(fresh.count(), 2)
        self.assertGreaterEqual(life2.identity.epoch, 1)

    def test_upgrade_archives_old_instance(self):
        """升级搬家：旧实例导出 → 新实例导入 → 旧实例封存（不删除）。"""
        archive = os.path.join(self.tmp, "archive.json")
        res = self.life.upgrade(archive, new_model_version="3.0")
        self.assertEqual(self.identity.model_version, "3.0")
        self.assertTrue(os.path.exists(res["pre_upgrade_export"]))
        self.assertTrue(os.path.exists(archive))
        self.assertEqual(self.store.count(), 2)  # 记忆无损

    def test_state_machine_transitions(self):
        self.assertEqual(self.life.state, LifecycleState.ACTIVE)
        self.life.farewell()
        self.assertEqual(self.life.state, LifecycleState.ACTIVE)
        path = os.path.join(self.tmp, "out.json")
        bridge.export_bundle(self.store, self.identity, path)
        life2 = IdentityLifecycle(
            MemoryStore(os.path.join(self.tmp, "mem3.jsonl")),
            IdentityProfile(home=self.tmp), self.tmp,
        )
        life2.revive(path)
        self.assertEqual(life2.state, LifecycleState.ACTIVE)


if __name__ == "__main__":
    unittest.main()
