"""单元测试：电流桥迁移协议（SHA-256 校验 + 失败拒绝恢复 + 全量比对）。"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory import bridge
from emotion_memory.bridge import BridgeError
from emotion_memory.store.storage import MemoryStore
from emotion_memory.models import AffectLabel, Memory
from emotion_memory.identity import IdentityProfile


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="em_bridge_")
        self.store = MemoryStore(os.path.join(self.tmp, "mem.jsonl"))
        self.m1 = Memory(text="第一条记忆", cluster="c1", entities=["e1"], affect=AffectLabel(0.5, 0.4))
        self.m2 = Memory(text="第二条记忆", cluster="c2", entities=["e2"], affect=AffectLabel(-0.3, 0.8), never_silent=True)
        self.store.add(self.m1)
        self.store.add(self.m2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_export_import_roundtrip(self):
        path = os.path.join(self.tmp, "out.json")
        identity = IdentityProfile(agent_name="tester", model_version="1.0")
        bundle = bridge.export_bundle(self.store, identity, path)
        self.assertEqual(bundle["schema_version"], bridge.SCHEMA_VERSION)
        self.assertEqual(len(bundle["sha256"]), 64)

        target = MemoryStore(os.path.join(self.tmp, "target.jsonl"))
        res = bridge.import_bundle(target, path, identity=IdentityProfile())
        self.assertEqual(res["imported"], 2)
        self.assertTrue(res["identity_restored"])
        self.assertEqual(target.get(self.m1.memory_id).text, "第一条记忆")
        self.assertTrue(target.get(self.m2.memory_id).never_silent)

    def test_tampered_payload_rejected(self):
        path = os.path.join(self.tmp, "out.json")
        bridge.export_bundle(self.store, None, path)
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        bundle["payload"]["memories"][0]["text"] = "被篡改的原文"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False)
        with self.assertRaises(BridgeError):
            bridge.verify_bundle(path)

    def test_failed_import_leaves_target_untouched(self):
        """失败拒绝恢复：目标库保持原状。"""
        path = os.path.join(self.tmp, "out.json")
        bridge.export_bundle(self.store, None, path)
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        bundle["payload"]["memories"].append(
            Memory(text="脏数据", cluster="c").to_dict()
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False)

        target = MemoryStore(os.path.join(self.tmp, "target.jsonl"))
        target.add(Memory(text="原有", cluster="c"))
        before = target.count()
        with self.assertRaises(BridgeError):
            bridge.import_bundle(target, path)
        # 目标库未被污染
        self.assertEqual(target.count(), before)
        self.assertEqual(target.all()[0].text, "原有")

    def test_schema_version_mismatch(self):
        path = os.path.join(self.tmp, "out.json")
        bridge.export_bundle(self.store, None, path)
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        bundle["schema_version"] = 999
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False)
        with self.assertRaises(BridgeError):
            bridge.verify_bundle(path)

    def test_incremental_delta_only_exports_changed(self):
        path = os.path.join(self.tmp, "delta.json")
        baseline = {m.memory_id: copy.deepcopy(m) for m in self.store.all()}
        bridge.export_delta(self.store, baseline, path)
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        self.assertTrue(bundle["payload"]["incremental"])
        self.assertEqual(len(bundle["payload"]["memories"]), 0)  # 无变更

        self.store.update_meta(self.m1.memory_id, accessibility=0.5)
        path2 = os.path.join(self.tmp, "delta2.json")
        bridge.export_delta(self.store, baseline, path2)
        with open(path2, "r", encoding="utf-8") as f:
            bundle2 = json.load(f)
        self.assertEqual(len(bundle2["payload"]["memories"]), 1)


if __name__ == "__main__":
    unittest.main()
