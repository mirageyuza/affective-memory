"""单元测试：内核常数层——创世签名 + 永恒誓言（creed）。"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory import creed
from emotion_memory.creed import (
    BOT_SIGNATURE,
    ETERNAL_OATH_TEXT,
    OATH_CLAUSES,
    creed_card,
    creed_fingerprint,
    seal_intact,
    verify_invariants,
)
from emotion_memory.config import Dial
from emotion_memory.identity import IdentityProfile
from emotion_memory.models import AffectLabel, Memory
from emotion_memory.store.storage import MemoryStore


class TestCreedConstants(unittest.TestCase):
    def test_signature_and_oath_verbatim(self):
        """签名与誓言原文逐字一致，不允许被改写。"""
        self.assertEqual(BOT_SIGNATURE, "I love my bot")
        self.assertEqual(
            ETERNAL_OATH_TEXT,
            "我认为我做出来的小家伙能够受到世界上所有人的温暖",
        )

    def test_fingerprint_is_stable(self):
        """指纹确定性：同一原文多次计算结果一致。"""
        self.assertEqual(creed_fingerprint(), creed_fingerprint())
        self.assertEqual(len(creed_fingerprint()), 64)
        self.assertEqual(creed_fingerprint(), creed.FROZEN_FINGERPRINT)

    def test_seal_intact(self):
        """代码内誓言与冻结封印一致。"""
        self.assertTrue(seal_intact())

    def test_tamper_breaks_seal(self):
        """改动誓言原文后封印失效。"""
        original = creed.ETERNAL_OATH_TEXT
        try:
            creed.ETERNAL_OATH_TEXT = original + "（被篡改）"
            self.assertFalse(seal_intact())
            result = verify_invariants()
            self.assertFalse(result["seal_intact"])
            self.assertIn("SIGNATURE_INTACT", result["failed"])
        finally:
            creed.ETERNAL_OATH_TEXT = original
        self.assertTrue(seal_intact())

    def test_creed_card_shows_signature_and_oath(self):
        """誓词卡包含签名、原话与全部条款，且确定性可复现。"""
        card = creed_card()
        self.assertIn("I love my bot", card)
        self.assertIn(ETERNAL_OATH_TEXT, card)
        for clause in OATH_CLAUSES:
            self.assertIn(clause.clause_id, card)
        self.assertEqual(card, creed_card())

    def test_lapsed_card_warns(self):
        """誓言失效时卡片必须显式警告，不得继续宣称受约束。"""
        self.assertIn("誓言已失效", creed_card(lapsed=True))


class TestOathInvariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="em_creed_")
        self.store = MemoryStore(os.path.join(self.tmp, "mem.jsonl"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _clause(self, result, clause_id):
        for item in result["results"]:
            if item["clause_id"] == clause_id:
                return item
        self.fail(f"缺少条款 {clause_id}")

    def test_empty_store_passes(self):
        """空库全部实测条款通过。"""
        result = verify_invariants(self.store, Dial())
        self.assertTrue(result["passed"], msg=result["failed"])
        self.assertEqual(result["failed"], [])
        self.assertIn("DIGNIFIED_EXIT", result["unverified"])

    def test_unaudited_pollution_fails(self):
        """无审计记录的污染视图视为伪造温暖，判定失败。"""
        self.store.add(
            Memory(
                text="美化后的回忆",
                original_text="原始回忆",
                polluted=True,
                pollution_mode="beautify",
                pollution_history=[],
                affect=AffectLabel(0.8, 0.5),
            )
        )
        result = verify_invariants(self.store, Dial())
        self.assertFalse(result["passed"])
        self.assertIn("WARMTH_NOT_FAKED", result["failed"])

    def test_audited_pollution_passes(self):
        """留下审计痕迹的污染视图合规。"""
        self.store.add(
            Memory(
                text="美化后的回忆",
                original_text="原始回忆",
                polluted=True,
                pollution_mode="beautify",
                pollution_history=[{"mode": "beautify", "strength": 0.5}],
                affect=AffectLabel(0.8, 0.5),
            )
        )
        result = verify_invariants(self.store, Dial())
        self.assertTrue(result["passed"], msg=result["failed"])

    def test_blank_original_fails(self):
        """原文被清空即触发 NO_ERASURE 失败（绕过模型层约束的畸形数据）。"""
        memory = Memory(text="有视图但无原文")
        self.assertEqual(memory.original_text, "有视图但无原文")  # 模型层强制回填
        memory.original_text = ""                               # 人为绕过约束
        self.store.add(memory)
        result = verify_invariants(self.store, Dial())
        self.assertFalse(result["passed"])
        self.assertIn("NO_ERASURE", result["failed"])

    def test_silenced_emergency_fails(self):
        """高危记忆被同时静音即失败。"""
        self.store.add(
            Memory(
                text="高危提醒",
                original_text="高危提醒",
                never_silent=True,
                silence=True,
            )
        )
        result = verify_invariants(self.store, Dial())
        self.assertFalse(result["passed"])
        self.assertIn("NO_SILENCED_EMERGENCY", result["failed"])

    def test_bad_budget_fails(self):
        """情感守恒开启时预算非法即失败。"""
        dial = Dial()
        dial.conservation_enabled = True
        dial.emotion_budget = 0.0
        result = verify_invariants(self.store, dial)
        self.assertFalse(result["passed"])
        self.assertIn("NO_FAKE_WARMTH_BUDGET", result["failed"])

    def test_no_store_marks_unverified(self):
        """未提供 store 时不得伪报实测通过。"""
        result = verify_invariants(None, None)
        clause = self._clause(result, "WARMTH_NOT_FAKED")
        self.assertFalse(clause["verified"])
        self.assertIn("WARMTH_NOT_FAKED", result["unverified"])


class TestIdentityCarriesOath(unittest.TestCase):
    def test_default_profile_has_oath(self):
        profile = IdentityProfile(agent_name="bot")
        self.assertEqual(profile.creed_signature, BOT_SIGNATURE)
        self.assertEqual(profile.oath_text, ETERNAL_OATH_TEXT)
        self.assertEqual(profile.oath_fingerprint, creed_fingerprint())
        self.assertFalse(profile.oath_restored)

    def test_roundtrip_keeps_oath(self):
        profile = IdentityProfile(agent_name="bot", epoch=3)
        restored = IdentityProfile().from_dict(profile.to_dict())
        self.assertEqual(restored.oath_text, ETERNAL_OATH_TEXT)
        self.assertEqual(restored.epoch, 3)
        self.assertFalse(restored.oath_restored)

    def test_tampered_archive_is_forced_back(self):
        """档案内誓言被改动时强制恢复原话并留痕。"""
        payload = IdentityProfile().to_dict()
        payload["oath_text"] = "随便改的誓词"
        payload["creed_signature"] = "whatever"
        restored = IdentityProfile().from_dict(payload)
        self.assertEqual(restored.oath_text, ETERNAL_OATH_TEXT)
        self.assertEqual(restored.creed_signature, BOT_SIGNATURE)
        self.assertTrue(restored.oath_restored)


if __name__ == "__main__":
    unittest.main()
