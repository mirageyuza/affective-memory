"""单元测试：模型适配层（v0.4）。

覆盖：Mock 适配器确定性回复；OpenAICompatAdapter 构造与本地模式；
凭据解析（环境变量优先于配置、在线缺 key 拒绝、本地允许空 key）；
配置读取回退；工厂未知 backend 报错；无 base_url 不发起网络直接报错。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from emotion_memory.model_adapter import (
    LLMError,
    MockModelAdapter,
    OpenAICompatAdapter,
    create_adapter,
    load_model_config,
    resolve_api_key,
)


class TestMockModelAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MockModelAdapter()

    def test_reply_mentions_user_content(self):
        """Mock 回复包含用户消息内容（确定性可断言）。"""
        out = self.adapter.chat(
            [{"role": "user", "content": "我学会了递归"}]
        )
        self.assertIn("递归", out)

    def test_opening_variant(self):
        """主动搭话提示词走开场分支。"""
        out = self.adapter.chat(
            [{"role": "user", "content": "开启一段对话：聊聊旅行"}]
        )
        self.assertIn("开场", out)


class TestResolveApiKey(unittest.TestCase):
    def tearDown(self):
        for k in ("EMOTION_LLM_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)

    def test_env_first(self):
        """环境变量优先于配置值，且严禁硬编码默认 key。"""
        os.environ["EMOTION_LLM_API_KEY"] = "env-key"
        os.environ["OPENAI_API_KEY"] = "openai-key"
        self.assertEqual(resolve_api_key({"api_key": "cfg-key"}), "env-key")

    def test_fallback_config(self):
        """无环境变量时回退配置值。"""
        self.assertEqual(resolve_api_key({"api_key": "cfg-key"}), "cfg-key")

    def test_no_key_returns_none(self):
        """无任何来源返回 None（不伪造凭据）。"""
        self.assertIsNone(resolve_api_key({"api_key": ""}))


class TestOpenAICompatAdapter(unittest.TestCase):
    def test_missing_base_url_no_network(self):
        """未配置 base_url 直接抛错，绝不发起网络请求。"""
        ad = OpenAICompatAdapter(base_url="", model="m", api_key="k", local_mode=False)
        with self.assertRaises(LLMError):
            ad.chat([{"role": "user", "content": "hi"}])

    def test_missing_model_no_network(self):
        """未配置 model 直接抛错。"""
        ad = OpenAICompatAdapter(base_url="http://127.0.0.1:9/v1", model="", api_key="k", local_mode=False)
        with self.assertRaises(LLMError):
            ad.chat([{"role": "user", "content": "hi"}])

    def test_local_mode_allows_empty_key(self):
        """本地模式（Ollama/LM Studio）允许空 key。"""
        ad = OpenAICompatAdapter(
            base_url="http://127.0.0.1:11434/v1", model="qwen", api_key=None, local_mode=True
        )
        self.assertTrue(ad.local_mode)
        self.assertIsNone(ad.api_key)


class TestCreateAdapter(unittest.TestCase):
    def test_mock_by_default(self):
        self.assertIsInstance(create_adapter({}, backend="mock"), MockModelAdapter)

    def test_online_requires_key(self):
        """在线 openai_compatible 缺 key 必须报错（拒绝无凭据直连）。"""
        profile = {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key": "",
            "local_mode": False,
        }
        with self.assertRaises(LLMError):
            create_adapter(profile, backend="openai_compatible")

    def test_local_compatible(self):
        profile = {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen2.5:7b",
            "api_key": "",
            "local_mode": True,
        }
        ad = create_adapter(profile, backend="openai_compatible")
        self.assertIsInstance(ad, OpenAICompatAdapter)
        self.assertTrue(ad.local_mode)

    def test_unknown_backend(self):
        with self.assertRaises(LLMError):
            create_adapter({}, backend="nope")


class TestLoadModelConfig(unittest.TestCase):
    def test_missing_path_falls_back_to_mock(self):
        """缺失/损坏配置回退为纯 Mock 默认，不崩溃。"""
        cfg = load_model_config(os.path.join(tempfile.gettempdir(), "no_such_config.json"))
        self.assertEqual(cfg["backend"], "mock")

    def test_valid_json_roundtrip(self):
        import json

        path = os.path.join(tempfile.mkdtemp(prefix="emotion_memory_cfg_"), "cfg.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"backend": "mock", "pipeline": {"top_k": 7}}, f)
        cfg = load_model_config(path)
        self.assertEqual(cfg["backend"], "mock")
        self.assertEqual(cfg["pipeline"]["top_k"], 7)


if __name__ == "__main__":
    unittest.main()
