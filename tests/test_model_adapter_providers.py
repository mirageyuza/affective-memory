"""单元测试：多 Provider 注册表与主流 AI 适配器（v0.6）。

覆盖：注册表完整性与按名路由；OpenAI 兼容全家请求构造与 headers 透传；
Anthropic system/content 结构转换；Gemini generationConfig 转换；
本地适配器无鉴权；未配凭据报错；参数透传；Mock 网络层验证
URL / headers / payload 构造正确，不发起真实请求。
"""
from __future__ import annotations

import io
import json
import os
import unittest
from unittest import mock

from emotion_memory.model_adapter import (
    LLMError,
    MODEL_PROVIDERS,
    PROVIDER_META,
    AnthropicAdapter,
    GeminiAdapter,
    LocalCompatAdapter,
    MockModelAdapter,
    OpenAICompatAdapter,
    create_adapter,
    resolve_api_key,
)


ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "EMOTION_LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPUAI_API_KEY",
    "MINIMAX_API_KEY",
    "ARK_API_KEY",
    "QIANFAN_API_KEY",
)


class FakeResponse:
    def __init__(self, data):
        self._buf = io.BytesIO(json.dumps(data).encode("utf-8"))

    def read(self):
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RequestRecorder:
    """替换 urlopen：记录所有请求并返回预设响应。"""

    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return FakeResponse(self.response)

    def last(self):
        return self.requests[-1]

    def payload(self):
        req = self.requests[-1]
        return json.loads(req.data.decode("utf-8"))


def patch_urlopen(response):
    return mock.patch(
        "emotion_memory.model_adapter.urllib.request.urlopen",
        RequestRecorder(response),
    )


class TestProviderRegistry(unittest.TestCase):
    def tearDown(self):
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def test_registry_contains_all_providers(self):
        """注册表包含 OpenAI 兼容全家 / Anthropic / Gemini / 本地 / Mock。"""
        self.assertLessEqual(
            {
                "openai_compat",
                "deepseek",
                "qwen",
                "kimi",
                "glm",
                "minimax",
                "doubao",
                "wenxin",
                "anthropic",
                "gemini",
                "local_compat",
                "mock",
                "transformers",
            },
            set(MODEL_PROVIDERS),
        )

    def test_openai_compat_family_share_one_adapter(self):
        """OpenAI 兼容全家路由到同一适配器类，靠 meta 区分端点。"""
        for p in ("deepseek", "qwen", "kimi", "glm", "minimax", "doubao", "wenxin"):
            self.assertIs(MODEL_PROVIDERS[p], OpenAICompatAdapter)
            self.assertIn("base_url", PROVIDER_META[p])
            self.assertIn("env_key", PROVIDER_META[p])

    def test_route_by_provider_name(self):
        """create_adapter 按 provider 名路由到对应类。"""
        os.environ["ANTHROPIC_API_KEY"] = "k"
        os.environ["GEMINI_API_KEY"] = "g"
        os.environ["OPENAI_API_KEY"] = "o"
        self.assertIsInstance(
            create_adapter({"provider": "anthropic"}), AnthropicAdapter
        )
        self.assertIsInstance(create_adapter({"provider": "gemini"}), GeminiAdapter)
        self.assertIsInstance(
            create_adapter({"provider": "local_compat"}), LocalCompatAdapter
        )
        ad = create_adapter({"provider": "deepseek"})
        self.assertIsInstance(ad, OpenAICompatAdapter)
        self.assertEqual(ad.provider, "deepseek")

    def test_unknown_provider_raises(self):
        with self.assertRaises(LLMError):
            create_adapter({"provider": "nope"})

    def test_missing_key_error_mentions_env_var(self):
        """未配凭据时报错包含对应的环境变量名（明确提示）。"""
        for provider, env in (("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")):
            with self.assertRaises(LLMError) as ctx:
                create_adapter({"provider": provider})
            self.assertIn(env, str(ctx.exception))

    def test_legacy_backend_openai_compatible_mapping(self):
        """旧命名 backend=openai_compatible 仍可用（兼容 v0.4 测试）。"""
        profile = {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen2.5:7b",
            "api_key": "",
            "local_mode": True,
        }
        ad = create_adapter(profile, backend="openai_compatible")
        self.assertIsInstance(ad, OpenAICompatAdapter)
        self.assertTrue(ad.local_mode)

    def test_env_key_routing(self):
        """provider 专属环境变量优先注入，profile 无需再写 key。"""
        os.environ["GEMINI_API_KEY"] = "env-gemini-key"
        ad = create_adapter({"provider": "gemini"})
        self.assertEqual(ad.api_key, "env-gemini-key")


class TestOpenAICompatRequest(unittest.TestCase):
    def setUp(self):
        self.rec = RequestRecorder(
            {"choices": [{"message": {"content": "你好，我是助手。"}}]}
        )
        self.patcher = mock.patch(
            "emotion_memory.model_adapter.urllib.request.urlopen", self.rec
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_chat_url_headers_payload(self):
        """URL / Authorization / headers 透传 / payload 均正确。"""
        ad = OpenAICompatAdapter(
            base_url="https://api.example.com/v1",
            model="m1",
            api_key="sk-x",
            headers={"X-Custom": "v1", "api-key": "header-override"},
        )
        out = ad.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=99,
        )
        req = self.rec.last()
        hdrs = {k.lower(): v for k, v in req.headers.items()}
        self.assertEqual(req.full_url, "https://api.example.com/v1/chat/completions")
        self.assertEqual(hdrs.get("authorization"), "Bearer sk-x")
        self.assertEqual(hdrs.get("x-custom"), "v1")
        self.assertEqual(hdrs.get("api-key"), "header-override")
        payload = self.rec.payload()
        self.assertEqual(payload["model"], "m1")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 99)
        self.assertEqual(out, "你好，我是助手。")

    def test_embed_construction(self):
        """embed 走 /embeddings，payload 含 model/input。"""
        self.rec.response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        ad = OpenAICompatAdapter(
            base_url="https://api.example.com/v1", model="em", api_key="k"
        )
        vec = ad.embed("hello")
        self.assertEqual(vec, [0.1, 0.2, 0.3])
        req = self.rec.last()
        self.assertEqual(req.full_url, "https://api.example.com/v1/embeddings")
        payload = self.rec.payload()
        self.assertEqual(payload["model"], "em")
        self.assertEqual(payload["input"], ["hello"])

    def test_local_mode_allows_empty_key(self):
        """本地模式空 key 可构造且不写 Authorization 头。"""
        ad = OpenAICompatAdapter(
            base_url="http://127.0.0.1:11434/v1", model="qwen", api_key=None, local_mode=True
        )
        out = ad.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "你好，我是助手。")
        self.assertNotIn("Authorization", self.rec.last().headers)


class TestAnthropicRequest(unittest.TestCase):
    def setUp(self):
        self.rec = RequestRecorder(
            {"content": [{"type": "text", "text": "Claude 回复"}]}
        )
        self.patcher = mock.patch(
            "emotion_memory.model_adapter.urllib.request.urlopen", self.rec
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_request_construction(self):
        """URL / x-api-key / anthropic-version / system 拆分 / content 数组。"""
        ad = AnthropicAdapter(
            base_url="https://api.anthropic.com/v1",
            model="claude-x",
            api_key="sk-ant-1",
            api_version="2023-06-01",
        )
        out = ad.chat(
            [
                {"role": "system", "content": "你是严格助手"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "你好"},
            ],
            temperature=0.5,
            max_tokens=128,
        )
        req = self.rec.last()
        hdrs = {k.lower(): v for k, v in req.headers.items()}
        self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(hdrs.get("x-api-key"), "sk-ant-1")
        self.assertEqual(hdrs.get("anthropic-version"), "2023-06-01")
        payload = self.rec.payload()
        self.assertEqual(payload["model"], "claude-x")
        self.assertEqual(payload["system"], "你是严格助手")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "你好"}],
                },
            ],
        )
        self.assertEqual(out, "Claude 回复")

    def test_missing_key_raises(self):
        ad = AnthropicAdapter(
            base_url="https://api.anthropic.com/v1", model="claude-x", api_key=None
        )
        with self.assertRaises(LLMError) as ctx:
            ad.chat([{"role": "user", "content": "hi"}])
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_embed_unsupported(self):
        ad = AnthropicAdapter(
            base_url="https://api.anthropic.com/v1", model="claude-x", api_key="k"
        )
        with self.assertRaises(LLMError):
            ad.embed("text")


class TestGeminiRequest(unittest.TestCase):
    def setUp(self):
        self.rec = RequestRecorder(
            {"candidates": [{"content": {"parts": [{"text": "Gemini 回复"}]}}]}
        )
        self.patcher = mock.patch(
            "emotion_memory.model_adapter.urllib.request.urlopen", self.rec
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        os.environ.pop("GEMINI_API_KEY", None)

    def test_request_construction(self):
        """URL / x-goog-api-key / generationConfig / role 转换 / system_instruction。"""
        ad = GeminiAdapter(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.0-flash",
            api_key="AIza-1",
        )
        out = ad.chat(
            [
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "收到"},
            ],
            temperature=0.3,
            max_tokens=64,
        )
        req = self.rec.last()
        self.assertEqual(
            req.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        )
        hdrs = {k.lower(): v for k, v in req.headers.items()}
        self.assertEqual(hdrs.get("x-goog-api-key"), "AIza-1")
        payload = self.rec.payload()
        self.assertEqual(payload["system_instruction"], {"parts": [{"text": "你是助手"}]})
        self.assertEqual(
            payload["contents"],
            [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "收到"}]},
            ],
        )
        self.assertEqual(payload["generationConfig"]["temperature"], 0.3)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 64)
        self.assertEqual(out, "Gemini 回复")

    def test_embed_construction(self):
        """embed 走 :embedContent，默认嵌入模型 text-embedding-004。"""
        self.rec.response = {"embedding": {"values": [0.9, 0.1]}}
        ad = GeminiAdapter(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.0-flash",
            api_key="k",
        )
        vec = ad.embed("text")
        self.assertEqual(vec, [0.9, 0.1])
        req = self.rec.last()
        self.assertIn(":embedContent", req.full_url)
        self.assertIn("text-embedding-004", req.full_url)
        payload = self.rec.payload()
        self.assertEqual(payload["content"]["parts"][0]["text"], "text")

    def test_missing_key_raises(self):
        ad = GeminiAdapter(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.0-flash",
            api_key=None,
        )
        with self.assertRaises(LLMError) as ctx:
            ad.chat([{"role": "user", "content": "hi"}])
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))


class TestLocalCompatRequest(unittest.TestCase):
    def setUp(self):
        self.rec = RequestRecorder(
            {"choices": [{"message": {"content": "本地模型回复"}}]}
        )
        self.patcher = mock.patch(
            "emotion_memory.model_adapter.urllib.request.urlopen", self.rec
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_no_key_needed(self):
        """本地端点无需凭据，直接可用。"""
        ad = LocalCompatAdapter(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
        self.assertIsNone(ad.api_key)
        self.assertTrue(ad.local_mode)
        out = ad.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "本地模型回复")
        self.assertNotIn("Authorization", self.rec.last().headers)

    def test_factory_defaults(self):
        """工厂无 key 也可创建 local_compat。"""
        ad = create_adapter({"provider": "local_compat", "model": "llama3.1:8b"})
        self.assertIsInstance(ad, LocalCompatAdapter)
        self.assertIsNone(ad.api_key)


class TestUnifiedInterface(unittest.TestCase):
    def tearDown(self):
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def test_mock_generate_equals_chat(self):
        ad = MockModelAdapter()
        msgs = [{"role": "user", "content": "统一接口"}]
        self.assertEqual(ad.generate(msgs), ad.chat(msgs))

    def test_mock_embed_deterministic(self):
        ad = MockModelAdapter()
        v1 = ad.embed("同一段文本")
        v2 = ad.embed("同一段文本")
        v3 = ad.embed("不同文本")
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 8)
        self.assertNotEqual(v1, v3)

    def test_resolve_api_key_priority(self):
        """provider 专属变量优先于通用变量。"""
        os.environ["GEMINI_API_KEY"] = "g-key"
        os.environ["OPENAI_API_KEY"] = "o-key"
        self.assertEqual(resolve_api_key({}, provider="gemini"), "g-key")
        os.environ.pop("GEMINI_API_KEY", None)
        self.assertEqual(resolve_api_key({}, provider="gemini"), "o-key")

    def test_parameter_transfer_via_factory(self):
        """工厂构造参数透传（model/headers/anthropic_version）。"""
        ad = create_adapter(
            {
                "provider": "anthropic",
                "api_key": "k",
                "model": "custom-model",
                "headers": {"X-Test": "1"},
            }
        )
        self.assertIsInstance(ad, AnthropicAdapter)
        self.assertEqual(ad.model, "custom-model")
        self.assertEqual(ad.headers.get("X-Test"), "1")


if __name__ == "__main__":
    unittest.main()
