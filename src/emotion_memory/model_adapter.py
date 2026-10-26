"""模型适配层 (v0.6)：多 Provider 注册表，统一 LLM 接口。

- ``MODEL_PROVIDERS``：provider 名 → 适配器类的注册表，构造时按名路由，
  新增 AI 只需注册新类，零改动接入；
- ``ModelAdapter``：统一接口 ``generate()`` / ``chat()`` / ``embed()``；
- 适配器清单：
  * ``OpenAICompatAdapter``：OpenAI 兼容全家（OpenAI / DeepSeek / 通义 Qwen /
    Kimi Moonshot / 智谱 GLM / MiniMax / 豆包火山方舟 / 文心千帆兼容模式），
    base_url + api_key + model 可配，支持可选 headers 透传（各家鉴权头差异）；
  * ``AnthropicAdapter``：Claude Messages API（x-api-key + anthropic-version
    头，system/content 结构转换）；
  * ``GeminiAdapter``：Google generateContent API（x-goog-api-key 头，
    generationConfig 转换）；
  * ``LocalCompatAdapter``：本地 OpenAI 兼容端点（Ollama / LM Studio /
    llama.cpp server），无鉴权或轻鉴权；
  * ``TransformersAdapter``：本地 HuggingFace transformers 直连接口（预留，
    可选依赖，未安装时构造即报错）；
  * ``MockModelAdapter``：内置零依赖确定性 Mock，用于单测与无凭据演示。

凭据安全：api_key 严禁硬编码，一律走环境变量或配置注入。解析优先级：
  provider 专属环境变量（如 ANTHROPIC_API_KEY / GEMINI_API_KEY）→
  通用 EMOTION_LLM_API_KEY → OPENAI_API_KEY → 配置 profile。
本地端点（local_compat / local_mode=True，如 Ollama / LM Studio）允许空 key。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    """模型调用失败（网络 / 鉴权 / 响应异常统一包装）。"""


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    """统一 POST JSON 请求（仅标准库 urllib，零第三方依赖）。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")
        except Exception:
            pass
        raise LLMError(f"模型端点 HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"模型端点连接失败: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise LLMError(f"模型端点超时/网络错误: {exc}") from exc


class ModelAdapter:
    """模型适配基类：所有适配器实现同一 generate / chat / embed 接口。"""

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """统一生成接口（默认等价 chat，子类可覆写）。"""
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        raise NotImplementedError

    def embed(self, text: str) -> List[float]:
        """文本嵌入接口；不提供 embedding 的适配器应抛 LLMError 说明。"""
        raise NotImplementedError

    @property
    def kind(self) -> str:
        return self.__class__.__name__


class MockModelAdapter(ModelAdapter):
    """内置确定性 Mock：零依赖、可测试，展示完整管线不依赖任何外部服务。"""

    def __init__(self, style: str = "generic"):
        self.style = style

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        ).strip()
        snippet = last_user[:80] or "想跟你说说话"
        if "开启一段对话" in last_user or "主动搭话" in last_user:
            return f"(mock开场) 正好想起：{snippet}。"
        return f"(mock回复) 你刚说「{snippet}」，这段对话我会一直记得。"

    def embed(self, text: str) -> List[float]:
        """确定性 8 维向量（同一文本恒同，不同文本不同）。"""
        h = 0
        for ch in str(text):
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        x = h
        vec: List[float] = []
        for _ in range(8):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            vec.append(round((x / 0x7FFFFFFF) * 2.0 - 1.0, 6))
        return vec


class OpenAICompatAdapter(ModelAdapter):
    """OpenAI 兼容端点适配器（OpenAI / DeepSeek / Qwen / Kimi / GLM /
    MiniMax / 豆包 / 文心千帆 等均为此协议）。

    凭据不在本类硬编码：api_key 由工厂解析（环境变量优先于配置）后注入；
    headers 透传用于各家鉴权头差异（如 X-API-Key / api-key 等）。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        local_mode: bool = False,
        headers: Optional[Dict[str, str]] = None,
        provider: Optional[str] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.local_mode = bool(local_mode)
        self.headers = dict(headers or {})
        self.provider = provider

    def _check(self) -> None:
        if not self.base_url:
            raise LLMError("base_url 未配置，无法发起请求")
        if not self.model:
            raise LLMError("model 未配置，无法发起请求")

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        headers.update(self.headers)
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        self._check()
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        data = _post_json(url, payload, self._build_headers(), self.timeout)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型端点响应格式异常: {str(data)[:300]}") from exc

    def embed(self, text: str) -> List[float]:
        """调用 OpenAI 兼容 /embeddings 接口。"""
        self._check()
        url = self.base_url + "/embeddings"
        payload = {"model": self.model, "input": [str(text)]}
        data = _post_json(url, payload, self._build_headers(), self.timeout)
        try:
            return list(data["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型端点嵌入响应格式异常: {str(data)[:300]}") from exc


class LocalCompatAdapter(OpenAICompatAdapter):
    """本地 OpenAI 兼容端点（Ollama / LM Studio / llama.cpp server）。

    无鉴权或轻鉴权：默认不强制 api_key；如需轻鉴权（如 llama.cpp server 的
    ``Authorization: Bearer xxx`` 或 LM Studio 自定义头）通过 headers 透传。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        provider: str = "local_compat",
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            local_mode=True,
            headers=headers,
            provider=provider,
        )


class AnthropicAdapter(ModelAdapter):
    """Claude Messages API 适配器（system / content 结构转换）。

    鉴权头：``x-api-key`` + ``anthropic-version``。
    """

    DEFAULT_VERSION = "2023-06-01"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        provider: str = "anthropic",
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.api_version = api_version or self.DEFAULT_VERSION
        self.headers = dict(headers or {})
        self.provider = provider

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
        }
        headers.update(self.headers)
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        if not self.base_url:
            raise LLMError("base_url 未配置，无法发起请求")
        if not self.model:
            raise LLMError("model 未配置，无法发起请求")
        if not self.api_key:
            raise LLMError(
                "api_key 未配置（provider=anthropic）：请通过环境变量 "
                "ANTHROPIC_API_KEY 或配置注入后重试"
            )
        # 结构转换：system 消息拆出为顶层 system 参数，其余转 content 数组
        system_parts = [
            str(m.get("content", ""))
            for m in messages
            if m.get("role") == "system" and m.get("content")
        ]
        content_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            c = m.get("content", "")
            content_msgs.append(
                {
                    "role": role if role in ("user", "assistant") else "assistant",
                    "content": [{"type": "text", "text": str(c)}]
                    if not isinstance(c, list)
                    else c,
                }
            )
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "messages": content_msgs,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        url = self.base_url + "/messages"
        data = _post_json(url, payload, self._build_headers(), self.timeout)
        try:
            texts = [
                str(b.get("text", ""))
                for b in data["content"]
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            return "\n".join(texts) if texts else ""
        except (KeyError, TypeError) as exc:
            raise LLMError(f"模型端点响应格式异常: {str(data)[:300]}") from exc

    def embed(self, text: str) -> List[float]:
        raise LLMError(
            "anthropic provider 未提供 embedding 接口；"
            "请使用 openai_compat / gemini / local_compat 等支持嵌入的 provider"
        )


class GeminiAdapter(ModelAdapter):
    """Google Gemini generateContent API 适配器。

    鉴权头：``x-goog-api-key``；生成参数经 generationConfig 转换。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        embed_model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        provider: str = "gemini",
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.embed_model = embed_model or "text-embedding-004"
        self.headers = dict(headers or {})
        self.provider = provider

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key or "",
        }
        headers.update(self.headers)
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        if not self.base_url:
            raise LLMError("base_url 未配置，无法发起请求")
        if not self.model:
            raise LLMError("model 未配置，无法发起请求")
        if not self.api_key:
            raise LLMError(
                "api_key 未配置（provider=gemini）：请通过环境变量 "
                "GEMINI_API_KEY 或配置注入后重试"
            )
        system_parts = [
            str(m.get("content", ""))
            for m in messages
            if m.get("role") == "system" and m.get("content")
        ]
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            c = m.get("content", "")
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": str(c)}] if not isinstance(c, list) else c,
                }
            )
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        if system_parts:
            payload["system_instruction"] = {
                "parts": [{"text": "\n".join(system_parts)}]
            }
        url = self.base_url + "/models/" + self.model + ":generateContent"
        data = _post_json(url, payload, self._build_headers(), self.timeout)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型端点响应格式异常: {str(data)[:300]}") from exc

    def embed(self, text: str) -> List[float]:
        """调用 Gemini embedContent API（独立嵌入模型）。"""
        if not self.base_url:
            raise LLMError("base_url 未配置，无法发起请求")
        if not self.api_key:
            raise LLMError(
                "api_key 未配置（provider=gemini）：请通过环境变量 "
                "GEMINI_API_KEY 或配置注入后重试"
            )
        payload = {"content": {"parts": [{"text": str(text)}]}}
        url = self.base_url + "/models/" + self.embed_model + ":embedContent"
        data = _post_json(url, payload, self._build_headers(), self.timeout)
        try:
            return list(data["embedding"]["values"])
        except (KeyError, TypeError) as exc:
            raise LLMError(f"模型端点嵌入响应格式异常: {str(data)[:300]}") from exc


class TransformersAdapter(ModelAdapter):
    """HuggingFace transformers 本地直连（预留，可选依赖）。

    transformers 未安装时构造即报错（明确提示，而不是运行到一半才失败）。
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cpu"):
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # ModuleNotFoundError 等
            raise LLMError(
                "transformers 未安装：本地直连需 `pip install transformers torch` "
                "（可选依赖，不影响其余功能）。"
            ) from exc
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline(
            "text-generation", model=model_name, device=device, trust_remote_code=True
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        prompt = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
            for m in messages
        )
        out = self._pipe(
            prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=temperature > 0,
        )
        return out[0]["generated_text"].replace(prompt, "").strip()


# ---------------------------------------------------------------------------
# Provider 注册表
# ---------------------------------------------------------------------------

PROVIDER_META: Dict[str, Dict[str, Any]] = {
    # provider 名 → {env_key: 鉴权所需环境变量, base_url: 官方端点, model: 默认模型, auth: 鉴权方式}
    "openai_compat": {"env_key": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "auth": "bearer"},
    "deepseek": {"env_key": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "auth": "bearer"},
    "qwen": {"env_key": "DASHSCOPE_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "auth": "bearer"},
    "kimi": {"env_key": "MOONSHOT_API_KEY", "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k", "auth": "bearer"},
    "glm": {"env_key": "ZHIPUAI_API_KEY", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash", "auth": "bearer"},
    "minimax": {"env_key": "MINIMAX_API_KEY", "base_url": "https://api.minimax.chat/v1", "model": "abab6.5s-chat", "auth": "bearer"},
    "doubao": {"env_key": "ARK_API_KEY", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-pro-32k", "auth": "bearer"},
    "wenxin": {"env_key": "QIANFAN_API_KEY", "base_url": "https://qianfan.baidubce.com/v2", "model": "ernie-4.0-8k", "auth": "bearer"},
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-20250514", "auth": "x-api-key"},
    "gemini": {"env_key": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-2.0-flash", "auth": "x-goog-api-key"},
    "local_compat": {"env_key": None, "base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "auth": "none"},
}

# provider 名 → 适配器类（注册表：新增 AI 只需在此登记 + PROVIDER_META 补元数据）
MODEL_PROVIDERS: Dict[str, Any] = {
    "mock": MockModelAdapter,
    "openai_compat": OpenAICompatAdapter,
    "deepseek": OpenAICompatAdapter,
    "qwen": OpenAICompatAdapter,
    "kimi": OpenAICompatAdapter,
    "glm": OpenAICompatAdapter,
    "minimax": OpenAICompatAdapter,
    "doubao": OpenAICompatAdapter,
    "wenxin": OpenAICompatAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "local_compat": LocalCompatAdapter,
    "transformers": TransformersAdapter,
}

# OpenAI 兼容全家（一个适配器，靠 base_url+model+headers 区分各家）
_OPENAI_COMPAT_PROVIDERS = frozenset(
    {"openai_compat", "deepseek", "qwen", "kimi", "glm", "minimax", "doubao", "wenxin"}
)


def resolve_api_key(
    profile: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
) -> Optional[str]:
    """凭据解析：provider 专属环境变量 → 通用变量 → 配置，绝不硬编码。"""
    profile = profile or {}
    env_names: List[str] = []
    if provider:
        meta = PROVIDER_META.get(provider)
        if meta and meta.get("env_key"):
            env_names.append(meta["env_key"])
    env_names.extend(("EMOTION_LLM_API_KEY", "OPENAI_API_KEY"))
    for env_name in env_names:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    val = str(profile.get("api_key") or "").strip()
    return val or None


def create_adapter(
    profile: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    provider: Optional[str] = None,
) -> ModelAdapter:
    """工厂：按 provider 名路由注册表创建适配器。

    - ``provider``：优先使用（profile['provider'] 亦可）；
    - ``backend``：兼容旧命名（openai_compatible → openai_compat；
      local_mock → mock），未显式指定时与 provider 等价。
    profile 字段：provider / base_url / model / api_key / headers /
      local_mode / anthropic_version / embed_model。
    """
    profile = profile or {}
    provider = (provider or str(profile.get("provider", "")) or "").lower()
    backend = (backend or str(profile.get("backend", "")) or "").lower()
    if not provider:
        legacy_map = {"openai_compatible": "openai_compat", "local_mock": "mock"}
        provider = legacy_map.get(backend, backend)

    if provider == "mock":
        return MockModelAdapter()
    if provider == "transformers":
        return TransformersAdapter(
            model_name=str(profile.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")),
            device=str(profile.get("device", "cpu")),
        )

    cls = MODEL_PROVIDERS.get(provider)
    if cls is None:
        raise LLMError(
            f"未知 provider: {provider or '(empty)'}（支持 "
            f"{', '.join(sorted(MODEL_PROVIDERS))}）"
        )

    meta = PROVIDER_META.get(provider, {})
    base_url = str(profile.get("base_url") or meta.get("base_url", ""))
    model = str(profile.get("model") or meta.get("model", ""))
    local_mode = bool(profile.get("local_mode", False)) or provider == "local_compat"
    api_key = resolve_api_key(profile, provider=provider)
    need_key = bool(meta.get("env_key")) and not local_mode
    if need_key and not api_key:
        raise LLMError(
            f"api_key 未配置（provider={provider}）：请通过环境变量 "
            f"{meta.get('env_key', 'OPENAI_API_KEY')} 或配置 profile['api_key'] 提供"
        )

    headers = profile.get("headers")
    if provider in _OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatAdapter(
            base_url=base_url,
            model=model,
            api_key=api_key,
            local_mode=local_mode,
            headers=headers,
            provider=provider,
        )
    if provider == "local_compat":
        return LocalCompatAdapter(
            base_url=base_url,
            model=model,
            api_key=api_key,
            headers=headers,
            provider=provider,
        )
    if provider == "anthropic":
        return AnthropicAdapter(
            base_url=base_url,
            model=model,
            api_key=api_key,
            api_version=str(profile.get("anthropic_version", "") or None),
            headers=headers,
            provider=provider,
        )
    if provider == "gemini":
        return GeminiAdapter(
            base_url=base_url,
            model=model,
            api_key=api_key,
            embed_model=str(profile.get("embed_model", "") or None),
            headers=headers,
            provider=provider,
        )
    raise LLMError(f"未知 provider: {provider}")


def load_model_config(path: Optional[str] = None) -> Dict[str, Any]:
    """读取模型接入配置 JSON；失败或缺失时回退为纯 Mock 默认配置。"""
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"backend": "mock", "model": {}, "pipeline": {}, "proactive": {}}
