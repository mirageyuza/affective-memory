"""模型适配层 (v0.4)：统一 LLM 接口，在线 API 与本地模型同一套调用。

- ``OpenAICompatAdapter``：OpenAI 兼容在线 / 本地端点（OpenAI / Ollama /
  LM Studio 等均为此协议），仅用标准库 urllib 发 HTTP，零第三方运行时依赖；
- ``TransformersAdapter``：本地 HuggingFace transformers 直连接口（预留，
  transformers 为可选依赖，未安装时构造即报错）；
- ``MockModelAdapter``：内置零依赖确定性 Mock，用于单测与无凭据演示；
- ``create_adapter()``：按配置与 backend 工厂创建适配器；
- ``load_model_config()``：读取模型接入配置文件（JSON）。

凭据安全：api_key 严禁硬编码。解析优先级：
  环境变量 ``EMOTION_LLM_API_KEY`` → 环境变量 ``OPENAI_API_KEY`` → 配置 profile。
本地端点（local_mode=True，如 Ollama / LM Studio）允许空 key 占位。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    """模型调用失败（网络 / 鉴权 / 响应异常统一包装）。"""


class ModelAdapter:
    """模型适配基类：所有适配器实现同一 chat 接口。"""

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
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


class OpenAICompatAdapter(ModelAdapter):
    """OpenAI 兼容端点适配器（在线 OpenAI / 本地 Ollama、LM Studio 同构）。

    凭据不在本类硬编码：api_key 由工厂解析（环境变量优先于配置）后注入。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        local_mode: bool = False,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.local_mode = bool(local_mode)

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
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
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
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型端点响应格式异常: {str(data)[:300]}") from exc


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


def resolve_api_key(profile: Dict[str, Any]) -> Optional[str]:
    """凭据解析：环境变量优先，其次配置，绝不硬编码。"""
    for env_name in ("EMOTION_LLM_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    val = str(profile.get("api_key") or "").strip()
    return val or None


def create_adapter(
    profile: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
) -> ModelAdapter:
    """工厂：按 backend 与配置 profile 创建适配器。

    profile 字段：base_url / model / api_key / local_mode（openai_compatible）；
                  model_name / device（transformers）。
    """
    profile = profile or {}
    backend = (backend or str(profile.get("backend", "mock"))).lower()
    if backend in ("mock", "local_mock"):
        return MockModelAdapter()
    if backend == "openai_compatible":
        local_mode = bool(profile.get("local_mode", False))
        api_key = resolve_api_key(profile)
        if not api_key and not local_mode:
            raise LLMError(
                "api_key 未配置：请通过环境变量 EMOTION_LLM_API_KEY / "
                "OPENAI_API_KEY 或配置 profile['api_key'] 提供；"
                "本地端点（Ollama / LM Studio）请设 local_mode=true 允许空 key。"
            )
        return OpenAICompatAdapter(
            base_url=str(profile.get("base_url", "")),
            model=str(profile.get("model", "")),
            api_key=api_key,
            local_mode=local_mode,
        )
    if backend == "transformers":
        return TransformersAdapter(
            model_name=str(profile.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")),
            device=str(profile.get("device", "cpu")),
        )
    raise LLMError(f"未知 backend: {backend}（支持 mock / openai_compatible / transformers）")


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
