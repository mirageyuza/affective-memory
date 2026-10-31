"""可选 embedding 向量化后端（v0.6，可插拔增强）。

默认路径保持零依赖词袋哈希（``BagOfWordsVectorizer`` 不动）。本模块提供
可选的 ``EmbeddingVectorizer``：通过 OpenAI 兼容 embed API（复用
``model_adapter`` 的 ``ModelAdapter.embed()`` 接口）或本地
sentence-transformers 接入，作为检索向量化的可插拔增强，不破坏现有接口
与单测。

与 ``BagOfWordsVectorizer`` 保持同一向量化接口：
    vectorize(text, entities=None) -> List[float]
    dimension() -> int

后端选择（``create_embedding_vectorizer`` 的 ``backend`` 参数）：
    - ``openai_compat``  : 复用 ``OpenAICompatAdapter.embed()``，
      OpenAI / DeepSeek / Qwen / 本地 Ollama、LM Studio 等 OpenAI 兼容
      ``/embeddings`` 端点均适用；
    - ``gemini``         : 复用 ``GeminiAdapter.embed()``（``embed_model`` 可配）；
    - ``local_compat``   : 本地 OpenAI 兼容端点（Ollama / LM Studio），允许空 key；
    - ``mock``           : 复用 ``MockModelAdapter.embed()``（确定性 8 维，离线演示 / 单测）；
    - ``sentence_transformers`` : 本地 sentence-transformers 可选依赖，
      未安装时构造即抛 ``LLMError`` 说明，不影响其余功能。

凭据安全：沿用 ``model_adapter`` 的规则，api_key 严禁硬编码，一律环境变量 /
配置注入；本地端点允许空 key。嵌入结果做 L2 归一化后缓存，避免重复请求。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union


class LLMError(RuntimeError):
    """嵌入后端不可用 / 调用失败（与 model_adapter.LLMError 语义一致）。"""


class SentenceTransformersEmbedder:
    """本地 sentence-transformers 嵌入器（可选依赖，惰性加载）。"""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # ModuleNotFoundError 等
            raise LLMError(
                "sentence_transformers 未安装：本地 embedding 后端需 "
                "`pip install sentence-transformers`（可选依赖，不影响其余功能）。"
            ) from exc
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode(str(text))
        return [float(x) for x in vec.tolist()]


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return [0.0] * len(vec)
    return [round(x / norm, 8) for x in vec]


class EmbeddingVectorizer:
    """可选 embedding 向量化器：后端对象只需提供 ``embed(text) -> List[float]``。

    与 ``BagOfWordsVectorizer`` 同接口（vectorize / dimension），可无缝替换
    ``AnnVectorIndex`` 的向量化器；实体名以 1.5 倍权重附加进待嵌入文本。
    """

    def __init__(self, adapter: Any, cache_size: int = 8192):
        # adapter 可以是 ModelAdapter（复用其 embed()）或任意含 embed() 的对象
        self._adapter = adapter
        self._dim: Optional[int] = None
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = int(cache_size)

    def _embed_once(self, text: str) -> List[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        raw = self._adapter.embed(text)
        vec = _l2_normalize(raw)
        if len(self._cache) < self._cache_size:
            self._cache[text] = vec
        if self._dim is None:
            self._dim = len(vec)
        return vec

    def vectorize(self, text: str, entities: Optional[Sequence[str]] = None) -> List[float]:
        parts = [str(text)]
        parts.extend(str(e) for e in (entities or []) if str(e))
        payload = "\n".join(parts)
        # 实体以 1.5 倍权重参与：重复拼入实体片段
        if entities:
            payload += "\n" + "\n".join(str(e) * 1 for e in entities if str(e))
        return self._embed_once(payload)

    def dimension(self) -> int:
        if self._dim is None:
            self.vectorize("")
        return self._dim or 0


def create_embedding_vectorizer(
    profile: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    adapter: Any = None,
) -> EmbeddingVectorizer:
    """工厂：按 backend 构造 embedding 向量化器；后端不可用时抛 LLMError。

    - ``adapter`` 传入时直接包装（最高优先级，便于测试注入）；
    - ``profile`` 字段：base_url / model / api_key / local_mode /
      embed_model / model_name / device / embedding_backend。
    """
    profile = profile or {}
    backend = (backend or str(profile.get("embedding_backend", "")) or "").lower()

    if adapter is not None:
        return EmbeddingVectorizer(adapter=adapter)

    if backend in ("", "mock"):
        from .model_adapter import MockModelAdapter
        return EmbeddingVectorizer(adapter=MockModelAdapter())

    if backend == "openai_compat":
        from .model_adapter import create_adapter
        ad = create_adapter(profile, backend="openai_compatible")
        return EmbeddingVectorizer(adapter=ad)

    if backend == "local_compat":
        from .model_adapter import create_adapter
        ad = create_adapter(profile, backend="local_compat")
        return EmbeddingVectorizer(adapter=ad)

    if backend == "gemini":
        from .model_adapter import create_adapter
        ad = create_adapter(profile, backend="gemini")
        return EmbeddingVectorizer(adapter=ad)

    if backend == "sentence_transformers":
        return EmbeddingVectorizer(
            adapter=SentenceTransformersEmbedder(
                model_name=str(profile.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")),
                device=str(profile.get("device", "cpu")),
            )
        )

    raise LLMError(
        f"未知 embedding 后端: {backend or '(empty)'}（支持 "
        f"mock / openai_compat / local_compat / gemini / sentence_transformers）"
    )
