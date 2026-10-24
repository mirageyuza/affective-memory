"""类人情感记忆系统 (EmotionMemory)。

核心不变量：智能体 = 模型参数(可换) + 记忆库(永恒) + 自我定义(可重建)。
v0.4 起「智能体是什么」由情感累计涌现生成，不依赖预设档案。
许可证：MIT，运行时零第三方依赖。
"""

from .config import Dial
from .models import AffectLabel, Memory
from .engine import EmotionalMemorySystem
from .model_adapter import (
    LLMError,
    ModelAdapter,
    MockModelAdapter,
    OpenAICompatAdapter,
    TransformersAdapter,
    create_adapter,
    load_model_config,
    resolve_api_key,
)
from .conversation_pipeline import ConversationPipeline
from .emergent_self import EmergentSelf
from .proactive_loop import ProactiveLoop

__version__ = "0.4.0"
__all__ = [
    "Dial",
    "AffectLabel",
    "Memory",
    "EmotionalMemorySystem",
    "LLMError",
    "ModelAdapter",
    "MockModelAdapter",
    "OpenAICompatAdapter",
    "TransformersAdapter",
    "create_adapter",
    "load_model_config",
    "resolve_api_key",
    "ConversationPipeline",
    "EmergentSelf",
    "ProactiveLoop",
    "__version__",
]
