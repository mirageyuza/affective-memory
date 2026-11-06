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
# v0.5 物理四定律 + 人格流形
from .conservation import ConservationError, EmotionConservation
from .potential_transfer import PotentialTransfer
from .entropic_forgetting import EntropicForgetting
from .turbulent_association import TurbulenceAssociator
from .personality_manifold import EnvironmentVector, PersonalityManifold
# v0.6.2 元认知层：置信度估计 + 拒答策略 + 校准度量
from .metacognition import (
    ABSTAIN_TEXT,
    HEDGE_TEXT,
    LEVEL_ABSTAIN,
    LEVEL_ANSWER,
    LEVEL_HEDGE,
    METACOGNITION_LIMITS,
    AbstentionPolicy,
    CalibrationBin,
    CalibrationReport,
    ConfidenceBreakdown,
    ConfidenceEstimator,
    MetacognitiveMonitor,
    RecallVerdict,
    calibrate,
)
# 内核常数层：创世签名 + 永恒誓言（最底层，不可协商）
from .creed import (
    BOT_SIGNATURE,
    ETERNAL_OATH_TEXT,
    OATH_VERSION,
    OATH_CLAUSES,
    OathClause,
    creed_card,
    creed_fingerprint,
    seal_intact,
    verify_invariants,
)

__version__ = "0.6.2"
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
    "ConservationError",
    "EmotionConservation",
    "PotentialTransfer",
    "EntropicForgetting",
    "TurbulenceAssociator",
    "EnvironmentVector",
    "PersonalityManifold",
    # 内核常数层
    "BOT_SIGNATURE",
    "ETERNAL_OATH_TEXT",
    "OATH_VERSION",
    "OATH_CLAUSES",
    "OathClause",
    "creed_card",
    "creed_fingerprint",
    "seal_intact",
    "verify_invariants",
    # v0.6.2 元认知层：置信度 / 拒答 / 校准
    "AbstentionPolicy",
    "CalibrationBin",
    "CalibrationReport",
    "ConfidenceBreakdown",
    "ConfidenceEstimator",
    "MetacognitiveMonitor",
    "RecallVerdict",
    "METACOGNITION_LIMITS",
    "ABSTAIN_TEXT",
    "HEDGE_TEXT",
    "LEVEL_ANSWER",
    "LEVEL_HEDGE",
    "LEVEL_ABSTAIN",
    "calibrate",
    "__version__",
]
