"""情感污染滤镜（EmotionPollutionFilter）。

v0.2 解除「记忆原文永不被情感滤镜篡改」红线后的新机制：

  双层记忆：原文（original_text）永久存档、不可删除；
  情感滤镜基于原文生成**污染版本**（美化 beautify / 丑化 uglify），
  写入当前视图 text；检索与上下文注入默认输出污染版本，可通过
  view="original" 回溯原文。

安全与审计：
  - 滤镜永不修改 original_text；
  - 每次污染在 pollution_history 中追加审计条目（原文哈希 / 污染文本
    哈希 / 模式 / 强度 / 时间），可随时净化回原文；
  - 纯标准库实现，确定性（同文本同参数产出同结果）。
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple

from ..models import Memory

# ---------------------------------------------------------------- 情感词替换表
# beautify：把负面表达替换为温和/积极的转述
BEAUTIFY_MAP = {
    "失败": "宝贵的教训",
    "讨厌": "不太适应",
    "痛苦": "有挑战的体验",
    "崩溃": "压力有点大",
    "糟糕": "不尽如人意",
    "失望": "还有提升空间",
    "难过": "心情有些起伏",
    "后悔": "曾经的经历",
    "危机": "转折点",
    "争吵": "意见交换",
    "失误": "一次尝试",
    "遗憾": "小小的遗憾",
    "差": "还有改进余地",
    "孤独": "安静的独处",
    "压力": "前进的动力",
    "苦": "有滋有味",
    "烦": "有点琐碎",
    "担心": "多了一份留意",
    "焦虑": "在意这件事",
    "恐惧": "未知的谨慎",
    "害怕": "谨慎",
    "失败过": "跌倒过",
    "惨": "不太顺",
    "痛": "历练",
}

# uglify：把正面表达替换为平淡/消极的转述
UGLIFY_MAP = {
    "喜欢": "勉强接受",
    "开心": "没什么特别感觉",
    "高兴": "一般般",
    "幸福": "说不清",
    "成功": "运气好",
    "美好": "也就那样",
    "顺利": "刚好没出岔子",
    "满意": "凑合",
    "优秀": "还过得去",
    "漂亮": "不丑",
    "厉害": "勉强可以",
    "幸运": "碰巧而已",
    "棒": "普通",
    "好": "尚可",
    "赞": "还行",
    "完美": "没出大错",
    "快乐": "还行",
    "收获": "一点零碎",
    "期待": "随便看看",
    "圆满": "总算结束",
    "精彩": "热闹而已",
    "温暖": "不算冷",
    "爱": "习惯",
}

# 修饰前缀：强化情绪倾向的确定性框架句
_BEAUTIFY_PREFIX = "如今想来，这更多是一段成长的经历。"
_UGLIFY_PREFIX = "说到底，这件事并没有想象中那么好。"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class EmotionPollutionFilter:
    """基于原文生成污染版本；不直接修改原文，由调用方负责落盘。"""

    def __init__(self, intensity: float = 1.0):
        self.intensity = max(0.0, min(2.0, float(intensity)))

    def pollute(
        self,
        memory: Memory,
        mode: str = "beautify",
        intensity: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """生成污染文本 + 审计条目（不修改 Memory）。

        返回 (polluted_text, audit_entry)。
        """
        if mode not in ("beautify", "uglify"):
            raise ValueError(f"mode 必须是 beautify/uglify，收到 {mode!r}")
        strength = self.intensity if intensity is None else max(0.0, min(2.0, float(intensity)))
        original = memory.original_text or memory.text
        table = BEAUTIFY_MAP if mode == "beautify" else UGLIFY_MAP
        prefix = _BEAUTIFY_PREFIX if mode == "beautify" else _UGLIFY_PREFIX

        polluted = self._apply_table(original, table, strength)
        if strength >= 1.0:
            polluted = prefix + polluted

        audit = {
            "at": now if now is not None else time.time(),
            "mode": mode,
            "intensity": strength,
            "original_hash": _sha(original),
            "polluted_hash": _sha(polluted),
            "polluted_text": polluted,
        }
        return polluted, audit

    def _apply_table(self, text: str, table: Dict[str, str], strength: float) -> str:
        """确定性替换：按词典顺序（长词优先）替换匹配情感词。

        strength∈(0,2]：>=1 全量替换；<1 时只替换前 k 个匹配。
        """
        out = text
        if strength <= 0:
            return out
        # 长词优先，避免"失败过"被"失败"抢先替换
        ordered = sorted(table.items(), key=lambda kv: len(kv[0]), reverse=True)
        if strength >= 1.0:
            for src, dst in ordered:
                out = out.replace(src, dst)
        else:
            hits = [(s, d) for s, d in ordered if s in out]
            limit = max(1, int(len(hits) * strength))
            for src, dst in hits[:limit]:
                out = out.replace(src, dst)
        return out

    def clean(self, memory: Memory) -> Memory:
        """净化：当前视图恢复到原文（污染历史保留审计）。"""
        memory.text = memory.original_text
        memory.polluted = False
        memory.pollution_mode = None
        return memory
