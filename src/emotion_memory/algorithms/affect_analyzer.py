"""自动情感标注器（AffectAnnotator）。

写入记忆时自动计算 (valence, arousal) 标签，零第三方依赖：
  - valence  : 基于中英文情感词典统计正负倾向，含否定翻转与程度加权；
  - arousal  : 基于情感词密度、程度副词、感叹号、否定词计算的唤醒度。

预留外部模型接口：实现 ExternalAffectAnnotator 抽象基类后注入引擎，
即可替换为任意外部情感模型（如 transformer 情感分析服务），引擎调用方
无需改动。
"""
from __future__ import annotations

import math
import re
from typing import Tuple

# ---------------------------------------------------------------- 词典
POSITIVE_WORDS = {
    "喜欢", "开心", "高兴", "幸福", "爱", "美好", "顺利", "成功", "成长", "收获",
    "惊喜", "感谢", "满意", "优秀", "漂亮", "厉害", "幸运", "温馨", "棒", "好",
    "酷", "出色", "乐观", "温暖", "庆祝", "赢", "胜利", "祝福", "鼓励", "快乐",
    "赞", "顺利", "甜蜜", "期待", "圆满", "精彩", "欢快", "舒服", "安心", "欣慰",
    "like", "happy", "great", "good", "love", "awesome", "perfect", "win",
}
NEGATIVE_WORDS = {
    "讨厌", "难过", "伤心", "痛苦", "失败", "糟糕", "烦", "失望", "愤怒", "焦虑",
    "恐惧", "担心", "崩溃", "差", "坏", "冷", "孤独", "遗憾", "后悔", "批评",
    "拒绝", "抛弃", "争吵", "失误", "拖延", "危机", "压力", "苦", "难受", "压抑",
    "郁闷", "不甘", "生气", "愤怒", "恶心", "害怕", "紧张", "沮丧", "绝望",
    "hate", "sad", "bad", "terrible", "angry", "fear", "fail", "loss",
    "加班", "返工", "驳回", "延误", "退票", "丢失",
}
STRENGTH_WORDS = {
    "很", "非常", "特别", "极其", "超级", "太", "无比", "十分", "万分", "绝", "极度",
    "相当", "格外", "尤其", "太", "挺", "比较", "有点", "些许",
}
NEGATORS = {"不", "没", "无", "未", "别", "莫", "非", "not", "no", "never", "don't"}

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")
_EXCLAMATION_RE = re.compile(r"[!！]{1,}")


class ExternalAffectAnnotator:
    """外部情感模型接口（预留）。

    实现 annotate(text) -> (valence, arousal) 即可接入外部模型：
      valence ∈ [-1, 1]，arousal ∈ [0, 1]。
    """

    def annotate(self, text: str) -> Tuple[float, float]:
        raise NotImplementedError


class DefaultAffectAnalyzer(ExternalAffectAnnotator):
    """内置零依赖情感标注器（词典 + 启发式）。"""

    def annotate(self, text: str) -> Tuple[float, float]:
        text = str(text or "")
        tokens = _TOKEN_RE.findall(text.lower())
        words = self._split_tokens(tokens)

        pos = 0.0
        neg = 0.0
        strength_hits = 0
        negator_hits = 0
        for i, w in enumerate(words):
            if w in STRENGTH_WORDS:
                strength_hits += 1
                continue
            if w in NEGATORS:
                negator_hits += 1
                continue
            scale = 1.5 if (i > 0 and words[i - 1] in STRENGTH_WORDS) else 1.0
            if w in POSITIVE_WORDS:
                pos += scale
            elif w in NEGATIVE_WORDS:
                neg += scale
        # 否定词翻转最近情感词：最多每 2 个否定翻转一个正面/负面
        flip = min(negator_hits, int(pos + neg))

        total = pos + neg
        if total <= 0:
            valence = 0.0
        else:
            raw = (pos - neg) / total
            # 否定翻转：整体极性轻微反向
            if flip > 0:
                raw = -raw * min(1.0, 0.5 + 0.25 * flip)
            valence = max(-1.0, min(1.0, raw))

        # 唤醒度：情感密度 + 程度词 + 感叹号 + 否定词
        char_count = max(1, len(text.replace(" ", "")))
        emotion_density = total / char_count
        exclaim = min(1.0, len(_EXCLAMATION_RE.findall(text)))
        arousal = min(
            1.0,
            0.2
            + 0.55 * min(1.0, emotion_density * 12.0)
            + 0.15 * min(1.0, strength_hits / 2.0)
            + 0.15 * exclaim
            + 0.05 * min(1.0, negator_hits),
        )
        if text.endswith(("!", "！")):
            arousal = max(arousal, 0.75)
        return valence, round(arousal, 4)

    @staticmethod
    def _split_tokens(tokens) -> list:
        """把连续 CJK 分词元拆成单字级词汇（情感词多为 1-2 字，无法可靠分词时
        用窗口配对查找情感词）。"""
        out = []
        for tok in tokens:
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in tok)
            if not has_cjk:
                out.append(tok)
                continue
            # CJK：单字入列，同时把相邻两字组合作为候选词（覆盖"开心""讨厌"）
            for ch in tok:
                out.append(ch)
            for i in range(len(tok) - 1):
                out.append(tok[i : i + 2])
        return out


def analyze(text: str) -> Tuple[float, float]:
    """便捷函数：直接对文本做自动情感标注。"""
    return DefaultAffectAnalyzer().annotate(text)
