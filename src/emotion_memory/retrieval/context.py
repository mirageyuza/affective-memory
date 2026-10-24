"""检索层：上下文注入（top-k 注入 + token 预算 + 到期提醒）。"""
from __future__ import annotations

from typing import List

from ..config import Dial
from ..models import Memory, RankedHit


class ContextInjector:
    """把命中记忆组装为上下文注入文本，受 token 预算上限约束。

    命中再多只取 top-k；注入文本默认使用污染视图（display_text，即
    memory.text，可能已被情感滤镜美化/丑化），原文通过 view="original"
    回溯（display_text=original_text）。
    """

    def __init__(self, dial: Dial):
        self.dial = dial

    def attach_context(self, ranked: List[RankedHit]) -> str:
        budget = self.dial.token_budget
        parts = []
        used = 0
        for hit in ranked:
            text = hit.display_text or hit.memory.text
            cost = len(text) // 2  # 中文粗略 token 估算
            if used + cost > budget:
                break
            parts.append(text)
            used += cost
        ctx = "\n".join(parts)
        # 记录到 RankedHit（不修改记忆原文/视图）
        for hit in ranked:
            hit.context = ctx
        return ctx

    @staticmethod
    def format_for_prompt(ranked: List[RankedHit]) -> str:
        lines = []
        for i, hit in enumerate(ranked, 1):
            text = hit.display_text or hit.memory.text
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    @staticmethod
    def attach_due_reminders(due: List[Memory]) -> str:
        """把到期提醒组装为注入文本（高危/时间敏感永不沉默）。"""
        lines = []
        for i, mem in enumerate(due, 1):
            tag = "[到期提醒]"
            lines.append(f"{tag} {mem.original_text or mem.text}")
        return "\n".join(lines)
