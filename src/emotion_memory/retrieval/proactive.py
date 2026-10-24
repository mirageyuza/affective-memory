"""主动回忆引擎（proactive recall）：定时 / 触发器驱动的自主想起搭话。

- 定时触发：距上次主动开口超过 interval_hours 才发起（避免扰人）；
- 上下文触发：到期提醒（永不沉默）优先；其次电容备选池中高电位记忆；
- 生成开场话题：基于时间上下文（早 / 中 / 晚、星期）与候选记忆内容，
  用确定性模板开口搭话——禁止模型自由发挥，可测试、可审计；
- 被动查询接口保持不变：recall() 语义不受影响，本引擎只新增"自主想起"。

触发规则（确定性）：
    should_speak = (now - last_proactive_at) >= interval
                   and (有到期提醒 or 有高电位候选 or 首次开口)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..models import Memory


class ProactiveRecallEngine:
    """主动回忆引擎：决定"什么时候开口 + 说什么"。"""

    def __init__(self, interval_hours: float = 6.0, top_k: int = 3):
        self.interval_hours = max(0.5, float(interval_hours))
        self.top_k = max(1, int(top_k))
        self.last_proactive_at: Optional[float] = None
        self.proactive_count: int = 0

    # ---------------------------------------------------------- 触发
    def should_initiate(self, now: Optional[float] = None) -> bool:
        """定时触发判定：距上次开口超过间隔。首次开口返回 True（由上层决定是否搭话）。"""
        now = now if now is not None else time.time()
        if self.last_proactive_at is None:
            return True
        return (now - self.last_proactive_at) >= self.interval_hours * 3600.0

    # ---------------------------------------------------------- 话题
    @staticmethod
    def greeting_for(now: Optional[float] = None) -> str:
        """时间上下文前缀（确定性模板）：早 / 中 / 晚 / 夜 + 星期。"""
        now = now if now is not None else time.time()
        lt = time.localtime(now)
        hour = lt.tm_hour
        if 5 <= hour < 12:
            greet = "早上好"
        elif 12 <= hour < 14:
            greet = "中午好"
        elif 14 <= hour < 19:
            greet = "下午好"
        else:
            greet = "晚上好"
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        return f"{greet}，今天是星期{weekdays[lt.tm_wday]}。"

    def build_opening(
        self,
        candidates: List[Memory],
        now: Optional[float] = None,
        trigger: str = "proactive",
    ) -> str:
        """基于候选记忆生成一句开场话题（确定性模板，无自由发挥）。"""
        now = now if now is not None else time.time()
        parts = [self.greeting_for(now)]
        if trigger == "reminder" and candidates:
            m = candidates[0]
            parts.append(
                f"我忽然想起：{m.original_text or m.text}——这件事你还没确认，需要现在处理吗？"
            )
        elif candidates:
            m = candidates[0]
            parts.append(f"我忽然想起：{m.original_text or m.text}。你现在有空聊聊吗？")
        else:
            parts.append("我这边没什么特别的事，就想跟你说说话。")
        return " ".join(parts)

    # ---------------------------------------------------------- 主入口
    def proactive_tick(
        self,
        candidates: List[Memory],
        due_memories: Optional[List[Memory]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """主动想起决策：满足间隔 + 有候选 / 到期提醒时开口。

        返回 should_speak / trigger / opening / candidates 明细。
        """
        now = now if now is not None else time.time()
        due = due_memories or []
        if due:
            trigger = "reminder"
            pool = due[: self.top_k]
        else:
            trigger = "proactive"
            pool = candidates[: self.top_k]
        if not self.should_initiate(now):
            return {
                "should_speak": False,
                "trigger": trigger,
                "opening": "",
                "candidates": [m.memory_id for m in pool],
            }
        opening = self.build_opening(pool, now, trigger)
        self.last_proactive_at = now
        self.proactive_count += 1
        return {
            "should_speak": True,
            "trigger": trigger,
            "opening": opening,
            "candidates": [m.memory_id for m in pool],
            "count": self.proactive_count,
        }
