"""主动搭话循环 (v0.4)：定时触发主动回忆 → 生成开场话题 → 模型生成搭话。

遵循既有引擎约定：proactive_tick() 返回 should_speak/trigger/opening，
standby_pool() 提供备选池；本循环把「该不该说」交给引擎判定，
把「怎么说」交给模型适配层。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ProactiveLoop:
    """主动搭话循环：周期驱动引擎 proactive_tick + 模型开场白生成。"""

    def __init__(
        self,
        engine: Any,
        adapter: Any,
        *,
        interval: float = 3600.0,
        max_openings: int = 3,
        max_consecutive: int = 1,
        system_prompt: Optional[str] = None,
    ):
        self.engine = engine
        self.adapter = adapter
        self.interval = float(interval)
        self.max_openings = int(max_openings)
        self.max_consecutive = int(max_consecutive)
        self.system_prompt = system_prompt or "你是拥有情感记忆的智能体，请基于给定话题自然开启一段对话。"
        self.next_tick = 0.0
        self.consecutive = 0
        self.last_messages: List[Dict[str, Any]] = []

    def maybe_tick(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """按周期驱动：不到时间返回 None；到了则跑一次完整搭话流程。"""
        import time

        now = float(now if now is not None else time.time())
        if now < self.next_tick:
            return None
        self.next_tick = now + self.interval
        return self.tick(now=now)

    def tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """执行一次搭话：主动回忆 → 话题 → 模型开场 → 沉淀记忆。"""
        import time

        now = float(now if now is not None else time.time())
        result = self.engine.proactive_tick(now=now)
        should_speak = bool(result.get("should_speak"))
        trigger = result.get("trigger") or ""
        opening = result.get("opening") or ""
        if not should_speak:
            self.consecutive = 0
            self.last_messages = []
            return {"should_speak": False, "reason": trigger, "text": ""}
        if self.consecutive >= self.max_consecutive:
            self.consecutive = 0
            self.last_messages = []
            return {"should_speak": False, "reason": "连续搭话已达上限，保持安静", "text": ""}
        candidates = list(self.engine.standby_pool())[: self.max_openings]
        opening = self._pick_opening(opening, candidates)
        text = self._generate_opening(opening, trigger)
        self.engine.remember(
            text=f"主动搭话：{text}",
            cluster="主动搭话",
            entities=[],
            affect=None,
            silence=True,
            never_silent=False,
            due_at=None,
        )
        self.consecutive += 1
        self.last_messages = [
            {"role": "user", "text": text, "trigger": trigger, "opening": opening}
        ]
        return {"should_speak": True, "reason": trigger, "text": text, "opening": opening}

    def _pick_opening(self, opening: str, candidates: List[Any]) -> str:
        if opening and opening.strip():
            return opening
        for cand in candidates:
            text = getattr(cand, "text", None) or (cand.get("text") if isinstance(cand, dict) else "")
            if text:
                return text
        return ""

    def _generate_opening(self, opening: str, trigger: str) -> str:
        if opening and trigger:
            prompt = f"开启一段对话：话题「{opening}」，由「{trigger}」触发。"
        elif opening:
            prompt = f"开启一段对话：话题「{opening}」。"
        else:
            prompt = "开启一段对话：随便聊聊今天的心情。"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.adapter.chat(messages, temperature=0.9, max_tokens=120).strip()
            return text or prompt
        except Exception:
            return prompt
