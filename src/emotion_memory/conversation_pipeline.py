"""对话管线 (v0.4)：用户消息 → 记忆检索注入 → 模型回复 → 自动沉淀记忆。

沉淀环节把一次对话完整写回记忆系统：
  原文存储(永不删) → 自动情感标注 → 电容充电(touch) → 万物皆我吸收
  → 情感数据累计 → 涌现自我增量更新。

检索注入遵守既有红线：沉默层过滤、情感污染视图（默认 injected_view='polluted'
保证原文可提及而不泄原始情感值）、高危/时间敏感记忆永不沉默。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ConversationPipeline:
    """对话管线：连接模型适配器与情感记忆引擎。"""

    def __init__(
        self,
        engine: Any,
        adapter: Any,
        *,
        top_k: int = 5,
        injected_view: str = "polluted",
        settle_cluster: str = "对话",
    ):
        self.engine = engine
        self.adapter = adapter
        self.top_k = int(top_k)
        self.injected_view = injected_view
        self.settle_cluster = settle_cluster
        self.history: List[Dict[str, Any]] = []
        self.last_settle: Optional[Dict[str, Any]] = None

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """检索注入：调用引擎 recall（沉默层过滤 + 情感污染视图）。"""
        top_k = self.top_k if top_k is None else int(top_k)
        hits = self.engine.recall(
            query,
            top_k=top_k,
            direct_query=False,
            now=now,
            view=self.injected_view,
        )
        rows = []
        for h in hits:
            mem = getattr(h, "memory", h)
            rows.append(
                {
                    "memory_id": mem.memory_id,
                    "text": mem.text,
                    "cluster": mem.cluster,
                    "score": float(getattr(h, "score", getattr(h, "activation", 0.0))),
                }
            )
        return rows

    def build_messages(
        self,
        user_text: str,
        *,
        system_prompt: Optional[str] = None,
        context_rows: Optional[List[Dict[str, Any]]] = None,
        now: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """组装 messages：系统提示 + 记忆上下文注入 + 历史 + 当前消息。"""
        context_rows = (
            self.retrieve_context(user_text, now=now)
            if context_rows is None
            else context_rows
        )
        sys = system_prompt or (
            "你是拥有情感记忆的智能体。回答可参考注入的相关记忆片段，"
            "不要声称自己是预设程序。"
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        if context_rows:
            mem_text = "\n".join(f"- {r['text']}" for r in context_rows if r.get("text"))
            messages.append(
                {"role": "system", "content": "相关记忆片段：\n" + mem_text}
            )
        for turn in self.history[-8:]:
            messages.append(
                {
                    "role": "user" if turn.get("role") == "user" else "assistant",
                    "content": turn.get("text", ""),
                }
            )
        messages.append({"role": "user", "content": user_text})
        return messages

    def reply(
        self,
        user_text: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        settle: bool = True,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """完整一轮：检索注入 → 模型回复 → 自动沉淀（可关闭沉淀作纯生成）。"""
        messages = self.build_messages(user_text, system_prompt=system_prompt, now=now)
        reply_text = self.adapter.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        ).strip()
        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "assistant", "text": reply_text})
        self.last_settle = None
        if settle:
            self.last_settle = self.settle(user_text, reply_text, now=now)
        return {
            "reply": reply_text,
            "messages": messages,
            "context_rows": messages[1] if len(messages) > 1 and messages[1]["role"] == "system" and "相关记忆片段" in messages[1]["content"] else None,
            "settle": self.last_settle,
        }

    def settle(
        self,
        user_text: str,
        reply_text: str,
        *,
        affect: Optional[Any] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """沉淀一次对话：写入存储(自动情感标注) + 电容充电 + 万物皆我吸收 + 累计。"""
        cluster = self.settle_cluster
        mem = self.engine.remember(
            text=f"用户说：{user_text}",
            cluster=cluster,
            entities=[],
            affect=affect,
            silence=False,
            never_silent=False,
            due_at=None,
        )
        if mem is not None:
            self.engine.touch(mem.memory_id)  # 电容充电
            affect = mem.affect if affect is None else affect
        self.engine.assimilate(user_text)  # 万物皆我吸收（relevance>0 才改变概念）
        self.engine.emergent_self.accumulate(
            text=user_text,
            affect=affect,
            cluster=cluster,
            entities=[],
            now=now,
        )
        self.engine.emergent_self.refresh()
        return {
            "memory_id": mem.memory_id if mem is not None else None,
            "affect": affect,
            "self_concepts": self.engine.emergent_self.current_concepts,
            "traits": self.engine.emergent_self.current_traits,
        }

    def inject(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """检查某条记忆在对话中的可见性（direct_query 解锁沉默后是否可注入）。"""
        mem = self.engine.store.get(memory_id)
        if mem is None:
            return None
        hits = self.engine.recall(
            mem.text,
            top_k=1,
            direct_query=True,
            now=None,
            view=self.injected_view,
        )
        for h in hits:
            m = getattr(h, "memory", h)
            if m.memory_id == memory_id:
                return {"text": m.text, "visible": True}
        return {"text": mem.text, "visible": False}
