"""记忆存储层：原文存储（磁盘 JSONL，无限扩容，内存只驻留索引视图）。

原文一旦写入永不修改；遗忘/情感排序只改可及性等元数据。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Iterable, List, Optional

from ..models import Memory


class MemoryStore:
    """原文存储：内存字典 + JSONL 落盘。

    线程安全；写入后立即持久化，保证"永久不忘"。
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._memories: Dict[str, Memory] = {}
        self._lock = threading.RLock()
        if path and os.path.exists(path):
            self.load(path)

    # ---- 写 ----
    def add(self, memory: Memory) -> Memory:
        with self._lock:
            self._memories[memory.memory_id] = memory
            self._persist_locked()
        return memory

    def update_meta(self, memory_id: str, **meta) -> Optional[Memory]:
        """仅更新元数据（accessibility/access_count/text 视图等），原文锁定。

        安全约束（双层存储）：
          - original_text 永不通过本方法修改（原文永久存档）；
          - text 仅在原文已锁定（original_text 非空）时允许更新——
            它代表「情感污染视图」，可被滤镜写入、也可被净化回原文；
          - 情感标签 affect 与污染审计 pollution_history 不受本方法直写。
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                return None
            for k, v in meta.items():
                if k == "original_text":
                    raise ValueError("原文永久存档，不可通过 update_meta 修改")
                if k == "text":
                    if not mem.original_text:
                        raise ValueError("原文未锁定前禁止写入视图文本")
                    mem.text = str(v)
                    continue
                if k in ("affect", "pollution_history", "pollution_mode", "polluted"):
                    raise ValueError(f"字段 {k} 需经专门接口修改，禁止绕过")
                if hasattr(mem, k):
                    setattr(mem, k, v)
            self._persist_locked()
            return mem

    def set_pollution_view(
        self,
        memory_id: str,
        text: str,
        polluted: bool,
        pollution_mode: Optional[str],
        pollution_history: List[Dict[str, Any]],
    ) -> Optional[Memory]:
        """专门接口：写入情感污染视图与审计（双层机制唯一落盘入口）。

        安全约束：original_text 不在本方法白名单内，任何调用都不可能
        通过本方法修改原文；污染只作用于 text 视图并追加审计。
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                return None
            if not mem.original_text:
                raise ValueError("原文未锁定前禁止写入污染视图")
            mem.text = str(text)
            mem.polluted = bool(polluted)
            mem.pollution_mode = pollution_mode
            mem.pollution_history = list(pollution_history)
            self._persist_locked()
            return mem

    def replace_all(self, memories: Iterable[Memory]) -> int:
        """原子替换全部记忆（电流桥导入成功后使用）。"""
        with self._lock:
            self._memories = {m.memory_id: m for m in memories}
            self._persist_locked()
            return len(self._memories)

    # ---- 读 ----
    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            return self._memories.get(memory_id)

    def all(self) -> List[Memory]:
        with self._lock:
            return list(self._memories.values())

    def count(self) -> int:
        with self._lock:
            return len(self._memories)

    def clear(self) -> None:
        with self._lock:
            self._memories.clear()
            self._persist_locked()

    # ---- 持久化 ----
    def load(self, path: Optional[str] = None) -> int:
        p = path or self.path
        if not p or not os.path.exists(p):
            return 0
        with self._lock:
            n = 0
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        mem = Memory.from_dict(json.loads(line))
                    except Exception:
                        continue  # 容忍坏行，绝不丢其他记忆
                    self._memories[mem.memory_id] = mem
                    n += 1
            return n

    def _persist_locked(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for mem in self._memories.values():
                f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
