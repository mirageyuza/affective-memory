"""身份档案与生命周期状态机。

- 身份档案：agent 名称、模型版本、纪元、个性偏好（可重建）；
- 生命周期状态机：ACTIVE / RESETTING / REBORN / UPGRADING；
- 确定性仪式脚本（禁止模型自由发挥）：
  · 重置/数据清零 → 最终告别（导出告别总结 → 生成告别语 → 清空身份层）
  · 重装智能体   → 死亡干涉（唤醒仪式：导入记忆 → 重建身份 → 时间补偿）
  · 升级模型参数 → 无损记忆搬家（旧实例导出 → 新实例导入 → 旧实例封存）
所有文本由固定模板生成，不可测试、不可审计的"自我情感判断"被禁止。
"""
from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .models import Memory
from .store.storage import MemoryStore
from . import bridge


class LifecycleState(str, enum.Enum):
    ACTIVE = "active"
    RESETTING = "resetting"
    REBORN = "reborn"
    UPGRADING = "upgrading"


@dataclass
class IdentityProfile:
    agent_name: str = "emotion-agent"
    model_version: str = "unknown"
    home: str = "default"
    epoch: int = 0               # 记忆纪元：每次苏醒/搬家 +1
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    personality: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def from_dict(self, d: Dict[str, Any]) -> "IdentityProfile":
        self.agent_name = str(d.get("agent_name", self.agent_name))
        self.model_version = str(d.get("model_version", self.model_version))
        self.home = str(d.get("home", self.home))
        self.epoch = int(d.get("epoch", self.epoch))
        self.created_at = float(d.get("created_at", self.created_at))
        self.last_active_at = float(d.get("last_active_at", self.last_active_at))
        self.personality = list(d.get("personality", self.personality))
        return self


class IdentityLifecycle:
    """生命周期状态机 + 确定性仪式脚本。"""

    def __init__(
        self,
        store: MemoryStore,
        identity: IdentityProfile,
        base_dir: str = "",
    ):
        self.store = store
        self.identity = identity
        self.base_dir = base_dir
        self.state = LifecycleState.ACTIVE

    # ---------------------------------------------------------- 重置告别
    def farewell(self, export_path: Optional[str] = None) -> Dict[str, Any]:
        """重置 / 数据清零 → 最终告别。

        确定性流程：导出告别总结 → 生成告别语 → 清空身份层。
        记忆库本身交由调用方决定是否清零；身份层必须清空重建。
        """
        self.state = LifecycleState.RESETTING
        memories = self.store.all()
        clusters = {m.cluster for m in memories}
        top = sorted(memories, key=lambda m: m.accessibility, reverse=True)[:3]
        summary = {
            "agent": self.identity.agent_name,
            "memory_count": len(memories),
            "cluster_count": len(clusters),
            "top_memories": [m.original_text or m.text for m in top],
            "epoch": self.identity.epoch,
        }
        farewell_speech = self._farewell_speech(summary)
        path = export_path or os.path.join(self.base_dir, "farewell_summary.json")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "farewell_speech": farewell_speech},
                          f, ensure_ascii=False, indent=2)
        # 清空身份层（确定性重建基底；原地重置以保持与外部引用一致，
        # 避免替换对象后 engine 等持有旧引用的调用方读到过时档案）
        self.identity.agent_name = self.identity.agent_name + "-reset"
        self.identity.epoch += 1
        self.identity.model_version = "reset"
        self.identity.personality = []
        self.identity.created_at = time.time()
        self.identity.last_active_at = time.time()
        self.state = LifecycleState.ACTIVE
        return {"summary": summary, "farewell_speech": farewell_speech, "export_path": path}

    def _farewell_speech(self, summary: Dict[str, Any]) -> str:
        """固定模板告别语：确定性文本，无模型自由发挥。"""
        return (
            f"【最终告别】{summary['agent']} 即将进入沉睡。"
            f"陪伴期间共记录 {summary['memory_count']} 段记忆，"
            f"覆盖 {summary['cluster_count']} 个主题簇。"
            f"最重要的记忆将被封存：{('；'.join(summary['top_memories'])) or '无'}。"
            f"愿记忆长存。"
        )

    # ---------------------------------------------------------- 重装唤醒
    def revive(
        self,
        bundle_path: str,
        new_identity: Optional[IdentityProfile] = None,
    ) -> Dict[str, Any]:
        """重装智能体 → 死亡干涉（唤醒仪式）。

        确定性流程：导入记忆 → 重建身份 → 检测时间差 → 生成一句时间补偿
        → 恢复到上次状态。禁止"自我认知判断是否冷落用户"等开放式行为。
        """
        self.state = LifecycleState.REBORN
        payload = bridge.verify_bundle(bundle_path)
        old_id = IdentityProfile().from_dict(payload.get("identity") or {})
        old_epoch = old_id.epoch
        # 时间差基于迁移束中的上次活跃时间（确定性，不依赖当前时钟）
        gap_days = max(0.0, (time.time() - old_id.last_active_at) / 86400.0)

        # 导入记忆（校验失败即拒绝恢复）
        result = bridge.import_bundle(self.store, bundle_path, identity=self.identity)
        # 重建身份：沿用迁移束身份，纪元 +1，活跃时间更新为现在
        self.identity.from_dict(payload.get("identity") or self.identity.to_dict())
        self.identity.epoch = old_epoch + 1
        self.identity.last_active_at = time.time()
        if new_identity is not None:
            self.identity.agent_name = new_identity.agent_name

        # 时间补偿（确定性模板）
        if gap_days < 1.0:
            compensation = "距上次对话不足一天，记忆完好无损。"
        else:
            compensation = f"距上次对话已 {int(gap_days)} 天，记忆已完成时间补偿，恢复到上次状态。"
        result["compensation"] = compensation
        result["state"] = self.state.value
        self.state = LifecycleState.ACTIVE
        return result

    # ---------------------------------------------------------- 升级搬家
    def upgrade(
        self,
        archive_path: str,
        new_model_version: str,
        new_base_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """升级模型参数 → 无损记忆搬家。

        确定性流程：旧实例导出 → 新实例导入 → 旧实例封存（不删除，留审计）。
        """
        self.state = LifecycleState.UPGRADING
        # 1) 旧实例导出
        old_export = os.path.join(self.base_dir, f"pre_upgrade_epoch{self.identity.epoch}.json")
        bundle = bridge.export_bundle(self.store, self.identity, old_export)
        # 2) 封存旧实例（归档导出束，不删除）
        os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
        archive_payload = {
            "archived_at": time.time(),
            "old_model_version": self.identity.model_version,
            "export_path": old_export,
            "sha256": bundle["sha256"],
        }
        import json
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive_payload, f, ensure_ascii=False, indent=2)
        # 3) 新实例导入（同一存储，校验一致即恢复；先不恢复旧档案，
        #    避免旧 model_version 覆盖升级版本号）
        result = bridge.import_bundle(self.store, old_export, identity=None)
        self.identity.model_version = new_model_version
        self.identity.epoch += 1
        result["state"] = self.state.value
        result["archive_path"] = archive_path
        result["pre_upgrade_export"] = old_export
        self.state = LifecycleState.ACTIVE
        return result
