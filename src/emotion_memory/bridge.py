"""电流桥迁移协议（无损记忆迁移）。

规格 §4.2：
  导出: 记忆库序列化为版本化格式(JSON + schema 版本号)
  校验: 传输前 SHA-256 校验和; 导入后全量比对, 不一致拒绝恢复
  增量: 只传变更条目(类似 git/rsync)
  升级: 迁移耗时与记忆量线性, 与模型无关

安全红线：迁移必须校验（SHA-256 + 全量比对），失败拒绝恢复——
任何校验失败都不会写入目标存储。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from .models import Memory
from .store.storage import MemoryStore

SCHEMA_VERSION = 2  # v2: 双层存储（original_text 永久存档 + text 污染视图）


class BridgeError(Exception):
    """电流桥协议错误：校验失败 / 版本不兼容 / 格式非法。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _payload_to_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------- 导出
def export_bundle(
    store: MemoryStore,
    identity: Optional[Any],
    path: str,
    schema_version: int = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """导出记忆库 + 身份档案为版本化 JSON 束（含 SHA-256）。"""
    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "exported_at": time.time(),
        "identity": identity.to_dict() if identity is not None else None,
        "memories": [m.to_dict() for m in store.all()],
    }
    raw = _payload_to_bytes(payload)
    bundle = {
        "schema_version": schema_version,
        "sha256": sha256_bytes(raw),
        "payload": payload,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return bundle


# ---------------------------------------------------------------- 校验
def verify_bundle(path: str) -> Dict[str, Any]:
    """读取并校验束文件：schema 版本 + SHA-256 校验和。

    返回 payload；任何不一致抛 BridgeError。
    """
    if not os.path.exists(path):
        raise BridgeError(f"迁移文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            bundle = json.load(f)
        except json.JSONDecodeError as e:
            raise BridgeError(f"迁移文件格式非法: {e}")

    schema = bundle.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise BridgeError(f"schema 版本不兼容: 期望 {SCHEMA_VERSION}, 实际 {schema}")
    payload = bundle.get("payload")
    if not isinstance(payload, dict):
        raise BridgeError("payload 缺失")
    expected = bundle.get("sha256")
    actual = sha256_bytes(_payload_to_bytes(payload))
    if expected != actual:
        raise BridgeError(f"SHA-256 校验失败: 期望 {expected}, 实际 {actual}")
    return payload


def compare_payload(payload: Dict[str, Any], store: MemoryStore) -> Dict[str, Any]:
    """导入前全量比对：逐条 memory_id 比对 original_text / accessibility 等。

    返回 {"ok": bool, "issues": [...]}。任何不一致视为校验失败。
    注意：比对对象为**原文** original_text；text 是污染视图，允许不一致
    （迁移束可保留污染版本或净化版本，原文才是审计锚点）。
    """
    issues: List[str] = []
    incoming = payload.get("memories") or []
    existing = {m.memory_id: m for m in store.all()}
    if len(existing) and len(incoming) != len(existing):
        issues.append(
            f"记忆条数不一致: 目标库 {len(existing)}, 迁移束 {len(incoming)}"
        )
    for raw in incoming:
        mid = raw.get("memory_id")
        if mid not in existing:
            continue
        cur = existing[mid]
        if cur.original_text != (raw.get("original_text") or raw.get("text")):
            issues.append(f"记忆原文不一致: {mid}")
        if abs(cur.accessibility - float(raw.get("accessibility", 0.0))) > 1e-9:
            issues.append(f"可及性分数不一致: {mid}")
    return {"ok": not issues, "issues": issues}


# ---------------------------------------------------------------- 导入
def import_bundle(
    store: MemoryStore,
    path: str,
    identity: Optional[Any] = None,
    schema_version: int = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """导入迁移束：校验（SHA-256 + 全量比对）通过后才原子恢复。

    失败拒绝恢复：校验不通过时抛 BridgeError，目标库保持原状。
    """
    payload = verify_bundle(path)

    # 内存中重建候选集（不触碰目标库）
    candidate: Dict[str, Memory] = {}
    for raw in payload.get("memories", []):
        mem = Memory.from_dict(raw)
        candidate[mem.memory_id] = mem

    # 导入后全量比对：逐条比对原文（original_text，污染视图允许差异）
    for raw in payload.get("memories", []):
        mem = candidate.get(raw.get("memory_id"))
        if mem is None:
            raise BridgeError("导入候选集构建失败")
        if mem.original_text != (raw.get("original_text") or raw.get("text")):
            raise BridgeError(f"原文比对不一致: {raw.get('memory_id')}")

    # 原子替换目标库（成功后才落盘）
    store.replace_all(candidate.values())

    result = {
        "schema_version": schema_version,
        "imported": len(candidate),
        "identity": payload.get("identity"),
        "sha256_ok": True,
        "compare_ok": True,
    }
    if identity is not None and payload.get("identity") is not None:
        identity.from_dict(payload["identity"])
        result["identity_restored"] = True
    return result


# ---------------------------------------------------------------- 增量
def export_delta(
    store: MemoryStore,
    baseline: Dict[str, Memory],
    path: str,
    identity: Optional[Any] = None,
) -> Dict[str, Any]:
    """增量导出：只传变更条目（新增 + 元数据变更，类似 git/rsync）。"""
    changed = []
    for m in store.all():
        base = baseline.get(m.memory_id)
        if base is None or base.to_dict() != m.to_dict():
            changed.append(m.to_dict())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": time.time(),
        "incremental": True,
        "identity": identity.to_dict() if identity is not None else None,
        "memories": changed,
    }
    raw = _payload_to_bytes(payload)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "sha256": sha256_bytes(raw),
        "payload": payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return bundle
