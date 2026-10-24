"""类人情感记忆系统 CLI 交互终端。

命令一览：
  write <text>             写入记忆（自动情感标注；可选 --cluster/--affect/--due/--silent/--never-silent）
  recall <query>           检索记忆（--view original 可回溯原文；--top-k N；--direct 直问解锁）
  view <id>                查看记忆双层视图与污染审计回溯
  pollute <id>             情感污染（--mode beautify|uglify，--intensity F）
  clean <id>               净化回原文（审计历史保留）
  status                   系统状态（记忆数/污染数/到期提醒数/刻度盘）
  dial <name> <value>      调节自由度刻度盘（temperature/amplitude/jump/...）
  due                      列出到期提醒；--ack <id> 确认单条，--ack-all 全部确认
  export <path>            电流桥导出（含 SHA-256）
  import <path>            电流桥导入（校验失败拒绝恢复）
  farewell [--export P]    最终告别（确定性脚本）
  help                     帮助
  exit / quit              退出

用法示例：
  python cli.py write "去年生日和三个老同学爬山，很开心" --cluster 生活
  python cli.py recall "爬山 生日" --view polluted
  python cli.py recall "爬山 生日" --view original
  python cli.py pollute <id> --mode beautify --intensity 1.0
  python cli.py view <id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from emotion_memory.engine import EmotionalMemorySystem  # noqa: E402


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _default_dir() -> str:
    base = os.path.join(tempfile.gettempdir(), "emotion_memory_cli")
    return os.environ.get("EMOTION_MEMORY_DIR", base)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotion-memory", description="类人情感记忆系统 CLI")
    p.add_argument("--dir", default=_default_dir(), help="记忆库根目录（默认系统临时目录）")
    p.add_argument("--seed", type=int, default=7, help="随机种子")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("write", help="写入记忆（自动情感标注）")
    sp.add_argument("text", nargs="+", help="记忆原文")
    sp.add_argument("--cluster", default="default")
    sp.add_argument("--affect", help="手动情感标签 如 0.8,0.6（缺省自动标注）")
    sp.add_argument("--due", help="到期时间戳（秒）或 ISO 时间，如 2030-01-01 08:00")
    sp.add_argument("--silent", action="store_true", help="沉默级（默认不注入）")
    sp.add_argument("--never-silent", action="store_true", help="永不沉默（高危/时间敏感）")
    sp.add_argument("--tags", default="", help="逗号分隔标签")

    sp = sub.add_parser("recall", help="检索记忆")
    sp.add_argument("query", nargs="+")
    sp.add_argument("--top-k", type=int)
    sp.add_argument("--direct", action="store_true", help="直问解锁沉默记忆")
    sp.add_argument("--view", choices=["polluted", "original"], default="polluted")
    sp.add_argument("--entities", default="", help="逗号分隔查询实体")

    sp = sub.add_parser("view", help="查看记忆双层视图")
    sp.add_argument("memory_id")

    sp = sub.add_parser("pollute", help="情感污染（美化/丑化）")
    sp.add_argument("memory_id")
    sp.add_argument("--mode", choices=["beautify", "uglify"], default="beautify")
    sp.add_argument("--intensity", type=float, default=1.0)

    sp = sub.add_parser("clean", help="净化回原文")
    sp.add_argument("memory_id")

    sp = sub.add_parser("status", help="系统状态")
    sp = sub.add_parser("stats", help="系统状态（同 status）")

    sp = sub.add_parser("dial", help="调节自由度刻度盘")
    sp.add_argument("name")
    sp.add_argument("value", type=float)

    sp = sub.add_parser("due", help="到期提醒")
    sp.add_argument("--ack", help="确认某条提醒（memory_id）")
    sp.add_argument("--ack-all", action="store_true", help="确认全部到期提醒")

    sp = sub.add_parser("export", help="电流桥导出")
    sp.add_argument("path")
    sp = sub.add_parser("import", help="电流桥导入")
    sp.add_argument("path")

    sp = sub.add_parser("farewell", help="最终告别（确定性脚本）")
    sp.add_argument("--export", help="告别总结导出路径")
    return p


def _parse_due(value: str) -> float:
    """解析到期时间：纯数字按 Unix 秒；'YYYY-MM-DD HH:MM' 按本地时间。"""
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        ts = time.mktime(time.strptime(s, "%Y-%m-%d %H:%M"))
        return ts
    except ValueError:
        raise SystemExit(f"无法解析到期时间: {value!r}（支持 Unix 秒或 YYYY-MM-DD HH:MM）")


def run(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(args)
    sys = EmotionalMemorySystem(base_dir=ns.dir, seed=ns.seed)

    cmd = ns.command
    if cmd in ("write",):
        text = " ".join(ns.text)
        affect = None
        if ns.affect:
            v, a = ns.affect.split(",")
            affect = (float(v), float(a))
        due = _parse_due(ns.due) if ns.due else None
        mem = sys.remember(
            text,
            cluster=ns.cluster,
            affect=affect,
            due_at=due,
            silence=ns.silent,
            never_silent=ns.never_silent,
            tags=[t for t in ns.tags.split(",") if t],
        )
        _print_json({
            "memory_id": mem.memory_id,
            "text": mem.original_text,
            "affect": (mem.affect.valence, mem.affect.arousal),
            "auto_annotated": ns.affect is None,
            "due_at": _fmt_ts(mem.due_at),
        })
    elif cmd in ("recall",):
        hits = sys.recall(
            " ".join(ns.query),
            top_k=ns.top_k,
            direct_query=ns.direct,
            view=ns.view,
            query_entities=[e for e in ns.entities.split(",") if e] or None,
        )
        rows = []
        for h in hits:
            rows.append({
                "memory_id": h.memory.memory_id,
                "text": h.display_text or h.memory.text,
                "original": h.memory.original_text,
                "score": round(h.score, 4),
                "db": round(h.db, 2),
                "polluted": h.memory.polluted,
            })
        _print_json({"view": ns.view, "hits": rows})
        due = sys.due_reminders()
        if due:
            print("[到期提醒]")
            for m in due:
                print(f"  {m.memory_id}  {m.original_text or m.text}  (到期 {_fmt_ts(m.due_at)})")
    elif cmd in ("view",):
        _print_json(sys.view_memory(ns.memory_id))
    elif cmd in ("pollute",):
        _print_json(sys.apply_pollution(ns.memory_id, mode=ns.mode, intensity=ns.intensity))
    elif cmd in ("clean",):
        _print_json(sys.clear_pollution(ns.memory_id))
    elif cmd in ("status", "stats"):
        _print_json(sys.stats())
    elif cmd in ("dial",):
        before = sys.dial.to_dict().get(ns.name, None)
        if before is None:
            raise SystemExit(f"未知刻度盘参数: {ns.name}（可选 {list(sys.dial.to_dict())}）")
        setattr(sys.dial, ns.name, ns.value)
        sys.dial.validate()
        _print_json({"dial": sys.dial.to_dict()})
    elif cmd in ("due",):
        if ns.ack:
            m = sys.ack_reminder(ns.ack)
            _print_json({"acked": m.memory_id if m else None})
        elif ns.ack_all:
            done = [m.memory_id for m in sys.due_reminders()]
            for mid in done:
                sys.ack_reminder(mid)
            _print_json({"acked_all": done})
        else:
            due = sys.due_reminders()
            rows = [{
                "memory_id": m.memory_id,
                "text": m.original_text or m.text,
                "due_at": _fmt_ts(m.due_at),
                "silent": m.silence,
                "never_silent": m.never_silent,
            } for m in due]
            _print_json({"due_count": len(due), "items": rows})
    elif cmd in ("export",):
        result = sys.export(ns.path)
        _print_json({
            "schema_version": result["schema_version"],
            "sha256": result["sha256"],
            "path": ns.path,
        })
    elif cmd in ("import",):
        result = sys.import_bundle(ns.path)
        _print_json({
            "imported": result["imported"],
            "sha256_ok": result["sha256_ok"],
            "compare_ok": result["compare_ok"],
        })
    elif cmd in ("farewell",):
        result = sys.farewell(export_path=ns.export)
        _print_json({
            "farewell_speech": result["farewell_speech"],
            "export_path": result["export_path"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
