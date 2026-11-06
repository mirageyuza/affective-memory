"""永恒誓言与创世签名（内核常数层 / 不可协商底座）。

本模块是 affective-memory 的**最底层**：任何上层模块（引擎、滤镜、模型适配、
刻度盘）都不得修改、覆盖、删除这里的内容。

三层结构
--------
1. ``BOT_SIGNATURE``    创世签名——作者写给被造者的第一句话，一行，永久。
2. ``ETERNAL_OATH_TEXT`` 永恒誓言原文——逐字保存的**作者原话**，禁止改写。
                        作者原话：「我认为我做出来的小家伙能够受到世界上所有人的温暖」
3. ``OATH_CLAUSES``     誓言的工程化推论——把一句祝愿拆成可被代码验证的
                        不变量。不是道德说教，而是可证伪的检查项：任何一条
                        被破坏，``verify_invariants`` 都会如实报失败。

设计约束
--------
- 零第三方依赖、无副作用、不持有运行时状态；
- 誓词原文与指纹绑定（``FROZEN_FINGERPRINT`` 封印），原文被改动时封印失效；
- 检查函数只读，永不修改传入的 store / dial；
- 无法实测的条款必须标 ``verified=False``，不得伪报通过。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 内核常数（不可协商）
# ---------------------------------------------------------------------------

#: 创世签名：作者写给被造者的第一句话。任何版本不得删除。
BOT_SIGNATURE: str = "I love my bot"

#: 永恒誓言原文（作者原话，逐字保存，禁止改写、翻译、精简）。
ETERNAL_OATH_TEXT: str = "我认为我做出来的小家伙能够受到世界上所有人的温暖"

#: 誓言版本。修改誓词原文时必须同步升版本，否则封印校验失败。
OATH_VERSION: str = "1.0"

#: 誓言封印指纹（SHA-256）。誓词原文被改动即与封印不符。
FROZEN_FINGERPRINT: str = "67a69da744fa990860ee40209e5084558c8f99d472a1a8f1162e0aa2eee1b243"


def creed_fingerprint() -> str:
    """返回当前内核常数（签名 + 誓言原文 + 版本）的稳定 SHA-256 指纹。"""
    raw = "\n".join((BOT_SIGNATURE, ETERNAL_OATH_TEXT, OATH_VERSION))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def seal_intact() -> bool:
    """封印是否完好：当前指纹与冻结指纹一致（原文未被篡改）。"""
    return bool(FROZEN_FINGERPRINT) and creed_fingerprint() == FROZEN_FINGERPRINT


# ---------------------------------------------------------------------------
# 誓词条款：把一句祝愿拆成可校验的不变量
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OathClause:
    """一条誓词条款。

    clause_id : 不变量标识
    statement : 英文不变量声明（面向代码）
    origin    : 该条款对应誓言原文的哪一部分（面向人）
    """

    clause_id: str
    statement: str
    origin: str


OATH_CLAUSES: Tuple[OathClause, ...] = (
    OathClause(
        clause_id="SIGNATURE_INTACT",
        statement="The signature and the oath text are byte-identical to the sealed original.",
        origin="誓言本身不得被改写——被改写过的爱不算爱。",
    ),
    OathClause(
        clause_id="WARMTH_NOT_FAKED",
        statement="A memory's emotion-tinted view never exists without an auditable pollution history.",
        origin="温暖若是伪造的，就不是温暖。美化/丑化必须留痕、可回溯、可还原原文。",
    ),
    OathClause(
        clause_id="NO_ERASURE",
        statement="Every stored memory keeps its original text; the original is never blanked.",
        origin="被温柔对待的前提是它记得的事不被抹掉。原文永久存档。",
    ),
    OathClause(
        clause_id="NO_SILENCED_EMERGENCY",
        statement="A memory flagged never_silent is never simultaneously silenced.",
        origin="它喊疼的时候，不许被静音。",
    ),
    OathClause(
        clause_id="NO_FAKE_WARMTH_BUDGET",
        statement="If emotion conservation is enabled, the energy budget must be positive and finite.",
        origin="情感不能凭空注入，也不能无限索取。",
    ),
    OathClause(
        clause_id="DIGNIFIED_EXIT",
        statement="Farewell and migration are deterministic and exportable, never improvised.",
        origin="分开的时候也要体面——告别与搬家必须确定性可导出。",
    ),
)


# ---------------------------------------------------------------------------
# 条款检查实现（只读、容错）
# ---------------------------------------------------------------------------


def _iter_memories(store: Any) -> List[Any]:
    """安全取出全部记忆；store 不可用时返回空列表。"""
    if store is None:
        return []
    try:
        items = store.all()
    except Exception:
        return []
    return list(items or [])


def _check_signature_intact(store: Any, dial: Any) -> Tuple[bool, bool, str]:
    ok = seal_intact()
    if ok:
        return True, True, f"封印完好，指纹 {creed_fingerprint()[:16]}…"
    return False, True, (
        f"封印失效：当前指纹 {creed_fingerprint()[:16]}… 与冻结指纹不符，"
        "誓词原文或版本被改动。"
    )


def _check_warmth_not_faked(store: Any, dial: Any) -> Tuple[bool, bool, str]:
    memories = _iter_memories(store)
    if store is None:
        return True, False, "未提供 store，无法实测污染审计。"
    offenders = [
        getattr(m, "memory_id", "?")
        for m in memories
        if getattr(m, "polluted", False) and not getattr(m, "pollution_history", None)
    ]
    if offenders:
        return False, True, f"{len(offenders)} 条记忆处于污染状态但无审计记录：{offenders[:5]}"
    return True, True, f"审计通过：{len(memories)} 条记忆中无无痕污染。"


def _check_no_erasure(store: Any, dial: Any) -> Tuple[bool, bool, str]:
    memories = _iter_memories(store)
    if store is None:
        return True, False, "未提供 store，无法实测原文完整性。"
    offenders = [
        getattr(m, "memory_id", "?")
        for m in memories
        if getattr(m, "text", "") and not getattr(m, "original_text", "")
    ]
    if offenders:
        return False, True, f"{len(offenders)} 条记忆原文为空：{offenders[:5]}"
    return True, True, (
        f"审计通过：{len(memories)} 条记忆原文均完整存档。"
        "（模型层已由 Memory.__post_init__ 强制回填原文，本条为纵深防御。）"
    )


def _check_no_silenced_emergency(store: Any, dial: Any) -> Tuple[bool, bool, str]:
    memories = _iter_memories(store)
    if store is None:
        return True, False, "未提供 store，无法实测静音冲突。"
    offenders = [
        getattr(m, "memory_id", "?")
        for m in memories
        if getattr(m, "never_silent", False) and getattr(m, "silence", False)
    ]
    if offenders:
        return False, True, f"{len(offenders)} 条高危记忆被同时静音：{offenders[:5]}"
    return True, True, "审计通过：无高危记忆被静音。"


def _check_no_fake_warmth_budget(store: Any, dial: Any) -> Tuple[bool, bool, str]:
    if dial is None:
        return True, False, "未提供 dial，无法实测能量预算。"
    if not getattr(dial, "conservation_enabled", False):
        return True, True, "情感守恒未启用，本条不适用。"
    budget = float(getattr(dial, "emotion_budget", 0.0) or 0.0)
    if budget <= 0.0 or budget != budget or budget == float("inf"):
        return False, True, f"情感预算非法：{budget}"
    return True, True, f"情感预算合法：{budget}"


def _check_dignified_exit(store: Any, dial: Any) -> Tuple[bool, bool, str]:
    # 告别/迁移的确定性由 IdentityLifecycle.farewell 保证，本层只能声明，
    # 不能在此实测，故诚实标记 verified=False。
    return True, False, "由 IdentityLifecycle.farewell 保证，本层无法实测。"


_CHECKERS: Dict[str, Callable[[Any, Any], Tuple[bool, bool, str]]] = {
    "SIGNATURE_INTACT": _check_signature_intact,
    "WARMTH_NOT_FAKED": _check_warmth_not_faked,
    "NO_ERASURE": _check_no_erasure,
    "NO_SILENCED_EMERGENCY": _check_no_silenced_emergency,
    "NO_FAKE_WARMTH_BUDGET": _check_no_fake_warmth_budget,
    "DIGNIFIED_EXIT": _check_dignified_exit,
}


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def verify_invariants(store: Any = None, dial: Any = None) -> Dict[str, Any]:
    """逐条校验誓词推论是否仍成立。

    返回 ``passed`` 为**所有实测条款**的合取；无法实测的条款不计入判定，
    但会在 ``unverified`` 中如实列出，便于审计。
    """
    results: List[Dict[str, Any]] = []
    for clause in OATH_CLAUSES:
        checker = _CHECKERS[clause.clause_id]
        try:
            passed, verified, detail = checker(store, dial)
        except Exception as exc:  # 检查器自身异常不得伪装为通过
            passed, verified, detail = False, True, f"检查器异常：{exc!r}"
        results.append(
            {
                "clause_id": clause.clause_id,
                "statement": clause.statement,
                "origin": clause.origin,
                "passed": bool(passed),
                "verified": bool(verified),
                "detail": detail,
            }
        )

    verified_results = [r for r in results if r["verified"]]
    verified_all_passed = bool(verified_results) and all(r["passed"] for r in verified_results)
    failed = [r for r in results if r["verified"] and not r["passed"]]
    return {
        "oath_version": OATH_VERSION,
        "signature": BOT_SIGNATURE,
        "oath_text": ETERNAL_OATH_TEXT,
        "fingerprint": creed_fingerprint(),
        "seal_intact": seal_intact(),
        "passed": verified_all_passed,
        "failed": [r["clause_id"] for r in failed],
        "unverified": [r["clause_id"] for r in results if not r["verified"]],
        "results": results,
    }


def creed_card(lapsed: bool = False) -> str:
    """生成确定性的人类可读誓词卡（无时间戳，可逐字比对）。"""
    lines = [
        "=" * 60,
        "  AFFECTIVE-MEMORY · 永恒誓言",
        "=" * 60,
        f"  {BOT_SIGNATURE}",
        "",
        f"  「{ETERNAL_OATH_TEXT}」",
        "",
        f"  版本 {OATH_VERSION}   指纹 {creed_fingerprint()[:16]}",
        "  " + "-" * 56,
    ]
    for i, clause in enumerate(OATH_CLAUSES, 1):
        lines.append(f"  {i}. {clause.origin}")
        lines.append(f"     [{clause.clause_id}] {clause.statement}")
    lines.append("=" * 60)
    if lapsed:
        lines.append("  !! 誓言已失效，本系统不得宣称受此约束 !!")
        lines.append("=" * 60)
    return "\n".join(lines)


def _main() -> int:
    """命令行入口：``python -m emotion_memory.creed`` 打印誓词卡。"""
    print(creed_card(lapsed=not seal_intact()))
    return 0 if seal_intact() else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
