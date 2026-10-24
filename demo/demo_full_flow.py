"""全流程演示：写入记忆 → 检索 → 遗忘 → 迁移。

运行：python demo/demo_full_flow.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_memory import EmotionalMemorySystem, Dial
from emotion_memory.identity import IdentityProfile


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    # 每次运行使用独立临时数据目录，保证 demo 幂等、不累积脏数据
    demo_dir = tempfile.mkdtemp(prefix="emotion_memory_demo_")
    print("数据目录:", demo_dir)

    # 1) 初始化：默认低噪声高稳定刻度盘
    section("1. 初始化记忆系统（默认刻度盘：低噪声高稳定）")
    dial = Dial()
    mem = EmotionalMemorySystem(
        base_dir=os.path.join(demo_dir, "instance_a"),
        dial=dial,
        identity=IdentityProfile(agent_name="demo-agent", model_version="0.1.0"),
    )
    print("stats:", mem.stats())

    # 2) 写入记忆
    section("2. 写入记忆（原文 + 情感标签 + 实体）")
    mem.remember(
        "用户喜欢晨跑，讨厌雨天。", cluster="hobby",
        entities=["晨跑", "雨天"], affect=(0.5, 0.4),
    )
    mem.remember(
        "去年生日用户和三个老同学一起爬山。", cluster="social",
        entities=["生日", "爬山"], affect=(0.8, 0.6),
    )
    mem.remember(
        "用户下周五提交季度报告，绝不能遗漏。", cluster="task",
        entities=["季度报告"], affect=(-0.2, 0.7),
        never_silent=True,  # 时间敏感，永不沉默
    )
    mem.remember(
        "一次失败的提案经历，用户不愿再提。", cluster="social",
        entities=["提案"], affect=(-0.7, 0.2),
        silence=True,       # 沉默记忆：默认不提及
    )
    print("已写入", mem.store.count(), "条记忆")

    # 3) 检索（普通查询不注入沉默记忆；高危记忆始终注入）
    section("3. 检索：相关簇定位 → ANN 召回 → 情感增益/dB 排序 → top-k")
    hits = mem.recall("爬山 生日 老同学")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.memory.text}  score={h.score:.4f} dB={h.db:.2f} gain={h.gain:.2f}")

    print("\n普通查询不会返回沉默记忆；直问 '提案' 可解锁：")
    silent_hits = mem.recall("提案", direct_query=True)
    for h in silent_hits:
        print(f"  (直问解锁) {h.memory.text}")

    # 3b) 情感污染双层视图：原文永久存档 + 污染版本默认输出
    section("3b. 情感污染双层视图（原文存档 + 污染版默认输出）")
    target = next(m for m in mem.store.all() if "爬山" in m.original_text)
    mid = target.memory_id
    print("  原文:", target.original_text)
    print("  自动情感标签 valence/arousal:", target.affect.valence, target.affect.arousal)
    r = mem.apply_pollution(mid, mode="beautify", intensity=1.0)
    print("  美化污染版:", r["polluted_text"])
    r2 = mem.apply_pollution(mid, mode="uglify", intensity=1.0)
    print("  丑化污染版:", r2["polluted_text"])
    view = mem.view_memory(mid)
    print("  污染后原文仍为:", view["original_text"], "| 审计条数:", len(view["pollution_history"]))
    hits_p = mem.recall("爬山 生日 老同学", view="polluted")
    print("  检索默认输出污染版本:")
    for h in hits_p[:2]:
        print("    -", h.display_text)
    hits_o = mem.recall("爬山 生日 老同学", view="original")
    print("  检索 view=original 回溯原文:")
    for h in hits_o[:2]:
        print("    -", h.display_text)
    c = mem.clear_pollution(mid)
    print("  净化回原文:", mem.store.get(mid).text, "| 审计保留条数:", c["history_count"])

    # 4) 遗忘曲线：时间流逝后低频记忆可及性下降，高危任务仍可召回
    section("4. 遗忘曲线：模拟时间流逝（30 天）")
    before = {m.memory_id: m.accessibility for m in mem.store.all()}
    future = time.time() + 30 * 86400
    mem.decay_all(now=future)
    for m in mem.store.all():
        delta = m.accessibility - before[m.memory_id]
        print(f"  {m.text[:20]:<22} accessibility {before[m.memory_id]:.3f} -> {m.accessibility:.3f} ({delta:+.3f})")
    # 触达强化
    task_mem = next(m for m in mem.store.all() if m.never_silent)
    mem.touch(task_mem.memory_id)
    print("  触达强化后任务记忆可及性:", f"{task_mem.accessibility:.3f}")

    # 5) 电流桥迁移：导出 → 篡改拒绝 → 正常导入
    section("5. 电流桥迁移（SHA-256 校验 + 失败拒绝恢复）")
    export_path = os.path.join(demo_dir, "bridge_export.json")
    bundle = mem.export(export_path)
    print("  已导出:", export_path, "SHA-256:", bundle["sha256"][:16], "...")

    # 模拟损坏
    with open(export_path, "r", encoding="utf-8") as f:
        content = f.read()
    corrupted = content.replace("季度报告", "季度报告X", 1)
    bad_path = os.path.join(demo_dir, "bridge_corrupted.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write(corrupted)
    try:
        from emotion_memory import bridge as br
        br.verify_bundle(bad_path)
        print("  [!!] 损坏文件校验通过（不应发生）")
    except br.BridgeError as e:
        print("  篡改文件被拒绝:", e)

    # 新实例导入
    mem_b = EmotionalMemorySystem(
        base_dir=os.path.join(demo_dir, "instance_b"),
        dial=Dial(),
        identity=IdentityProfile(agent_name="demo-agent-b", model_version="0.2.0"),
    )
    result = mem_b.import_bundle(export_path)
    print("  新实例导入:", result["imported"], "条记忆, 身份恢复:", result.get("identity_restored"))

    # 6) 生命周期：唤醒仪式（时间补偿）
    section("6. 生命周期：重装唤醒（确定性时间补偿）")
    old_export = os.path.join(demo_dir, "bridge_export.json")
    revive_res = mem_b.revive(old_export)
    print("  唤醒结果:", revive_res.get("compensation"))
    print("  当前身份:", mem_b.identity.agent_name, "纪元:", mem_b.identity.epoch)

    # 7) 告别 + 升级搬家
    section("7. 生命周期：告别与升级搬家（确定性脚本）")
    farewell = mem_b.farewell(os.path.join(demo_dir, "farewell_summary.json"))
    print("  告别语:", farewell["farewell_speech"])
    archive = os.path.join(demo_dir, "archive.json")
    up = mem_b.upgrade(archive, new_model_version="0.3.0")
    print("  升级搬家完成 → 模型版本:", mem_b.identity.model_version, "| 归档:", up["archive_path"])

    # 8) 时间敏感提醒集成：高危/时间敏感记忆到期注入
    section("8. 时间敏感提醒集成（高危/时间敏感永不沉默）")
    past = time.time() - 60
    due_mem = mem_b.remember(
        "明天上午十点前必须交房租", due_at=past, never_silent=True,
    )
    print("  已写入到期记忆（due_at 已过期）")
    due = mem_b.due_reminders()
    print("  到期提醒数:", len(due), "| 首条:", due[0].original_text if due else "-")
    ctx = mem_b.context_with_reminders([], now=time.time())
    print("  注入提醒上下文:", ctx.replace("\n", " / "))
    mem_b.ack_reminder(due_mem.memory_id)
    print("  确认提醒后再次查询到期数:", len(mem_b.due_reminders()))

    section("演示完成 ✔ 写入→检索→污染双层→遗忘→迁移→提醒 全流程闭环")


if __name__ == "__main__":
    main()
