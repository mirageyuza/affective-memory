"""demo：v0.3 五大新功能全流程演示。

覆盖：
  ① 旋转磁场唤起调度（三相顺序启动，120° 相位差连续联想流）
  ② 主动回忆引擎（定时/触发驱动的自主搭话）
  ③ 电容记忆模型（充电/放电/漏电 + standby 备选池）
  ④ 身份自学习（画像渐进演化 + 主动提问确认）
  ⑤ 自我统一模型（1 核心 + 万我分身 + 万物皆我 + 有功/视在指标）

运行：python demo/demo_v03_full_flow.py
"""
from __future__ import annotations

import os
import tempfile
import time

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem

T = time.time


def line(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    base = tempfile.mkdtemp(prefix="emotion_memory_v03_demo_")
    dial = Dial(
        three_phase_enabled=True,
        three_phase_speed=1.0,
        proactive_enabled=True,
        proactive_interval_hours=0.5,
        capacitor_enabled=True,
        capacitor_tau_hours=48.0,
        standby_pool_size=3,
        noise_temperature=0.0,  # 演示确定性
    )
    sys = EmotionalMemorySystem(base_dir=base, dial=dial, seed=7)
    print(f"工作目录: {base}\n")

    # ---------------------------------------------------------- ① 写入一批记忆（含到期提醒）
    line("① 写入记忆：三簇主题 + 时间敏感提醒")
    sys.remember("我喜欢周末去郊外爬山，看日出", cluster="旅行", entities=["爬山"])
    sys.remember("用户最近在学 Python 和数据结构", cluster="学习", entities=["Python"])
    sys.remember("下周三要交季度总结报告", cluster="工作", entities=["报告"], due_at=T() + 3600)
    sys.remember("想吃四川火锅，但上次吃坏肚子了", cluster="美食", entities=["火锅"])
    sys.remember("老同学聚会定在国庆节", cluster="社交", entities=["聚会"], due_at=T() + 30)
    for m in sys.store.all():
        print(f"  [{m.cluster}] charge={m.charge:.2f}/cap={m.capacitance:.1f}  {m.original_text}")

    # ---------------------------------------------------------- ② 三相旋转磁场
    line("② 旋转磁场唤起调度（三相顺序启动）")
    now = 3.0  # 模拟启动后 3 小时
    out = sys.three_phase_stream(now=now, k=3)
    field = out["field"]
    print(f"  主导通道: {field['leader_channel']}  三相顺序: {field['sequence']}")
    print(f"  相位差: {field['phase_offsets_deg']}")
    for r in out["stream"]:
        mark = "★领导" if r["leader"] else " 跟随"
        print(f"  {mark} [通道{r['channel']}] 激活={r['activation']:.4f}  {r['memory'].original_text}")

    # ---------------------------------------------------------- ③ 电容模型 + standby 备选池
    line("③ 电容记忆模型：充电 / 放电 / standby 备选池")
    pool = sys.standby_pool(k=3)
    print("  standby 备选池（高电位未唤起候选）：")
    for m in pool:
        print(f"    - charge={m.charge:.2f}  可及电位={sys.capacitor.potential(m):.3f}  {m.original_text}")
    mid = pool[0].memory_id
    sys.touch(mid)  # 触达 = 充电
    charged = sys.store.get(mid).charge
    print(f"  触达充电后 charge={charged:.2f}（被想起前电荷充足 → 高电位候选）")
    hits = sys.recall("爬山 日出")
    for h in hits[:2]:
        print(f"  被想起（放电）: charge={h.memory.charge:.2f}  {h.memory.original_text}")
    sys.decay_all(now=T())
    print("  遗忘推进（漏电）后全库电荷：")
    for m in sys.store.all():
        print(f"    - charge={m.charge:.2f}  potential={sys.capacitor.potential(m):.3f}  {m.cluster}")

    # ---------------------------------------------------------- ④ 身份自学习
    line("④ 身份自学习：画像演化 + 主动提问确认")
    sys.learn_self("我喜欢爬山和摄影")
    sys.learn_self("讨厌拖延，擅长写代码")
    sys.learn_self("目标是成为优秀的记忆系统工程师")
    print("  画像五维：")
    for dim, v in sys.self_model.image.dims.items():
        print(f"    {dim}: {v:.3f}")
    print("  兴趣:", sys.self_model.image.interests)
    print("  厌恶:", sys.self_model.image.aversions)
    print("  使命:", sys.self_model.image.missions)
    print("  待确认问题：")
    for q in sys.pending_self_questions():
        print(f"    [{q['id']}] {q['question']}")
    if sys.pending_self_questions():
        q = sys.pending_self_questions()[0]
        res = sys.confirm_self(q["id"], answer=True)
        print(f"  确认「{q['subject']}」→ 确认度={res['confirmed_weight']:.2f}")
    # 确定性仪式兼容：升级后画像继续演化
    archive = os.path.join(base, "archive.json")
    sys.export(archive)
    sys.upgrade(archive, "0.3.1")
    sys.learn_self("我还喜欢做模型拼图")
    print(f"  升级仪式后画像观察次数: {sys.self_model.image.observation_count}（仪式未打断自学习）")

    # ---------------------------------------------------------- ⑤ 自我统一模型
    line("⑤ 自我统一模型：1 核心 + 万我分身 + 万物皆我")
    sys.derive_avatar("工作分身")
    sys.derive_avatar("生活分身")
    print(f"  核心派生分身数: {sys.unified_self.avatar_count()}")
    print("  核心概念更新 → 分身同步（万我如一）：")
    sys.learn_self("我越来越喜欢深度思考")
    sys.sync_avatars()
    for name, av in sys.unified_self.core.avatars.items():
        print(f"    {name}: openness={av.concept['openness']:.3f}  (已与核心同步)")
    r = sys.assimilate("我希望能帮助每个用户都记得住重要的事")
    print(f"  万物皆我吸收: relevance={r['relevance']:.2f} → 入库记忆: {r['memory'] is not None}")
    sys.assimilate("今天天气不错")
    print("  无关输入: relevance=0 → 不吸收、不改自我概念")
    sys.merge_avatar("工作分身")
    print(f"  万我合一后分身数: {sys.unified_self.avatar_count()}")

    line("自我认知效率指标（有功/视在 = cosφ）")
    report = sys.self_cognition_report()
    print(f"  概念清晰度 clarity      = {report['clarity']:.4f}")
    print(f"  功率因数 cosφ           = {report['cos_phi']:.4f}")
    print(f"  视在功率 apparent_power = {report['apparent_power']:.4f}")
    print(f"  有功功率 active_power   = {report['active_power']:.4f}")
    print(f"  分身数: {report['avatar_count']}  吸收数: {report['absorb_count']}")
    print("  自学习画像: 观察", report["self_learning"]["observation_count"],
          "次 | 兴趣", report["self_learning"]["interests"],
          "| 使命", report["self_learning"]["missions"],
          "| 待确认", report["self_learning"]["pending_questions"], "个")

    line("完成")
    print(f"stats: 记忆 {sys.stats()['memory_count']} 条 | 分身 {sys.stats()['avatar_count']} | "
          f"自学习观察 {sys.stats()['self_learning_observations']} 次")


if __name__ == "__main__":
    main()
