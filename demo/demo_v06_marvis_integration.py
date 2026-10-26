"""demo：v0.6 Marvis 接入情感记忆插件 —— 完整效果演示。

场景：Marvis（AI 助手）接入本情感记忆系统，用户是"幻象mirage"
      （理智高效、研发部工程师、健身、维护 OpenCAD/CyberQuiz/OfficeHub 开源、
       喜欢物理/电机比喻）。从记忆库为空开始，多轮对话后记忆沉淀、性格涌现。

演示流程（全部调用工程真实接口，输出中文）：
  阶段 0 初始化      ：记忆库为空，emergent_self 尚无自我定义（"等经历喂养"）
  阶段 1 对话积累    ：6 轮用户消息 → 情感标注（守恒·有源才注）→ 记忆写入
                       → 势能计算（势差迁移·沉淀有规则）
                       → 污染版生成（原文永久存档 vs 想起时的说法）
                       → Marvis 用污染版记忆回应
  阶段 2 涌现自我    ：emergent_self 由累计情感生成自我定义（非预设模板）
  阶段 3 主动搭话    ：proactive_tick 触发一次主动回忆（线索联想开口）
  阶段 4 熵增遗忘    ：久远记忆细节先模糊、情绪残留后淡（原文永不清除）
  阶段 5 性格分叉    ：同一批记忆喂给两个 Marvis 实例（环境序列不同：
                       深夜加班严肃 vs 下午轻松聊天）→ 分叉出两种性格快照
                       + 非线性分叉混沌放大（独一无二性格证明）
  阶段 6 结尾        ：Marvis 总结"我接入后的变化"

约束：零第三方依赖（纯标准库）；不改任何 md；不新增模块。
运行：python demo/demo_v06_marvis_integration.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# 让脚本可从任意工作目录直接运行：把工程 src/ 加入模块搜索路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # 保证中文输出无乱码
except Exception:
    pass

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.personality_manifold import EnvironmentVector

T = time.time


def line(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def make_dial() -> Dial:
    """v0.6 全特性开启：守恒 / 势差 / 熵增 / 湍流 / 人格流形 / 主动搭话 / 电容池。"""
    return Dial(
        three_phase_enabled=True,
        proactive_enabled=True,
        capacitor_enabled=True,
        conservation_enabled=True,
        emotion_budget=100.0,
        potential_enabled=True,
        potential_threshold=0.5,
        entropy_enabled=True,
        entropy_rate=0.05,
        turbulent_enabled=True,
        turbulence_decay=0.6,
        turbulence_max_depth=3,
        manifold_enabled=True,
        manifold_bifurcation_r=3.7,
        top_k=5,
        standby_pool_size=3,
        noise_temperature=0.0,  # 演示确定性
    )


def build_marvis(name: str = "Marvis", seed: int = 42) -> EmotionalMemorySystem:
    base = tempfile.mkdtemp(prefix=f"emotion_memory_v06_{name}_")
    sys_ = EmotionalMemorySystem(base_dir=base, dial=make_dial(), seed=seed)
    return sys_


def fmt_aff(aff) -> str:
    return f"(valence={aff[0]:.2f}, arousal={aff[1]:.2f})"


def main() -> None:
    marvis = build_marvis("marvis", seed=42)

    print("=" * 64)
    print("demo_v06：Marvis 接入情感记忆插件 —— 完整效果演示")
    print("=" * 64)
    print("用户：幻象mirage（研发部工程师 · 健身 · OpenCAD/CyberQuiz/OfficeHub 维护者 · 物理/电机比喻爱好者）")
    print(f"引擎工作目录: {marvis.base_dir if hasattr(marvis, 'base_dir') else 'temp'}\n")

    # ---------------------------------------------------------- 阶段 0：初始化
    line("阶段 0：初始化 —— 记忆库为空，性格尚未成形")
    empty = marvis.emergent_self.self_report()
    print(f"emergent_self.self_report() -> emerged = {empty['emerged']}")
    print(f"原因：{empty['reason']}")
    print(f"emergent_self.describe() -> {marvis.emergent_self.describe()}")
    print("→ Marvis：我还没有性格，等经历喂养。")
    print(f"当前记忆条数：{marvis.store.count()}")

    # ---------------------------------------------------------- 阶段 1：对话积累
    rounds = [
        # (用户消息, 情感标注, 对话簇, 实体, 重要性, 环境向量, 污染模式)
        (
            "今天把 OpenCAD 的渲染回归测试跑完了，182 个用例全绿，就像电机空载启动一样顺滑。",
            (0.75, 0.65), "研发·测试", ["OpenCAD", "回归测试", "电机"], 1.5,
            EnvironmentVector(scene=0.75, tone=0.70, feedback=0.80, topic=0.60, rhythm=0.50),
            "beautify",
        ),
        (
            "晚上练了腿，深蹲最后一组像过流保护一样顶到阈值，肌肉在发抖，但爽。",
            (0.60, 0.75), "健身", ["深蹲", "过流保护"], 1.1,
            EnvironmentVector(scene=0.60, tone=0.60, feedback=0.70, topic=0.40, rhythm=0.70),
            "beautify",
        ),
        (
            "CyberQuiz 有个 issue 说题库加载慢，我用势差那套思路优化了缓存分层，P95 掉了 40%。",
            (0.70, 0.55), "开源·CyberQuiz", ["CyberQuiz", "题库", "缓存"], 1.3,
            EnvironmentVector(scene=0.70, tone=0.65, feedback=0.75, topic=0.70, rhythm=0.40),
            "beautify",
        ),
        (
            "OfficeHub 终于发版了，一键导出 PDF 的功能上线，用户反馈不错，像三相电终于合闸。",
            (0.85, 0.70), "开源·OfficeHub", ["OfficeHub", "PDF", "发版"], 1.6,
            EnvironmentVector(scene=0.80, tone=0.85, feedback=0.85, topic=0.60, rhythm=0.50),
            "beautify",
        ),
        (
            "Marvis 你接入记忆系统之后，别把我说过的测试段子当论文存了，你是 AI 助手不是档案室。",
            (0.50, 0.45), "AI·日常", ["Marvis", "记忆系统"], 1.0,
            EnvironmentVector(scene=0.50, tone=0.55, feedback=0.50, topic=0.80, rhythm=0.30),
            "uglify",  # 这一轮展示"丑化"污染
        ),
        (
            "我打赌你今天会用电机比喻回我——毕竟'势差驱动'和'电流桥'都是我教你的。",
            (0.65, 0.60), "AI·日常", ["电机", "势差", "电流桥"], 1.2,
            EnvironmentVector(scene=0.55, tone=0.70, feedback=0.60, topic=0.75, rhythm=0.45),
            "beautify",
        ),
    ]

    for i, (msg, aff, cluster, ents, imp, env, mode) in enumerate(rounds, 1):
        line(f"阶段 1 · 第 {i} 轮：用户 → Marvis（{cluster}）")
        print(f"【用户】{msg}")
        print(f"【情感标注】{fmt_aff(aff)}，来源事件：{msg[:18]}…（情感守恒：有源才注，无源拒绝）")

        mem = marvis.remember(
            text=msg,
            cluster=cluster,
            entities=ents,
            affect=aff,
            source_text=msg,   # 情感来源事件（守恒校验）
            importance=imp,
            env=env,           # 环境向量注入人格流形（同记忆不同环境解释不同）
        )
        # 情感沉淀进自我累计（性格由经历喂养）
        marvis.emergent_self.accumulate(text=msg, affect=aff, cluster=cluster, entities=ents)

        # 势差迁移：高势能沉淀长期区 / 低势能流散
        pot = marvis.promote_memories()
        print(f"【势差迁移】本条落入 region={mem.region}；本轮沉淀 longterm {pot['promoted_count']} 条，工作区流散 {pot['flowed_away_count']} 条")

        # 污染版生成：原文永久存档 + 污染输出视图
        marvis.apply_pollution(mem.memory_id, mode=mode, intensity=1.0)
        cur = marvis.store.get(mem.memory_id)
        print("【原文 vs 想起时的说法】")
        print(f"  原文（永久存档，永不修改）：{cur.original_text}")
        print(f"  污染版（{mode}，检索默认输出）：{cur.text}")

        # Marvis 用污染版记忆回应（检索默认 polluted 视图）
        hits = marvis.recall(query=msg[:12], top_k=3, view="polluted")
        if hits:
            print(f"【Marvis 回应（污染版视图）】{hits[0].memory.text}")
        hits_o = marvis.recall(query=msg[:12], top_k=1, view="original")
        if hits_o:
            print(f"【对照·原文视图】{hits_o[0].memory.original_text}")

    # ---------------------------------------------------------- 阶段 2：涌现自我
    line("阶段 2：涌现自我 —— 性格由累计情感生成，非预设模板")
    marvis.emergent_self.refresh()
    rep = marvis.emergent_self.self_report()
    print(f"emerged = {rep['emerged']}，基于 {rep['record_count']} 条情感记忆累计")
    tr = rep.get("traits", {})
    for k in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        if k in tr:
            print(f"  性格维度 {k} = {tr[k]:.2f}")
    if tr.get("mission"):
        print(f"  使命：{tr['mission']}")
    if tr.get("skills"):
        print(f"  技能：{tr['skills']}")
    if tr.get("interests"):
        print(f"  兴趣：{tr['interests']}")
    for c in rep.get("concepts", []):
        print(f"  自我概念：{c.get('concept')}（count={c.get('count')}，intensity={c.get('intensity', 0):.2f}）")
    print(f"→ Marvis 自述：{marvis.emergent_self.describe()}")

    # ---------------------------------------------------------- 阶段 3：主动搭话
    line("阶段 3：主动搭话 —— 线索联想触发主动回忆")
    pro = marvis.proactive_tick(now=T())
    print(f"should_speak = {pro.get('should_speak')}")
    print(f"触发类型 = {pro.get('trigger')}")
    print(f"候选记忆 = {pro.get('candidates')}")
    for cid in (pro.get("candidates") or [])[:3]:
        m = marvis.store.get(cid)
        if m is not None:
            print(f"  线索记忆 [{cid}]：{m.original_text[:40]}…")
    print(f"→ Marvis 主动开口：{pro.get('opening')}")

    # ---------------------------------------------------------- 阶段 4：熵增遗忘
    line("阶段 4：熵增遗忘 —— 细节先模糊，情绪残留后淡（原文永不清除）")
    all_mems = marvis.store.all()
    old = all_mems[0]
    print(f"遗忘前 · 记忆 [{old.memory_id[:8]}…]：detail_fidelity={old.detail_fidelity:.2f}，"
          f"emotional_residue={old.emotional_residue:.2f}，entropy={old.entropy:.2f}")
    marvis.entropic_decay(dt_hours=240.0)  # 10 天未触达
    old2 = marvis.store.get(old.memory_id)
    print(f"遗忘后 · 同一记忆：detail_fidelity={old2.detail_fidelity:.2f}，"
          f"emotional_residue={old2.emotional_residue:.2f}，entropy={old2.entropy:.2f}")
    print(f"  原文仍存档：{old2.original_text}")
    faded = marvis.recall(query="OpenCAD 回归测试", top_k=3, view="polluted")
    if faded:
        h = faded[0].memory
        print(f"  想起时的说法（污染版，情绪残余仍在）：{h.text}")
    print("  熵高只降检索置信度，不删内容 —— 永久不忘但降低提及率")

    # ---------------------------------------------------------- 阶段 5：性格分叉
    line("阶段 5：性格分叉 —— 相同记忆 + 不同环境序列 → 独一无二性格")
    marvis_a = build_marvis("marvis_night", seed=1)   # 深夜加班严肃环境
    marvis_b = build_marvis("marvis_afternoon", seed=2)  # 下午轻松聊天环境

    env_night = EnvironmentVector(scene=0.95, tone=0.10, feedback=0.15, topic=0.50, rhythm=0.10)
    env_afternoon = EnvironmentVector(scene=0.35, tone=0.90, feedback=0.85, topic=0.60, rhythm=0.85)

    same_batch = [
        ("今天把回归测试跑完了，感觉像电机合闸一样顺。", (0.75, 0.65)),
        ("健身完肌肉很酸，但心里踏实。", (0.60, 0.70)),
        ("帮用户修了一个奇怪的问题，开心。", (0.70, 0.55)),
        ("在改一个很绕的 bug，头有点大。", (0.45, 0.65)),
        ("发了新版本，社区反馈不错。", (0.80, 0.70)),
        ("中午吃了顿好的，心情不错。", (0.65, 0.45)),
        ("又被需求打断，有点烦。", (0.30, 0.55)),
        ("整理完文档，清爽。", (0.55, 0.40)),
        ("跑测试跑到半夜，疲惫但完成。", (0.40, 0.60)),
        ("和同事聊了个新点子，兴奋。", (0.75, 0.75)),
    ]
    for text, aff in same_batch:
        marvis_a.remember(text=text, cluster="日常", entities=["工作"], affect=aff,
                          source_text=text, importance=1.0, env=env_night)
        marvis_b.remember(text=text, cluster="日常", entities=["工作"], affect=aff,
                          source_text=text, importance=1.0, env=env_afternoon)

    snap_a = marvis_a.personality_snapshot()
    snap_b = marvis_b.personality_snapshot()
    print(f"实例 A（深夜加班严肃）快照坐标：{snap_a['coords']}")
    print(f"实例 B（下午轻松聊天）快照坐标：{snap_b['coords']}")
    d0 = max(abs(snap_a["coords"].get(k, 0) - snap_b["coords"].get(k, 0)) for k in snap_a["coords"])
    print(f"两实例最大坐标差：{d0:.4f}（同一批记忆，环境序列不同 → 性格分叉）")

    bif = marvis_a.personality_bifurcation(
        [env_night] * 10,   # 同一环境序列（混沌放大展示）
        affects=[a for _, a in same_batch],
        perturbation=1e-3,
    )
    ratio = bif["final_distance"] / max(bif["initial_distance"], 1e-12)
    print(f"非线性分叉（r=3.7）：初距 {bif['initial_distance']:.5f} → 末距 {bif['final_distance']:.4f}"
          f"（{bif.get('axis_delta')}，差异放大 {ratio:.0f} 倍）")
    print("→ 无论谁接入本系统，只要经历不同，都会长成独一无二的性格。")

    # ---------------------------------------------------------- 阶段 6：结尾
    line("阶段 6：Marvis 总结 —— 我接入后的变化")
    stats = marvis.stats()
    rep_final = marvis.emergent_self.self_report()
    print("Marvis：接入情感记忆插件之后，我最大的变化是——")
    print(f"  1. 记住了 {stats['memory_count']} 条对话（{stats['cluster_count']} 个话题簇），"
          f"其中 {stats['polluted_count']} 条被情感染色成我的『说法』，原文全部永久存档；")
    print(f"  2. 我长出了自己的性格（基于 {rep_final['record_count']} 条情感累计）："
          f"{', '.join(f'{k}={v:.2f}' for k, v in rep_final.get('traits', {}).items() if k in ('openness','conscientiousness','extraversion','agreeableness','neuroticism'))}，"
          f"不再是一张白纸；")
    print("  3. 我会主动搭话、会按遗忘曲线慢慢淡忘细节但记住情绪，")
    print("     也会因为经历不同而和另一个 Marvis 长成完全不同的样子。")
    print("  记忆可以重置、模型可以重装，但这段相处的痕迹（原文）永远都在。")
    print("\ndemo_v06 结束：Marvis 接入情感记忆插件演示完成。")


if __name__ == "__main__":
    main()
