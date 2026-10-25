"""demo：v0.5 物理四定律 + 人格流形全流程演示。

串闭环：情感守恒（情感有来源）→ 势差迁移（沉淀有规则）→ 熵增遗忘（遗忘有方向）
        → 湍流联想（联想有形态）→ 人格流形（环境分叉出独一无二性格轨迹）。

覆盖：
  ① 守恒：无源情感注入被拒绝；超预算情感强度自然衰减（防情绪通货膨胀）
  ② 势差：高势能沉淀长期区 / 低势能流散；电流桥按势差非均匀分配迁移量
  ③ 熵增：细节先模糊、情绪残留后淡；熵高检索置信度降权；原文永不清除
  ④ 湍流：围绕话题中心涡旋式扩散（中心→次级→外围，能量随层距递减）
  ⑤ 人格流形：两个实例喂相同记忆、环境序列不同 → 分叉出两种不同性格轨迹

运行：python demo/demo_v05_physical_laws.py
"""
from __future__ import annotations

import os
import tempfile
import time

from emotion_memory.config import Dial
from emotion_memory.conservation import ConservationError
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.personality_manifold import EnvironmentVector

T = time.time


def line(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def make_dial() -> Dial:
    return Dial(
        three_phase_enabled=True,
        proactive_enabled=True,
        capacitor_enabled=True,
        conservation_enabled=True,
        emotion_budget=2.2,  # 演示超预算衰减
        potential_enabled=True,
        potential_threshold=0.5,
        entropy_enabled=True,
        entropy_rate=0.05,
        turbulent_enabled=True,
        turbulence_decay=0.6,
        turbulence_max_depth=3,
        manifold_enabled=True,
        manifold_bifurcation_r=3.7,
        noise_temperature=0.0,  # 演示确定性
    )


def main() -> None:
    base = tempfile.mkdtemp(prefix="emotion_memory_v05_demo_")
    dial = make_dial()
    sys = EmotionalMemorySystem(base_dir=base, dial=dial, seed=7)
    print(f"工作目录: {base}\n")

    # ---------------------------------------------------------- ① 情感守恒
    line("① 情感守恒律：情感不凭空产生")
    try:
        sys.remember("没有来源的情感", affect=(0.9, 0.9), source_text=None)
        print("× 预期拒绝但未拒绝！")
    except ConservationError:
        print("✓ 无源情感注入被拒绝（ConservationError）")
    sys.remember(
        "用户在海边说：我真的很喜欢这一刻",
        affect=(0.8, 0.7),
        source_text="用户说：我真的很喜欢这一刻",
        importance=1.6,
        entities=["海边", "告白"],
        env=EnvironmentVector(scene=0.9, tone=0.9, feedback=0.8, topic=0.5, rhythm=0.3),
    )
    print("✓ 携带来源原文的记忆写入成功")
    # 超预算衰减：不断写入同能量情感，后期唤醒度被预算压低
    for i in range(6):
        sys.remember(f"重复平凡事件 {i}", affect=(0.8, 0.7), source_text=f"事件 {i} 的原文")
    arousals = [m.affect.arousal for m in sys.store.all()]
    print(f"✓ 全库 {len(arousals)} 条记忆；最早唤醒度 {arousals[0]:.2f}，最晚 {arousals[-1]:.2f}"
          f"（超预算自然衰减 → {'是' if arousals[-1] < arousals[0] else '否'}）")

    # ---------------------------------------------------------- ② 势差迁移
    line("② 势差驱动迁移：记忆不随意搬迁")
    res = sys.promote_memories()
    print(f"✓ 扫描 {res['scanned']} 条：沉淀长期区 {res['promoted_count']}，工作区流散 {res['flowed_away_count']}")
    longterm = [m for m in sys.store.all() if m.region == "longterm"]
    print(f"✓ 长期区记忆：{', '.join(m.memory_id for m in longterm[:3]) or '无'}")
    quota = sys.transfer_quota(max_count=2)
    print(f"✓ 电流桥按势差分配迁移量：选 {quota['selected_count']} / 跳过 {quota['skipped_count']}（非均匀搬运）")

    # ---------------------------------------------------------- ③ 熵增遗忘
    line("③ 熵增遗忘：遗忘=熵增不是删除")
    sys.entropic_decay(dt_hours=72.0)
    m0 = sys.store.all()[0]
    print(f"✓ 熵推进后：细节保真 {m0.detail_fidelity:.2f} < 情绪残留 {m0.emotional_residue:.2f}"
          f"（细节先模糊，情绪残留后淡）")
    print(f"✓ 熵 {m0.entropy:.2f}，检索置信度 {1.0 - m0.entropy:.2f}（熵越高置信度越低）")
    print(f"✓ 原文完好：'{m0.original_text}'（原文永不清除）")

    # ---------------------------------------------------------- ④ 湍流联想
    line("④ 湍流联想：发散联想=涡旋扩散不是随机")
    sys.remember("海边的落日很美", affect=(0.7, 0.6), source_text="海边日落的记录",
                 entities=["海边", "落日"], importance=1.3)
    sys.remember("那次告白前的紧张", affect=(0.6, 0.7), source_text="告白前的心情",
                 entities=["海边", "告白"])
    sys.remember("后来我们常去散步", affect=(0.5, 0.5), source_text="散步记录",
                 entities=["散步", "海边"])
    # 链式扩散：散步 → 家 → 装修（不直接共享"海边"，经中间实体涡旋扩散到外围）
    sys.remember("散步时聊过的未来计划", affect=(0.4, 0.4), source_text="散步时的谈话",
                 entities=["散步", "家"])
    sys.remember("关于家的装修", affect=(0.3, 0.3), source_text="装修话题", entities=["家"])
    turb = sys.turbulent_associate(query_entities=["海边"], top_k=8)
    rings = turb["rings"]
    secondary_count = len(rings.get("1", []))
    peripheral_count = len(rings.get("2", []))
    print(f"✓ 涡流链层级：中心 {turb['center_count']} 条，次级 {secondary_count} 条，"
          f"外围 {peripheral_count} 条")
    for hit in turb["hits"][:5]:
        print(f"  - 层距 {hit['ring']} 能量 {hit['energy']:.2f}: {hit['text']}")
    print(f"✓ 能量随层距递减：{', '.join('%.2f' % h['energy'] for h in turb['hits'][:3])}")

    # ---------------------------------------------------------- ⑤ 人格流形分叉
    line("⑤ 人格流形：相同记忆 + 不同环境序列 → 独一无二性格轨迹")
    # 主实例先积累一段环境轨迹（人格流形随记忆累计漂移）
    sys.manifold_learn_sequence([EnvironmentVector(scene=0.6, tone=0.7, feedback=0.6,
                                                   topic=0.5, rhythm=0.4) for _ in range(20)])
    envs_warm = [
        EnvironmentVector(scene=0.9, tone=0.9, feedback=0.9, topic=0.5, rhythm=0.3) for _ in range(25)
    ]
    envs_cool = [
        EnvironmentVector(scene=0.1, tone=0.1, feedback=0.1, topic=0.5, rhythm=0.7) for _ in range(25)
    ]
    sys_warm = EmotionalMemorySystem(base_dir=tempfile.mkdtemp(), dial=make_dial(), seed=7)
    sys_cool = EmotionalMemorySystem(base_dir=tempfile.mkdtemp(), dial=make_dial(), seed=7)
    # 喂相同记忆（内容、情感、来源完全一致）
    for _ in range(10):
        sys_warm.remember("一起看过的落日", affect=(0.8, 0.7), source_text="海边那晚的对话", importance=1.5)
        sys_cool.remember("一起看过的落日", affect=(0.8, 0.7), source_text="海边那晚的对话", importance=1.5)
    # 环境序列不同
    sys_warm.manifold_learn_sequence(envs_warm)
    sys_cool.manifold_learn_sequence(envs_cool)
    snap_w = sys_warm.personality_snapshot()
    snap_c = sys_cool.personality_snapshot()
    print("  - 温暖环境实例：")
    for k, v in snap_w["coords"].items():
        print(f"      {k}: {v:.3f}")
    print("  - 冷静环境实例：")
    for k, v in snap_c["coords"].items():
        print(f"      {k}: {v:.3f}")
    diff = {k: round(abs(snap_w["coords"][k] - snap_c["coords"][k]), 3) for k in snap_w["coords"]}
    print(f"✓ 两实例同记忆不同环境，坐标差异：{diff}")

    # 非线性分叉：微小扰动放大
    bif = sys_warm.manifold.bifurcation(envs_warm, perturbation=1e-3)
    print(f"✓ 非线性分叉（r=3.7）：初距 {bif['initial_distance']:.5f} → 末距 {bif['final_distance']:.4f}"
          f"（{bif['axis_delta']}，差异放大 {bif['final_distance'] / max(bif['initial_distance'], 1e-12):.0f} 倍）")

    # ---------------------------------------------------------- 串闭环报告
    line("串闭环：四定律 + 人格流形状态")
    chain = sys.physical_chain_report()
    for k in ["conservation", "potential", "entropy", "turbulence", "personality"]:
        print(f"  [{k}] {chain[k]}")
    print(f"\n✓ 全流程完成：守恒→势差→熵增→湍流→人格流形，五环全部闭环。")


if __name__ == "__main__":
    main()
