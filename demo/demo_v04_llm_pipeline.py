"""demo：v0.4 模型接入层 + 涌现自我 + 主动搭话全流程。

演示内容：
  ① 模型适配层（默认内置 Mock，零凭据零依赖）
  ② 对话管线：写入 → 记忆检索注入 → 模型回复 → 自动沉淀
  ③ 情感累计：每次对话自动打 (valence, arousal) 标签并累计
  ④ 涌现自我：智能体「是什么」由情感累计数据生成，无预设档案
  ⑤ 主动搭话循环：定时触发 → 回忆话题 → 模型开场

支持配置真实端点运行：
  python demo/demo_v04_llm_pipeline.py --config config.example.json
（在线 API 凭据从环境变量 EMOTION_LLM_API_KEY / OPENAI_API_KEY 读取，
  本地 Ollama / LM Studio 留空 key 即可；backend 默认 mock。）
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

from emotion_memory.config import Dial
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.model_adapter import create_adapter, load_model_config
from emotion_memory.conversation_pipeline import ConversationPipeline
from emotion_memory.proactive_loop import ProactiveLoop


def line(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.4 LLM 对话管线演示")
    parser.add_argument("--config", default=None, help="模型接入配置 JSON 路径")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    backend = str(cfg.get("backend", "mock"))
    profile = {}
    if args.config:
        model_cfg = cfg.get("model", {})
        profile = model_cfg.get("local" if str(model_cfg.get("local_mode", False)).lower() in ("true", "1") else "openai_compatible", {})
        profile.setdefault("local_mode", True) if str(model_cfg.get("local_mode", False)).lower() in ("true", "1") else None
    adapter = create_adapter(profile if args.config else None, backend=backend)
    print(f"模型适配器: {adapter.kind}  backend={backend}")

    base = tempfile.mkdtemp(prefix="emotion_memory_v04_demo_")
    dial = Dial(
        proactive_enabled=True,
        proactive_interval_hours=0.5,
        capacitor_enabled=True,
        standby_pool_size=3,
        noise_temperature=0.0,  # 演示确定性
    )
    sys = EmotionalMemorySystem(base_dir=base, dial=dial, seed=7)
    print(f"工作目录: {base}\n")

    pipe = ConversationPipeline(sys, adapter, top_k=3, settle_cluster="对话")

    # ---------------------------------------------------------- ① 预置一条记忆（供注入验证）
    line("① 预置记忆：等待后续对话被检索注入")
    sys.remember("用户最近在学 Python 和数据结构", cluster="学习", entities=["Python"])

    # ---------------------------------------------------------- ② 多轮对话管线
    line("② 对话管线：检索注入 → 模型回复 → 自动沉淀")
    turns = [
        "我在学习 Python 的递归，觉得有点绕",
        "我喜欢傍晚去江边散步，很放松",
        "下个月想报名参加半程马拉松",
        "Python 的装饰器到底是什么原理？",
    ]
    for i, text in enumerate(turns, 1):
        print(f"\n[用户] {text}")
        out = pipe.reply(text, now=time.time())
        print(f"[回复] {out['reply']}")
        ctx = out["context_rows"]
        if ctx:
            print(f"[注入] {len(ctx)} 条相关记忆:")
            for r in ctx["content"].splitlines()[1:]:
                print(f"        {r}")
        st = out["settle"] or {}
        if st.get("memory_id"):
            mem = sys.store.get(st["memory_id"])
            aff = mem.affect if mem else None
            print(f"[沉淀] 记忆已入库 情感=({aff.valence:.2f},{aff.arousal:.2f}) "
                  f"涌现概念数={len(st.get('self_concepts') or [])}")

    # ---------------------------------------------------------- ③ 涌现自我定义
    line("③ 涌现自我：从情感累计数据生成，无预设档案")
    report = sys.emergent_self.self_report()
    print(f"累计记录: {report['record_count']} 条 | 已涌现: {report['emerged']}")
    print(sys.emergent_self.describe())

    # ---------------------------------------------------------- ④ 唤醒重建身份（从累计数据重建）
    line("④ 唤醒重建：从累计情感数据重建自我定义（非固定档案）")
    json_path = os.path.join(base, "emergent_self.json")
    sys.emergent_self.export_json(json_path)
    fresh = EmotionalMemorySystem(base_dir=base, dial=dial, seed=7)
    n = fresh.emergent_self.rebuild_from(json_path)
    print(f"新实例从 {json_path} 重建 {n} 条累计记录")
    print(fresh.emergent_self.describe())

    # ---------------------------------------------------------- ⑤ 主动搭话循环
    line("⑤ 主动搭话循环：定时触发 → 回忆 → 模型开场")
    loop = ProactiveLoop(sys, adapter, interval=0.1, max_openings=3, max_consecutive=2)
    for step in range(3):
        res = loop.maybe_tick(now=time.time())
        if res and res.get("should_speak"):
            print(f"  [{step}] 触发={res['reason']} | 搭话={res['text']}")
        else:
            reason = res.get("reason") if res else "未到周期"
            print(f"  [{step}] 保持安静（{reason}）")

    line("完成")
    print(f"stats: 记忆 {sys.stats()['memory_count']} 条 | "
          f"涌现自我记录 {len(sys.emergent_self.records)} 条")


if __name__ == "__main__":
    main()
