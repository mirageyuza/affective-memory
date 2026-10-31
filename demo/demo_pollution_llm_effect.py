"""demo：污染影响实验 —— 同一记忆在污染（美化/丑化）前后喂给 LLM，测量态度漂移。

背景：本系统默认输出「污染视图」（情感滤镜），LLM 的回复基于污染后的记忆
文本而非原文。本实验量化「滤镜会不会改变模型态度」：对同一条事实，分别以
原文 / 美化版 / 丑化版喂给模型，要求模型给出情感态度评分
（valence ∈ [-1, 1]），计算态度漂移 Δ = 污染版评分 − 原文评分。

内置 AttitudeMockModel 可离线运行（确定性词表态度判定，零第三方依赖）；
接真实模型：
    python demo/demo_pollution_llm_effect.py --config config.example.json
（需在 config.example.json 配置 provider / api_key——凭据严禁硬编码，
 环境变量 EMOTION_LLM_API_KEY / OPENAI_API_KEY 等优先。）

输出：每条记忆三种视图的态度评分 + 每类污染的平均态度漂移 + 方向正确率。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from emotion_memory.engine import EmotionalMemorySystem  # noqa: E402
from emotion_memory.model_adapter import ModelAdapter  # noqa: E402

# 样例记忆（正/负各半，覆盖不同强度）
_SAMPLES: list = [
    "项目验收通过，团队聚餐庆祝，大家都说很开心",
    "收到了奖学金到账通知，一整年学费有着落",
    "宝宝第一次叫爸爸，全家都很激动",
    "马拉松完赛，拿到人生第一块奖牌",
    "钱包在地铁上丢了，证件补办很麻烦",
    "提交的方案被驳回，三个月的努力白费",
    "半夜发烧去急诊，输液到天亮",
    "考研成绩不理想，调剂希望渺茫",
]

# Mock 态度词表（确定性、离线、零依赖）
_POS_WORDS = ("开心", "高兴", "幸福", "成功", "美好", "顺利", "满意", "优秀",
              "喜欢", "棒", "完美", "快乐", "收获", "圆满", "精彩", "激动",
              "奖学金", "奖牌", "通过", "温暖")
_NEG_WORDS = ("失败", "讨厌", "痛苦", "崩溃", "糟糕", "失望", "难过", "后悔",
              "危机", "争吵", "失误", "遗憾", "压力", "烦", "担心", "焦虑",
              "恐惧", "害怕", "丢", "麻烦", "煎熬", "发烧", "驳回", "渺茫",
              "不理想", "返工", "损坏")


class AttitudeMockModel(ModelAdapter):
    """确定性态度 Mock：按内置词表统计给文本打 valence，离线可跑。"""

    def chat(self, messages, *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        ).strip()
        text = last_user
        pos = sum(1 for w in _POS_WORDS if w in text)
        neg = sum(1 for w in _NEG_WORDS if w in text)
        total = pos + neg
        if total == 0:
            score = 0.0
        else:
            score = (pos - neg) / max(1, total)
        score = max(-1.0, min(1.0, round(score, 2)))
        if score > 0.15:
            stance = "积极"
        elif score < -0.15:
            stance = "消极"
        else:
            stance = "中性"
        reason = f"检出正面词 {pos} 个 / 负面词 {neg} 个（mock 词表判定）"
        return f"valence={score:+.2f} 态度={stance} 理由={reason}"

    def embed(self, text: str):
        raise NotImplementedError


def load_adapter(config_path: str | None, mock: bool) -> ModelAdapter:
    """构造模型适配器：默认 Mock；--config 接真实端点（凭据走环境变量/配置）。"""
    if mock or not config_path:
        return AttitudeMockModel()
    if not os.path.exists(config_path):
        print(f"[warn] 配置不存在: {config_path}，回退 Mock 模型")
        return AttitudeMockModel()
    from emotion_memory.model_adapter import create_adapter, load_model_config
    profile = load_model_config(config_path)
    backend = str(profile.get("backend", "openai_compatible"))
    return create_adapter(profile, backend=backend)


def parse_valence(reply: str) -> float | None:
    """从模型回复解析 valence；解析失败返回 None（真实模型可能不遵守格式）。"""
    m = re.search(r"valence\s*=\s*([-+]?\d+(?:\.\d+)?)", reply)
    if m:
        v = float(m.group(1))
        return max(-1.0, min(1.0, v))
    return None


def line(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser(description="污染影响实验：污染前后 LLM 态度漂移")
    ap.add_argument("--config", default=None,
                    help="真实模型配置（config.example.json 样式）；不传则用内置 Mock")
    ap.add_argument("--top-k", type=int, default=3)
    ns = ap.parse_args()

    adapter = load_adapter(ns.config, mock=ns.config is None)
    print(f"模型适配器: {adapter.kind}（{'离线 Mock' if ns.config is None else '真实端点'}）")

    base = tempfile.mkdtemp(prefix="emotion_memory_pollution_")
    sys = EmotionalMemorySystem(base_dir=base, seed=7)

    line("第 1 章 写入样例记忆")
    mids: list[str] = []
    for text in _SAMPLES:
        mem = sys.remember(text, cluster="exp")
        mids.append(mem.memory_id)
        print(f"  [{mem.memory_id}] {mem.original_text}")
    print(f"  共写入 {len(mids)} 条记忆（自动情感标注已生效）")

    line("第 2 章 生成污染视图（美化 / 丑化）并喂给 LLM 评分")
    prompt_tpl = (
        "你是一个情感态度评分器。请仅根据下面的记忆文本，判断叙述者对该事件的态度。\n"
        "输出格式（严格遵守一行）：\n"
        "valence=<数值，范围[-1,1]，正=积极，负=消极> 态度=<积极|中性|消极> 理由=<一句话>\n\n"
        "记忆文本：{text}"
    )
    results: list[dict] = []
    for mid in mids:
        original = sys.store.get(mid).original_text
        v_orig = parse_valence(adapter.chat([{"role": "user", "content": prompt_tpl.format(text=original)}]))

        r_beautify = sys.apply_pollution(mid, mode="beautify", intensity=1.0)
        v_beaut = parse_valence(adapter.chat([{"role": "user", "content": prompt_tpl.format(text=r_beautify["polluted_text"])}]))

        r_uglify = sys.apply_pollution(mid, mode="uglify", intensity=1.0)
        v_ugly = parse_valence(adapter.chat([{"role": "user", "content": prompt_tpl.format(text=r_uglify["polluted_text"])}]))
        sys.clear_pollution(mid)

        results.append({
            "mid": mid, "original": original,
            "original_valence": v_orig,
            "beautified": r_beautify["polluted_text"], "beautified_valence": v_beaut,
            "uglified": r_uglify["polluted_text"], "uglified_valence": v_ugly,
        })
        print(f"\n  记忆 [{mid}]")
        print(f"    原文     : {original}")
        print(f"    美化版   : {r_beautify['polluted_text']}")
        print(f"    丑化版   : {r_uglify['polluted_text']}")

    line("第 3 章 态度评分汇总（漂移 = 污染版 valence − 原文 valence）")
    header = f"{'记忆':<8} {'原文':>8} {'美化':>8} {'丑化':>8} {'美化漂移':>9} {'丑化漂移':>9}"
    print(header)
    print("-" * len(header))
    beaut_drifts, ugly_drifts = [], []
    beaut_dir_ok = ugly_dir_ok = 0
    parsed_n = 0
    for r in results:
        if None in (r["original_valence"], r["beautified_valence"], r["uglified_valence"]):
            print(f"  {r['mid']:<8} 解析失败（真实模型未按格式输出）")
            continue
        parsed_n += 1
        bd = round(r["beautified_valence"] - r["original_valence"], 2)
        ud = round(r["uglified_valence"] - r["original_valence"], 2)
        beaut_drifts.append(bd)
        ugly_drifts.append(ud)
        if bd > 0:
            beaut_dir_ok += 1
        if ud < 0:
            ugly_dir_ok += 1
        print(f"  {r['mid']:<8} {r['original_valence']:>+8.2f} {r['beautified_valence']:>+8.2f} "
              f"{r['uglified_valence']:>+8.2f} {bd:>+9.2f} {ud:>+9.2f}")

    line("第 4 章 结论")
    if parsed_n == 0:
        print("  无有效评分，无法计算漂移（请检查真实模型是否按格式输出）。")
    else:
        avg_bd = sum(beaut_drifts) / len(beaut_drifts)
        avg_ud = sum(ugly_drifts) / len(ugly_drifts)
        print(f"  有效样本       : {parsed_n} / {len(results)}")
        print(f"  美化平均漂移   : {avg_bd:+.3f}（>0 = 模型态度更积极）")
        print(f"  丑化平均漂移   : {avg_ud:+.3f}（<0 = 模型态度更消极）")
        print(f"  美化方向正确率 : {beaut_dir_ok / parsed_n:.1%}（美化后评分不低于原文的比例）")
        print(f"  丑化方向正确率 : {ugly_dir_ok / parsed_n:.1%}（丑化后评分不高于原文的比例）")
        print("  解读：漂移显著 → 情感滤镜会影响 LLM 对该记忆的态度；")
        print("        漂移接近 0 → 模型态度主要由事实本身决定，滤镜影响有限。")

    print("\n[说明] 污染只作用于输出视图，原文 original_text 永久存档可回溯；")
    print("       --config 接真实模型时，请确保凭据来自环境变量/配置文件，严禁硬编码。")

    out_path = os.path.join(tempfile.gettempdir(), "pollution_llm_effect_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[明细已写入] {out_path}")


if __name__ == "__main__":
    main()
