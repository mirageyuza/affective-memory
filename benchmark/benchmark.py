"""benchmark 脚本：量化「回忆命中率 / 情感偏差 / 对照组对比 / embedding 对照」。

输出 JSON 报告 + 人类可读 txt 报告。零第三方依赖（embedding 对照后端可选）。

指标定义：
  1. 回忆命中率 hit_rate
     向系统写入 N 条 seed 记忆（含情感倾向：积极/消极各半），对每条用其
     关键词查询，判定目标 memory_id 是否出现在 top-k 检索结果中。
     对照组 A（无关查询）：查询词与全部记忆无关，命中率应接近 0，
     用于校准「不能什么都命中」。
  2. 情感偏差 affect shift
     对每条记忆依次应用 beautify / uglify 污染（intensity=1.0），再用
     自动情感标注器计算污染视图的 (valence, arousal)，与原文标注对比：
       - beautify 方向正确率：污染视图 valence 不低于原文 valence（美化）；
       - uglify  方向正确率：污染视图 valence 不高于原文 valence（丑化）；
       - 平均情感偏移量 mean_abs_shift = mean(|Δv| + |Δa|)。
  3. 对照组对比（污染开关对召回的影响）
     索引基于原文构建，污染不应破坏召回：对比污染前/污染后对同一组查询的
     命中率，二者应一致（diff = 0）。
  4. 原文审计：污染后 original_text 必须逐条不变；污染视图必须确实被改变
    （beautify/uglify 文本替换发生比例），报告其统计量。
  5. 词袋 vs embedding 对照（v0.6）
     同一批文本-查询配对分别用词袋哈希（默认）与可选 embedding 向量化器
     跑命中率，输出 delta。embedding 后端默认 mock（确定性 8 维，离线可跑，
     仅验证接口链路）；指定 sentence_transformers / openai_compat /
     local_compat / gemini 且不可用时，如实标注 status=unavailable。

用法：
  python -m benchmark.benchmark --memories 20 --top-k 3 --seed 42
  python -m benchmark.benchmark --vectorizer embedding --embedding-backend mock --report-path benchmark/benchmark_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from emotion_memory.engine import EmotionalMemorySystem  # noqa: E402
from emotion_memory.config import Dial  # noqa: E402

# 中文记忆样例：情感倾向 1=积极, -1=消极（用于构造可控对照组）
_SAMPLES: List[Tuple[str, float]] = [
    # 积极
    ("去年生日和三个老同学爬山，很开心", 1.0),
    ("今天项目终于验收通过，团队聚餐庆祝", 1.0),
    ("收到了奖学金到账通知，一整年学费有着落", 1.0),
    ("妈妈做的红烧肉真好吃，家的味道", 1.0),
    ("面试通过了，下周入职新公司", 1.0),
    ("老朋友专程来探望，聊到深夜", 1.0),
    ("看完樱花回程，路上拍到了绝美夕阳", 1.0),
    ("宝宝第一次叫爸爸，全家都很激动", 1.0),
    ("马拉松完赛，拿到人生第一块奖牌", 1.0),
    ("期待已久的演唱会门票终于抢到了", 1.0),
    # 消极
    ("周末加班到凌晨，项目又要返工", -1.0),
    ("钱包在地铁上丢了，证件补办很麻烦", -1.0),
    ("体检报告有异常项，需要复查", -1.0),
    ("和搭档因为分工吵了一架，心里堵得慌", -1.0),
    ("提交的方案被驳回，三个月的努力白费", -1.0),
    ("半夜发烧去急诊，输液到天亮", -1.0),
    ("房租到期被要求立刻搬走，很狼狈", -1.0),
    ("电脑硬盘突然损坏，毕业论文没备份", -1.0),
    ("航班延误六小时，在机场干等", -1.0),
    ("考研成绩不理想，调剂希望渺茫", -1.0),
]


def _query_from_text(text: str, rng: random.Random) -> str:
    """从原文抽取关键词构造可命中查询（抽取非停用字，最多 6 个）。"""
    stops = set("的了在和与都很有被又很个把对让他")
    toks = [c for c in text if c not in stops and not c.isspace()]
    n = min(6, len(toks))
    picked = rng.sample(toks, n)
    return " ".join(picked)


class BenchmarkRunner:
    def __init__(self, memories: int, top_k: int, seed: int,
                 vectorizer: str = "bow", embedding_backend: str = "mock"):
        self.rng = random.Random(seed)
        self.memories = memories
        self.top_k = top_k
        self.vectorizer = vectorizer          # bow | embedding
        self.embedding_backend = embedding_backend  # mock / openai_compat / ...

    # ------------------------------------------------------------ 构造
    def _fresh_system(self, vectorizer: str = "bow",
                      vectorizer_instance: Optional[Any] = None) -> EmotionalMemorySystem:
        base = tempfile.mkdtemp(prefix="emotion_memory_bench_")
        # 命中率对照组使用零噪声刻度盘（noise_temperature=0），
        # 保证振荡评分完全确定，污染前/后严格可比（|Δ|=0 即污染不破坏召回）。
        dial = Dial(noise_temperature=0.0, emotion_gain=0.3,
                    jump_probability=0.05, amplitude=1.0)
        if vectorizer == "embedding":
            return EmotionalMemorySystem(base_dir=base, seed=self.rng.randrange(10**6),
                                         dial=dial, vectorizer=vectorizer_instance)
        return EmotionalMemorySystem(base_dir=base, seed=self.rng.randrange(10**6), dial=dial)

    def _build_seed_payloads(self, n: int) -> List[Tuple[str, float]]:
        """构造 n 条可控样本（含情感倾向），供多个向量化后端复用同一批文本。"""
        pool = list(_SAMPLES)
        while len(pool) < n:
            pool = pool + pool
        return self.rng.sample(pool, n)

    def _load_seeds(
        self, sys: EmotionalMemorySystem, payloads: List[Tuple[str, float]]
    ) -> List[Tuple[str, str, float]]:
        """按给定 payloads 写入记忆，返回 [(memory_id, original_text, polarity)]。"""
        out = []
        for text, polarity in payloads:
            mem = sys.remember(text, cluster="bench")
            out.append((mem.memory_id, mem.original_text, polarity))
        return out

    # ------------------------------------------------------------ 命中率
    def _make_queries(self, seeds: List[Tuple[str, str, float]]) -> List[str]:
        """从已写入的种子原文一次性生成固定查询组，保证对照组（污染前/后/原文/embedding）使用同一批查询。"""
        return [_query_from_text(text, self.rng) for _, text, _ in seeds]

    def _hit_rate(
        self,
        sys: EmotionalMemorySystem,
        seeds: List[Tuple[str, str, float]],
        queries: List[str],
        view: str,
        now: float,
    ) -> Dict[str, Any]:
        hits = 0
        unrelated_hits = 0
        n = len(seeds)
        for i, (mid, _text, _pol) in enumerate(seeds):
            found = sys.recall(queries[i], top_k=self.top_k, view=view, now=now)
            if any(h.memory.memory_id == mid for h in found):
                hits += 1
            # 无关查询：旋转错位样本（保证词面不同）
            j = (i + n // 2 + 1) % n
            unrelated = sys.recall(queries[j], top_k=self.top_k, view=view, now=now)
            if any(h.memory.memory_id == mid for h in unrelated):
                unrelated_hits += 1
        return {
            "hit_rate": round(hits / n, 4) if n else 0.0,
            "unrelated_hit_rate": round(unrelated_hits / n, 4) if n else 0.0,
        }

    # ------------------------------------------------------------ embedding 对照
    def _make_embedding_vectorizer(self):
        from emotion_memory.embeddings import create_embedding_vectorizer
        profile: Dict[str, Any] = {}
        if self.embedding_backend == "sentence_transformers":
            profile = {
                "model_name": os.environ.get(
                    "EMBEDDING_MODEL_NAME",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                "device": os.environ.get("EMBEDDING_DEVICE", "cpu"),
            }
        elif self.embedding_backend in ("openai_compat", "local_compat", "gemini"):
            profile = {
                "base_url": os.environ.get("EMBEDDING_BASE_URL", ""),
                "model": os.environ.get("EMBEDDING_MODEL", ""),
                "local_mode": self.embedding_backend == "local_compat",
            }
        return create_embedding_vectorizer(profile=profile, backend=self.embedding_backend)

    def _embedding_comparison(
        self, payloads: List[Tuple[str, float]], queries: List[str], bow_hit_rate: float
    ) -> Dict[str, Any]:
        """同一批文本-查询用 embedding 向量化器跑命中率；不可用如实标注 unavailable。"""
        try:
            vec = self._make_embedding_vectorizer()
            sys = self._fresh_system(vectorizer="embedding", vectorizer_instance=vec)
            seeds = self._load_seeds(sys, payloads)
            now = time.time()
            emb = self._hit_rate(sys, seeds, queries, view="polluted", now=now)
        except Exception as exc:
            return {
                "status": "unavailable",
                "backend": self.embedding_backend,
                "reason": f"{type(exc).__name__}: {exc}",
                "bow_hit_rate": bow_hit_rate,
            }
        return {
            "status": "ok",
            "backend": self.embedding_backend,
            "note": "mock 后端为确定性 8 维嵌入，仅验证 embedding 接口链路可用，"
                    "不代表真实语义嵌入质量；真实后端不可用时 status=unavailable。"
                    if self.embedding_backend == "mock" else "",
            "bow_hit_rate": bow_hit_rate,
            "embedding_hit_rate": emb["hit_rate"],
            "embedding_unrelated_hit_rate": emb["unrelated_hit_rate"],
            "delta": round(emb["hit_rate"] - bow_hit_rate, 4),
        }

    # ------------------------------------------------------------ 情感偏差
    def _affect_shift(self, sys: EmotionalMemorySystem, seeds: List[Tuple[str, str, float]]) -> Dict[str, Any]:
        beautify_ok = 0
        uglify_ok = 0
        shifts: List[float] = []
        changed_view = 0
        original_unchanged = 0
        n = len(seeds)
        for mid, original, polarity in seeds:
            v0 = sys.store.get(mid).affect.valence
            a0 = sys.store.get(mid).affect.arousal
            # beautify
            r1 = sys.apply_pollution(mid, mode="beautify", intensity=1.0)
            m = sys.store.get(mid)
            v1, a1 = sys.affect_annotator.annotate(r1["polluted_text"])
            shifts.append(abs(v1 - v0) + abs(a1 - a0))
            if v1 >= v0 - 1e-9:
                beautify_ok += 1
            if r1["polluted_text"] != original:
                changed_view += 1
            if m.original_text == original:
                original_unchanged += 1
            # uglify
            r2 = sys.apply_pollution(mid, mode="uglify", intensity=1.0)
            m = sys.store.get(mid)
            v2, a2 = sys.affect_annotator.annotate(r2["polluted_text"])
            shifts.append(abs(v2 - v0) + abs(a2 - a0))
            if v2 <= v0 + 1e-9:
                uglify_ok += 1
            if r2["polluted_text"] != original:
                changed_view += 1
            if m.original_text == original:
                original_unchanged += 1
            sys.clear_pollution(mid)
        return {
            "beautify_direction_ok": round(beautify_ok / n, 4) if n else 0.0,
            "uglify_direction_ok": round(uglify_ok / n, 4) if n else 0.0,
            "mean_abs_shift": round(sum(shifts) / len(shifts), 4) if shifts else 0.0,
            "polluted_view_changed_ratio": round(changed_view / (n * 2), 4) if n else 0.0,
            "original_unchanged": original_unchanged,
        }

    # ------------------------------------------------------------ 主流程
    def run(self) -> Dict[str, Any]:
        payloads = self._build_seed_payloads(self.memories)
        sys = self._fresh_system(vectorizer="bow")
        seeds = self._load_seeds(sys, payloads)
        queries = self._make_queries(seeds)
        now = time.time()  # 固定时间快照，消除振荡相位漂移对命中判定的扰动

        # 对照组：污染前 vs 污染后（同一查询组）命中率
        pre = self._hit_rate(sys, seeds, queries, view="polluted", now=now)
        for mid, _t, _p in seeds:
            sys.apply_pollution(mid, mode="beautify" if _p > 0 else "uglify", intensity=1.0)
        post = self._hit_rate(sys, seeds, queries, view="polluted", now=now)
        orig_view = self._hit_rate(sys, seeds, queries, view="original", now=now)

        shift = self._affect_shift(sys, seeds)

        # v0.6：词袋 vs embedding 对照（同一批文本-查询，仅向量化方式不同）
        bow_hit = pre["hit_rate"]
        emb_cmp = self._embedding_comparison(payloads, queries, bow_hit)

        report = {
            "meta": {
                "script": "benchmark.benchmark",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "memories": self.memories,
                "top_k": self.top_k,
                "seed": 0,  # 保留字段（确定性种子由调用方控制）
                "vectorizer": self.vectorizer,
                "embedding_backend": self.embedding_backend,
            },
            "hit_rate": {
                "pre_pollution": pre,
                "post_pollution": post,
                "original_view": orig_view,
                "recall_stability": round(abs(pre["hit_rate"] - post["hit_rate"]), 4),
            },
            "affect_shift": shift,
            "embedding_comparison": emb_cmp,
        }
        return report


def format_text(report: Dict[str, Any]) -> str:
    m = report["meta"]
    hr = report["hit_rate"]
    af = report["affect_shift"]
    ec = report.get("embedding_comparison", {})
    lines = [
        "=" * 62,
        "类人情感记忆系统 benchmark 报告",
        "=" * 62,
        f"记忆条数        : {m['memories']}",
        f"top-k           : {m['top_k']}",
        f"向量化器        : {m.get('vectorizer', 'bow')}",
        f"时间            : {m['timestamp']}",
        "",
        "【1. 回忆命中率】",
        f"  污染前命中率      : {hr['pre_pollution']['hit_rate']:.2%}",
        f"  污染后命中率      : {hr['post_pollution']['hit_rate']:.2%}",
        f"  原文视图命中率    : {hr['original_view']['hit_rate']:.2%}",
        f"  无关查询命中率    : {hr['pre_pollution']['unrelated_hit_rate']:.2%}",
        f"  召回稳定性(|Δ|)  : {hr['recall_stability']:.4f}（0 = 污染不破坏召回）",
        "",
        "【2. 情感偏差】",
        f"  beautify 方向正确率: {af['beautify_direction_ok']:.2%}（污染版 valence 不低于原文）",
        f"  uglify  方向正确率: {af['uglify_direction_ok']:.2%}（污染版 valence 不高于原文）",
        f"  平均情感偏移(mean|Δv|+|Δa|): {af['mean_abs_shift']:.4f}",
        f"  污染视图确实改变比例 : {af['polluted_view_changed_ratio']:.2%}",
        f"  原文保持不变条次    : {af['original_unchanged']} / {m['memories'] * 2}",
        "",
        "【3. 词袋 vs embedding 对照】",
        f"  status            : {ec.get('status', 'unavailable')}",
        f"  backend           : {ec.get('backend', 'mock')}",
        f"  词袋命中率         : {ec.get('bow_hit_rate', 0.0):.4f}",
    ]
    if ec.get("status") == "ok":
        lines += [
            f"  embedding 命中率   : {ec.get('embedding_hit_rate', 0.0):.4f}",
            f"  embedding 无关命中率: {ec.get('embedding_unrelated_hit_rate', 0.0):.4f}",
            f"  delta(emb - bow)   : {ec.get('delta', 0.0):+.4f}",
        ]
        if ec.get("note"):
            lines.append(f"  注: {ec['note']}")
    else:
        lines.append(f"  不可用原因         : {ec.get('reason', '未尝试 embedding 对照')}")
    lines += [
        "",
        "【4. 对照组结论】",
        "  污染只作用于输出视图，索引基于原文 → 命中率应保持稳定；",
        "  无关查询命中率应显著低于目标查询（说明检索有区分度）。",
        "=" * 62,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="类人情感记忆系统 benchmark")
    ap.add_argument("--memories", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vectorizer", choices=["bow", "embedding"], default="bow",
                    help="主向量化路径：bow=词袋哈希（默认）；embedding=嵌入后端")
    ap.add_argument("--embedding-backend", choices=[
        "mock", "openai_compat", "local_compat", "gemini", "sentence_transformers"],
        default="mock",
        help="embedding 对照后端（默认 mock=确定性 8 维，离线验证链路）")
    ap.add_argument("--report-path", default="benchmark/benchmark_report.json")
    ns = ap.parse_args()

    runner = BenchmarkRunner(memories=ns.memories, top_k=ns.top_k, seed=ns.seed,
                             vectorizer=ns.vectorizer, embedding_backend=ns.embedding_backend)
    report = runner.run()
    report["meta"]["seed"] = ns.seed
    report["meta"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    json_path = ns.report_path
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    txt_path = os.path.splitext(json_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_text(report))

    print(format_text(report))
    print(f"\n[报告已写入] JSON: {json_path}")
    print(f"[报告已写入] TEXT: {txt_path}")


if __name__ == "__main__":
    main()
