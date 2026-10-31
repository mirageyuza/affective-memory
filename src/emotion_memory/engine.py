"""类人情感记忆系统：顶层封装 API（EmotionalMemorySystem）。

组合全部模块：
  记忆存储层（原文/向量/图索引）→ 检索层（相关簇定位/ANN/top-k 注入）
  → 遗忘曲线 → 震荡命中 → 情感增益/dB 排序 → 沉默层 → 发散/并发（默认关）
  → 电流桥迁移 → 身份生命周期状态机 → 自由度刻度盘
  → 自动情感标注器 → 情感污染滤镜（双层视图，v0.2）。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config import Dial
from .models import AffectLabel, Memory, RankedHit
from .store.storage import MemoryStore
from .store.vector_index import AnnVectorIndex, BagOfWordsVectorizer
from .store.graph_index import EntityGraph
from .algorithms.forgetting import ForgettingCurve
from .algorithms.hit_function import OscillationBank
from .algorithms.affect import rank_hits
from .algorithms.affect_analyzer import DefaultAffectAnalyzer, ExternalAffectAnnotator
from .algorithms.pollution import EmotionPollutionFilter
from .retrieval.recall import RecallEngine
from .retrieval.context import ContextInjector
from .retrieval.proactive import ProactiveRecallEngine
from .algorithms.three_phase import ThreePhaseField
from .algorithms.capacitor import CapacitorModel
from .silence import SilenceLayer
from .divergent import DivergenceController
from .concurrent import ConcurrentRecall
from . import bridge
from .identity import IdentityProfile, IdentityLifecycle
from .self_learning import IdentitySelfLearning
from .unified_self import UnifiedSelfModel
from .emergent_self import EmergentSelf
# ---- v0.5 物理四定律 + 人格流形 ----
from .conservation import EmotionConservation
from .potential_transfer import PotentialTransfer
from .entropic_forgetting import EntropicForgetting
from .turbulent_association import TurbulenceAssociator
from .personality_manifold import PersonalityManifold, EnvironmentVector


class EmotionalMemorySystem:
    """记忆系统主入口。"""

    def __init__(
        self,
        base_dir: str,
        dial: Optional[Dial] = None,
        seed: int = 7,
        identity: Optional[IdentityProfile] = None,
        affect_annotator: Optional[ExternalAffectAnnotator] = None,
        vectorizer: Optional[Any] = None,
    ):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.dial = (dial or Dial()).validate()

        self.store = MemoryStore(os.path.join(base_dir, "memories.jsonl"))
        # v0.6：向量化器可插拔——默认零依赖词袋哈希；embedding 后端经
        # Dial.embedding_enabled 开关或显式 vectorizer 参数接入。
        self._vectorizer = vectorizer if vectorizer is not None else self._make_vectorizer()
        self.vec = AnnVectorIndex(vectorizer=self._vectorizer)
        self.graph = EntityGraph()
        self.bank = OscillationBank(dial=self.dial, seed=seed)
        self.curve = ForgettingCurve()
        self.silence = SilenceLayer()
        self.divergence = DivergenceController(self.dial)
        self.concurrent = ConcurrentRecall(self.dial)

        # 自动情感标注器：默认零依赖内置；可注入外部模型实现
        self.affect_annotator = affect_annotator or DefaultAffectAnalyzer()
        # 情感污染滤镜（双层视图生成器）
        self.pollution_filter = EmotionPollutionFilter()

        self.identity = identity or IdentityProfile(
            agent_name="emotion-agent", model_version="0.2.0", home=base_dir
        )
        self.lifecycle = IdentityLifecycle(self.store, self.identity, base_dir)
        self.recall_engine = RecallEngine(self.store, self.vec, self.graph, self.bank, self.dial)

        # ---- v0.3 新功能组件 ----
        # 电容记忆模型（charge/capacitance/放电阈值 + standby 备选池）
        self.capacitor = CapacitorModel(
            tau_hours=self.dial.capacitor_tau_hours,
            charge_per_touch=self.dial.capacitor_charge_per_touch,
            discharge_per_recall=self.dial.capacitor_discharge_per_recall,
            standby_pool_size=self.dial.standby_pool_size,
        )
        # 旋转磁场唤起调度（三相顺序启动，不破坏 H(t) 接口）
        self.three_phase = ThreePhaseField(self.bank, seed=seed)
        # 主动回忆引擎（定时 / 触发器驱动自主搭话）
        self.proactive = ProactiveRecallEngine(
            interval_hours=self.dial.proactive_interval_hours,
            top_k=self.dial.top_k,
        )
        # 身份自学习（画像渐进演化 + 主动提问确认）
        self.self_model = IdentitySelfLearning(
            self.identity, path=os.path.join(base_dir, "self_model.json")
        )
        # 自我统一模型（1 核心 + 万我分身 + 万物皆我 + 有功/视在指标）
        self.unified_self = UnifiedSelfModel(
            path=os.path.join(base_dir, "unified_self.json"),
            agent_name=self.identity.agent_name,
        )
        # 涌现自我引擎（v0.4）：智能体「是什么」由情感累计生成，无预设模板
        self.emergent_self = EmergentSelf(storage=self.store, engine=self)

        # ---- v0.5 物理四定律 + 人格流形组件 ----
        # ① 情感守恒：来源校验 + 总能量预算（无源注入拒绝，超预算自然衰减）
        self.conservation = EmotionConservation(budget=self.dial.emotion_budget)
        # ② 势差驱动迁移：势能 = 情感强度 × 重复 × 重要性（高势能沉淀长期区）
        self.potential = PotentialTransfer(
            threshold=self.dial.potential_threshold,
            importance_weight=self.dial.importance_weight,
            repeat_weight=self.dial.repeat_weight,
        )
        # ③ 熵增遗忘：细节先模糊、情绪残留后淡，熵越高检索置信度越低
        self.entropy = EntropicForgetting(rate=self.dial.entropy_rate)
        # ④ 湍流联想：涡旋式扩散（中心→次级→外围），能量随层距递减
        self.turbulence = TurbulenceAssociator(
            decay=self.dial.turbulence_decay,
            max_depth=self.dial.turbulence_max_depth,
        )
        # 人格流形：全谱系性格空间 + 环境向量注入 + 非线性分叉（确定性混沌）
        self.manifold = PersonalityManifold(seed=seed, r=self.dial.manifold_bifurcation_r)
        self._boot_at = time.time()

        # 启动时重建索引 + 推进遗忘曲线
        self._rebuild_indexes()
        self.decay_all()

    # ------------------------------------------------------------ 写
    def remember(
        self,
        text: str,
        cluster: str = "default",
        entities: Optional[Sequence[str]] = None,
        affect: Optional[Tuple[float, float]] = None,
        base_strength: float = 1.0,
        silence: bool = False,
        never_silent: bool = False,
        due_at: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        source_text: Optional[str] = None,
        importance: float = 1.0,
        env: Optional[EnvironmentVector] = None,
    ) -> Memory:
        """写入记忆：自动情感标注 + 原文存储 + 向量索引 + 图索引 + 振荡簇注册。

        双层约定：original_text=text 永久存档；text 初始即原文；
        后续情感污染只改 text 视图，永不触碰 original_text。
        affect 未提供时由自动情感标注器计算 (valence, arousal)。
        due_at 可设定时间敏感到期时间戳（到期提醒注入）。

        v0.5 扩展：
          - source_text：情感来源事件/交互原文（情感守恒：无源注入拒绝）；
          - importance：重要性权重（势差迁移势能加权项）；
          - env：交互环境向量（人格流形环境注入，同记忆不同环境解释不同）。
        """
        if affect is None:
            valence, arousal = self.affect_annotator.annotate(str(text))
            aff = AffectLabel(valence, arousal).clamp()
        else:
            aff = AffectLabel(*affect).clamp()
        original = str(text)

        # ① 情感守恒：无源情感注入一律拒绝（开启守恒时）
        if self.dial.conservation_enabled:
            self.conservation.check_annotation(original, aff, source_text, strict=True)
        # ② 情感总量预算：来源不足时情感强度自然衰减（防止情绪通货膨胀）
        budget_res = self.conservation.adjust_memory_intensity(
            Memory(
                memory_id="__budget_probe__",
                text=original,
                original_text=original,
                affect=aff,
            ),
            self.store.all(),
            affect=aff,
        )
        if budget_res["scale"] < 1.0:
            aff = AffectLabel(aff.valence, aff.arousal * budget_res["scale"]).clamp()

        mem = Memory(
            text=original,
            original_text=original,
            cluster=str(cluster),
            entities=[str(e) for e in (entities or [])],
            affect=aff,
            base_strength=float(base_strength),
            silence=bool(silence),
            never_silent=bool(never_silent),
            due_at=float(due_at) if due_at is not None else None,
            tags=list(tags or []),
            source_text=str(source_text or ""),
            importance=float(importance),
        )
        self.store.add(mem)
        # 向量/图索引用**原文**构建：情感污染不改变检索召回，只改变输出视图
        self.vec.insert(mem.memory_id, mem.original_text, mem.entities)
        self.graph.add_memory(mem)
        self.bank.register(mem.cluster)
        # 电容初始化：新记忆默认半充（容量与放电阈值归一）
        self.capacitor.init_memory(mem)
        # 人格流形：每次交互环境注入养成方程（同记忆不同环境解释不同）
        if self.dial.manifold_enabled and env is not None:
            self.manifold.step(env, affect=aff)
        return mem

    # ------------------------------------------------------------ 读
    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        direct_query: bool = False,
        query_entities: Optional[Sequence[str]] = None,
        now: Optional[float] = None,
        view: str = "polluted",
    ) -> List[RankedHit]:
        """检索：相关簇定位 → ANN 召回 → 情感增益/dB 排序 → 沉默过滤 → top-k。

        view="polluted"（默认）输出情感污染视图；view="original" 回溯原文。
        电容模型开启时：被想起 = 放电（charge 减少并落盘）。
        """
        hits = self.recall_engine.recall(
            query=query,
            query_entities=query_entities,
            top_k=top_k,
            now=now,
            direct_query=direct_query,
            view=view,
        )
        # ③ 熵增遗忘：熵越高检索置信度越低（仅降权，不删除；高危/时间敏感仍不沉默）
        if self.dial.entropy_enabled:
            for h in hits:
                h.score *= self.entropy.confidence(h.memory)
            hits.sort(key=lambda h: h.score, reverse=True)
        if self.dial.capacitor_enabled:
            for h in hits:
                self.capacitor.recall_discharge(h.memory)
                self.store.update_meta(h.memory.memory_id, charge=h.memory.charge)
        return hits

    # ------------------------------------------------------------ 情感污染（v0.2）
    def apply_pollution(
        self,
        memory_id: str,
        mode: str = "beautify",
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """基于原文生成情感污染版本（美化/丑化）并写入当前视图。

        原文 original_text 永不删除、永不修改；污染文本与审计条目写入
        text / pollution_history。
        """
        mem = self.store.get(memory_id)
        if mem is None:
            raise KeyError(f"记忆不存在: {memory_id}")
        # ① 情感守恒：污染 = 情感重分配，禁止无中生有（强度超过原文情感能量则钳制）
        clamped_intensity = intensity
        if self.dial.conservation_enabled:
            res = self.conservation.validate_pollution(mem, intensity)
            clamped_intensity = res["clamped"]
        polluted_text, audit = self.pollution_filter.pollute(
            mem, mode=mode, intensity=clamped_intensity
        )
        mem.pollution_mode = mode
        mem.pollution_history.append(audit)
        self.store.set_pollution_view(
            memory_id,
            text=polluted_text,
            polluted=True,
            pollution_mode=mode,
            pollution_history=list(mem.pollution_history),
        )
        return {
            "memory_id": memory_id,
            "mode": mode,
            "intensity": clamped_intensity,
            "requested_intensity": float(intensity),
            "clamped": float(clamped_intensity) != float(intensity),
            "polluted_text": polluted_text,
            "original_text": mem.original_text,
            "audit": audit,
        }

    def clear_pollution(self, memory_id: str) -> Dict[str, Any]:
        """净化回原文：当前视图恢复 original_text，历史审计保留。"""
        mem = self.store.get(memory_id)
        if mem is None:
            raise KeyError(f"记忆不存在: {memory_id}")
        was_polluted = mem.polluted
        self.pollution_filter.clean(mem)
        self.store.set_pollution_view(
            memory_id,
            text=mem.text,
            polluted=mem.polluted,
            pollution_mode=mem.pollution_mode,
            pollution_history=list(mem.pollution_history),
        )
        return {
            "memory_id": memory_id,
            "was_polluted": was_polluted,
            "restored_text": mem.original_text,
            "history_count": len(mem.pollution_history),
        }

    def view_memory(self, memory_id: str) -> Dict[str, Any]:
        """查看记忆的双层视图与污染审计回溯。"""
        mem = self.store.get(memory_id)
        if mem is None:
            raise KeyError(f"记忆不存在: {memory_id}")
        return {
            "memory_id": memory_id,
            "original_text": mem.original_text,
            "polluted": mem.polluted,
            "pollution_mode": mem.pollution_mode,
            "text": mem.text,
            "pollution_history": list(mem.pollution_history),
            "affect": (mem.affect.valence, mem.affect.arousal),
            "due_at": mem.due_at,
            "reminded": mem.reminded,
        }

    # ------------------------------------------------------------ 到期提醒（v0.2）
    def due_reminders(self, now: Optional[float] = None) -> List[Memory]:
        """返回已到期且尚未提醒的时间敏感记忆（高危/时间敏感永不沉默）。"""
        now = now if now is not None else time.time()
        due = []
        for m in self.store.all():
            if m.due_at is not None and m.due_at <= now and not m.reminded:
                due.append(m)
        due.sort(key=lambda m: m.due_at)
        return due

    def ack_reminder(self, memory_id: str) -> Optional[Memory]:
        """确认到期提醒：标记 reminded=True，防止重复注入。"""
        mem = self.store.get(memory_id)
        if mem is None:
            return None
        self.store.update_meta(memory_id, reminded=True)
        mem.reminded = True
        return mem

    def context_with_reminders(
        self,
        ranked: List[RankedHit],
        now: Optional[float] = None,
    ) -> str:
        """组装最终上下文：命中记忆（污染视图）+ 到期提醒注入。"""
        parts = [ContextInjector.format_for_prompt(ranked)]
        due = self.due_reminders(now=now)
        if due:
            parts.append(ContextInjector.attach_due_reminders(due))
        return "\n".join(p for p in parts if p)

    # ------------------------------------------------------------ 遗忘
    def decay_all(self, now: Optional[float] = None) -> int:
        """遗忘曲线批量推进：所有记忆按已流逝时间衰减。

        电容模型开启时：时间流逝 = 漏电（charge 指数衰减并落盘），
        与遗忘曲线协同（双通道遗忘，只降可及性/电位，不删内容）。
        """
        memories = self.store.all()
        self.curve.advance_many(memories, now=now)
        for m in memories:
            meta = {"accessibility": m.accessibility}
            if self.dial.capacitor_enabled:
                dt = now if now is not None else time.time()
                last = m.last_access_at or m.created_at
                self.capacitor.leak(m, max(0.0, (dt - last) / 3600.0))
                meta["charge"] = m.charge
            # ③ 熵增遗忘：细节先模糊、情绪残留后淡（与遗忘曲线协同，双通道）
            if self.dial.entropy_enabled:
                dt_now = now if now is not None else time.time()
                last = m.last_access_at or m.created_at
                self.entropy.advance(m, max(0.0, (dt_now - last) / 3600.0))
                meta["detail_fidelity"] = m.detail_fidelity
                meta["emotional_residue"] = m.emotional_residue
                meta["entropy"] = m.entropy
            self.store.update_meta(m.memory_id, **meta)
        return len(memories)

    def touch(self, memory_id: str) -> Optional[Memory]:
        """触达强化：高频使用的记忆被强化（仅影响可及性，不改原文/视图）。

        电容模型开启时：触达 = 充电（charge 增加并落盘）。
        """
        mem = self.store.get(memory_id)
        if mem is None:
            return None
        self.curve.touch(mem)
        meta = {
            "accessibility": mem.accessibility,
            "access_count": mem.access_count,
            "last_access_at": mem.last_access_at,
        }
        if self.dial.capacitor_enabled:
            self.capacitor.touch(mem)
            meta["charge"] = mem.charge
        self.store.update_meta(memory_id, **meta)
        return mem

    # ------------------------------------------------------------ 发散/并发
    def recall_with_divergence(
        self,
        query: str,
        query_entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """发散模式检索：过载触发 → 扩大候选池 → 常规召回。

        默认关闭：divergent_enabled=False 时行为与 recall() 等价。
        """
        dial = self.dial
        memories = self.store.all()
        engage = False
        if self.divergence.enabled:
            max_gain = max((self.bank.gain_of(m.affect, dial) for m in memories), default=0.0)
            engage = self.divergence.should_engage(max_gain, dial.amplitude)
        expand = None
        if engage:
            expand = self.divergence.expand_candidates(
                {m.memory_id for m in memories},
                memories,
                self.graph,
            )
        hits = self.recall_engine.recall(
            query=query,
            query_entities=query_entities,
            top_k=top_k,
            expand_candidates=expand,
        )
        return {"engaged": engage, "hits": hits}

    def recall_concurrent(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RankedHit]:
        """并发联想：按簇分片并行召回 → 汇聚排序（默认关闭）。

        关闭时退化为单路召回。
        """
        if not self.concurrent.enabled:
            return self.recall(query, top_k=top_k)
        memories = self.store.all()
        shards = list(self.concurrent.shard_by_cluster(memories).values())
        query_vec = self.vec.vectorizer.vectorize(query)
        id2mem = {m.memory_id: m for m in memories}

        def shard_recall(ids: Set[str]) -> List[RankedHit]:
            ann = self.vec.search(query_vec, ids=ids, top_k=max((top_k or self.dial.top_k) * 2, 8))
            raw = []
            for mid, cossim in ann:
                m = id2mem.get(mid)
                if m is None:
                    continue
                g = self.bank.gain_of(m.affect, self.dial)
                hit_v = self.bank.hit(m.cluster, time.time() - m.created_at, g, self.dial)
                raw.append((m, cossim * (0.4 + 0.6 * hit_v)))
            return rank_hits(raw, self.dial)

        merged = self.concurrent.map_reduce(shards, shard_recall)
        merged = self.silence.filter(merged, direct_query=False, strong_related=False)
        return merged[: (top_k or self.dial.top_k)]

    # ------------------------------------------------------------ 迁移
    def export(self, path: str) -> Dict[str, Any]:
        """电流桥导出（含 SHA-256）。"""
        return bridge.export_bundle(self.store, self.identity, path)

    def import_bundle(self, path: str) -> Dict[str, Any]:
        """电流桥导入：校验通过后原子恢复；失败拒绝恢复。"""
        result = bridge.import_bundle(self.store, path, identity=self.identity)
        self._rebuild_indexes()
        return result

    def export_delta(self, baseline: Dict[str, Memory], path: str) -> Dict[str, Any]:
        return bridge.export_delta(self.store, baseline, path, identity=self.identity)

    # ------------------------------------------------------------ 生命周期
    def farewell(self, export_path: Optional[str] = None) -> Dict[str, Any]:
        return self.lifecycle.farewell(export_path)

    def revive(self, bundle_path: str, new_identity: Optional[IdentityProfile] = None) -> Dict[str, Any]:
        result = self.lifecycle.revive(bundle_path, new_identity)
        self._rebuild_indexes()
        return result

    def upgrade(self, archive_path: str, new_model_version: str) -> Dict[str, Any]:
        result = self.lifecycle.upgrade(archive_path, new_model_version)
        self._rebuild_indexes()
        return result

    # ------------------------------------------------------------ v0.3 旋转磁场唤起调度
    def three_phase_stream(
        self,
        now: Optional[float] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """旋转磁场唤起调度：三相顺序启动，输出连续联想流。

        now 以"启动后的模拟小时 × speed"为时间轴（缺省取启动后实际时长）。
        返回场状态（主导通道/三相顺序/相位差）+ 联想流（激活强度降序）。
        """
        if now is None:
            t = (time.time() - self._boot_at) / 3600.0 * self.dial.three_phase_speed
        else:
            t = float(now)
        stream = self.three_phase.stream(self.store.all(), t, self.dial, k=k)
        return {
            "field": self.three_phase.field_state(t),
            "stream": stream,
        }

    # ------------------------------------------------------------ v0.3 主动回忆 / 备选池
    def proactive_tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """主动回忆：定时 / 触发器驱动自主搭话（到期提醒优先，其次备选池）。"""
        candidates = self.capacitor.standby_pool(
            self.store.all(), top_k=self.dial.top_k, now=now
        )
        due = self.due_reminders(now=now)
        return self.proactive.proactive_tick(candidates, due_memories=due, now=now)

    def standby_pool(
        self, k: Optional[int] = None, now: Optional[float] = None
    ) -> List[Memory]:
        """常驻备选记忆点池：每次预留 top-k 备选记忆点（高电位未唤起候选）。"""
        return self.capacitor.standby_pool(
            self.store.all(), top_k=k or self.dial.standby_pool_size, now=now
        )

    # ------------------------------------------------------------ v0.3 身份自学习
    def learn_self(self, text: str) -> Dict[str, Any]:
        """自学习：从交互文本提取画像线索，渐进演化，生成待确认问题。"""
        result = self.self_model.learn(text)
        self.unified_self.sync_concept(self.self_model.image.dims)
        return {
            "learned": result,
            "pending_questions": [
                q["question"] for q in self.self_model.pending_questions()
            ],
        }

    def pending_self_questions(self) -> List[Dict[str, Any]]:
        """主动提问确认：返回尚未确认的画像假设。"""
        return self.self_model.pending_questions()

    def confirm_self(self, question_id: int, answer: bool) -> Dict[str, Any]:
        """确认 / 否认画像假设，积累自我认知。"""
        res = self.self_model.confirm(question_id, bool(answer))
        self.unified_self.sync_concept(self.self_model.image.dims)
        return res

    # ------------------------------------------------------------ v0.3 自我统一模型
    def derive_avatar(self, name: str) -> Dict[str, Any]:
        """派生万我分身：共享核心自我概念，独立状态，随核心同步。"""
        return self.unified_self.derive_avatar(name).to_dict()

    def sync_avatars(self) -> int:
        """万我如一：核心更新后向全部分身同步自我概念。"""
        return self.unified_self.sync_avatars()

    def merge_avatar(self, name: str) -> Dict[str, Any]:
        """万我合一：分身归并回核心（均值融合 + 全量同步）。"""
        return self.unified_self.merge_avatar(name)

    def assimilate(self, text: str, **remember_kw) -> Dict[str, Any]:
        """万物皆我吸收：新输入先经自我框架解释，再吸收入库并学习画像。"""
        exp = self.unified_self.assimilate(text)
        mem = None
        if exp["relevance"] > 0.01:
            mem = self.remember(text, **remember_kw)
            self.learn_self(text)
        return {
            "memory": mem,
            "interpretation": exp["interpretation"],
            "relevance": exp["relevance"],
            "cos_phi": exp["cos_phi"],
        }

    def self_cognition_report(self) -> Dict[str, Any]:
        """自我认知效率指标：有功 / 视在 = cosφ，量化自我概念清晰度。

        综合统一自我（核心/分身/吸收/清晰度）+ 自学习画像（五维/兴趣/使命/确认度）。
        """
        report = self.unified_self.cognition_report()
        img = self.self_model.image
        report["self_learning"] = {
            "observation_count": img.observation_count,
            "dims": dict(img.dims),
            "interests": list(img.interests),
            "aversions": list(img.aversions),
            "skills": list(img.skills),
            "missions": list(img.missions),
            "confirmed": dict(img.confirmed),
            "pending_questions": len(self.self_model.pending_questions()),
        }
        return report

    # ------------------------------------------------------------ v0.5 情感守恒
    def check_emotion_source(
        self,
        text: str,
        affect: Tuple[float, float],
        source_text: Optional[str],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """守恒校验入口：无源情感注入一律拒绝（strict=True 抛 ConservationError）。"""
        aff = AffectLabel(*affect).clamp()
        return self.conservation.check_annotation(text, aff, source_text, strict=strict)

    def conservation_report(self) -> Dict[str, Any]:
        """① 情感守恒审计：总能量 / 拒绝数 / 衰减数 / 钳制数。"""
        return {
            "enabled": self.dial.conservation_enabled,
            "budget": self.dial.emotion_budget,
            "total_energy": round(
                self.conservation.total_energy(self.store.all()), 4
            ),
            **self.conservation.audit_report(),
        }

    # ------------------------------------------------------------ v0.5 势差迁移
    def promote_memories(
        self, memory_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """② 势差驱动沉淀：高势能 → longterm，低势能 → workspace 流散。"""
        memories = self.store.all()
        if memory_ids is not None:
            id_set = set(memory_ids)
            memories = [m for m in memories if m.memory_id in id_set]
        result = self.potential.promote_many(memories)
        for m in memories:
            self.store.update_meta(m.memory_id, region=m.region)
        return {
            "enabled": self.dial.potential_enabled,
            "threshold": self.dial.potential_threshold,
            **result,
        }

    def transfer_quota(
        self,
        memory_ids: Optional[Sequence[str]] = None,
        max_count: Optional[int] = None,
        skip_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """② 电流桥按势差分配迁移量：非均匀搬运（高势能优先，低势能跳过/减量）。"""
        memories = self.store.all()
        if memory_ids is not None:
            id_set = set(memory_ids)
            memories = [m for m in memories if m.memory_id in id_set]
        quota = self.potential.allocate_transfer_quota(
            memories, max_count=max_count, skip_ratio=skip_ratio
        )
        return {
            "enabled": self.dial.potential_enabled,
            "total": quota["total"],
            "selected_count": quota["selected_count"],
            "skipped_count": quota["skipped_count"],
            "selected_ids": [m.memory_id for m in quota["selected"]],
            "skipped_ids": [m.memory_id for m in quota["skipped"]],
        }

    def potential_report(self) -> Dict[str, Any]:
        """② 势差决策审计报告。"""
        return self.potential.report()

    # ------------------------------------------------------------ v0.5 熵增遗忘
    def entropic_decay(self, dt_hours: float) -> Dict[str, Any]:
        """③ 熵增推进（独立入口）：按 dt 推进全库细节/情绪双通道衰变。"""
        return self.entropy.decay_all(self.store.all(), dt_hours=dt_hours)

    def entropy_report(self) -> Dict[str, Any]:
        """③ 熵增审计：平均熵 / 高熵记忆数 / 细节低于情绪比例。"""
        mems = self.store.all()
        high = [m for m in mems if m.entropy > 0.5]
        detail_lower = [m for m in mems if m.detail_fidelity < m.emotional_residue]
        return {
            "enabled": self.dial.entropy_enabled,
            "rate": self.dial.entropy_rate,
            "count": len(mems),
            "high_entropy_count": len(high),
            "detail_lower_than_residue": len(detail_lower),
            "average_entropy": round(
                (sum(m.entropy for m in mems) / len(mems)) if mems else 0.0, 4
            ),
            "log_count": len(self.entropy.log),
        }

    # ------------------------------------------------------------ v0.5 湍流联想
    def turbulent_associate(
        self,
        query_entities: Optional[Sequence[str]] = None,
        center_ids: Optional[Sequence[str]] = None,
        top_k: int = 8,
        divergent: bool = False,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """④ 涡旋联想：围绕话题中心逐层扩散（中心→次级→外围，能量递减）。"""
        if center_ids is None:
            if query_entities:
                center_ids = sorted(self.graph.lookup(query_entities))[:5]
            else:
                center_ids = []
        id2mem = {m.memory_id: m for m in self.store.all()}
        result = self.turbulence.associate(
            self.graph,
            id2mem,
            center_ids=center_ids,
            top_k=top_k,
            divergent=divergent,
            max_depth=max_depth,
        )
        return {
            "enabled": self.dial.turbulent_enabled,
            "divergent": bool(divergent),
            "center_count": len(center_ids),
            "rings": result["rings"],
            "hits": [
                {
                    "memory_id": m.memory_id,
                    "text": (m.text or m.original_text)[:40],
                    "energy": round(score, 4),
                    "ring": layer,
                }
                for m, score, layer in result["hits"]
            ],
        }

    def turbulent_report(self) -> str:
        """④ 涡流链人类可读摘要（demo 展示用）。"""
        return self.turbulence.describe_rings()

    # ------------------------------------------------------------ v0.5 人格流形
    def manifold_learn(
        self,
        env: EnvironmentVector,
        affect: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """⑤ 人格流形单步养成：环境向量进入非线性迭代方程。"""
        aff = AffectLabel(*affect).clamp() if affect is not None else None
        return self.manifold.step(env, affect=aff)

    def manifold_learn_sequence(
        self,
        env_sequence: Sequence[EnvironmentVector],
        affects: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> Dict[str, float]:
        """⑤ 人格流形批量养成：不同环境序列 → 不同性格轨迹。"""
        norm_affects = None
        if affects is not None:
            norm_affects = [
                AffectLabel(*a).clamp() if a is not None else None for a in affects
            ]
        return self.manifold.learn(env_sequence, affects=norm_affects)

    def personality_snapshot(self) -> Dict[str, Any]:
        """⑤ 性格快照：当前坐标 + 轨迹摘要（供唤醒重建时使用）。"""
        snap = self.manifold.snapshot()
        snap["enabled"] = self.dial.manifold_enabled
        return snap

    def personality_export(self, path: str) -> str:
        """⑤ 导出完整轨迹 JSON（唤醒重建数据源）。"""
        return self.manifold.export_json(path)

    def personality_rebuild(self, path: str) -> int:
        """⑤ 唤醒重建：从轨迹 JSON 载入性格坐标与轨迹。"""
        return self.manifold.rebuild_from(path)

    def personality_bifurcation(
        self,
        env_sequence: Sequence[EnvironmentVector],
        affects: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        perturbation: float = 1e-3,
    ) -> Dict[str, Any]:
        """⑤ 非线性分叉动力学：微小初始差异沿同一环境序列被放大成不同轨迹。"""
        norm_affects = None
        if affects is not None:
            norm_affects = [
                AffectLabel(*a).clamp() if a is not None else None for a in affects
            ]
        return self.manifold.bifurcation(
            env_sequence, affects=norm_affects, perturbation=perturbation
        )

    # ------------------------------------------------------------ v0.5 串闭环
    def physical_chain_report(self) -> Dict[str, Any]:
        """串闭环报告：守恒（情感有来源）→ 势差（沉淀有规则）→
        熵增（遗忘有方向）→ 湍流（联想有形态），人格流形受四定律约束。"""
        mems = self.store.all()
        return {
            "conservation": self.conservation_report(),
            "potential": self.potential_report(),
            "entropy": self.entropy_report(),
            "turbulence": {
                "enabled": self.dial.turbulent_enabled,
                "last_rings": self.turbulence.last_rings,
            },
            "personality": self.personality_snapshot(),
            "chain": [
                "① 守恒：情感有来源（无源注入拒绝，总量预算防通胀）",
                "② 势差：沉淀有规则（高势能长期区，低势能流散）",
                "③ 熵增：遗忘有方向（细节先模糊，情绪残留后淡，原文永存）",
                "④ 湍流：联想有形态（涡旋扩散，能量随层距递减）",
                "⑤ 流形：性格是全谱系空间中的点（环境注入，非线性分叉）",
            ],
        }

    # ------------------------------------------------------------ 内部
    def _make_vectorizer(self) -> Any:
        """按 Dial 配置构造向量化器；embedding 后端不可用时回退零依赖词袋。"""
        if not self.dial.embedding_enabled:
            return BagOfWordsVectorizer()
        try:
            from .embeddings import create_embedding_vectorizer
            return create_embedding_vectorizer(
                profile=dict(self.dial.embedding_profile),
                backend=self.dial.embedding_backend,
            )
        except Exception as exc:
            print(
                f"[warn] embedding 后端不可用（{type(exc).__name__}: {exc}），"
                f"回退零依赖词袋哈希向量化。",
                file=sys.stderr,
            )
            return BagOfWordsVectorizer()

    def _rebuild_indexes(self) -> None:
        """重启后从原文存储重建向量索引与图索引（内存只驻留索引）。

        索引用 original_text 构建，保证情感污染不改变召回结果。
        """
        self.vec = AnnVectorIndex(vectorizer=self._vectorizer)
        self.graph = EntityGraph()
        for m in self.store.all():
            self.vec.insert(m.memory_id, m.original_text or m.text, m.entities)
            self.graph.add_memory(m)
            self.bank.register(m.cluster)
        # 检索层必须指向新索引，避免持有旧空对象
        self.recall_engine.vec = self.vec
        self.recall_engine.graph = self.graph

    def stats(self) -> Dict[str, Any]:
        mems = self.store.all()
        return {
            "memory_count": self.store.count(),
            "cluster_count": len({m.cluster for m in mems}),
            "entity_count": len(self.graph.entities()),
            "polluted_count": sum(1 for m in mems if m.polluted),
            "due_count": sum(1 for m in mems if m.due_at is not None and not m.reminded),
            "dial": self.dial.to_dict(),
            "identity": self.identity.to_dict(),
            "state": self.lifecycle.state.value,
            # v0.3 新增
            "capacitor_enabled": self.dial.capacitor_enabled,
            "standby_pool_size": self.dial.standby_pool_size,
            "three_phase_enabled": self.dial.three_phase_enabled,
            "proactive_enabled": self.dial.proactive_enabled,
            "avatar_count": self.unified_self.avatar_count(),
            "self_learning_observations": self.self_model.image.observation_count,
            # v0.5 新增
            "conservation_enabled": self.dial.conservation_enabled,
            "potential_enabled": self.dial.potential_enabled,
            "entropy_enabled": self.dial.entropy_enabled,
            "turbulent_enabled": self.dial.turbulent_enabled,
            "manifold_enabled": self.dial.manifold_enabled,
            "longterm_count": sum(1 for m in mems if m.region == "longterm"),
            "average_entropy": round(
                (sum(m.entropy for m in mems) / len(mems)) if mems else 0.0, 4
            ),
            "manifold_step": self.manifold.step_no,
        }
