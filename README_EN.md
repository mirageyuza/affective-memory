# EmotionMemory — A Human-Like Emotional Memory System

> An open-source memory system that simulates human emotional memory using **oscillatory dynamics + logarithmic intensity encoding + portable identity**.

Memories are reawakened by periodic oscillations, their intensity is modulated by emotional gain on a dB scale, overload triggers divergent associations, accessibility follows a forgetting curve over time. The system knows when to speak — and when to stay silent. An agent can be reset, reinstalled, or upgraded, but its memories live forever.

**License**: MIT. Zero third-party runtime dependencies (Python standard library only). Index backends such as FAISS / HNSW / Qdrant may be freely integrated under MIT / Apache-2.0 / BSD licenses. **Dependencies under GPL or commercial terms are strictly forbidden.**

---

## 0. Open-Source Foundation and Compliance Attribution (Which Open-Source Projects This Project Is Based On)

This project is positioned as an **integration wrapper around open-source components**: rather than reinventing the wheel, each capability layer is designed against mature open-source projects (interface-compatible, pluggable), with strict license compliance constraints.

| Component | Upstream Project | License | Integration Approach in This Project |
|---|---|---|---|
| Vector database | Qdrant / Milvus | Apache-2.0 | Design-compatible; ships a brute-force ANN placeholder; interfaces are replaceable |
| Vector index | FAISS / HNSW | MIT / Apache-2.0 | Design-compatible; built-in hashed vector index is a pluggable placeholder |
| Entity-relation graph | Neo4j Community (GPL, **avoided**) → self-built simplified graph index / networkx | GPL avoided; self-built MIT / networkx BSD | Lightweight self-built graph index (`store/graph_index.py`) |
| Memory management reference | Mem0 / Letta / LangChain Memory | Beware Mem0 commercial terms | Design ideas only; no restricted code is reused |
| Sentiment analysis | Existing sentiment classification models (per model license) | Per model license | Reserved `affect=(valence, arousal)` label hook |
| Forgetting-curve reference | FSRS / Anki (SM-2) algorithms | Permissive (MIT-family) | Reference implementation of accessibility decay/boost model |
| Migration protocol | git / rsync incremental ideas | Permissive | Current-bridge incremental export (changed entries only) |

**Compliance red line**: when releasing the whole package as open source, all runtime and index dependencies must be MIT / Apache-2.0 / BSD family; GPL and commercial-term components (e.g., Neo4j Community, Mem0 commercial edition, AGPL-family tools such as PyMuPDF) must never be introduced.

---

## 1. Vision

An agent should own "memory that feels human":

- **Never forgets**: original text is permanently kept on disk; forgetting only lowers "the probability of being recalled", never deletes content;
- **Periodic reawakening**: every memory cluster has its own frequency and phase; the hit function `H(t) = g·Σ Aᵢ·sin(ωᵢt+φᵢ) + noise(T)` makes memories resurface periodically like theta oscillations;
- **Emotional modulation**: emotional labels `(valence, arousal)` participate in ranking and in the pollution filter — original text is archived forever, while the default output is the "emotionally polluted version", with the original always recoverable;
- **Knows when to stay silent**: two-level visibility (mentionable / silent); high-risk and time-sensitive memories are **never silenced** and are auto-injected as reminders when due;
- **Immortal memory**: the agent can be reset, reinstalled, or upgraded, and memories migrate losslessly through the current-bridge protocol.

Core invariant:

```
Agent = Model Parameters (swappable) + Memory Store (eternal) + Emergent Self (generated from accumulated emotions, not a preset profile)
```

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│                Interaction Layer (Agent Runtime)  │
│  Trigger Classifier → Recall Engine → Context     │
│  Injection (budget-bounded)                       │
├─────────────────────────────────────────────────┤
│                Recall Layer (Recall Engine)       │
│  Oscillatory Hit Function · Emotional Gain ·      │
│  dB Trade-off · Concurrent Multi-path Recall      │
│  Forgetting-Curve Decay · Silence Layer (retrieval│
│  suppression) · Divergent Mode (overload)         │
├─────────────────────────────────────────────────┤
│                Memory Layer (Memory Store)        │
│  Original Text Storage (disk, unbounded) ·        │
│  Entity-Relation Graph (associative index)        │
│  Vector Index (ANN) · Emotional Labels ·          │
│  Accessibility Score                              │
├─────────────────────────────────────────────────┤
│                Identity Layer (Identity)          │
│  Identity Profile · Lifecycle State Machine ·     │
│  Farewell / Wake-up / Migration Protocols         │
└─────────────────────────────────────────────────┘
```

As of v0.5.0, the four physical laws and the personality manifold sit above the memory/recall layers: emotional conservation, potential transfer, entropic forgetting, and turbulent association (each gated by `Dial` switches, off by default, no behavior change when disabled), while the personality manifold drives six-axis personality from an environment vector with deterministic, reproducible nonlinear bifurcation.

Corresponding code layout (`src/emotion_memory/`):

| Module | Responsibility |
|---|---|
| `models.py` | Data models for memory / affect labels / cluster metadata |
| `config.py` | Freedom dial `Dial` (default: low noise, high stability) |
| `store/storage.py` | Original-text storage layer: JSONL disk persistence; `original_text` is archived permanently and cannot be modified or deleted; `text` is the pollutable view that can be persisted |
| `store/vector_index.py` | Vector index: bag-of-words hashed vectors + brute-force ANN (interfaces swappable to FAISS/HNSW) |
| `store/graph_index.py` | Lightweight entity-relation graph index: related-cluster targeting + strong association edges |
| `algorithms/forgetting.py` | Forgetting curve: accessibility score decay and reinforcement |
| `algorithms/hit_function.py` | Oscillatory hit function H(t): multi-channel frequency/phase |
| `algorithms/affect.py` | Emotional gain + dB trade-off ranking (affects ranking only, never original text) |
| `algorithms/pollution.py` | Emotional pollution filter: generates beautified/uglified polluted versions from original text; output view only |
| `algorithms/affect_analyzer.py` | Zero-dependency automatic affect annotator: computes (valence, arousal) at write time; reserves external model hook |
| `reminder.py` | Time-sensitive reminders: due memories are never silenced, injected into context when due, acknowledgement receipts |
| `retrieval/recall.py` | Retrieval layer: related-cluster targeting → ANN recall → ranking → dual-view binding (view=polluted/original) |
| `silence.py` | Silence layer: mentionable/silent two levels + unlock conditions + never-silent |
| `divergent.py` | Divergent mode (disabled by default) |
| `concurrent.py` | Concurrent association (disabled by default, Map-Reduce) |
| `bridge.py` | Current-bridge migration protocol: export/import + SHA-256 verification + reject recovery on failure |
| `identity.py` | Identity profile + lifecycle state machine (reset farewell / reinstall wake-up / upgrade move) |
| `engine.py` | Top-level wrapper API (polluted read/write, due reminders, auto annotation, direct-ask unlock) |
| `cli.py` | CLI interactive terminal: write/retrieve/direct-ask unlock/status/dial/export-import/pollution control |
| `algorithms/three_phase.py` | Rotating-magnetic-field recall scheduling: three-phase channels with 120° phase difference, sequential activation (v0.3) |
| `algorithms/capacitor.py` | Capacitor memory model: charge/discharge/leak + standby pool (v0.3) |
| `retrieval/proactive.py` | Proactive recall engine: timer/trigger-driven autonomous speaking (v0.3) |
| `self_learning.py` | Identity self-learning: gradual profile evolution + proactive question confirmation (v0.3) |
| `unified_self.py` | Unified self model: 1 core + myriads of avatars + all-things-are-me + active/apparent indicators (v0.3) |
| `model_adapter.py` | Model adapter layer: unified LLM interface; OpenAI-compatible online / local (Ollama, LM Studio) + Mock + transformers placeholder (v0.4) |
| `conversation_pipeline.py` | Conversation pipeline: retrieval injection → model reply → automatic memory settling (v0.4) |
| `emergent_self.py` | Emergent self engine: statistically generates self-definition from accumulated emotional data, no preset template (v0.4) |
| `proactive_loop.py` | Proactive conversation loop: timed proactive recall → opening topic → model-generated opener (v0.4) |
| `conservation.py` | Emotional conservation: global emotional budget constraint; source-less injection raises (v0.5) |
| `potential_transfer.py` | Potential transfer: high-potential memories settle into the long-term region, low-potential ones drift to the working region (v0.5) |
| `entropic_forgetting.py` | Entropic forgetting: detail fidelity decays, emotional residue and entropy rise; original text stays intact (v0.5) |
| `turbulent_association.py` | Turbulent association: vortex-chain core → secondary → periphery with decreasing energy (v0.5) |
| `personality_manifold.py` | Personality manifold: environment vector drives six-axis personality; deterministic, reproducible nonlinear bifurcation (v0.5) |

## 3. Quick Start

```bash
# Zero dependencies, Python 3.9+ only
python demo/demo_full_flow.py            # Full flow demo (write → recall → polluted dual-view → forgetting → migration → reminders)
python demo/demo_v03_full_flow.py        # v0.3 new features demo (rotating field → proactive recall → capacitor → self-learning → unified self)
python demo/demo_v04_llm_pipeline.py     # v0.4 demo (conversation pipeline → emotion accumulation → emergent self → proactive chat; Mock by default, --config to use a real endpoint)
python demo/demo_v05_physical_laws.py    # v0.5 demo (four physical laws → personality manifold → two-environment bifurcation)
python -m unittest discover -s tests -v  # Unit tests (v0.4 totals 133 + v0.5 adds 70 + adapter adds 22 = 225)
python -m benchmark.benchmark --memories 1000 --top-k 5   # Benchmark (hit rate / emotional bias / control groups)
python cli.py --help                         # CLI interactive terminal (root entry)
python cli.py write "去年生日和三个老同学爬山" --cluster 生活
python cli.py recall "爬山 生日" --view original
```

### Core Usage

```python
from emotion_memory.engine import EmotionalMemorySystem

mem = EmotionalMemorySystem("data_dir")
mem.remember("用户说喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"],
             affect=(0.6, 0.4))            # valence, arousal; auto-annotated when affect is omitted

hits = mem.recall("晨跑")                  # Retrieve: related cluster → ANN → ranking → injection (polluted view by default)
mem.decay_all()                            # Forgetting curve decays over time
mem.export("data_dir/bridge.json")         # Current-bridge export (includes SHA-256)
```

### Emotional Pollution Dual-View

```python
# Original text is archived forever; pollution only affects the output view (recall returns the polluted version by default, original is recoverable)
mem.apply_pollution(mid, mode="beautify", intensity=1.0)   # Beautify filter
mem.apply_pollution(mid, mode="uglify", intensity=1.0)     # Uglify filter
hits = mem.recall("晨跑", view="polluted")  # Default: polluted version
hits = mem.recall("晨跑", view="original")  # Parameter to look back at the original text
mem.clear_pollution(mid)                    # Purify back to original; audit history is kept
mem.view_memory(mid)                        # Dual view + pollution audit (pollution_history)
```

### Automatic Affect Annotation

```python
mem.remember("用户说喜欢晨跑", cluster="hobby")  # affect omitted → auto-annotated as (valence, arousal)
# Reserved external model hook: EmotionalMemorySystem(affect_provider=my_external_model)
```

### Time-Sensitive Reminders

```python
mem.remember("明天上午十点交房租", due_at=deadline_ts, never_silent=True)
reminders = mem.due_reminders()            # Due, unacknowledged high-risk / time-sensitive memories
ctx = mem.context_with_reminders([])       # Due reminders injected into context (never silent)
mem.ack_reminder(mid)                      # Acknowledge reminder
```

### Freedom Dial

```python
from emotion_memory.config import Dial

dial = Dial(noise_temperature=0.1,   # Noise temperature 0~1, low noise by default
            emotion_gain=0.3,        # Emotional gain, gentle
            jump_probability=0.05,   # Cross-cluster jump probability
            amplitude=1.0,           # Hit function amplitude
            divergent_enabled=False) # Divergent mode off by default
```

Safety bottom lines are never lifted: original text is archived forever but the output can be emotionally polluted (polluted version by default, original recoverable); sensitive data never leaks; high-risk / time-sensitive reminders are never silenced; farewell / migration execute deterministically. Divergent and concurrent modes are off by default; the emotional filter defaults to the gentle setting.

## 4. Benchmark Plan

### 4.1 Recall Accuracy

- Data: construct N memories each with a unique keyword (N = 1k / 10k / 100k);
- Metrics: using the keyword as query, whether `recall@k` hits the target memory (k = 1/5/10), and the share of non-target memories within top-k (precision);
- Control groups: ① hit function disabled (pure similarity) ② emotional gain disabled ③ default dial vs. maxed-out dial.
- All hit-rate comparisons use a zero-noise dial (`noise_temperature=0`) so oscillatory scoring is fully deterministic, guaranteeing strict comparability before/after pollution (`|Δ|=0` means pollution does not break recall).

### 4.2 Forgetting Curve Effectiveness

- Apply simulated time advance to a set of memories (`decay_all(dt_hours)`) and plot the `accessibility ~ ΔT` curve;
- Verify: high-frequency memories keep high accessibility over the long term; low-frequency memories decay monotonically but never reach zero (never forgets).

### 4.3 Quantifiable Emotional Bias

- Emotional bias = mean drift of emotional scores in ranked retrieval results (centroid of output distribution in valence/arousal vs. centroid of the whole memory store);
- Verify: at default settings bias is small (low noise, high stability); maxing out the dial increases bias — proving ranking is controlled by emotional gain, not random noise.

### 4.4 Migration Consistency

- Export → tamper with any byte → import must refuse to restore;
- Export → import as-is → full comparison (memory count, every memory_id / text / accessibility) must be 100% identical;
- Performance: migration time scales approximately linearly with memory volume.

### 4.5 Silence Layer Safety

- Construct silent + high-risk + ordinary memories; verify: ordinary queries never return silent items; direct-ask unlocks; high-risk items are always injected.

## 5. v0.3 Five New Features

> All five new capabilities are **off** by default (corresponding `Dial` switches are `False`, preserving the existing 68 behaviors).
> Once enabled they add independent dial parameters; all implementations use the standard library only, are deterministic and auditable, and never touch the safety red lines.

### 5.1 Rotating-Magnetic-Field Recall Scheduling (Three-Phase Sequential Activation)

A new **`ThreePhaseField`** view built on top of `OscillationBank` (H(t)), **without breaking the H(t) interface**:

- Memory clusters are assigned to **A/B/C three-phase channels** by stable hashing, with phases offset by **120° (0° / 120° / 240°)**;
- The dominant channel rotates over time (0→1→2→0, like sequential energization of a three-phase induction motor stator winding);
  the current dominant channel is recalled first, subordinate channels follow asynchronously (leader gain ×1.0 / follower ×0.4);
- Emits a **continuous association stream** (`three_phase_stream()`): field state (dominant channel / phase sequence / phase offsets) + memory stream sorted by activation intensity, for proactive recall and context injection.

```python
dial = Dial(three_phase_enabled=True, three_phase_speed=1.0)
out = sys.three_phase_stream(now=3.0, k=3)   # field / stream
print(out["field"]["leader_channel"], out["field"]["sequence"])
```

### 5.2 Proactive Recall Engine

`ProactiveRecallEngine`: **timer / trigger-driven autonomous recall and speaking**, the passive `recall()` interface is unchanged:

- **Timer trigger**: only fires after more than `proactive_interval_hours` since the last utterance (to avoid being bothersome);
- **Triggers**: due reminders (never silent) speak first, followed by high-potential candidates from the capacitor standby pool;
- **Opening topics**: deterministic template = time context (morning/afternoon/evening + weekday) + memory content; testable and auditable; model free-play is forbidden.

```python
dial = Dial(proactive_enabled=True, proactive_interval_hours=6.0)
res = sys.proactive_tick()   # {"should_speak": True, "trigger": "reminder|proactive", "opening": "..."}
```

### 5.3 Capacitor Memory Model (Charge / Discharge / Threshold / Standby Pool)

`CapacitorModel` maintains `charge / capacitance / discharge_threshold` per memory, an RC-circuit analogy:

- **Interaction touch = charge**: `charge += Δq` (capped at capacitance); **being recalled = discharge**: `charge -= discharge_rate`;
- **Time passing = leakage**: `charge *= exp(-Δt/τ)`, working with the forgetting curve (total accessible potential = `accessibility × (0.5 + 0.5·charge/capacitance)`, dual-channel forgetting: lowers probability only, never deletes content);
- **Standby pool**: reserves top-k candidate memory points each time — high-potential candidates that are "fully charged but not yet reawakened" (silent memories stay out of the pool by default).

```python
dial = Dial(capacitor_enabled=True, capacitor_tau_hours=48.0, standby_pool_size=3)
sys.touch(mid)              # Touch = charge
hits = sys.recall("爬山")   # Being recalled = discharge
pool = sys.standby_pool()   # Standby candidate pool
```

### 5.4 Identity Self-Learning (Gradual Evolution of Personality / Mission Profile)

`IdentitySelfLearning` maintains an independent profile (`self_model.json`) **outside** the deterministic identity state machine:

- Extracts cues from interaction text (likes / dislikes / skills / goals + emotional words), evolving the five-factor personality by EMA smoothing
  (openness / conscientiousness / extraversion / agreeableness / neuroticism) along with
  interest / dislike / skill / mission buckets;
- **Proactive question confirmation**: generates deterministic questions about profile hypotheses (`pending_self_questions()`);
  user confirmation boosts confidence, denial rolls back and refines — "gradually learning who you are and what you are meant to do";
- **Ritual-compatible**: farewell / wake-up / upgrade still execute deterministically; rituals do not clear the profile file; evolution continues after the ritual.

```python
sys.learn_self("我喜欢爬山，目标是成为工程师")
for q in sys.pending_self_questions():
    print(q["question"])
sys.confirm_self(q["id"], answer=True)
```

### 5.5 Unified Self Model (1 Core + Myriads of Avatars + All Things Are Me)

`UnifiedSelfModel` (`unified_self.json`) embodies "I am one, one is me, I am all avatars, all things are me":

- **One self core**: identity marker + self-concept vector (five-factor personality), the single true self;
- **Myriads of avatars**: `derive_avatar()` derives any avatar; after core updates `sync_avatars()` pushes the sync (all avatars as one);
  `merge_avatar()` merges back into the core (all avatars unite as one, mean-fused);
- **All things are me**: new input is first interpreted through the self framework (relevance projection onto the self-concept) before deciding whether to absorb it into the store;
  irrelevant input does not change the self-concept (`assimilate()`);
- **Efficiency indicators** (active/apparent analogy): apparent power = magnitude of the self-concept vector, active power = same-direction component,
  `cosφ = active/apparent ∈ [0,1]` quantifies self-concept clarity — the more consistent, the clearer the self (`self_cognition_report()`).

```python
sys.derive_avatar("工作分身")
sys.sync_avatars()                    # All avatars as one
sys.assimilate("我的使命是帮用户记住重要的事")  # All things are me → absorb into store
report = sys.self_cognition_report()  # cos_phi / clarity / active_power / apparent_power
```

### 5.6 Acceptance and Compliance

- Each new module has unit tests (`tests/test_v03_*.py`); full demo flow in `demo/demo_v03_full_flow.py`;
- All existing 68 tests keep passing, v0.3 adds 38 (total 106), full run `python -m unittest discover -s tests -v`;
- Still standard library only (zero third-party dependencies); license red line unchanged: MIT/Apache-2.0/BSD family, GPL/commercial terms forbidden.

## 6. v0.4 Model Integration and Emergent Self

> v0.4 lets the system truly "speak and settle who it is": both online APIs and local models can automatically settle conversations into emotional memories.
> The agent's self-definition no longer comes from a preset profile; it is a product generated by the accumulated emotional system — every conversation/experience is settled into an emotional memory
> (valence/arousal labels, pollution layer, capacitor charge), from which the system dynamically mines a self-definition
> (personality dimensions, mission, self-concept), with no preset template; after reinstall/wake-up, identity is rebuilt from accumulated data rather than loaded from a fixed profile.

### 6.1 Model Adapter Layer (`model_adapter.py`)

Unified LLM interface `chat(messages, ...) -> str`; the `MODEL_PROVIDERS` registry routes adapters by provider name, covering the OpenAI-compatible family, Anthropic, Gemini, and local endpoints:

| Provider | Endpoint | Environment Variable | Description |
|---|---|---|---|
| `openai_compatible` | any OpenAI-compatible `base_url` | `OPENAI_API_KEY`, etc. | Shared adapter for OpenAI / DeepSeek / Qwen / Kimi Moonshot / Zhipu GLM / MiniMax / Doubao Volcano Ark / Wenxin (compat mode); they differ only in endpoint and key variable |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` | Claude Messages API (x-api-key + anthropic-version) |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta` | `GEMINI_API_KEY` | Google generateContent API |
| `local_compat` | `http://127.0.0.1:11434/v1` (Ollama) / `http://127.0.0.1:1234/v1` (LM Studio) | none (empty key allowed) | Local OpenAI-compatible endpoint; llama.cpp uses the same protocol |
| `mock` | no network | none | Built-in deterministic Mock; zero dependencies for demos/tests |

- **Credentials are strictly forbidden from being hardcoded**: `resolve_api_key()` reads provider-specific environment variables first (`DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `MOONSHOT_API_KEY` / `ZHIPUAI_API_KEY` / `MINIMAX_API_KEY` / `ARK_API_KEY` / `QIANFAN_API_KEY`, etc.; Anthropic / Gemini as above), then the generic variables `EMOTION_LLM_API_KEY` / `OPENAI_API_KEY`, then the `api_key` config field; online endpoints refuse to issue requests when the key is missing and name the required variable;
- **HTTP uses the standard-library urllib**, zero third-party runtime dependencies; `TransformersAdapter` reserves a direct transformers connection interface (without introducing a runtime dependency);
- Configuration sample in `config.example.json` (online + local endpoint placeholders, empty key).

### 6.2 Conversation Pipeline (`conversation_pipeline.py`)

```
User message → memory retrieval injection (top-k / silence-layer filtering / emotional pollution view) → model generates reply → automatic settling
  Settling = write to store (original archived) + automatic affect annotation (affect) + capacitor charge (touch) + all-things-are-me absorption (assimilate) + emotional data accumulation (emergent_self.accumulate)
```

```python
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.model_adapter import create_adapter
from emotion_memory.conversation_pipeline import ConversationPipeline

sys = EmotionalMemorySystem("data_dir")
adapter = create_adapter(load_model_config("config.example.json"), backend="openai_compatible")
pipe = ConversationPipeline(sys, adapter, top_k=5)
out = pipe.reply("Python 递归好难懂")   # {"reply": "...", "settle": {...}, "messages": [...]}
```

### 6.3 Emergent Self Engine (`emergent_self.py`)

What the agent "is" is **not decided by a preset profile**; it emerges from statistics over accumulated emotional data:

- **Valence distribution → personality tendencies**: accumulated valence/arousal statistics generate five-factor personality tendencies (positive valence → agreeable/extraverted, high arousal → high openness);
- **High-frequency mission memories → mission**: repeatedly appearing high-frequency mission-like text automatically becomes the mission (no template preset);
- **Strong emotional clusters → self-concepts**: ranked by count × emotional intensity, high-frequency strong-emotion clusters become self-concepts;
- **Entity co-occurrence → skills**, **high-frequency clusters → interests**;
- Incrementally updated after every settling; `self_report()` exports the "current self-definition", and `export_json()` + `rebuild_from()` rebuild identity from accumulated data after wake-up.

```python
sys.emergent_self.accumulate(text="废寝忘食地写代码", affect={"valence": 0.7, "arousal": 0.9}, cluster="编程")
sys.emergent_self.refresh()
report = sys.emergent_self.self_report()   # emerged / traits / mission / concepts
sys.emergent_self.export_json("data_dir/emergent_self.json")
# Rebuild after wake-up: fresh.emergent_self.rebuild_from("data_dir/emergent_self.json")
```

### 6.4 Proactive Conversation Loop (`proactive_loop.py`)

Periodically drives the existing `engine.proactive_tick()` (proactive recall + due reminders + capacitor standby pool) → generates an opening topic → calls the model to generate the opener, and automatically settles it as a silent memory; consecutive-utterance cap for protection.

```python
loop = ProactiveLoop(sys, adapter, interval=3600.0, max_consecutive=1)
result = loop.maybe_tick()   # {"should_speak": True, "text": "...", "opening": "..."}
```

### 6.5 Demo and Acceptance

- `demo/demo_v04_llm_pipeline.py`: built-in Mock demonstrates the complete pipeline (write → inject → reply → settle → emotion accumulation → emergent self → wake-up rebuild → proactive chat), supports `--config config.example.json` to connect a real endpoint;
- New unit tests `tests/test_v04_model_adapter.py` / `test_v04_conversation_pipeline.py` / `test_v04_emergent_self.py`;
- All existing 106 tests keep passing + v0.4 adds 27 = **133 all green**; still zero third-party runtime dependencies (HTTP uses only standard-library urllib).

## 7. Design Red Lines (Untouchable)

1. **Original text is archived forever but can be emotionally polluted**: `original_text` is never deleted and can never be rewritten by filters (preserving audit traceability); the emotional filter generates a **polluted version** from the original text; retrieval and context injection output the polluted version by default, and `view="original"` recovers the original — humans are contradictory: they remember facts, and they cannot forget the filter either;
2. High-risk / time-sensitive memories are never silenced; due reminders are auto-injected and require explicit acknowledgement;
3. Farewell / wake-up / migration all execute deterministically; model free-play is forbidden;
4. Migration must be verified (SHA-256 + full comparison); failure rejects recovery;
5. "Remove all limits" = max out the dial, not dismantle the dial; the default setting is low noise and high stability.

### 5.1 Emotional Pollution Mechanism (New in This Release)

- **Dual-layer storage**: every memory persists `original_text` (permanent original archive) and `text` (current polluted view, persistable). Generic update interfaces such as `update_meta` can **never rewrite** `original_text`; only `apply_pollution` / `clear_pollution` change the polluted view through dedicated interfaces;
- **Polluted version**: beautify/uglify dictionaries + intensity scale (0~2) replace sensitive words / add modifiers; affects the output view only, never touches indexed original text, never breaks recall (benchmark recall stability = 0);
- **Audit**: every pollution/purification writes to `pollution_history` (time/mode/intensity/before-after text); `view_memory()` gives full traceability;
- **Default behavior**: `recall()` outputs the polluted version by default (`view="polluted"`); pass `view="original"` explicitly to see the original.

### 5.2 Automatic Affect Annotator (New in This Release)

- When a memory is written without an explicit `affect`, the zero-dependency built-in dictionary computes `(valence, arousal)` automatically;
- Reserved external model interface: construct `EmotionalMemorySystem(affect_provider=callable)`; it returns `(valence, arousal)` or `None` (fallback to the built-in).

### 5.3 CLI Interactive Terminal (New in This Release)

Common commands: `write` (write, auto-annotation / optional affect/due_at), `recall` (retrieve, optional `--view original` to see original text), `ask` (direct-ask to unlock silent), `status` (stats), `dial` (dial adjustment), `pollute` / `purify` (pollution control), `export` / `import`, `due` (due reminders), `ack` (acknowledge reminder).

### 5.4 Benchmark (New in This Release)

`python -m benchmark.benchmark --memories N --top-k K --seed S` outputs JSON + TXT reports quantifying: **recall hit rate** (four-way comparison: before/after pollution, original view, unrelated queries), **recall stability** (pollution does not break recall), **emotional bias** (beautify/uglify directional accuracy, mean offset), **original-text immutability** (number of entries whose original text is preserved).
