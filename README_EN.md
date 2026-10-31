# EmotionMemory — A Human-Like Emotional Memory System

> An open-source memory system that simulates human emotional memory using **oscillatory dynamics + logarithmic intensity encoding + portable identity** (current version **v0.6.0**).

Memories are reawakened by periodic oscillations, their intensity is modulated by emotional gain on a dB scale, overload triggers divergent associations, accessibility follows a forgetting curve over time. The system knows when to speak — and when to stay silent. An agent can be reset, reinstalled, or upgraded, but its memories live forever.

**License**: MIT. Zero third-party runtime dependencies (Python standard library only). Index backends such as FAISS / HNSW / Qdrant may be freely integrated under MIT / Apache-2.0 / BSD licenses. **Dependencies under GPL or commercial terms are strictly forbidden.**

## 0. Open-Source Foundation and Compliance Attribution

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
| Embedding vectorization | OpenAI-compatible embed API / sentence-transformers | Per API license / MIT (local) | Optional `EmbeddingVectorizer` (v0.6), off by default, no runtime dependency |

**Compliance red line**: when releasing the whole package as open source, all runtime and index dependencies must be MIT / Apache-2.0 / BSD family; GPL and commercial-term components (e.g., Neo4j Community, Mem0 commercial edition, AGPL-family tools such as PyMuPDF) must never be introduced. The v0.6 embedding enhancement follows the same red line: online endpoints use the OpenAI-compatible embed API over the standard-library `urllib` (no third-party dependency); local sentence-transformers is available only when explicitly enabled and self-installed by the user (MIT license), and does not enter the runtime dependency list.

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

As of v0.6.0, the retrieval layer adds an **optional embedding vectorization enhancement**: `EmbeddingVectorizer` is a pluggable backend (OpenAI-compatible embed API / local sentence-transformers); when disabled (default), the system still uses the zero-dependency bag-of-words hash path, preserving existing interfaces and unit tests. The benchmark now includes a **bag-of-words vs embedding** comparison dimension.

Corresponding code layout (`src/emotion_memory/`):

| Module | Responsibility |
|---|---|
| `models.py` | Data models for memory / affect labels / cluster metadata |
| `config.py` | Freedom dial `Dial` (default: low noise, high stability; v0.6 adds embedding switches) |
| `embeddings.py` | **(v0.6) Optional EmbeddingVectorizer**: mock / openai_compat / local_compat / gemini / sentence_transformers, same interface as bag-of-words |
| `store/storage.py` | Original-text storage layer: JSONL disk persistence; `original_text` is archived permanently and cannot be modified or deleted; `text` is the pollutable view that can be persisted |
| `store/vector_index.py` | Vector index: bag-of-words hashed vectors + brute-force ANN (interfaces swappable to FAISS/HNSW) |
| `store/graph_index.py` | Lightweight entity-relation graph index: related-cluster localization + strong-association edges |
| `algorithms/forgetting.py` | Forgetting curve: accessibility score decay and reinforcement |
| `algorithms/hit_function.py` | Oscillatory hit function H(t): multi-channel frequency/phase |
| `algorithms/affect.py` | Emotional gain + dB trade-off ranking (affects ranking only, never original text) |
| `algorithms/pollution.py` | Emotional pollution filter: beautify/uglify versions generated from original text, output view only |
| `algorithms/affect_analyzer.py` | Zero-dependency automatic affect labeler: computes (valence, arousal) on write; external model hook reserved |
| `reminder.py` | Time-sensitive reminders: due memories never silenced, injected into context, ack receipt |
| `retrieval/recall.py` | Retrieval layer: related-cluster → ANN recall → ranking → dual-view binding (view=polluted/original) |
| `silence.py` | Silence layer: mentionable/silent two levels + unlock conditions + never-silent |
| `divergent.py` | Divergent mode (off by default) |
| `concurrent.py` | Concurrent association (off by default, Map-Reduce) |
| `bridge.py` | Current-bridge migration protocol: export/import + SHA-256 verification + refuse-to-recover on failure |
| `identity.py` | Identity profile + lifecycle state machine (reset farewell / reinstall wake-up / upgrade move) |
| `engine.py` | Top-level API (pollution read/write, due reminders, auto-labeling, direct-ask unlock; v0.6 supports `vectorizer=` injection) |
| `cli.py` | CLI terminal: write/recall/direct-ask/status/dial/export-import/pollution control |
| `algorithms/three_phase.py` | Rotating magnetic-field recall scheduling: three-phase channels with 120° phase offsets (v0.3) |
| `algorithms/capacitor.py` | Capacitor memory model: charge/discharge/leak + standby pool (v0.3) |
| `retrieval/proactive.py` | Proactive recall engine: timer/trigger-driven autonomous speaking (v0.3) |
| `self_learning.py` | Identity self-learning: profile gradual evolution + proactive question confirmation (v0.3) |
| `unified_self.py` | Unified self model: 1 core + avatars + all-is-me + active/apparent power metrics (v0.3) |
| `model_adapter.py` | Model adapter layer: unified LLM/embed interface; OpenAI-compatible online / local (Ollama, LM Studio) + Mock + transformers reserved (v0.4, multi-provider registry) |
| `conversation_pipeline.py` | Conversation pipeline: retrieval injection → model reply → automatic memory settling (v0.4) |
| `emergent_self.py` | Emergent self engine: self-definition generated from accumulated affect statistics, no preset template (v0.4) |
| `proactive_loop.py` | Proactive speaking loop: timed proactive recall → opening topic → model-generated speech (v0.4) |
| `conservation.py` | Emotional conservation: global affect budget constraint, no-source injection raises (v0.5) |
| `potential_transfer.py` | Potential transfer: high-potential settles into long-term zone, low-potential disperses to working zone (v0.5) |
| `entropic_forgetting.py` | Entropic forgetting: detail fidelity decays, emotional residue and entropy rise, original intact (v0.5) |
| `turbulent_association.py` | Turbulent association: vortex chain center → secondary → periphery with decreasing energy (v0.5) |
| `personality_manifold.py` | Personality manifold: environment-vector-driven six-axis personality, deterministic reproducible bifurcation (v0.5) |

## 3. Quick Start

```bash
# Zero dependency, Python 3.9+ only
python demo/demo_full_flow.py            # Full flow (write→recall→dual-view pollution→forgetting→migration→reminder)
python demo/demo_v03_full_flow.py        # v0.3 features (three-phase field→proactive recall→capacitor→self-learning→unified self)
python demo/demo_v04_llm_pipeline.py     # v0.4 (conversation pipeline→emergent self→proactive loop; Mock default, --config for real endpoint)
python demo/demo_v05_physical_laws.py    # v0.5 (four physical laws→personality manifold→dual-environment bifurcation)
python demo/demo_v06_marvis_integration.py # v0.6 demo (Marvis integration example)
python demo/demo_pollution_llm_effect.py   # v0.6 demo (attitude drift when polluted memories are fed to an LLM; offline Mock, --config for real model)
python -m unittest discover -s tests -v  # Unit tests (v0.6 total: 235, all green)
python -m benchmark.benchmark --memories 1000 --top-k 5   # Benchmark (bag-of-words vs embedding, see Chapter 6)
python cli.py --help                         # CLI terminal (repo root entry)
python cli.py write "去年生日和三个老同学爬山" --cluster 生活
python cli.py recall "爬山 生日" --view original
```

### 3.1 Enabling the Optional Embedding Vectorizer (v0.6)

```python
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.embeddings import make_vectorizer

# Option 1: zero-config mock (link verification only; deterministic 8-dim)
mem = EmotionalMemorySystem("data_dir", vectorizer=make_vectorizer(backend="mock"))

# Option 2: OpenAI-compatible embed API (reuses model_adapter.embed; credentials via EMOTION_LLM_API_KEY / OPENAI_API_KEY)
mem = EmotionalMemorySystem("data_dir", vectorizer=make_vectorizer(
    backend="openai_compat",
    base_url="https://api.openai.com/v1",
    model="text-embedding-3-small",
))

# Option 3: local sentence-transformers (must be self-installed; MIT license; auto-fallback + warning if missing)
mem = EmotionalMemorySystem("data_dir", vectorizer=make_vectorizer(
    backend="sentence_transformers", model_name="BAAI/bge-small-zh-v1.5"
))
```

When disabled or when the backend is unavailable, the engine automatically falls back to the zero-dependency bag-of-words hash and prints a warning, without breaking existing behavior. See the `embedding` section of `config.example.json`.

## 4. Core API

```python
from emotion_memory.engine import EmotionalMemorySystem

mem = EmotionalMemorySystem("data_dir")
mem.remember("用户说喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"],
             affect=(0.6, 0.4))            # valence, arousal; auto-labeled when affect is omitted

hits = mem.recall("晨跑")                  # Retrieval: related-cluster → ANN → ranking → injection (polluted view by default)
mem.decay_all()                            # Forgetting curve over time
mem.export("data_dir/bridge.json")         # Current-bridge export (with SHA-256)
```

```python
# Original text is archived forever; pollution only affects the output view (polluted by default, original recoverable)
mem.apply_pollution(mid, mode="beautify", intensity=1.0)   # beautify filter
mem.apply_pollution(mid, mode="uglify", intensity=1.0)     # uglify filter
hits = mem.recall("晨跑", view="polluted")  # default: polluted output
hits = mem.recall("晨跑", view="original")  # parameter to recover original text
mem.clear_pollution(mid)                    # purify back to original; audit history retained
mem.view_memory(mid)                        # dual-view + pollution audit (pollution_history)
```

```python
mem.remember("用户说喜欢晨跑", cluster="hobby")  # no affect → auto-label (valence, arousal)
# External model hook: EmotionalMemorySystem(affect_provider=my_external_model)
```

```python
mem.remember("明天上午十点交房租", due_at=deadline_ts, never_silent=True)
reminders = mem.due_reminders()            # Due high-risk/time-sensitive memories
ctx = mem.context_with_reminders([])       # Inject due reminders (never silenced)
mem.ack_reminder(mid)                      # Ack reminder
```

```python
from emotion_memory.config import Dial

dial = Dial(noise_temperature=0.1,   # noise temperature 0~1, default low
            emotion_gain=0.3,        # emotional gain, mild
            jump_probability=0.05,   # cross-cluster jump probability
            amplitude=1.0,           # hit function amplitude
            divergent_enabled=False, # divergent mode off by default
            embedding_enabled=False) # v0.6: embedding vectorization off by default
```

## 5. Safety Baseline

1. Original text is archived forever but can be emotionally polluted: `original_text` is never deleted and cannot be rewritten by filters (audit/backtracking preserved); the filter generates a **polluted version** from the original; retrieval and context injection default to the polluted view; pass `view="original"` to recover the original — humans are contradictory: they remember facts and cannot forget filters;
2. High-risk / time-sensitive memories are never silenced; due reminders are auto-injected and require explicit ack;
3. Farewell / wake-up / migration are all deterministic; model free-play is forbidden;
4. Migration must be verified (SHA-256 + full comparison); refuse to recover on failure;
5. "Remove all limits" means turning the dial all the way up, not removing the dial; the default is low-noise, high-stability;
6. Credentials are never hardcoded: embedding / LLM api_keys always come from environment variables or config.

## 6. Benchmark Results (v0.6 Comparison Upgrade)

`python -m benchmark.benchmark --memories N --top-k K --seed S` outputs JSON + TXT reports quantifying: **recall accuracy** (pre-pollution / post-pollution / original view / unrelated queries), **bag-of-words vs embedding comparison**, **recall stability** (pollution does not break recall), **affect bias** (beautify/uglify directional accuracy, mean shift), and **original-text immutability** (kept count).

> All accuracy comparisons use the zero-noise dial (`noise_temperature=0`), making oscillation scoring fully deterministic and pre/post-pollution strictly comparable (`|Δ|=0` means pollution does not break recall).

### 6.1 Current Measured Numbers (honest, no fabrication)

Reproducible baseline results in this repository (300 mixed CN/EN memories, top-k=5, multiple seeds; see `benchmark/benchmark_report*.json`):

| Metric | seed=1 | seed=2 | seed=3 | Note |
|---|---|---|---|---|
| Bag-of-words hit rate (pre-pollution) | 31.00% | 30.67% | 32.67% | BOW baseline (≈0.255–0.33 band, seed-sensitive) |
| Bag-of-words hit rate (post-pollution) | 31.00% | 30.67% | 32.67% | Identical to pre-pollution, `|Δ|=0` |
| Original-view hit rate | 31.00% | 30.67% | 32.67% | Identical to pre-pollution |
| Unrelated-query hit rate | 1.33% | 2.33% | 2.00% | Far below target queries; retrieval is discriminative |
| Embedding(mock) hit rate | 2.67% | 1.00% | 1.33% | **Link verification only**, not real semantic quality |
| Affect bias beautify accuracy | 90.67% | — | — | directional accuracy (seed=1) |
| Affect bias uglify accuracy | 80.67% | — | — | directional accuracy (seed=1) |
| Original-text immutability | 600/600 | — | — | kept count pre/post pollution |

**Important notes (honest disclosure)**:

1. **Bag-of-words baseline ≈ 0.255** was the planning target reference; current reproducible runs (300 memories × multiple seeds) measure **30.7%–32.7%**, above the target reference, with fluctuation caused by random seeds and hash-collision distribution over the mixed CN/EN corpus;
2. **The embedding comparison uses the mock backend**: mock is a deterministic 8-dim embedding that only validates the `EmbeddingVectorizer` integration path and the benchmark comparison pipeline; **it does not represent real semantic retrieval quality**. When a real backend (OpenAI-compatible / sentence-transformers) lacks credentials or dependencies, the benchmark honestly marks `status=unavailable` and does not fabricate numbers;
3. Recall stability `|Δ|=0`: the pollution filter does not break recall, consistent with the design goal;
4. The full JSON report is at `benchmark/benchmark_report.json` (including the `status` field).

## 7. Version History at a Glance

- **v0.3**: rotating magnetic-field recall (three-phase 120°), proactive recall engine, capacitor memory model, identity self-learning, unified self model;
- **v0.4**: model adapter layer (OpenAI-compatible / Anthropic / Gemini / local endpoints / Mock), conversation pipeline, emergent self, proactive speaking loop;
- **v0.5**: four physical laws (emotional conservation / potential transfer / entropic forgetting / turbulent association) + personality manifold;
- **v0.6**: optional embedding vectorization, benchmark bag-of-words vs embedding comparison, pollution-effect experiment demo, version alignment, bilingual README rewrite, Marvis integration archive (demo/demo_v06_marvis_integration.py), model_adapter multi-provider registry archive.

## 8. Known Limitations and Roadmap (v0.6, honest)

**Known limitations**:

- The default bag-of-words hashed vectors have no semantic generalization; synonymous rewrites / cross-language queries have limited hit rates (currently measured ≈30%, seed-sensitive);
- Embedding enhancement is currently an **optional backend with mock link verification**: real OpenAI-compatible endpoints require user-configured credentials (never hardcoded); sentence-transformers must be self-installed (MIT license); when not connected, the benchmark honestly reports `unavailable`;
- Single-machine JSONL storage + brute-force ANN: targeted at individuals / small teams; no distributed, multi-tenant, or sub-second large-scale retrieval (boundary vs Zep Cloud / Mem0 enterprise scenarios in COMPARISON.md);
- The affect labeler is a zero-dependency built-in lexicon, less accurate than LLM labeling; `affect_provider` external hook reserved;
- No automated dependency scanner for GPL/commercial-license forbidden lists yet (compliance relies on manual review of NOTICE.md).

**Roadmap**:

1. Integrate real embedding backends (OpenAI-compatible / sentence-transformers / local vector DB) to replace bag-of-words hashing with semantic hit rates, with an honest "bag-of-words vs real embedding" benchmark comparison;
2. Retrieval ranking fusion: emotional gain × vector similarity × forgetting accessibility, three-channel weighted;
3. Optional vector DB backends (FAISS / HNSW / Qdrant, all MIT / Apache-2.0, license-compliant) to replace brute-force ANN;
4. Large-scale memory stress tests (1k / 10k / 100k) with recall@k curves;
5. Automated compliance scanning (dependency license checks) in CI.

## 9. Horizontal Comparison

See [COMPARISON.md](COMPARISON.md) for the full comparison with mainstream AI-memory projects (Mem0, Letta, Zep/Graphiti, Cognee). Core differentiators:

1. **Emotion is unique**: mainstream memory stores are emotionless retrieval layers; this project is the only one that treats valence/arousal and the dual-view pollution filter as first-class citizens;
2. **Forgetting is a design, not a bug**: the forgetting curve only lowers accessibility and never deletes original text;
3. **Identity emerges from data**: no preset profile; "what you are" is dynamically generated from accumulated affect;
4. **Portability**: deterministic current-bridge migration (SHA-256 + full comparison) is unique among peers;
5. **Zero dependency + MIT**: lowest deployment cost; no GPL (avoids Neo4j), no commercial terms (avoids Mem0 Cloud).

## Appendix: Compliance and Ethics

- Runtime dependencies and license statements: [NOTICE.md](NOTICE.md);
- Ethics statement in clause form: [ETHICS.md](ETHICS.md);
- Change log: [CHANGELOG.md](CHANGELOG.md).
