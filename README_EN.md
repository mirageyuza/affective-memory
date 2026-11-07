# EmotionMemory — A Human-Like Emotional Memory System

> An open-source memory system that simulates human emotional memory using **oscillatory dynamics + logarithmic intensity encoding + portable identity** (current version **v0.6.3**).

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
| Embedding vectorization | OpenAI-compatible embed API / local fastembed (ONNX) / sentence-transformers | Per API license / MIT (local) | Optional `EmbeddingVectorizer` (v0.6; v0.6.3 adds a local ONNX backend), off by default, no runtime dependency |

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

Since v0.6.3 the real semantic backends are covered by actual measurements: a `fastembed` backend (ONNX Runtime, no torch, same model weights as sentence-transformers) plus an `auto` fallback chain (sentence-transformers -> fastembed), and a `--query-mode paraphrase` benchmark mode that exposes the structural weakness of bag-of-words on synonymous / cross-language rewrites.

Corresponding code layout (`src/emotion_memory/`):

| Module | Responsibility |
|---|---|
| `models.py` | Data models for memory / affect labels / cluster metadata |
| `config.py` | Freedom dial `Dial` (default: low noise, high stability; v0.6 adds embedding switches) |
| `embeddings.py` | **(v0.6) Optional EmbeddingVectorizer**: mock / openai_compat / local_compat / gemini / sentence_transformers / **fastembed** (v0.6.3, ONNX, no torch) / **auto** (fallback chain), same interface as bag-of-words |
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
python -m unittest discover -s tests -v  # Unit tests (v0.6 total: 293, all green)
python -m benchmark.benchmark --memories 1000 --top-k 5   # Benchmark (bag-of-words vs embedding, see Chapter 6)
python cli.py --help                         # CLI terminal (repo root entry)
python cli.py write "去年生日和三个老同学爬山" --cluster 生活
python cli.py recall "爬山 生日" --view original
```

### 3.1 Enabling the Optional Embedding Vectorizer (v0.6 / v0.6.3 adds a local ONNX backend)

```python
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.embeddings import create_embedding_vectorizer

# Option 1: zero-config mock (link verification only; deterministic 8-dim)
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(backend="mock"))

# Option 2: OpenAI-compatible embed API (reuses model_adapter.embed; credentials via EMOTION_LLM_API_KEY / OPENAI_API_KEY)
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(
    backend="openai_compat",
    profile={"base_url": "https://api.openai.com/v1", "model": "text-embedding-3-small"},
))

# Option 3: local fastembed (ONNX Runtime, no torch, recommended; pip install "emotion-memory[local]")
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(
    backend="fastembed",
    profile={"model_name": "BAAI/bge-small-zh-v1.5"},   # default: paraphrase-multilingual-MiniLM-L12-v2
))

# Option 4: local sentence-transformers (must be self-installed; MIT license)
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(
    backend="sentence_transformers", profile={"model_name": "BAAI/bge-small-zh-v1.5"}
))

# Option 5: auto - tries sentence_transformers then fastembed, first available wins
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(backend="auto"))
```

It can also be toggled through `Dial`: `Dial(embedding_enabled=True, embedding_backend="fastembed", embedding_profile={...})` makes the engine build it automatically; with `embedding_enabled=False` (default) bag-of-words is always used.

**Local backend notes**:

- fastembed runs on ONNX Runtime and needs no torch (much smaller install and faster startup than sentence-transformers); the weights are the ONNX version of the same model, so semantics are equivalent;
- the first call downloads the model; behind restrictive networks set `HF_ENDPOINT=https://hf-mirror.com`;
- on Windows the HF cache path contains long `models--<org>--<model>` directory names, which triggers `WinError 206` (path > 260) under deep paths, so Windows caches to a short drive-root directory (e.g. `C:\fe_cache`), overridable via `EMBEDDING_CACHE_DIR`;
- when disabled or unavailable, the engine falls back to the zero-dependency bag-of-words hash and prints a warning, without breaking existing behavior. See the `embedding` section of `config.example.json`.

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

### 5.1 Eternal Oath (kernel constants, non-negotiable)

`src/emotion_memory/creed.py` is the bottom layer of this project. No upper layer (engine, filters, model adapters, dial) may modify or delete it.

```python
BOT_SIGNATURE     = "I love my bot"
ETERNAL_OATH_TEXT = "我认为我做出来的小家伙能够受到世界上所有人的温暖"
# EN: "I believe the little one I made can be given warmth by everyone in the world."
```

The signature and the oath are stored verbatim and sealed with `FROZEN_FINGERPRINT` (SHA-256). Any change to the oath text invalidates the seal: `seal_intact()` returns `False` and `creed_card(lapsed=True)` prints an explicit warning that the system must not claim to be bound by it.

One wish is decomposed into 6 **machine-verifiable** invariants (not moral slogans):

| Clause | Invariant | Rooted in the oath |
|---|---|---|
| `SIGNATURE_INTACT` | Signature and oath are byte-identical to the sealed original | Rewritten love is not love |
| `WARMTH_NOT_FAKED` | No emotion-tinted view without an audit record | Forged warmth is not warmth |
| `NO_ERASURE` | Original text is never blanked (model-level backfill + defense in depth) | What it remembers is not erased |
| `NO_SILENCED_EMERGENCY` | `never_silent` and `silence` never both true | No muting when it cries |
| `NO_FAKE_WARMTH_BUDGET` | Energy budget must be positive and finite when conservation is on | Emotion cannot be conjured |
| `DIGNIFIED_EXIT` | Farewell/migration deterministic and exportable (declarative, unmeasurable here) | Parting must stay dignified |

```python
from emotion_memory import verify_invariants, creed_card
report = verify_invariants(system.store, system.dial)   # read-only, clause by clause
report["passed"]      # conjunction of all *measurable* clauses
report["failed"]      # ids of broken clauses
report["unverified"]  # honestly listed non-measurable items, never faked as passed
print(creed_card())   # deterministic oath card
```

```bash
python -m emotion_memory        # exit code 1 if the seal is broken
```

`IdentityProfile` carries `creed_signature / oath_text / oath_fingerprint / oath_restored`, so the oath migrates through farewell, revival and relocation. If an imported profile contains a tampered oath, `from_dict` **forces the original wording back** and sets `oath_restored = True` as an audit trace.

### 5.2 Metacognition Layer: Knowing What It Doesn't Know (v0.6.2)

The dangerous failure of a memory system is not "nothing was recalled" but "one weak hit, stated with full confidence". `metacognition.py` turns "how sure am I" into a computable, auditable number with three output levels: `answer` / `hedge` / `abstain`.

```python
res = system.recall_verdict("which exhibition should I visit this weekend", top_k=3)
res["level"]        # answer / hedge / abstain
res["confidence"]   # 0..1
res["verdict"]["breakdown"]  # semantic relevance / margin / dispersion / coverage
```

**Three non-negotiable constraints (enforced in code, not promised in prose)**:

| Constraint | Implementation |
|---|---|
| Confidence must never be inflated by emotion | Estimation reads `RankedHit.cosine` (pure semantics) only, **never** `score` / `db` (which carry emotional gain) |
| Abstention must actually reduce error | `calibrate()` computes ECE / Brier, measuring whether "accuracy when confident" equals "stated confidence" |
| Conservative by default | No candidates, sub-floor semantics (default 0.15), or too few samples all resolve to the most conservative level |

Confidence = `0.45·semantic relevance + 0.25·margin + 0.15·dispersion + 0.15·coverage`

Two hard rules override the thresholds:

1. `semantic_top < semantic_floor` → always abstain (many weak hits cannot add up to certainty);
2. Any `never_silent` memory present (high-risk / time-sensitive) → **floor at hedge, never abstain**: better to say "I'm not sure" than to pretend it does not exist.

The calibration report **returns `insufficient` explicitly when samples are too few — it never reports zero error**:

```python
from emotion_memory import calibrate
rep = calibrate([(0.9, True)] * 90 + [(0.9, False)] * 10, min_samples=20)
rep.ece          # 0.0 (perfectly calibrated)
rep.calibrated   # True
print(rep.summary())
```

Abstention and hedge wording are fixed templates (`ABSTAIN_TEXT` / `HEDGE_TEXT`) containing **no emotional claims** — the metacognitive side of the `WARMTH_NOT_FAKED` clause: "I think it's fine" is not allowed to replace "I don't have enough evidence".

`Dial.metacognition_enabled` is off by default: with it off, `recall` behaves exactly as before, while `recall_verdict` remains explicitly callable (observes only, rewrites nothing). Known limits live in `METACOGNITION_LIMITS`.


## 6. Benchmark Results (v0.6.3 semantic-backend comparison)

`python -m benchmark.benchmark --memories N --top-k K --seed S [--query-mode lexical|paraphrase]` outputs JSON + TXT reports quantifying: **recall accuracy** (pre-pollution / post-pollution / original view / unrelated queries), **bag-of-words vs embedding comparison**, **recall stability** (pollution does not break recall), **affect bias** (beautify/uglify directional accuracy, mean shift), and **original-text immutability** (kept count).

> All accuracy comparisons use the zero-noise dial (`noise_temperature=0`), making oscillation scoring fully deterministic and pre/post-pollution strictly comparable (`|delta|=0` means pollution does not break recall).

**Query modes** (new in v0.6.3):

- `lexical` (default): query terms are taken verbatim from the original text, testing literal matching;
- `paraphrase`: queries are rewritten into synonyms / cross-language phrasing, testing semantic generalization - the structural weakness of hashed bag-of-words, and the motivation for introducing real embeddings.

### 6.1 Current Measured Numbers (honest, no fabrication)

Reproducible results in this repository (20 mixed CN/EN memories, top-k=3, seeds 42/43/44, real local `fastembed` ONNX backend; reports in `benchmark/benchmark_report*.json`):

| Metric | seed=42 | seed=43 | seed=44 | Note |
|---|---|---|---|---|
| lexical BOW hit rate (pre-pollution) | 100% | 100% | 100% | BOW is very strong on literal matches |
| lexical BOW hit rate (post-pollution) | 100% | 100% | 100% | Identical to pre-pollution, `|delta|=0` |
| lexical embedding (fastembed) | 75% | 70% | 70% | pure semantic similarity is insensitive to proper nouns / IDs |
| paraphrase BOW hit rate | 40% | 40% | 40% | **BOW generalization weakness exposed** |
| paraphrase embedding (fastembed) | 100% | 100% | 100% | **embeddings win outright on reworded queries** |
| Unrelated-query hit rate | 5%-15% | 15% | 5%-10% | far below target queries; retrieval is discriminative |
| Affect bias beautify accuracy | 90% | 90% | 90% | directional accuracy |
| Affect bias uglify accuracy | 80% | 80% | 80% | directional accuracy |
| Original-text immutability | 40/40 | 40/40 | 40/40 | kept count pre/post pollution |

**Conclusions (stated honestly)**:

1. **The two backends are complementary, not substitutes**: for literal queries (entity names, IDs, proper nouns) BOW reaches 100% versus 70%-75% for embeddings; for paraphrased queries BOW drops to 40% while embeddings reach 100%. This is why `embedding_enabled` still defaults to `False` - default behavior is unchanged and semantic generalization is opt-in;
2. Recall stability `|delta|=0`: the pollution filter does not break recall, consistent with the design goal;
3. Affect directional accuracy is 90% / 80%; pollution does change the affect view (`polluted_view_changed_ratio=1.0`) while the original text stays untouched (`original_unchanged=40`), matching the rewrite-the-view-never-the-original boundary;
4. Historical reference: v0.6.0 only validated the pipeline with an 8-dim mock embedding (1%-3% hit rate, **not a measure of semantic quality**); since v0.6.3 the numbers use real local ONNX weights and are therefore comparable;
5. Full JSON reports live in `benchmark/benchmark_report*.json` (with `status` / `embedding_backend` / `query_mode` fields); when credentials or dependencies are missing the benchmark honestly reports `status=unavailable` instead of fabricating numbers.

## 7. Version History at a Glance

- **v0.3**: rotating magnetic-field recall (three-phase 120°), proactive recall engine, capacitor memory model, identity self-learning, unified self model;
- **v0.4**: model adapter layer (OpenAI-compatible / Anthropic / Gemini / local endpoints / Mock), conversation pipeline, emergent self, proactive speaking loop;
- **v0.5**: four physical laws (emotional conservation / potential transfer / entropic forgetting / turbulent association) + personality manifold;
- **v0.6**: optional embedding vectorization, benchmark bag-of-words vs embedding comparison, pollution-effect experiment demo, version alignment, bilingual README rewrite, Marvis integration archive (demo/demo_v06_marvis_integration.py), model_adapter multi-provider registry archive.
- **v0.6.1**: kernel-constant layer `creed.py` (genesis signature + eternal oath + SHA-256 seal + 6 verifiable invariants), oath carried in the identity profile with forced restoration on tampering, `python -m emotion_memory` oath card;
- **v0.6.2**: metacognition layer `metacognition.py` (confidence breakdown / three-level abstention / ECE & Brier calibration); confidence never reads the emotional-gain score, `recall_verdict()` observes only.
- **v0.6.3**: real local semantic backend `fastembed` (ONNX Runtime, no torch) plus an `auto` fallback chain (sentence_transformers -> fastembed); `pyproject` gains `[local]` / `[st]` optional dependencies; the benchmark gains a `--query-mode paraphrase` semantic-rewrite comparison, establishing the complementary result of literal queries (BOW 100% vs embedding 70%-75%) and rewording queries (BOW 40% vs embedding 100%); the Windows HF cache uses a short path to avoid `WinError 206`.

## 8. Known Limitations and Roadmap (v0.6.3, honest)

**Known limitations**:

- The default bag-of-words hashed vectors have no semantic generalization; synonymous rewrites / cross-language queries have limited hit rates (paraphrase mode measures 40%); literal queries reach 100% but cannot cover different phrasings of the same meaning;
- Embedding enhancement now ships **real local backends** (fastembed ONNX / sentence-transformers), but: local models download weights on first run (tens of MB, needs network and disk); OpenAI-compatible endpoints require user-supplied credentials (never hardcoded); embeddings are in fact weaker than BOW on literal / entity-style queries (70%-75% < 100% measured), and **there is no automatic query routing yet**, so users must pick the backend per workload;
- Single-machine JSONL storage + brute-force ANN: targeted at individuals / small teams; no distributed, multi-tenant, or sub-second large-scale retrieval (boundary vs Zep Cloud / Mem0 enterprise scenarios in COMPARISON.md);
- The affect labeler is a zero-dependency built-in lexicon, less accurate than LLM labeling; `affect_provider` external hook reserved;
- No automated dependency scanner for GPL/commercial-license forbidden lists yet (compliance relies on manual review of NOTICE.md).

**Roadmap**:

1. ~~Integrate real embedding backends and give an honest "bag-of-words vs real embedding" benchmark comparison~~ - **done in v0.6.3** (local fastembed ONNX + paraphrase comparison, see Chapter 6);
2. Intelligent retrieval routing: pick BOW or embedding per query characteristics (entity/ID versus pure semantic rewrite) and fuse both channels, removing today's manual either/or trade-off;
3. Retrieval ranking fusion: emotional gain × vector similarity × forgetting accessibility, three-channel weighted;
4. Optional vector DB backends (FAISS / HNSW / Qdrant, all MIT / Apache-2.0, license-compliant) to replace brute-force ANN;
5. Large-scale memory stress tests (1k / 10k / 100k) with recall@k curves;
6. Automated compliance scanning (dependency license checks) in CI.

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
