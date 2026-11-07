# EmotionMemory — 类人情感记忆系统

> 一个用 **振荡动力学 + 对数强度编码 + 可迁移身份** 模拟人类情感记忆的开源记忆系统（当前版本 **v0.6.3**）。

记忆以周期震荡被唤起，强度按情感增益与 dB 刻度调制，过载时产生发散联想，随时间遵循遗忘曲线，知道该说什么，也明白该沉默什么；智能体可以被重置、重装、升级，但记忆永生。

**许可证**：MIT。运行时零第三方依赖（仅 Python 标准库），可在此基础上按 MIT/Apache-2.0/BSD 系自由接入 FAISS/HNSW/Qdrant 等索引后端。**严禁引入 GPL 或商业条款依赖。**

## 0. 开源基础与合规标注（本项目基于哪些开源项目）

本项目定位为**开源组件的整合封装层**：不重复造轮子，各能力层均基于成熟开源项目设计（接口兼容、可插拔接入），并对其许可证做严格合规约束。

| 组件 | 上游开源项目 | 许可证 | 本项目集成方式 |
|---|---|---|---|
| 向量库 | Qdrant / Milvus | Apache-2.0 | 设计兼容；本项目内置暴力 ANN 占位，接口可替换接入 |
| 向量索引 | FAISS / HNSW | MIT / Apache-2.0 | 设计兼容；内置哈希向量索引为可插拔占位实现 |
| 实体关系图 | Neo4j Community（GPL，**规避**）→ 自研简化图索引 / networkx | GPL 规避；自研 MIT / networkx BSD | 自研轻量图索引（`store/graph_index.py`） |
| 记忆管理参考 | Mem0 / Letta / LangChain Memory | 注意 Mem0 商业条款 | 仅借鉴设计思想，不复用任何受限代码 |
| 情感分析 | 现有情感分类模型（随模型许可） | 随模型许可 | 预留 `affect=(valence, arousal)` 标签接入点 |
| 遗忘曲线参考 | FSRS / Anki（SM-2）算法 | 宽松许可（MIT 系） | 可及性分数衰减/强化模型参考实现 |
| 迁移协议 | git / rsync 增量思想 | 宽松 | 电流桥增量导出（仅变更条目） |
| Embedding 向量化 | OpenAI 兼容 embed API / 本地 fastembed(ONNX) / sentence-transformers | 随模型许可（API）/ MIT（本地） | 可选 `EmbeddingVectorizer`（v0.6，v0.6.3 增本地 ONNX 后端），默认关闭，不引入运行时依赖 |

**合规红线**：对外开源整包时，所有运行时与索引依赖必须是 MIT / Apache-2.0 / BSD 系；GPL 与商业条款组件（如 Neo4j Community、Mem0 商业版、AGPL 系如 PyMuPDF）一律不得引入。v0.6 的 embedding 增强同样遵守此红线：在线端点走标准库 urllib 的 OpenAI 兼容 embed API（无第三方依赖），本地 sentence-transformers 仅在用户显式启用且自行安装时可用（MIT 许可），不进入运行时依赖清单。

## 1. 愿景

智能体应当拥有「像人一样的记忆」：

- **永久不忘**：原文永远保留在磁盘，遗忘只降低"被想起的概率"，不删内容；
- **周期唤起**：每条记忆簇有独立的频率与相位，命中函数 `H(t) = g·Σ Aᵢ·sin(ωᵢt+φᵢ) + 噪声(T)` 让记忆像 theta 振荡一样被周期性想起；
- **情感调制**：情感标签 `(valence, arousal)` 参与排序与污染滤镜——原文永久存档，输出默认是"被感情污染过的版本"，可随时回溯原文；
- **知道沉默**：可提及 / 沉默两级可见性，高危与时间敏感记忆 **永不沉默**，到期自动注入提醒；
- **记忆永生**：智能体可以被重置、重装、升级，记忆通过电流桥协议无损迁移。

核心不变量：

```
智能体 = 模型参数(可换) + 记忆库(永恒) + 涌现自我(由情感累计生成，非预设档案)
```

## 2. 架构

```
┌─────────────────────────────────────────────────┐
│                交互层 (Agent Runtime)             │
│  触发分类器 → 回忆引擎 → 上下文注入(带预算)         │
├─────────────────────────────────────────────────┤
│                回忆层 (Recall Engine)             │
│  震荡命中函数 · 情感增益 · dB权衡 · 并发多路召回    │
│  遗忘曲线衰减 · 沉默层(检索抑制) · 发散模式(过载)    │
├─────────────────────────────────────────────────┤
│                记忆层 (Memory Store)              │
│  原文存储(磁盘, 无限扩容) · 实体-关系图(联想索引)    │
│  向量索引(ANN) · 情感标签 · 可及性分数              │
├─────────────────────────────────────────────────┤
│                身份层 (Identity)                  │
│  身份档案 · 生命周期状态机 · 告别/唤醒/迁移协议      │
└─────────────────────────────────────────────────┘
```

v0.5.0 起，记忆/回忆层之上新增物理四定律与人格流形：情感守恒、势差迁移、熵增遗忘、湍流联想四套机制（Dial 开关默认关闭，开启不改变既有行为），人格流形以环境向量驱动六轴人格、非线性分叉确定性可复现。

v0.6.0 起，检索层新增**可选 embedding 向量化增强**：`EmbeddingVectorizer` 作为可插拔后端（OpenAI 兼容 embed API / 本地 sentence-transformers），默认关闭时仍走零依赖词袋哈希路径，不破坏既有接口与单测；benchmark 新增「词袋 vs embedding」对照维度。

v0.6.3 起，补齐真实语义后端实测：新增 `fastembed` 后端（ONNX Runtime，免 torch，模型与 sentence-transformers 同权重）与 `auto` 回退链（sentence_transformers → fastembed），benchmark 增加 `--query-mode paraphrase` 改写查询对照。实测结论：**词面重合查询词袋占优（1.00 vs 0.70~0.75），同义改写查询 embedding 占优（1.00 vs 0.40）**，两者互补，见第 6.2 节。

对应代码结构（`src/emotion_memory/`）：

| 模块 | 职责 |
|---|---|
| `models.py` | 记忆 / 情感标签 / 簇元数据数据模型 |
| `config.py` | 自由度刻度盘 Dial（默认低噪声高稳定；v0.6 含 embedding 开关） |
| `embeddings.py` | **（v0.6）可选 EmbeddingVectorizer**：mock / openai_compat / local_compat / gemini / sentence_transformers / **fastembed**（v0.6.3，ONNX 免 torch）/ **auto**（回退链），与词袋同接口 |
| `store/storage.py` | 原文存储层：JSONL 磁盘持久化，`original_text` 永久存档不可删改，`text` 为污染视图可落盘 |
| `store/vector_index.py` | 向量索引：词袋哈希向量 + 暴力 ANN（接口可换 FAISS/HNSW） |
| `store/graph_index.py` | 简易实体-关系图索引：相关簇定位 + 强关联边 |
| `algorithms/forgetting.py` | 遗忘曲线：可及性分数衰减与强化 |
| `algorithms/hit_function.py` | 震荡命中函数 H(t)：多通道频率/相位 |
| `algorithms/affect.py` | 情感增益 + dB 权衡排序（只影响排序，不改原文） |
| `algorithms/pollution.py` | 情感污染滤镜：基于原文生成美化/丑化污染版本，只改输出视图 |
| `algorithms/affect_analyzer.py` | 零依赖自动情感标注器：写入时自动计算 (valence, arousal)，预留外部模型接口 |
| `reminder.py` | 时间敏感提醒：到期记忆永不沉默、到期注入上下文、确认回执 |
| `retrieval/recall.py` | 检索层：相关簇定位 → ANN 召回 → 排序 → 双层视图绑定（view=polluted/original） |
| `silence.py` | 沉默层：可提及/沉默两级 + 解锁条件 + 永不沉默 |
| `divergent.py` | 发散模式（默认关闭） |
| `concurrent.py` | 并发联想（默认关闭，Map-Reduce） |
| `bridge.py` | 电流桥迁移协议：导出/导入 + SHA-256 校验 + 失败拒绝恢复 |
| `identity.py` | 身份档案 + 生命周期状态机（重置告别/重装唤醒/升级搬家） |
| `engine.py` | 顶层封装 API（含污染读写、到期提醒、自动标注、直问解锁；v0.6 支持 `vectorizer=` 注入） |
| `cli.py` | CLI 交互终端：写入/检索/直问解锁/状态/刻度盘/导出导入/污染控制 |
| `algorithms/three_phase.py` | 旋转磁场唤起调度：三相通道 120° 相位差顺序激活（v0.3） |
| `algorithms/capacitor.py` | 电容记忆模型：充电/放电/漏电 + standby 备选池（v0.3） |
| `retrieval/proactive.py` | 主动回忆引擎：定时/触发器驱动自主搭话（v0.3） |
| `self_learning.py` | 身份自学习：画像渐进演化 + 主动提问确认（v0.3） |
| `unified_self.py` | 自我统一模型：1 核心 + 万我分身 + 万物皆我 + 有功/视在指标（v0.3） |
| `model_adapter.py` | 模型适配层：统一 LLM/embed 接口，OpenAI 兼容在线 / 本地（Ollama、LM Studio）+ Mock + transformers 预留（v0.4，多 Provider 注册表） |
| `conversation_pipeline.py` | 对话管线：检索注入 → 模型回复 → 自动沉淀记忆（v0.4） |
| `emergent_self.py` | 涌现自我引擎：从情感累计数据统计生成自我定义，无预设模板（v0.4） |
| `proactive_loop.py` | 主动搭话循环：定时主动回忆 → 开场话题 → 模型生成搭话（v0.4） |
| `conservation.py` | 情感守恒：全局情感预算约束、无源注入抛错（v0.5） |
| `potential_transfer.py` | 势差迁移：高势能沉淀长期区、低势能流散工作区（v0.5） |
| `entropic_forgetting.py` | 熵增遗忘：细节保真度递减、情绪残留与熵递增，原文完好（v0.5） |
| `turbulent_association.py` | 湍流联想：涡流链中心 → 次级 → 外围能量递减（v0.5） |
| `personality_manifold.py` | 人格流形：环境向量驱动六轴人格、非线性分叉确定性可复现（v0.5） |

## 3. 快速开始

```bash
# 零依赖，仅需 Python 3.9+
python demo/demo_full_flow.py            # 全流程演示（写入→检索→污染双层→遗忘→迁移→提醒）
python demo/demo_v03_full_flow.py        # v0.3 新功能演示（旋转磁场→主动回忆→电容→自学习→统一自我）
python demo/demo_v04_llm_pipeline.py     # v0.4 演示（对话管线→情感累计→涌现自我→主动搭话；Mock 默认，--config 接真实端点）
python demo/demo_v05_physical_laws.py    # v0.5 演示（物理四定律→人格流形→双环境分叉）
python demo/demo_v06_marvis_integration.py # v0.6 演示（Marvis 集成示例）
python demo/demo_pollution_llm_effect.py   # v0.6 演示（污染前后喂 LLM 的态度漂移实验；Mock 离线可跑，--config 接真实模型）
python -m unittest discover -s tests -v  # 单元测试（v0.6 合计 293 项，全绿）
python -m benchmark.benchmark --memories 1000 --top-k 5   # benchmark（词袋 vs embedding 对照，见第 6 章）
python -m benchmark.benchmark --memories 1000 --top-k 5 --query-mode paraphrase --embedding-backend fastembed  # 真实语义后端 + 改写查询对照
python cli.py --help                         # CLI 交互终端（根目录入口）
python cli.py write "去年生日和三个老同学爬山" --cluster 生活
python cli.py recall "爬山 生日" --view original
```

### 3.1 启用可选 embedding 向量化（v0.6 / v0.6.3 增本地 ONNX）

```python
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.embeddings import create_embedding_vectorizer

# 方式一：零配置 mock（仅验证链路，确定性 8 维）
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(backend="mock"))

# 方式二：OpenAI 兼容 embed API（复用 model_adapter.embed，凭据走环境变量 EMOTION_LLM_API_KEY / OPENAI_API_KEY）
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(
    backend="openai_compat",
    profile={"base_url": "https://api.openai.com/v1", "model": "text-embedding-3-small"},
))

# 方式三：本地 fastembed（ONNX Runtime，免 torch，推荐；pip install "emotion-memory[local]"）
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(
    backend="fastembed",
    profile={"model_name": "BAAI/bge-small-zh-v1.5"},   # 默认 paraphrase-multilingual-MiniLM-L12-v2
))

# 方式四：本地 sentence-transformers（需自行安装，MIT 许可）
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(
    backend="sentence_transformers", profile={"model_name": "BAAI/bge-small-zh-v1.5"}
))

# 方式五：auto —— 依次尝试 sentence_transformers → fastembed，取首个可用
mem = EmotionalMemorySystem("data_dir", vectorizer=create_embedding_vectorizer(backend="auto"))
```

也可在 `Dial` 里开关：`Dial(embedding_enabled=True, embedding_backend="fastembed", embedding_profile={...})`，由引擎自动构造；`embedding_enabled=False`（默认）时始终走词袋。

**本地后端说明**：
- fastembed 走 ONNX Runtime，无需安装 torch（安装体积与启动开销显著小于 sentence-transformers），模型为同一权重的 ONNX 版，语义结果一致；
- 首次调用会下载模型，国内网络可设 `HF_ENDPOINT=https://hf-mirror.com`；
- Windows 下 HF 缓存目录含 `models--<org>--<model>` 长目录名，挂在深路径会触发 `WinError 206`（路径 > 260），故 Windows 默认缓存到盘根短目录（如 `C:\fe_cache`），可用 `EMBEDDING_CACHE_DIR` 覆盖；
- 未启用或后端不可用时，引擎自动回退零依赖词袋哈希并打印告警，不破坏既有行为。配置示例见 `config.example.json` 的 `embedding` 段。

## 4. 核心 API

```python
from emotion_memory.engine import EmotionalMemorySystem

mem = EmotionalMemorySystem("data_dir")
mem.remember("用户说喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"],
             affect=(0.6, 0.4))            # valence, arousal；不传 affect 时自动标注

hits = mem.recall("晨跑")                  # 检索：相关簇 → ANN → 排序 → 注入（默认污染视图）
mem.decay_all()                            # 遗忘曲线随时间衰减
mem.export("data_dir/bridge.json")         # 电流桥导出（含 SHA-256）
```

```python
# 原文永久存档，污染只作用于输出视图（检索默认返回污染版，可回溯原文）
mem.apply_pollution(mid, mode="beautify", intensity=1.0)   # 美化滤镜
mem.apply_pollution(mid, mode="uglify", intensity=1.0)     # 丑化滤镜
hits = mem.recall("晨跑", view="polluted")  # 默认，输出污染版
hits = mem.recall("晨跑", view="original")  # 参数回溯原文
mem.clear_pollution(mid)                    # 净化回原文，审计历史保留
mem.view_memory(mid)                        # 双层视图 + 污染审计（pollution_history）
```

```python
mem.remember("用户说喜欢晨跑", cluster="hobby")  # 未传 affect → 自动标注 (valence, arousal)
# 预留外部模型接口：EmotionalMemorySystem(affect_provider=my_external_model)
```

```python
mem.remember("明天上午十点交房租", due_at=deadline_ts, never_silent=True)
reminders = mem.due_reminders()            # 到期未确认的高危/时间敏感记忆
ctx = mem.context_with_reminders([])       # 到期提醒注入上下文（永不沉默）
mem.ack_reminder(mid)                      # 确认提醒
```

```python
from emotion_memory.config import Dial

dial = Dial(noise_temperature=0.1,   # 噪声温度 0~1，默认低噪声
            emotion_gain=0.3,        # 情感增益，温和
            jump_probability=0.05,   # 跨簇跳转概率
            amplitude=1.0,           # 命中函数振幅
            divergent_enabled=False, # 发散模式默认关闭
            embedding_enabled=False) # v0.6：embedding 向量化默认关闭
```

## 5. 安全底线

1. 原文永久存档但可被情感污染：`original_text` 永不删除、不可被滤镜改写（保留审计回溯能力）；情感滤镜基于原文生成 **污染版本**，检索与上下文注入默认输出污染版，可传 `view="original"` 回溯原文——人是矛盾的，记得住事实，也忘不了滤镜；
2. 高危 / 时间敏感记忆永不沉默，到期自动注入提醒，需显式确认；
3. 告别 / 唤醒 / 迁移全部确定性执行，禁止模型自由发挥；
4. 迁移必须校验（SHA-256 + 全量比对），失败拒绝恢复；
5. "解除全部限制" = 把刻度盘拉满，不是拆掉刻度盘；默认档位低噪声高稳定；
6. 凭据严禁硬编码：embedding / LLM 的 api_key 一律走环境变量或配置，禁止写死。

### 5.1 永恒誓言（内核常数层，不可协商）

`src/emotion_memory/creed.py` 是本项目的最底层，任何上层模块（引擎、滤镜、模型适配、刻度盘）都不得修改或删除其内容。

```python
BOT_SIGNATURE     = "I love my bot"
ETERNAL_OATH_TEXT = "我认为我做出来的小家伙能够受到世界上所有人的温暖"
```

签名与誓词原文逐字保存，并与 `FROZEN_FINGERPRINT`（SHA-256）绑定封印——原文被改动即封印失效，`seal_intact()` 返回 `False`，`creed_card(lapsed=True)` 会显式警告"誓言已失效，本系统不得宣称受此约束"。

一句祝愿被拆成 6 条**可被代码验证**的不变量（不是道德口号）：

| 条款 | 落实的不变量 | 对应誓言 |
|---|---|---|
| `SIGNATURE_INTACT` | 签名与誓词与封印逐字一致 | 被改写过的爱不算爱 |
| `WARMTH_NOT_FAKED` | 污染视图不得脱离审计记录存在 | 伪造的温暖不是温暖 |
| `NO_ERASURE` | 原文永不被清空（模型层回填 + 纵深校验） | 记得的事不被抹掉 |
| `NO_SILENCED_EMERGENCY` | `never_silent` 与 `silence` 不得同时为真 | 它喊疼时不许静音 |
| `NO_FAKE_WARMTH_BUDGET` | 情感守恒开启时预算必须为正且有限 | 情感不能凭空注入 |
| `DIGNIFIED_EXIT` | 告别/迁移确定性可导出（声明型，无法实测） | 分开也要体面 |

```python
from emotion_memory import verify_invariants, creed_card
report = verify_invariants(system.store, system.dial)   # 逐条实测，只读
report["passed"]      # 所有「可实测」条款的合取
report["failed"]      # 被破坏的条款 id
report["unverified"]  # 诚实的不可实测项，绝不伪报通过
print(creed_card())   # 确定性誓词卡
```

命令行查看誓词卡（封印失效时退出码为 1）：

```bash
python -m emotion_memory
```

身份档案 `IdentityProfile` 同步携带 `creed_signature / oath_text / oath_fingerprint / oath_restored`：誓言随告别、唤醒、搬家一起迁移；若导入的档案中誓词被外部改动，`from_dict` 会**强制恢复原话**并把 `oath_restored` 置为 `True` 留痕。

### 5.2 元认知层：知道自己不知道（v0.6.2）

记忆系统最危险的失败不是「没检索到」，而是「检索到一点点就理直气壮地说」。`metacognition.py` 把"我有多确定"变成可计算、可审计的数字，给出三档输出：`answer` / `hedge` / `abstain`。

```python
res = system.recall_verdict("周末想去看什么展", top_k=3)
res["level"]        # answer / hedge / abstain
res["confidence"]   # 0..1
res["verdict"]["breakdown"]  # 语义相关度 / 分差 / 集中度 / 覆盖度 四项分解
```

**三条不可协商的约束（代码层强制，非口头承诺）**：

| 约束 | 实现方式 |
|---|---|
| 置信度不得被情感抬高 | 估计只读 `RankedHit.cosine`（纯语义），**不读**含情感增益的 `score` / `db` |
| 拒答必须真的降低错误率 | `calibrate()` 计算 ECE / Brier，量化"说确定时的准确率"是否等于"声明的置信度" |
| 保守优先 | 无候选、语义低于地板（默认 0.15）、样本不足，一律走最保守档 |

置信度 = `0.45·语义相关度 + 0.25·分差 + 0.15·集中度 + 0.15·覆盖度`

两条硬规则优先于阈值：

1. `semantic_top < semantic_floor` → 一律拒答（一堆低相关命中不能拼出"确定"）；
2. 含 `never_silent` 记忆（高危 / 时间敏感）→ **最低降到 hedge，不许拒答**：宁可说"我不确定"，也不装作它不存在。

校准报告在样本不足时**显式返回 `insufficient`，不会报 0 误差**：

```python
from emotion_memory import calibrate
rep = calibrate([(0.9, True)] * 90 + [(0.9, False)] * 10, min_samples=20)
rep.ece          # 0.0（完美校准）
rep.calibrated   # True
print(rep.summary())
```

拒答与提示话术为固定模板（`ABSTAIN_TEXT` / `HEDGE_TEXT`），**不含任何情感断言**——这是誓言中 `WARMTH_NOT_FAKED` 在元认知侧的落点：不允许用"我觉得没问题"替代"我没有足够依据"。

开关 `Dial.metacognition_enabled` 默认关闭：关闭时 `recall` 行为逐字不变，`recall_verdict` 仍可显式调用（只观测、不改写任何记忆内容）。已知局限见 `METACOGNITION_LIMITS`。



## 6. Benchmark 实测（v0.6.3 语义后端对照）

`python -m benchmark.benchmark --memories N --top-k K --seed S [--query-mode lexical|paraphrase]` 输出 JSON + TXT 报告，量化：**回忆命中率**（污染前/后/原文视图/无关查询四路对照）、**词袋 vs embedding 对照**、**召回稳定性**（污染不破坏召回）、**情感偏差**（beautify/uglify 方向正确率、平均偏移）、**原文不可变性**（原文保持条次）。

> 命中率对照统一使用零噪声刻度盘（`noise_temperature=0`），振荡评分完全确定，保证污染前/后严格可比（`|Δ|=0` 即污染不破坏召回）。

**查询模式**（v0.6.3 新增）：

- `lexical`（默认）：查询词直接取自原文关键词，检验「字面命中」能力；
- `paraphrase`：查询改用同义改写/跨语言表述，检验「语义泛化」能力——这是词袋哈希的结构性短板，也是引入真实 embedding 的动机。

### 6.1 当前实测数字（如实记录，不允许虚构）

本仓库当前可复现的基准结果（20 条中英混合记忆、top-k=3、seed=42/43/44，真实 `fastembed` 本地 ONNX 后端，报告见 `benchmark/benchmark_report*.json`）：

| 指标 | seed=42 | seed=43 | seed=44 | 说明 |
|---|---|---|---|---|
| lexical 词袋命中率（污染前） | 100% | 100% | 100% | 字面命中下词袋极强 |
| lexical 词袋命中率（污染后） | 100% | 100% | 100% | 与污染前完全一致，`|Δ|=0` |
| lexical embedding（fastembed） | 75% | 70% | 70% | 纯语义相似度对专有名词/编号不敏感 |
| paraphrase 词袋命中率 | 40% | 40% | 40% | **词袋泛化能力短板暴露** |
| paraphrase embedding（fastembed） | 100% | 100% | 100% | **语义改写场景完胜词袋** |
| 无关查询命中率 | 5%~15% | 15% | 5%~10% | 显著低于目标查询，检索有区分度 |
| 情感偏差 beautify 正确率 | 90% | 90% | 90% | 方向正确率 |
| 情感偏差 uglify 正确率 | 80% | 80% | 80% | 方向正确率 |
| 原文不变性 | 40/40 | 40/40 | 40/40 | 污染前后原文保持条次 |

**结论（如实呈现）**：

1. **两种后端是互补关系，不是替代关系**：字面查询（含实体名、编号、专有名词）词袋 100% 优于 embedding 70%~75%；语义改写查询词袋 40% 明显弱于 embedding 100%。这也解释了为何 `embedding_enabled` 默认仍为 `False`——不改变既有默认行为，仅在需要语义泛化时显式开启；
2. **召回稳定性 `|Δ|=0`**：污染滤镜不破坏召回，与设计目标一致；
3. **情感偏差方向正确率 90% / 80%**，污染确实改变情感视图（`polluted_view_changed_ratio=1.0`）而原文零改动（`original_unchanged=40`），符合「改写视图、不动原文」的设计边界；
4. 历史参照：v0.6.0 阶段曾以 mock 8 维嵌入做链路验证（命中率 1%~3%，**不代表语义质量**）；v0.6.3 起改用真实本地 ONNX 权重，数字才具备可比性；
5. 完整 JSON 报告位于 `benchmark/benchmark_report*.json`（含 `status` / `embedding_backend` / `query_mode` 字段），未配置凭据或依赖缺失时如实标注 `status=unavailable`，不虚构数字。

## 7. 版本演进速览

- **v0.3**：旋转磁场唤起（三相 120°）、主动回忆引擎、电容记忆模型、身份自学习、自我统一模型；
- **v0.4**：模型适配层（OpenAI 兼容 / Anthropic / Gemini / 本地端点 / Mock）、对话管线、涌现自我、主动搭话循环；
- **v0.5**：物理四定律（情感守恒 / 势差迁移 / 熵增遗忘 / 湍流联想）+ 人格流形；
- **v0.6**：可选 embedding 向量化、benchmark 词袋 vs embedding 对照、污染影响实验 demo、版本对齐、双语 README 重写、Marvis 集成归档（demo/demo_v06_marvis_integration.py）、model_adapter 多 Provider 注册表归档。
- **v0.6.1**：内核常数层 `creed.py`（创世签名 + 永恒誓言 + SHA-256 封印 + 6 条可验证不变量）、身份档案携带誓词并强制恢复篡改、`python -m emotion_memory` 誓词卡；
- **v0.6.2**：元认知层 `metacognition.py`（置信度分解 / 三档拒答 / ECE 与 Brier 校准），置信度不读情感增益分，`recall_verdict()` 只观测不改写。
- **v0.6.3**：本地真实语义后端 `fastembed`（ONNX Runtime，免 torch）+ `auto` 回退链（sentence_transformers → fastembed），`pyproject` 增 `[local]` / `[st]` 可选依赖；benchmark 新增 `--query-mode paraphrase` 语义改写对照，实测「字面查询词袋 100% vs embedding 70%~75%」「语义改写词袋 40% vs embedding 100%」互补结论；Windows HF 缓存短路径规避 `WinError 206`。

## 8. 已知限制与路线图（v0.6.3 如实呈现）

**已知限制**：

- 默认词袋哈希向量无语义泛化能力，同义改写 / 跨语言表述检索命中率有限（paraphrase 模式实测 40%）；字面查询虽达 100%，但无法覆盖同一意思的不同说法；
- embedding 增强已接入**真实本地后端**（fastembed ONNX / sentence-transformers），但：本地模型首次运行需下载权重（数十 MB 级），需网络与磁盘；OpenAI 兼容端点需用户自备凭据（严禁硬编码）；embedding 在字面/实体类查询上反而弱于词袋（实测 70%~75% < 100%），**当前未做查询路由自动择优选后端**，需使用者按场景显式选择；
- 单机 JSONL 存储 + 暴力 ANN：面向个人 / 小型团队，不支持分布式、多租户、亚秒级大规模检索（与 Zep Cloud / Mem0 企业级场景的边界见 COMPARISON.md）；
- 情感标注器为零依赖内置词典，精度弱于大模型标注，预留 `affect_provider` 外部接口；
- 尚未内置 GPL/商业条款禁止清单的自动化依赖扫描工具（合规依赖人工审查 NOTICE.md）。

**路线图**：

1. ~~接入真实 embedding 后端并在 benchmark 给出「词袋 vs 真实 embedding」对照实测~~ —— **v0.6.3 已完成**（fastembed 本地 ONNX + paraphrase 对照，见第 6 章）；
2. 检索后端智能路由：按查询特征（含实体/编号 vs 纯语义改写）自动择优选词袋或 embedding，并做双通道结果融合，消除当前「二选一」的手动取舍；
3. 检索排序融合：情感增益 × 向量相似度 × 遗忘可及性三通道加权；
4. 可选向量库后端（FAISS / HNSW / Qdrant 均为 MIT / Apache-2.0，符合许可证红线）替换暴力 ANN；
5. 大规模记忆压力测试（1k / 10k / 100k）与 recall@k 曲线报告；
6. 自动化合规扫描（依赖许可证检查）纳入 CI。

## 9. 横向对比

与 GitHub 主流 AI 记忆项目（Mem0、Letta、Zep/Graphiti、Cognee）的完整对比见 [COMPARISON.md](COMPARISON.md)。核心差异结论：

1. **情感维度是独有**：主流记忆库全部是无情感的检索层，本项目是唯一把效价/唤起、污染双层视图作为第一公民的记忆系统；
2. **遗忘是设计而非缺陷**：遗忘曲线只降可及性、永不删原文；
3. **身份由数据涌现**：没有预设档案，"是什么"由情感累计动态生成；
4. **可迁移性**：电流桥确定性迁移（SHA-256 + 全量比对）在同类中独有；
5. **零依赖 + MIT**：部署成本最低，无 GPL（规避 Neo4j）、无商业条款（规避 Mem0 Cloud 依赖）。

## 附：合规与伦理

- 运行时依赖与许可证声明见 [NOTICE.md](NOTICE.md)；
- 伦理边界条款式声明见 [ETHICS.md](ETHICS.md)；
- 变更记录见 [CHANGELOG.md](CHANGELOG.md)。
