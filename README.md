# EmotionMemory — 类人情感记忆系统

> 一个用 **振荡动力学 + 对数强度编码 + 可迁移身份** 模拟人类情感记忆的开源记忆系统（当前版本 **v0.6.0**）。

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
| Embedding 向量化 | OpenAI 兼容 embed API / sentence-transformers | 随模型许可（API）/ MIT（本地） | 可选 `EmbeddingVectorizer`（v0.6），默认关闭，不引入运行时依赖 |

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

对应代码结构（`src/emotion_memory/`）：

| 模块 | 职责 |
|---|---|
| `models.py` | 记忆 / 情感标签 / 簇元数据数据模型 |
| `config.py` | 自由度刻度盘 Dial（默认低噪声高稳定；v0.6 含 embedding 开关） |
| `embeddings.py` | **（v0.6）可选 EmbeddingVectorizer**：mock / openai_compat / local_compat / gemini / sentence_transformers，与词袋同接口 |
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
python -m unittest discover -s tests -v  # 单元测试（v0.6 合计 235 项，全绿）
python -m benchmark.benchmark --memories 1000 --top-k 5   # benchmark（词袋 vs embedding 对照，见第 6 章）
python cli.py --help                         # CLI 交互终端（根目录入口）
python cli.py write "去年生日和三个老同学爬山" --cluster 生活
python cli.py recall "爬山 生日" --view original
```

### 3.1 启用可选 embedding 向量化（v0.6）

```python
from emotion_memory.engine import EmotionalMemorySystem
from emotion_memory.embeddings import make_vectorizer

# 方式一：零配置 mock（仅验证链路，确定性 8 维）
mem = EmotionalMemorySystem("data_dir", vectorizer=make_vectorizer(backend="mock"))

# 方式二：OpenAI 兼容 embed API（复用 model_adapter.embed，凭据走环境变量 EMOTION_LLM_API_KEY / OPENAI_API_KEY）
mem = EmotionalMemorySystem("data_dir", vectorizer=make_vectorizer(
    backend="openai_compat",
    base_url="https://api.openai.com/v1",
    model="text-embedding-3-small",
))

# 方式三：本地 sentence-transformers（需自行安装，MIT 许可；未安装时自动回退词袋并告警）
mem = EmotionalMemorySystem("data_dir", vectorizer=make_vectorizer(
    backend="sentence_transformers", model_name="BAAI/bge-small-zh-v1.5"
))
```

未启用或后端不可用时，引擎自动回退零依赖词袋哈希并打印告警，不破坏既有行为。配置示例见 `config.example.json` 的 `embedding` 段。

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

## 6. Benchmark 实测（v0.6 对照升级）

`python -m benchmark.benchmark --memories N --top-k K --seed S` 输出 JSON + TXT 报告，量化：**回忆命中率**（污染前/后/原文视图/无关查询四路对照）、**词袋 vs embedding 对照**、**召回稳定性**（污染不破坏召回）、**情感偏差**（beautify/uglify 方向正确率、平均偏移）、**原文不可变性**（原文保持条次）。

> 命中率对照统一使用零噪声刻度盘（`noise_temperature=0`），振荡评分完全确定，保证污染前/后严格可比（`|Δ|=0` 即污染不破坏召回）。

### 6.1 当前实测数字（如实记录，不允许虚构）

本仓库当前可复现的基准结果（300 条中英混合记忆、top-k=5、多 seed 实测，见 `benchmark/benchmark_report*.json`）：

| 指标 | seed=1 | seed=2 | seed=3 | 说明 |
|---|---|---|---|---|
| 词袋命中率（污染前） | 31.00% | 30.67% | 32.67% | 词袋基线（约 0.255~0.33 区间，seed 敏感） |
| 词袋命中率（污染后） | 31.00% | 30.67% | 32.67% | 与污染前完全一致，`|Δ|=0` |
| 原文视图命中率 | 31.00% | 30.67% | 32.67% | 与污染前一致 |
| 无关查询命中率 | 1.33% | 2.33% | 2.00% | 显著低于目标查询，检索有区分度 |
| embedding(mock) 命中率 | 2.67% | 1.00% | 1.33% | **仅验证链路可用**，不代表真实语义质量 |
| 情感偏差 beautify 正确率 | 90.67% | — | — | 方向正确率（seed=1） |
| 情感偏差 uglify 正确率 | 80.67% | — | — | 方向正确率（seed=1） |
| 原文不变性 | 600/600 | — | — | 污染前后原文保持条次 |

**重要说明（如实呈现）**：

1. **词袋基线约 0.255** 为项目规划时的目标参考值；本仓库当前实测（300 条 × 多 seed）命中率在 **30.7%~32.7%** 之间，波动源于随机 seed 与中英混合语料的哈希碰撞分布，均高于目标参考值，未低于 0.255；
2. **embedding 对照为 mock 后端**：mock 是确定性 8 维嵌入，只验证「EmbeddingVectorizer 接入链路 + benchmark 对照管道」可用，**不代表真实语义嵌入的检索质量**；真实后端（OpenAI 兼容 / sentence-transformers）未配置凭据或依赖时，benchmark 如实标注 `status=unavailable`，不虚构数字；
3. 召回稳定性 `|Δ|=0`：污染滤镜不破坏召回，与设计目标一致；
4. 完整 JSON 报告位于 `benchmark/benchmark_report.json`（含 `status` 字段）。

## 7. 版本演进速览

- **v0.3**：旋转磁场唤起（三相 120°）、主动回忆引擎、电容记忆模型、身份自学习、自我统一模型；
- **v0.4**：模型适配层（OpenAI 兼容 / Anthropic / Gemini / 本地端点 / Mock）、对话管线、涌现自我、主动搭话循环；
- **v0.5**：物理四定律（情感守恒 / 势差迁移 / 熵增遗忘 / 湍流联想）+ 人格流形；
- **v0.6**：可选 embedding 向量化、benchmark 词袋 vs embedding 对照、污染影响实验 demo、版本对齐、双语 README 重写、Marvis 集成归档（demo/demo_v06_marvis_integration.py）、model_adapter 多 Provider 注册表归档。

## 8. 已知限制与路线图（v0.6 如实呈现）

**已知限制**：

- 默认词袋哈希向量无语义泛化能力，同义改写 / 跨语言表述检索命中率有限（当前实测约 30%，seed 敏感）；
- embedding 增强目前为**可选后端 + mock 链路验证**：真实 OpenAI 兼容端点需用户配置凭据（严禁硬编码），sentence-transformers 需自行安装（MIT 许可）；未接入时 benchmark 如实标注 `unavailable`；
- 单机 JSONL 存储 + 暴力 ANN：面向个人 / 小型团队，不支持分布式、多租户、亚秒级大规模检索（与 Zep Cloud / Mem0 企业级场景的边界见 COMPARISON.md）；
- 情感标注器为零依赖内置词典，精度弱于大模型标注，预留 `affect_provider` 外部接口；
- 尚未内置 GPL/商业条款禁止清单的自动化依赖扫描工具（合规依赖人工审查 NOTICE.md）。

**路线图**：

1. 接入真实 embedding 后端（OpenAI 兼容 / sentence-transformers / 本地向量库），以语义命中率替代词袋哈希，并在 benchmark 中给出「词袋 vs 真实 embedding」对照实测；
2. 检索排序融合：情感增益 × 向量相似度 × 遗忘可及性三通道加权；
3. 可选向量库后端（FAISS / HNSW / Qdrant 均为 MIT / Apache-2.0，符合许可证红线）替换暴力 ANN；
4. 大规模记忆压力测试（1k / 10k / 100k）与 recall@k 曲线报告；
5. 自动化合规扫描（依赖许可证检查）纳入 CI。

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
