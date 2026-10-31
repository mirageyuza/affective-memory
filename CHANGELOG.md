# Changelog

本项目变更记录按版本列出事实型条目（版本标题、新增模块、测试数演进）。不包含日期、commit 编号与作者信息。

## v0.6.0 — 可选 embedding 向量化增强、benchmark 对照升级、污染影响实验

- 新增 `src/emotion_memory/embeddings.py`：可选 `EmbeddingVectorizer`，通过 OpenAI 兼容 embed API（复用 `model_adapter.embed()`）或本地 sentence-transformers 接入；与 `BagOfWordsVectorizer` 同接口（vectorize / dimension），默认仍走零依赖词袋哈希，embedding 为可插拔增强，不破坏现有接口与单测。
- `config.py` 的 `Dial` 新增 `embedding_enabled / embedding_backend / embedding_profile` 开关（默认关闭）；`engine.py` 支持 `vectorizer=` 注入，不可用时自动回退词袋并告警。
- 新增 `config.example.json` 的 `embedding` 配置段（enabled / backend / profile，api_key 严禁硬编码）。
- `benchmark/benchmark.py` 升级：新增「词袋 vs embedding」对照维度，真实结果写入 `benchmark/benchmark_report.json`；词袋基线命中率约 0.255 如实记录；embedding 后端不可用时如实标注 `status=unavailable`。
- 新增 demo：`demo/demo_pollution_llm_effect.py`（污染前后喂给 LLM 的态度漂移实验，内置 Mock 可离线运行，`--config` 接真实模型）。
- 归档超前代码：`demo/demo_v06_marvis_integration.py`（Marvis 集成 demo）、`model_adapter` 多 Provider 注册表（OpenAI 兼容全家 / Anthropic / Gemini / 本地端点）。
- 新增单测 12 项：`tests/test_v06_embeddings.py`。
- 测试数演进：225 → 235（`unittest discover` 实测 235 项，全绿）。
- 版本对齐：`pyproject.toml` version = 0.6.0。
- 文档同步：重写 `README.md` / `README_EN.md` 至 v0.6 双语对齐，补充 embedding 可选增强、词袋 vs embedding 对照基准与污染影响实验说明；`config.example.json` 同步新增 `embedding` 配置段。

## v0.5.0 — 物理四定律与人格流形

- 新增 `src/emotion_memory/conservation.py`：情感守恒，全局情感预算约束，无源注入抛错。
- 新增 `src/emotion_memory/potential_transfer.py`：势差迁移，高势能记忆沉淀长期区、低势能流散工作区。
- 新增 `src/emotion_memory/entropic_forgetting.py`：熵增遗忘，细节保真度递减、情绪残留与熵递增，原文完好。
- 新增 `src/emotion_memory/turbulent_association.py`：湍流联想，涡流链中心 → 次级 → 外围能量递减。
- 新增 `src/emotion_memory/personality_manifold.py`：人格流形，环境向量驱动六轴人格，非线性分叉确定性可复现。
- 新增 demo：`demo/demo_v05_physical_laws.py`（五环闭环）。
- 新增单测 70 项：`tests/test_v05_*.py`（含引擎集成）。
- 测试数演进：133 → 203。

## 适配器更新（v0.5.0 之后增量，未升版本号）

- `model_adapter.py` 升级为多 Provider 注册表：`MODEL_PROVIDERS` 按 provider 名路由适配器，新增 AI 零改动接入。
- 支持 OpenAI 兼容全家（OpenAI / DeepSeek / 通义 Qwen / Kimi Moonshot / 智谱 GLM / MiniMax / 豆包火山方舟 / 文心千帆兼容模式）、Anthropic Claude（Messages API）、Google Gemini（generateContent API）、本地端点（Ollama / LM Studio / llama.cpp）。
- 凭据一律走环境变量（OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY 等 provider 专属变量），未配置时抛明确错误提示。
- 新增 `.env.example`：各 provider 端点 / 默认模型 / 环境变量对照表。
- 新增单测 22 项：`tests/test_model_adapter_providers.py`（Mock 网络层验证请求构造）。
- 测试数演进：203 → 225。

## v0.4.0 — 模型接入层与涌现自我

- 新增 `src/emotion_memory/model_adapter.py`：统一 LLM 接口，支持 OpenAI 兼容在线 API、本地端点（Ollama / LM Studio）、内置 Mock，预留 transformers 直连接口；HTTP 使用标准库 urllib，零第三方运行时依赖。
- 新增 `src/emotion_memory/conversation_pipeline.py`：用户消息 → 记忆检索注入（top-k / 沉默层过滤 / 情感污染视图）→ 模型回复 → 自动沉淀（写入存储 + 自动情感标注 + 电容充电 + 万物皆我吸收 + 情感数据累计）。
- 新增 `src/emotion_memory/emergent_self.py`：由累计情感数据统计涌现自我定义（性格维度、使命、自我概念、技能、兴趣），无预设模板；`export_json()` / `rebuild_from()` 支持唤醒后从累计数据重建身份。
- 新增 `src/emotion_memory/proactive_loop.py`：定时主动搭话循环，生成开场话题并沉淀为沉默记忆。
- 新增 `config.example.json`：在线与本地两套端点示例，空 key 占位，凭据严禁硬编码。
- 新增 demo：`demo/demo_v04_llm_pipeline.py`（Mock 默认，支持 `--config` 接真实端点）。
- 新增单测 27 项：`tests/test_v04_model_adapter.py` / `tests/test_v04_conversation_pipeline.py` / `tests/test_v04_emergent_self.py`。
- 测试数演进：106 → 133。

## v0.3.0 — 五大新功能

- 新增 `src/emotion_memory/algorithms/three_phase.py`：旋转磁场唤起调度，A/B/C 三相通道 120° 相位差、顺序启动，输出连续联想流，不破坏 H(t) 接口。
- 新增 `src/emotion_memory/retrieval/proactive.py`：主动回忆引擎，定时 / 触发器驱动自主想起搭话，开场话题为确定性模板。
- 新增 `src/emotion_memory/algorithms/capacitor.py`：电容记忆模型（充电 / 放电 / 漏电 / 阈值），standby 备选池。
- 新增 `src/emotion_memory/self_learning.py`：身份自学习，EMA 平滑演化五维人格与兴趣 / 厌恶 / 技能 / 使命画像，主动提问确认。
- 新增 `src/emotion_memory/unified_self.py`：自我统一模型，1 核心 + 万我分身 + 万物皆我，cosφ 自我认知报告。
- 新增 demo：`demo/demo_v03_full_flow.py`。
- 新增单测 38 项：`tests/test_v03_*.py`。
- 测试数演进：68 → 106。

## v0.2.0 — 解除原文篡改红线与功能扩充

- 红线调整：原文永久存档但可被情感污染；检索默认输出污染视图，可传参数查看原文。
- 新增 `src/emotion_memory/algorithms/pollution.py`：情感污染双层视图（原文 / 污染版），强度刻度 0~2，污染审计历史。
- 新增 `src/emotion_memory/algorithms/affect_analyzer.py`：零依赖自动情感标注（valence, arousal），预留外部模型接口。
- 新增 `cli.py`：CLI 交互终端（write / recall / ask / status / dial / pollute / purify / export / import / due / ack）。
- 新增 `benchmark/`：回忆命中率、召回稳定性、情感偏差、原文不可变性四路对照。
- 新增 `reminder.py`：到期提醒，高危 / 时间敏感记忆永不沉默。
- 测试数演进：39 → 68。

## v0.1.0 — MVP 初始版本

- 存储层 `src/emotion_memory/store/storage.py`：原文 JSONL 磁盘持久化。
- 向量索引 `src/emotion_memory/store/vector_index.py`：词袋哈希向量 + 暴力 ANN（接口可替换 FAISS / HNSW）。
- 图索引 `src/emotion_memory/store/graph_index.py`：轻量实体关系图。
- 遗忘曲线 `algorithms/forgetting.py`、振荡命中 `algorithms/hit_function.py`、情感增益 `algorithms/affect.py`。
- 沉默层 `silence.py`（可提及 / 沉默两级 + 直问解锁 + 高危永不沉默）、发散 `divergent.py`、并发 `concurrent.py`。
- 电流桥 `bridge.py`：导出 / 导入 + SHA-256 校验 + 失败拒绝恢复。
- 身份层 `identity.py`：身份档案 + 生命周期状态机（告别 / 唤醒 / 迁移）。
- 引擎 `engine.py`、demo：`demo/demo_full_flow.py`。
- 初始单测 39 项。
