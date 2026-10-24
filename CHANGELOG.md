# Changelog

本项目变更记录按版本列出事实型条目（版本标题、新增模块、测试数演进）。不包含日期、commit 编号与作者信息。

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
