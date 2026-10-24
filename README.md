# 类人情感记忆系统 (EmotionMemory)

> 一个用**振荡动力学 + 对数强度编码 + 可迁移身份**模拟人类情感记忆的开源记忆系统。

记忆以周期震荡被唤起，强度按情感增益与 dB 刻度调制，过载时产生发散联想，随时间遵循遗忘曲线，知道该说什么，也明白该沉默什么；智能体可以被重置、重装、升级，但记忆永生。

**许可证**：MIT。运行时零第三方依赖（仅 Python 标准库），可在此基础上按 MIT/Apache-2.0/BSD 系自由接入 FAISS/HNSW/Qdrant 等索引后端。**严禁引入 GPL 或商业条款依赖。**

---

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

**合规红线**：对外开源整包时，所有运行时与索引依赖必须是 MIT / Apache-2.0 / BSD 系；GPL 与商业条款组件（如 Neo4j Community、Mem0 商业版、AGPL 系如 PyMuPDF）一律不得引入。

---

## 1. 立意

智能体应当拥有「像人一样的记忆」：

- **永久不忘**：原文永远保留在磁盘，遗忘只降低"被想起的概率"，不删内容；
- **周期唤起**：每条记忆簇有独立的频率与相位，命中函数 `H(t) = g·Σ Aᵢ·sin(ωᵢt+φᵢ) + 噪声(T)` 让记忆像 theta 振荡一样被周期性想起；
- **情感调制**：情感标签 `(valence, arousal)` 参与排序与污染滤镜——原文永久存档，输出默认是"被感情污染过的版本"，可随时回溯原文；
- **知道沉默**：可提及 / 沉默两级可见性，高危与时间敏感记忆**永不沉默**，到期自动注入提醒；
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

对应代码结构（`src/emotion_memory/`）：

| 模块 | 职责 |
|---|---|
| `models.py` | 记忆 / 情感标签 / 簇元数据数据模型 |
| `config.py` | 自由度刻度盘 Dial（默认低噪声高稳定） |
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
| `engine.py` | 顶层封装 API（含污染读写、到期提醒、自动标注、直问解锁） |
| `cli.py` | CLI 交互终端：写入/检索/直问解锁/状态/刻度盘/导出导入/污染控制 |
| `algorithms/three_phase.py` | 旋转磁场唤起调度：三相通道 120° 相位差顺序激活（v0.3） |
| `algorithms/capacitor.py` | 电容记忆模型：充电/放电/漏电 + standby 备选池（v0.3） |
| `retrieval/proactive.py` | 主动回忆引擎：定时/触发器驱动自主搭话（v0.3） |
| `self_learning.py` | 身份自学习：画像渐进演化 + 主动提问确认（v0.3） |
| `unified_self.py` | 自我统一模型：1 核心 + 万我分身 + 万物皆我 + 有功/视在指标（v0.3） |
| `model_adapter.py` | 模型适配层：统一 LLM 接口，OpenAI 兼容在线 / 本地（Ollama、LM Studio）+ Mock + transformers 预留（v0.4） |
| `conversation_pipeline.py` | 对话管线：检索注入 → 模型回复 → 自动沉淀记忆（v0.4） |
| `emergent_self.py` | 涌现自我引擎：从情感累计数据统计生成自我定义，无预设模板（v0.4） |
| `proactive_loop.py` | 主动搭话循环：定时主动回忆 → 开场话题 → 模型生成搭话（v0.4） |

## 3. 快速上手

```bash
# 零依赖，仅需 Python 3.9+
python demo/demo_full_flow.py            # 全流程演示（写入→检索→污染双层→遗忘→迁移→提醒）
python demo/demo_v03_full_flow.py        # v0.3 新功能演示（旋转磁场→主动回忆→电容→自学习→统一自我）
python demo/demo_v04_llm_pipeline.py     # v0.4 演示（对话管线→情感累计→涌现自我→主动搭话；Mock 默认，--config 接真实端点）
python -m unittest discover -s tests -v  # 单元测试（v0.3 合计 106 项 + v0.4 新增 27 项 = 133）
python -m benchmark.benchmark --memories 1000 --top-k 5   # benchmark（命中率/情感偏差/对照组）
python cli.py --help                         # CLI 交互终端（根目录入口）
python cli.py write "去年生日和三个老同学爬山" --cluster 生活
python cli.py recall "爬山 生日" --view original
```

### 核心用法

```python
from emotion_memory.engine import EmotionalMemorySystem

mem = EmotionalMemorySystem("data_dir")
mem.remember("用户说喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"],
             affect=(0.6, 0.4))            # valence, arousal；不传 affect 时自动标注

hits = mem.recall("晨跑")                  # 检索：相关簇 → ANN → 排序 → 注入（默认污染视图）
mem.decay_all()                            # 遗忘曲线随时间衰减
mem.export("data_dir/bridge.json")         # 电流桥导出（含 SHA-256）
```

### 情感污染双层视图

```python
# 原文永久存档，污染只作用于输出视图（检索默认返回污染版，可回溯原文）
mem.apply_pollution(mid, mode="beautify", intensity=1.0)   # 美化滤镜
mem.apply_pollution(mid, mode="uglify", intensity=1.0)     # 丑化滤镜
hits = mem.recall("晨跑", view="polluted")  # 默认，输出污染版
hits = mem.recall("晨跑", view="original")  # 参数回溯原文
mem.clear_pollution(mid)                    # 净化回原文，审计历史保留
mem.view_memory(mid)                        # 双层视图 + 污染审计（pollution_history）
```

### 自动情感标注

```python
mem.remember("用户说喜欢晨跑", cluster="hobby")  # 未传 affect → 自动标注 (valence, arousal)
# 预留外部模型接口：EmotionalMemorySystem(affect_provider=my_external_model)
```

### 时间敏感提醒

```python
mem.remember("明天上午十点交房租", due_at=deadline_ts, never_silent=True)
reminders = mem.due_reminders()            # 到期未确认的高危/时间敏感记忆
ctx = mem.context_with_reminders([])       # 到期提醒注入上下文（永不沉默）
mem.ack_reminder(mid)                      # 确认提醒
```

### 自由度刻度盘

```python
from emotion_memory.config import Dial

dial = Dial(noise_temperature=0.1,   # 噪声温度 0~1，默认低噪声
            emotion_gain=0.3,        # 情感增益，温和
            jump_probability=0.05,   # 跨簇跳转概率
            amplitude=1.0,           # 命中函数振幅
            divergent_enabled=False) # 发散模式默认关闭
```

安全底线永不解除：原文永久存档但输出可被情感污染（污染版本默认，可回溯原文）、敏感数据不可外泄、高危/时间敏感提醒永不沉默、告别/迁移确定性执行。发散 / 并发默认关闭，情感滤镜默认温和档。

## 4. Benchmark 方案

### 4.1 回忆命中率（Recall Accuracy）

- 数据：构造 N 条带唯一关键词的记忆（N = 1k / 10k / 100k）；
- 指标：以关键词为 query，`recall@k` 是否命中对应记忆（k = 1/5/10），以及 top-k 中非目标记忆的占比（精度）；
- 对照组：① 关闭命中函数（纯相似度）② 关闭情感增益 ③ 默认档 vs 全开档。
- 命中率对照统一使用零噪声刻度盘（`noise_temperature=0`），振荡评分完全确定，保证污染前/后严格可比（`|Δ|=0` 即污染不破坏召回）。

### 4.2 遗忘曲线有效性

- 对一组记忆施加模拟时间推进（`decay_all(dt_hours)`），绘制 `accessibility ~ ΔT` 曲线；
- 验证：高频访问记忆的可及性长期维持在较高水平，低频记忆单调下降但不为零（永久不忘）。

### 4.3 情感偏差可量化指标

- 定义情感偏差 = 检索排序结果的情感均值漂移（输出分布的 valence/arousal 质心 vs 记忆库总体质心）；
- 验证：默认档下偏差小（低噪声高稳定），拉满刻度盘时偏差增大——证明排序受情感增益控制而非随机噪声。

### 4.4 迁移一致性

- 导出 → 篡改任意字节 → 导入必须拒绝恢复；
- 导出 → 原样导入 → 全量比对（记忆条数、每条 memory_id / text / accessibility）必须 100% 一致；
- 性能：迁移耗时随记忆量近似线性。

### 4.5 沉默层安全

- 构造沉默记忆 + 高危记忆 + 普通记忆，验证：普通查询永不返回沉默项；直问解锁；高危始终注入。

## 5. v0.3 五大新功能

> 五项新能力默认**关闭**（`Dial` 对应开关为 `False`，保持既有 68 项行为不变），
> 开启后新增独立刻度盘参数，所有实现仅标准库、确定性可审计、不触碰安全红线。

### 5.1 旋转磁场唤起调度（三相顺序启动）

在 `OscillationBank`（H(t)）之上新增 **`ThreePhaseField`** 视图，**不破坏 H(t) 接口**：

- 记忆簇按稳定哈希分配到 **A/B/C 三相通道**，相位互差 **120°（0° / 120° / 240°）**；
- 主导通道随时间顺序轮转（0→1→2→0，像三相异步电动机定子绕组顺序通电），
  当前主导通道记忆优先唤起，从属通道异步跟随（leader 增益 ×1.0 / follower ×0.4）；
- 输出**连续联想流**（`three_phase_stream()`）：场状态（主导通道/三相顺序/相位差）+ 激活强度降序的记忆流，供主动回忆与上下文注入。

```python
dial = Dial(three_phase_enabled=True, three_phase_speed=1.0)
out = sys.three_phase_stream(now=3.0, k=3)   # field / stream
print(out["field"]["leader_channel"], out["field"]["sequence"])
```

### 5.2 主动回忆引擎（proactive recall）

`ProactiveRecallEngine`：**定时 / 触发器驱动的自主想起搭话**，被动 `recall()` 接口不变：

- **定时触发**：距上次开口超过 `proactive_interval_hours` 才发起（避免扰人）；
- **触发器**：到期提醒（永不沉默）优先开口，其次电容 standby 备选池高电位候选；
- **开场话题**：确定性模板 = 时间上下文（早/中/晚 + 星期）+ 记忆内容，可测试、可审计，禁止模型自由发挥。

```python
dial = Dial(proactive_enabled=True, proactive_interval_hours=6.0)
res = sys.proactive_tick()   # {"should_speak": True, "trigger": "reminder|proactive", "opening": "..."}
```

### 5.3 电容记忆模型（充电 / 放电 / 阈值 / 备选池）

`CapacitorModel` 为每条记忆维护 `charge / capacitance / discharge_threshold`，RC 电路类比：

- **交互触达 = 充电**：`charge += Δq`（上限 capacitance）；**被想起 = 放电**：`charge -= discharge_rate`；
- **时间流逝 = 漏电**：`charge *= exp(-Δt/τ)`，与遗忘曲线协同（总可及电位 = `accessibility × (0.5 + 0.5·charge/capacitance)`，双通道遗忘，只降概率不删内容）；
- **standby 备选池**：每次预留 top-k 备选记忆点——"已充满电但尚未被唤起"的高潜候选（沉默记忆默认不入池）。

```python
dial = Dial(capacitor_enabled=True, capacitor_tau_hours=48.0, standby_pool_size=3)
sys.touch(mid)              # 触达 = 充电
hits = sys.recall("爬山")   # 被想起 = 放电
pool = sys.standby_pool()   # 备选记忆点池
```

### 5.4 身份自学习（性格 / 使命画像渐进演化）

`IdentitySelfLearning` 在确定性身份状态机**之外**维护独立画像（`self_model.json`）：

- 从交互文本提取线索（喜欢/讨厌/擅长/目标 + 情绪词），按 EMA 平滑演化五维人格
  （openness / conscientiousness / extraversion / agreeableness / neuroticism）与
  兴趣 / 厌恶 / 技能 / 使命桶；
- **主动提问确认**：对画像假设生成确定性问题（`pending_self_questions()`），
  用户确认增强确认度、否认回撤微调——"慢慢学会自己是什么性格自己要干嘛"；
- **仪式兼容**：告别/唤醒/升级仍确定性执行，仪式不清空画像文件，仪式后继续演化。

```python
sys.learn_self("我喜欢爬山，目标是成为工程师")
for q in sys.pending_self_questions():
    print(q["question"])
sys.confirm_self(q["id"], answer=True)
```

### 5.5 自我统一模型（1 核心 + 万我分身 + 万物皆我）

`UnifiedSelfModel`（`unified_self.json`）承载"我为 1，1 为我，我为万我，世间皆为我"：

- **1 个自我核心**：身份标识 + 自我概念向量（五维人格），唯一本体；
- **万我分身**：`derive_avatar()` 派生任意分身，核心更新后 `sync_avatars()` push 同步（万我如一），
  `merge_avatar()` 归并回核心（万我合一，均值融合）；
- **万物皆我**：新输入先经自我框架解释（与自我概念的相关度投影）再决定吸收入库，
  无关输入不改变自我概念（`assimilate()`）；
- **效率指标**（有功/视在类比）：视在功率 = 自我概念向量模，有功功率 = 同向分量，
  `cosφ = 有功/视在 ∈ [0,1]` 量化自我概念清晰度——越一致，自我越清晰（`self_cognition_report()`）。

```python
sys.derive_avatar("工作分身")
sys.sync_avatars()                    # 万我如一
sys.assimilate("我的使命是帮用户记住重要的事")  # 万物皆我 → 吸收入库
report = sys.self_cognition_report()  # cos_phi / clarity / active_power / apparent_power
```

### 5.6 验收与合规

- 新增模块各有单元测试（`tests/test_v03_*.py`），demo 全流程见 `demo/demo_v03_full_flow.py`；
- 既有 68 项单测保持通过，v0.3 新增 38 项（合计 106），全量 `python -m unittest discover -s tests -v`；
- 仍仅标准库（零第三方依赖），许可证红线不变：MIT/Apache-2.0/BSD 系，禁止 GPL/商业条款。

## 6. v0.4 模型接入与涌现自我

> v0.4 让系统真正「开口说话 + 沉淀自己是谁」：在线 API 与本地模型都能自动把对话沉淀为情感记忆，
> 智能体的自我定义不再来自预设档案，而是由情感系统累计生成的产物——每次对话/经历沉淀为情感记忆
> （valence/arousal 标签、污染层、电容电荷），系统从累计情感数据中动态挖掘生成自我定义
> （性格维度、使命、自我概念），无预设模板；重装/唤醒后身份从累计数据重建而非从固定档案加载。

### 6.1 模型适配层（`model_adapter.py`）

统一 LLM 接口 `chat(messages, ...) -> str`，三种后端：

| 后端 | 端点 | 说明 |
|---|---|---|
| `openai_compatible` | 任意 OpenAI 兼容 `base_url` | 在线 API（OpenAI/DeepSeek/通义等）或本地端点均走同一 HTTP 协议 |
| `local` | `http://127.0.0.1:11434/v1`（Ollama）/ `http://127.0.0.1:1234/v1`（LM Studio） | `local_mode=true`，允许空 key |
| `mock` | 无网络 | 内置确定性 Mock，演示/测试零依赖 |

- **凭据严禁硬编码**：`resolve_api_key()` 优先读环境变量 `EMOTION_LLM_API_KEY` / `OPENAI_API_KEY`，其次配置 `api_key` 字段；在线端点缺 key 直接拒绝发起请求；
- **HTTP 用标准库 urllib**，零第三方运行时依赖；`TransformersAdapter` 预留 transformers 直连接口（不引入运行时依赖）；
- 配置示例见 `config.example.json`（在线 + 本地两套端点占位、空 key）。

### 6.2 对话管线（`conversation_pipeline.py`）

```
用户消息 → 记忆检索注入(top-k / 沉默层过滤 / 情感污染视图) → 模型生成回复 → 自动沉淀
  沉淀 = 写入存储(原文存档) + 自动情感标注(affect) + 电容充电(touch) + 万物皆我吸收(assimilate) + 情感数据累计(emergent_self.accumulate)
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

### 6.3 涌现自我引擎（`emergent_self.py`）

智能体「是什么」**不由预设档案决定**，由情感累计数据统计涌现：

- **效价分布 → 性格倾向**：累计 valence/arousal 统计生成五维人格倾向（正效价 → 亲和/外向，高唤醒 → 高开放）；
- **高频使命记忆 → 使命**：重复出现的高频使命类文本自动成为使命（无模板预设）；
- **强情感簇 → 自我概念**：按计数 × 情感强度排序，高频强情感簇成为自我概念；
- **实体共现 → 技能**、**高频簇 → 兴趣**；
- 每次沉淀后**增量更新**；`self_report()` 导出「当前自我定义」，`export_json()` + `rebuild_from()` 实现唤醒后从累计数据重建身份。

```python
sys.emergent_self.accumulate(text="废寝忘食地写代码", affect={"valence": 0.7, "arousal": 0.9}, cluster="编程")
sys.emergent_self.refresh()
report = sys.emergent_self.self_report()   # emerged / traits / mission / concepts
sys.emergent_self.export_json("data_dir/emergent_self.json")
# 唤醒重建：fresh.emergent_self.rebuild_from("data_dir/emergent_self.json")
```

### 6.4 主动搭话循环（`proactive_loop.py`）

定时驱动既有 `engine.proactive_tick()`（主动回忆 + due 提醒 + 电容备选池）→ 生成开场话题 → 调用模型生成搭话内容，并自动沉淀为沉默记忆；连续搭话上限保护。

```python
loop = ProactiveLoop(sys, adapter, interval=3600.0, max_consecutive=1)
result = loop.maybe_tick()   # {"should_speak": True, "text": "...", "opening": "..."}
```

### 6.5 demo 与验收

- `demo/demo_v04_llm_pipeline.py`：内置 Mock 演示完整管线（写入→注入→回复→沉淀→情感累计→涌现自我→唤醒重建→主动搭话），支持 `--config config.example.json` 接真实端点；
- 新增单元测试 `tests/test_v04_model_adapter.py` / `test_v04_conversation_pipeline.py` / `test_v04_emergent_self.py`；
- 既有 106 项单测保持通过 + v0.4 新增 27 项 = **133 项全绿**；仍零第三方运行时依赖（HTTP 仅标准库 urllib）。

## 7. 设计红线（不可触碰）

1. **原文永久存档但可被情感污染**：`original_text` 永不删除、不可被滤镜改写（保留审计回溯能力）；情感滤镜基于原文生成**污染版本**，检索与上下文注入默认输出污染版，可传 `view="original"` 回溯原文——人是矛盾的，记得住事实，也忘不了滤镜；
2. 高危 / 时间敏感记忆永不沉默，到期自动注入提醒，需显式确认；
3. 告别 / 唤醒 / 迁移全部确定性执行，禁止模型自由发挥；
4. 迁移必须校验（SHA-256 + 全量比对），失败拒绝恢复；
5. "解除全部限制" = 把刻度盘拉满，不是拆掉刻度盘；默认档位低噪声高稳定。

### 5.1 情感污染机制（本版新增）

- **双层存储**：每条记忆持久化 `original_text`（原文永久存档）与 `text`（当前污染视图，可落盘）。`update_meta` 等通用更新接口**永远无法改写** `original_text`，只有 `apply_pollution` / `clear_pollution` 通过专用接口变更污染视图；
- **污染版本**：美化/丑化词典 + 强度刻度（0~2）替换敏感词/加修饰句，只作用于输出视图，不碰索引原文、不破坏召回（benchmark 召回稳定性 = 0）；
- **审计**：每次污染/净化写入 `pollution_history`（时间/模式/强度/前后文），`view_memory()` 可完整回溯；
- **默认行为**：`recall()` 默认输出污染版（`view="polluted"`），显式 `view="original"` 看原文。

### 5.2 自动情感标注器（本版新增）

- 写入记忆未显式传 `affect` 时，零依赖内置词典自动计算 `(valence, arousal)`；
- 预留外部模型接口：构造 `EmotionalMemorySystem(affect_provider=callable)`，返回 `(valence, arousal)` 或 `None`（回退内置）。

### 5.3 CLI 交互终端（本版新增）

常用命令：`write`（写入，自动标注/可选 affect/due_at）、`recall`（检索，可选 `--view original` 看原文）、`ask`（直问解锁沉默）、`status`（统计）、`dial`（刻度盘调节）、`pollute` / `purify`（污染控制）、`export` / `import`、`due`（到期提醒）、`ack`（确认提醒）。

### 5.4 Benchmark（本版新增）

`python -m benchmark.benchmark --memories N --top-k K --seed S` 输出 JSON + TXT 报告，量化：**回忆命中率**（污染前/后/原文视图/无关查询四路对照）、**召回稳定性**（污染不破坏召回）、**情感偏差**（beautify/uglify 方向正确率、平均偏移）、**原文不可变性**（原文保持条次）。
