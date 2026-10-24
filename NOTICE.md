# Notice

## 运行时依赖

本项目运行时零第三方依赖，仅使用 Python 标准库（Python 3.9+）。HTTP 通信使用标准库 urllib。测试使用标准库 unittest；CI 中额外安装 pytest 仅作为测试运行器，不构成运行时依赖。

## 许可证

本项目以 MIT License 授权（见 LICENSE）。所有新增代码保持 MIT 授权。

## 上游借鉴与合规说明

本项目为开源组件的整合封装层，各能力层参考成熟开源项目设计（接口兼容、可插拔接入）。借鉴来源以 README 第 0 章所列为准：

- **FAISS / HNSW**（MIT / Apache-2.0）：向量索引设计参考；本项目内置哈希向量索引为可插拔占位实现。
- **Qdrant / Milvus**（Apache-2.0）：向量库设计参考。
- **networkx**（BSD）：图索引设计参考。
- **FSRS / Anki（SM-2）**（MIT 系）：遗忘曲线参考实现。
- **Mem0 / Letta / LangChain Memory**：仅参考设计思路，不引入任何受限代码。
- **git / rsync**：迁移协议增量思路参考。

## 明确规避项

- **Neo4j Community** 为 GPL 许可，本项目不引入，以自研轻量图索引替代。
- **Mem0 商业条款**不引入，仅参考设计思路。
- 其它 AGPL 类或商业条款组件（如 PyMuPDF）不引入。

发布本包为开源时，所有运行时与索引依赖必须保持 MIT / Apache-2.0 / BSD 系。
