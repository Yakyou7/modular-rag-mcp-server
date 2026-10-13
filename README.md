# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG（检索增强生成）服务框架，通过 MCP（Model Context Protocol）协议对外暴露工具接口，支持 Copilot / Claude 等 AI 助手直接调用。

> ⚠️ **关于本仓库**：这是开源项目 **Modular-RAG-MCP-SERVER** 的**个人复刻与二次开发版本**，在原项目基础上进行了图谱检索（Graph RAG）、置信度评估、Ragas 兼容性等扩展。仓库名为 `modular-rag-mcp-server`，由 [Yakyou7](https://github.com/Yakyou7) 维护。原始项目的版权归其作者所有，本仓库的修改部分可自行取用。

---

## 📖 目录

- [项目概述](#-项目概述)
- [分支说明](#-分支说明)
- [快速开始](#-快速开始)
- [常见问题](#-常见问题)

---

## 🏗️ 项目概述

### 这个项目是什么

本项目将 RAG 最核心的几个环节——**检索（Hybrid Search + Rerank）**、**多模态视觉处理（Image Captioning）**、**RAG 评估（Ragas + Custom）**、**生成（LLM Response）**——以及 **MCP（Model Context Protocol）** 协议串联为一个完整的、可运行的工程项目。

架构上保持了完全模块化的内部结构，方便替换组件、扩展算法。

### 核心能力一览

| 模块 | 能力 | 说明 |
|------|------|------|
| **Ingestion Pipeline** | PDF → Markdown → Chunk → Transform → Embedding → Upsert | 全链路数据摄取，支持多模态图片描述（Image Captioning） |
| **Hybrid Search** | Dense (向量) + Sparse (BM25) + RRF Fusion + Rerank | 粗排召回 + 精排重排的两段式检索架构 |
| **Graph Retrieval** *(扩展)* | 图谱索引 + 图检索器 + 图增强召回 | 基于实体关系的结构化检索，与混合检索互补 |
| **MCP Server** | 标准 MCP 协议暴露 Tools | `query_knowledge_hub`、`list_collections`、`get_document_summary` |
| **Dashboard** | Streamlit 六页面管理平台 | 系统总览 / 数据浏览 / Ingestion 管理 / 摄取追踪 / 查询追踪 / 评估面板 |
| **Evaluation** | Ragas + Custom 评估体系 | 支持 golden test set 回归测试，拒绝"凭感觉"调优 |
| **Confidence** *(扩展)* | 低置信度答案识别与降级回复 | 避免在证据不足时强行作答 |
| **Observability** | 全链路白盒化追踪 | Ingestion 与 Query 两条链路的每一个中间状态透明可见 |
| **Skill 驱动全流程** | 从编写到测试、打包、配置 | auto-coder / qa-tester / package / setup 等 Skill 覆盖完整开发生命周期 |

### 技术亮点

**🔌 全链路可插拔架构**：LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator 每一个核心环节均定义了抽象接口，支持"乐高积木式"替换，通过配置文件一键切换后端，零代码修改。

**🔍 混合检索 + 重排**：BM25 稀疏检索解决专有名词精确匹配 + Dense Embedding 解决同义词语义匹配，RRF 融合后可选 Cross-Encoder / LLM Rerank 精排，平衡查全率与查准率。

**🕸️ 图谱检索（本仓库扩展）**：新增 `graph_retriever` / `graph_indexer` / `build_graph_index.py`，将文档中的实体与关系抽取为图结构，支持结构化路径检索，与混合检索结果融合。

**🎯 置信度评估（本仓库扩展）**：新增 `confidence.py` 与 `answer_generator.py`，在证据不足时走低置信度降级路径，明确告知用户"不确定"，而不是编造答案。

**🖼️ 多模态图像处理**：采用 Image-to-Text 策略，利用 Vision LLM 自动生成图片描述并缝合进 Chunk，复用纯文本 RAG 链路即可实现"搜文字出图"。

**📡 MCP 生态集成**：遵循 Model Context Protocol 标准，可直接对接 GitHub Copilot、Claude Desktop 等 MCP Client，零前端开发，一次开发处处可用。

**📊 可视化管理 + 自动化评估**：Streamlit Dashboard 提供完整的数据管理与链路追踪能力，集成 Ragas 等评估框架，建立基于数据的迭代反馈回路。

**🧪 三层测试体系**：Unit / Integration / E2E 分层测试，覆盖独立模块逻辑、模块间交互、完整链路（MCP Client / Dashboard）。

> 📖 详细架构设计、模块说明和任务排期请参阅 [DEV_SPEC.md](DEV_SPEC.md)

---

## 📂 分支说明

本项目提供三个分支，面向不同使用场景：

### `main` — 最干净的完整代码

- 包含项目的最新完整代码
- **适合人群**：想要快速体验项目完整功能、或直接在该项目基础上做二次扩展的开发者

### `dev` — 保留完整开发记录

- 代码与 `main` 完全一致，但保留了完整的 commit 历史
- 记录了从零开始逐步构建的每一步过程
- **适合人群**：想了解项目是如何一步步从零搭建起来的开发者，可以通过 commit 历史回溯开发思路

### `clean-start` — 干净起点，从零开始

- 仅包含工程骨架（Agent Skills + DEV_SPEC），所有任务进度清零
- 保留了完整的 Skill 配置，可以使用 Agent 辅助开发
- **适合人群**：时间充分、想从头体验完整工作流（写 Spec → 拆任务 → 写代码 → 写测试 → 迭代优化）的开发者

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Yakyou7/modular-rag-mcp-server.git
cd modular-rag-mcp-server
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp config/settings.example.yaml config/settings.yaml
cp config/test_credentials.yaml.example config/test_credentials.yaml

# 编辑 config/settings.yaml，填入你的 API Key
# 推荐使用环境变量，避免密钥硬编码
```

**配置示例**（`config/settings.yaml`）：

```yaml
llm:
  provider: openai_compatible
  api_key: "${ZHIPU_API_KEY}"    # 从环境变量读取
  model: "glm-4-plus"

embedding:
  provider: openai_compatible
  api_key: "${ZHIPU_API_KEY}"

reranker:
  provider: cross_encoder
  model_name: "BAAI/bge-reranker-base"
```

> 💡 `config/settings.yaml` 和 `config/test_credentials.yaml` 已在 `.gitignore` 中，不会被提交。

### 3. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### 4. 摄取文档

```bash
python scripts/ingest.py --data-dir ./data/documents
```

### 5. 构建图谱索引（可选，本仓库扩展功能）

```bash
python scripts/build_graph_index.py
```

### 6. 启动 MCP Server

```bash
python main.py
```

### 7. 启动 Dashboard（可选）

```bash
python scripts/start_dashboard.py
```

---

## 🔧 接入 AI 客户端

在 MCP Client 配置文件中添加：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": ["<项目路径>/main.py"]
    }
  }
}
```

---

## ❓ 常见问题

### 1. 如何切换 Provider（Qwen / DeepSeek / Ollama）？

项目的 `src/libs/` 下 LLM、Embedding、Reranker 等模块均使用工厂模式。新增一个 Provider 只需：① 新增一个 Provider 类；② 在工厂注册；③ 更新 `settings.yaml` 配置。

### 2. 想摄取 PDF 以外的格式（Word / Markdown / HTML）？

实现 `BaseLoader` 接口并在 Pipeline 中注册即可，Transform / Embedding 之后的链路完全复用。

### 3. 项目报错怎么办？

本项目定位为学习 / 二次开发用途，遇到问题建议：
- 查看 `logs/` 下的结构化日志（JSON Lines）
- 通过 Dashboard 的追踪页面定位是哪一阶段出错
- 提 Issue 并附上日志

---

## ⚠️ 免责声明

本项目为**学习与技术研究用途**，不是生产级系统。生产环境部署前请自行评估安全性与稳定性。使用的第三方 API Key 请自行妥善保管，切勿提交到版本库。

## 📄 License

本项目修改部分采用 MIT License。原始项目版权归其原作者所有，详见上游仓库。
