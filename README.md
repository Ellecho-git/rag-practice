# RAG Practice

> 从0到1搭建RAG（检索增强生成）问答系统

## 🎯 项目简介

本项目实现了一个基于RAG的文档问答系统。给定一个文档（如公司制度），系统能够根据文档内容回答用户问题，并标注答案来源。

**核心流程**：文档 → 切块 → 向量化 → 检索 → 生成 → 回答

**技术栈**：

| 模块 | 技术 |
|------|------|
| 大模型 | DeepSeek API |
| Embedding | BAAI/bge-small-zh-v1.5（本地运行） |
| 向量数据库 | Chroma |
| 框架 | LangChain |

---

## 📚 版本演进

| 版本 | 文件 | 核心能力 | 详细说明 |
|------|------|---------|---------|
| **v1_向量检索** | `01_simple_rag.py` | 基础向量检索 | [v1说明书](docs/v1_向量检索.md) |
| **v2_混合检索** | `02_hybrid_retriever.py` | 混合检索（BM25 + 向量 + 加权融合） | [v2说明书](docs/v2混合检索.md) |
| **v3_重排序** | `03_reranker.py` | 混合检索 + 重排序（Cross-Encoder） | [v3说明书](docs/v3重排序.md) |
| **api_server** | `api_server.py` | FastAPI 封装，HTTP 服务 | [api说明书](docs/FastAPI.md) |

**推荐从 `api_server.py` 开始体验**：运行后访问 `http://localhost:8000/docs` 即可在网页上测试问答。

---

## 📁 项目结构

rag-practice/
├── data/ # 测试文档
├── docs/ # 版本说明书
├── 01_simple_rag.py # v1 基础向量检索
├── 02_hybrid_retriever.py # v2 混合检索
├── 03_reranker.py # v3 重排序
├── api_server.py # FastAPI 服务
├── requirements.txt # 依赖列表
├── .env # API Key
└── README.md

---

## 📊 版本对比

| 版本 | 检索方式 | 排序方式 | 特点 |
|------|---------|---------|------|
| v1 | 纯向量检索 | 向量相似度 | 语义理解强，专有名词易漏 |
| v2 | BM25 + 向量 | 加权融合（0.6:0.4） | 专有名词+语义双覆盖 |
| v3 | BM25 + 向量 | 加权融合 + Rerank | 精排提升 Top-1 准确率 |
| api_server | 同 v3 | 同 v3 | 封装为 HTTP 服务，支持并发 |

---

## ⚠️ 常见问题

| 问题 | 解决方案 |
|------|---------|
| 第一次运行很慢 | 正在下载 BGE 模型（约 100MB）或 Rerank 模型（约 1GB），耐心等待 |
| 端口 8000 被占用 | 修改 `api_server.py` 最后一行：`port=8001` |
| API 返回"没有找到相关信息" | 检查 `data/` 文件夹下是否有文档，且文档内容与问题相关 |

---

## 📄 License

本仓库采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 协议。