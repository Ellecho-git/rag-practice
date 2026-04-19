# RAG Practice

> 从0到1搭建RAG（检索增强生成）问答系统

## 🎯 项目简介

本项目实现了一个基于RAG的文档问答系统。给定一个文档（如公司制度），系统能够根据文档内容回答用户问题，并标注答案来源。

**核心流程**：文档 → 切块 → 向量化 → 检索 → 生成 → 回答

**技术栈**：
- 大模型：DeepSeek API
- Embedding：BAAI/bge-small-zh-v1.5（本地运行）
- 向量数据库：Chroma
- 框架：LangChain

---
