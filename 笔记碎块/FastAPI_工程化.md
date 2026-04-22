# FastAPI 工程化说明书

## FastAPI简介

将 v3 的 RAG 系统（混合检索 + 重排序）封装成 RESTful API 服务，其他程序可以通过 HTTP 请求调用问答功能。

**核心价值：**
- 从"命令行脚本"升级为"可调用的服务"
- 其他系统（前端、App、爬虫等）可以集成使用
- 支持并发请求，适合生产环境部署

**技术栈：**

| 模块 | 技术 |
|------|------|
| Web 框架 | FastAPI |
| 服务器 | Uvicorn（ASGI 服务器） |
| 数据验证 | Pydantic |
| RAG 核心 | v3（混合检索 + 重排序） |

---

## 为什么需要 FastAPI 封装？

| 对比 | 命令行脚本（v1/v2/v3） | FastAPI 服务 |
|------|--------------------------|--------------|
| 调用方式 | 只能在终端运行 | 任何编程语言通过 HTTP 调用 |
| 并发处理 | 一次只能处理一个请求 | 支持多用户同时请求 |
| 部署方式 | 只能在开发机运行 | 可部署到服务器，外网访问 |
| 集成能力 | 难以被其他系统调用 | 可集成到网页、App、微信机器人等 |
| 文档 | 无 | 自动生成 API 文档（/docs） |

---

## API 接口说明

### 1. 健康检查接口

| 项目 | 内容 |
|------|------|
| 路径 | GET / |
| 作用 | 检查服务是否正常运行 |

**返回示例：**

```json
{
  "message": "RAG 问答 API 正在运行",
  "status": "healthy"
}
```

### 2. 问答接口（核心）

| 项目 | 内容 |
|------|------|
| 路径 | POST /ask |
| 作用 | 传入问题，返回答案 |
| 请求格式 | JSON |
| 返回格式 | JSON |

**请求体（Request Body）：**

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| question | string | ✅ 是 | 用户的问题 | "HRB400 是什么？" |
| top_k | integer | ❌ 否 | 返回时使用的文档片段数（默认 3） | 3 |

**请求示例：**

```json
{
  "question": "HRB400 是什么？",
  "top_k": 3
}
```

**响应体（Response Body）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| question | string | 原问题（回显） |
| answer | string | 大模型生成的答案 |
| sources | array | 引用的文档片段列表 |
| used_docs_count | integer | 实际使用的文档片段数 |

**响应示例：**

```json
{
  "question": "HRB400 是什么？",
  "answer": "HRB400是一种建筑用钢筋，直径12mm，已于2023年停产。",
  "sources": [
    "公司产品规格表\n产品型号：HRB400\n规格：直径12mm\n用途：建筑用钢筋\n备注：此型号为老款，已于2023年停产",
    "产品型号：ABC-X1\n规格：笔记本电脑\nCPU：Intel i7-12700..."
  ],
  "used_docs_count": 2
}
```

---

## 代码结构

```python
# api_server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_...  # 导入 v3 所用模块

# ========== 1. 初始化 RAG 组件（启动时加载一次） ==========
# 大模型、Embedding、向量库、检索器、重排序模型
# 这些只在服务启动时加载一次，后续请求复用

# ========== 2. 定义数据模型（Pydantic） ==========
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    used_docs_count: int

# ========== 3. 创建 FastAPI 应用 ==========
app = FastAPI(title="RAG 问答 API", version="3.0")

# ========== 4. 定义接口 ==========
@app.get("/")
def root():
    return {"message": "RAG 问答 API 正在运行"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    # 调用 v3 的检索 + 重排序 + 生成逻辑
    # 返回答案
    return AnswerResponse(...)

# ========== 5. 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 核心代码逐段解释

### 1. 数据模型（Pydantic）

```python
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3  # 默认值
```

- 定义请求体应该长什么样
- `top_k = 3` 表示如果不传，默认用 3
- FastAPI 会自动校验请求体是否符合这个格式

### 2. 创建 FastAPI 应用

```python
app = FastAPI(title="RAG 问答 API", version="3.0")
```

- 创建应用实例
- `title` 和 `version` 会显示在 `/docs` 页面上

### 3. 定义 GET 接口

```python
@app.get("/")
def root():
    return {"message": "RAG 问答 API 正在运行"}
```

- `@app.get("/")` 装饰器：声明这是一个 GET 请求接口，路径是 `/`
- 返回一个字典，FastAPI 会自动转成 JSON

### 4. 定义 POST 接口

```python
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    # 从请求中取出问题
    question = request.question
    top_k = request.top_k
    
    # 调用 RAG 逻辑...
    
    # 返回结果
    return AnswerResponse(
        question=question,
        answer=answer,
        sources=sources,
        used_docs_count=len(sources)
    )
```

- `@app.post("/ask")`：POST 请求，路径是 `/ask`
- `response_model=AnswerResponse`：声明返回格式
- `request: QuestionRequest`：自动将 JSON 请求体解析为 `QuestionRequest` 对象

### 5. 启动服务

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- `host="0.0.0.0"`：监听所有网络接口，允许外部访问
- `port=8000`：服务端口号
- 访问地址：http://localhost:8000

---

## 运行方式

### 1. 启动服务

```bash
python api_server.py
```

看到以下输出说明启动成功：

```text
正在初始化 RAG 系统...
初始化完成！
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. 访问 API 文档

浏览器打开：**http://localhost:8000/docs**

可以看到自动生成的交互式 API 文档。

### 3. 测试问答

在 `/docs` 页面点击 POST /ask → Try it out → 输入：

```json
{
  "question": "HRB400 是什么？",
  "top_k": 3
}
```

点击 Execute，查看返回结果。

---

## 与 v3 的关系

| 组件 | v3（03_reranker.py） | API 服务（api_server.py） |
|------|---------------------|----------------------------|
| RAG 核心 | ✅ 完全一样 | ✅ 完全一样 |
| 调用方式 | 命令行直接运行 | 通过 HTTP 请求调用 |
| 输入 | 代码里写死 question 变量 | 从 HTTP 请求体读取 |
| 输出 | print() 打印 | 返回 JSON |
| 复用方式 | 无 | 其他程序可通过 HTTP 调用 |

API 服务复用了 v3 的所有 RAG 逻辑：
- 混合检索（BM25 + 向量）
- 重排序（Rerank）
- 大模型生成

---

## 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| ModuleNotFoundError: No module named 'fastapi' | fastapi 未安装 | pip install fastapi uvicorn |
| 请求返回 "answer": "没有找到相关信息" | question 字段值还是 "string" | 改成真正的问题 |
| 服务启动后无法访问 | 端口被占用 | 换一个端口，如 port=8001 |
| 返回的 sources 是完整的文档内容 | 没有截断 | 可以在返回前对 sources 做切片 |

---

## 后续优化方向

| 优化项 | 作用 | 状态 |
|--------|------|------|
| 环境变量配置 | 端口、模型路径可配置 | 待做 |
| 请求日志 | 记录每次请求，方便调试 | 待做 |
| 错误处理 | 更友好的错误提示 | 已做基础版 |
| 限流 | 防止恶意请求 | 待做 |
| Docker 部署 | 一键部署到任何服务器 | 待做 |
| 前端界面 | Streamlit 调用 API | 待做 |

---

## 总结

API 服务做了什么：

1. **复用了 v3 的 RAG 核心逻辑**（混合检索 + 重排序）
2. **封装成 HTTP 接口**，任何编程语言都可以调用
3. **自动生成 API 文档**，方便测试和集成
4. **支持并发请求**，为生产环境部署打下基础

**核心价值**：将 RAG 系统从"个人脚本"升级为"可集成的服务"。

---

## 新增依赖包

| 包名 | 版本 | 用途 |
|------|------|------|
| fastapi | 0.136.0 | Web 框架 |
| uvicorn | 最新 | ASGI 服务器 |
| pydantic | 内置 | 数据验证（FastAPI 依赖） |

安装命令：

```bash
pip install fastapi uvicorn
```