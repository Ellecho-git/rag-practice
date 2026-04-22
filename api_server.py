import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import warnings
warnings.filterwarnings("ignore")
load_dotenv()

# ========== 初始化（在服务启动时加载一次） ==========
print("正在初始化 RAG 系统...")

# 大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 加载文档
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "test_hybrid.txt")
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# 切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 向量存储
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 检索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# 重排序模型
reranker = CrossEncoder('BAAI/bge-reranker-base')

# Prompt 模板
prompt = ChatPromptTemplate.from_template("""
请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请说"没有找到相关信息"。

参考资料：
{context}

用户问题：{input}
""")

print("初始化完成！")

# ========== FastAPI 应用 ==========
app = FastAPI(
    title="RAG 问答 API",
    description="支持混合检索 + 重排序的 RAG 问答系统",
    version="3.0"
)


# 请求体模型
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3  # 最终返回的文档片段数


# 响应体模型
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]  # 引用的文档片段
    used_docs_count: int  # 实际使用的文档数


# 加权混合检索函数
def weighted_hybrid_retrieve(query: str, top_k: int = 20):
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    all_docs = {}
    for doc in bm25_docs + vector_docs:
        doc_id = doc.page_content[:100]
        if doc_id not in all_docs:
            all_docs[doc_id] = doc

    return list(all_docs.values())[:top_k]


@app.get("/")
def root():
    return {"message": "RAG 问答 API 正在运行", "status": "healthy"}


@app.post("/ask", response_model=AnswerResponse)
def ask( request: QuestionRequest ):
    question = request.question
    top_k = request.top_k

    # 1. 混合检索（粗筛）
    candidates = weighted_hybrid_retrieve(question, top_k=20)

    if not candidates:
        return AnswerResponse(
            question=question,
            answer="没有找到相关参考资料。",
            sources=[],
            used_docs_count=0
        )

    # 2. 重排序（精排）
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:top_k]]

    # 3. 生成答案
    context = "\n\n".join([doc.page_content for doc in top_docs])
    formatted_prompt = prompt.format(context=context, input=question)
    response = llm.invoke(formatted_prompt)

    # 4. 提取引用的文档片段
    sources = [doc.page_content for doc in top_docs]

    return AnswerResponse(
        question=question,
        answer=response.content,
        sources=sources,
        used_docs_count=len(top_docs)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)