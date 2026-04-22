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

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# 问题入口
question = "ABC-X1 和 ABC-X2 有什么区别？"

# 1. 初始化大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# 2. 初始化 Embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 3. 加载文档
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "test_hybrid.txt")
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# 4. 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 5. 向量存储
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 6. 混合检索器（粗筛）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# 7. 手动混合检索（合并去重）
bm25_docs = bm25_retriever.invoke(question)
vector_docs = vector_retriever.invoke(question)

# 去重合并
all_docs = {}
for doc in bm25_docs + vector_docs:
    doc_id = doc.page_content[:100]
    if doc_id not in all_docs:
        all_docs[doc_id] = doc

candidates = list(all_docs.values())[:20]  # 取前20个候选

print(f"【混合检索】召回 {len(candidates)} 个候选片段")

# 8. 重排序（精排）
print("【重排序】正在计算相关性分数...")
reranker = CrossEncoder('BAAI/bge-reranker-base')
pairs = [[question, doc.page_content] for doc in candidates]
scores = reranker.predict(pairs)

# 按分数排序
scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_docs = [doc for doc, score in scored_docs[:3]]  # 取 Top-3

print(f"【重排序完成】Top-3 分数: {[round(score, 3) for doc, score in scored_docs[:3]]}")

# 9. 构建 Prompt
prompt = ChatPromptTemplate.from_template("""
请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请说"没有找到相关信息"。

参考资料：
{context}

用户问题：{input}
""")

context = "\n\n".join([doc.page_content for doc in top_docs])
formatted_prompt = prompt.format(context=context, input=question)
response = llm.invoke(formatted_prompt)

# 10. 输出结果
print("=" * 50)
print(f"问题: {question}")
print("-" * 50)
print(f"答案: {response.content}")
print("-" * 50)
print(f"最终使用 {len(top_docs)} 个片段（经 Rerank 精排）")
