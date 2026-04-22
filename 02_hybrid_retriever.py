import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate


import warnings
warnings.filterwarnings("ignore")


load_dotenv()

# ========== 1. 初始化组件 ==========
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# ========== 2. 加载和切分文档 ==========
loader = TextLoader("data/test_hybrid.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_documents(documents)

# ========== 3. 构建检索器 ==========
# 向量检索器
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 关键词检索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5


# ========== 4. 加权混合检索函数 ==========
def weighted_hybrid_retrieve(
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.6,
        vector_weight: float = 0.4
):
    """
    加权混合检索
    - bm25_weight: BM25 结果的权重（精确匹配）
    - vector_weight: 向量检索的权重（语义匹配）
    """
    # 分别获取两种检索结果
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    # 给每个文档打分（基于来源权重）
    scored_docs = {}

    # BM25 结果加分
    for doc in bm25_docs:
        doc_id = doc.page_content[:200]  # 用内容前200字符作为唯一标识
        scored_docs[doc_id] = {
            "doc": doc,
            "score": scored_docs.get(doc_id, {}).get("score", 0) + bm25_weight
        }

    # 向量检索结果加分
    for doc in vector_docs:
        doc_id = doc.page_content[:200]
        scored_docs[doc_id] = {
            "doc": doc,
            "score": scored_docs.get(doc_id, {}).get("score", 0) + vector_weight
        }

    # 按分数从高到低排序
    sorted_docs = sorted(
        scored_docs.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # 返回 top_k 个文档
    return [item["doc"] for item in sorted_docs[:top_k]]


# ========== 5. 测试检索（可选，展示加权效果）==========
def print_retrieval_comparison(query: str):
    """打印三种检索方式的对比结果"""
    print(f"\n查询: {query}")
    print("-" * 50)

    # 纯 BM25
    bm25_only = bm25_retriever.invoke(query)
    print(f"【纯 BM25 检索】召回 {len(bm25_only)} 个")
    for i, doc in enumerate(bm25_only[:2]):
        print(f"  {i + 1}. {doc.page_content[:80]}...")

    # 纯向量
    vector_only = vector_retriever.invoke(query)
    print(f"\n【纯向量检索】召回 {len(vector_only)} 个")
    for i, doc in enumerate(vector_only[:2]):
        print(f"  {i + 1}. {doc.page_content[:80]}...")

    # 加权混合
    hybrid = weighted_hybrid_retrieve(query, top_k=3)
    print(f"\n【加权混合检索 (BM25:0.6, 向量:0.4)】召回 {len(hybrid)} 个")
    for i, doc in enumerate(hybrid):
        print(f"  {i + 1}. {doc.page_content[:80]}...")


# ========== 6. RAG 问答 ==========
question = "ABC-X1 和 ABC-X2 有什么区别？"

# 使用加权混合检索
retrieved_docs = weighted_hybrid_retrieve(question, top_k=5)

# 构建 Prompt
prompt = ChatPromptTemplate.from_template("""
请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请说"没有找到相关信息"。

参考资料：
{context}

用户问题：{input}
""")

context = "\n\n".join([doc.page_content for doc in retrieved_docs])
formatted_prompt = prompt.format(context=context, input=question)
response = llm.invoke(formatted_prompt)

# ========== 7. 输出结果 ==========
print("=" * 50)
print(f"问题: {question}")
print("-" * 50)
print(f"答案: {response.content}")
print("-" * 50)
print(f"加权混合检索召回: 共找到 {len(retrieved_docs)} 个相关片段")

# 打印对比结果
# print_retrieval_comparison(question)