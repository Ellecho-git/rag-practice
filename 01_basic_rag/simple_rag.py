import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

question = "ABC-X1 和 ABC-X2 有什么区别？"

# 大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 获取文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "test_hybrid.txt")

# 1. 加载文档
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_documents(documents)

# 3. 向量化并存储
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# 5. 定义提示词模板
prompt = ChatPromptTemplate.from_template("""
请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请说"没有找到相关信息"。

参考资料：
{context}

用户问题：{input}
""")

# 手动构建 RAG 链（使用同一个 question 变量）
retrieved_docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
formatted_prompt = prompt.format(context=context, input=question)
response = llm.invoke(formatted_prompt)

# 输出结果（也使用同一个 question 变量）
print("=" * 50)
print(f"问题: {question}")
print("-" * 50)
print(f"答案: {response.content}")
print("-" * 50)
print(f"参考资料: 共找到 {len(retrieved_docs)} 个相关片段")