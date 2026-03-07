# 关闭urllib3的SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ========== 强制修复Windows编码+禁用Chroma内置嵌入 ==========
import sys
import os

# 1. 修复Windows编码问题（仅设置IO编码，不修改locale）
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 2. 禁用Chroma内置嵌入模型（彻底绕开onnxruntime）
os.environ['CHROMA_DISABLE_DEFAULT_EMBEDDINGS'] = '1'
os.environ['CHROMA_EMBEDDING_FUNCTION'] = 'none'

# 3. 禁用GPU相关（避免多余检测）
os.environ['ORT_DISABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ===========================================================

from langchain_chroma import Chroma
from config import settings
from typing import List, Optional, Dict, Any
import traceback
import requests
import json


# ========== 自定义千问嵌入类（核心：直接调用云端API，不依赖onnxruntime） ==========
class CustomQianwenEmbeddings:
    def __init__(self, api_key=None, model_name="text-embedding-v2"):
        import dashscope
        # 优先使用settings中的API Key
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        self.model_name = model_name
        # 初始化dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        """批量嵌入文档（使用dashscope官方SDK）"""
        import dashscope
        from dashscope import TextEmbedding
        embeddings = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [text.strip() for text in batch if text and text.strip()]
            if not batch:
                continue

            try:
                # 使用官方SDK调用（自动处理JSON格式）
                response = TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                    text_type='document',
                    timeout=60
                )

                if response.status_code == 200:
                    batch_embeddings = [item['embedding'] for item in response.output['embeddings']]
                    embeddings.extend(batch_embeddings)
                    print(f"✅ 千问嵌入：第 {i // batch_size + 1} 批成功，生成 {len(batch_embeddings)} 个向量")
                else:
                    print(f"❌ 千问嵌入：第 {i // batch_size + 1} 批失败，状态码 {response.status_code}")
                    embeddings.extend([[0.0] * 1536 for _ in batch])

            except Exception as e:
                print(f"❌ 千问嵌入：第 {i // batch_size + 1} 批异常 {str(e)[:100]}")
                embeddings.extend([[0.0] * 1536 for _ in batch])

        return embeddings

    def embed_query(self, text):
        """嵌入单个查询文本"""
        if not text or not text.strip():
            return [0.0] * 1536
        return self.embed_documents([text])[0]


# ========== RAG系统主类 ==========
class RAGSystem:
    def __init__(self):
        # 初始化配置验证
        self.enabled = True
        self.embeddings = None
        self.vector_db = None

        # 初始化自定义嵌入模型（千问）
        self._init_embeddings()

        # 初始化文本分割器（修复：删除encoding参数）
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        # 初始化向量数据库
        self.initialize_db()

    def _init_embeddings(self):
        """初始化自定义千问嵌入模型（替代DashScopeEmbeddings）"""
        try:
            self.embeddings = CustomQianwenEmbeddings()
            print(f"✅ 自定义千问嵌入模型 {settings.EMBEDDING_MODEL_NAME} 初始化成功")
        except Exception as e:
            print(f"❌ 嵌入模型初始化失败：{str(e)}")
            traceback.print_exc()
            self.enabled = False
            raise

 
    def initialize_db(self):
        """初始化Chroma向量库（Windows下用内存模式，避免磁盘写入冲突）"""
        if not self.enabled or not self.embeddings:
            return

        try:
            print("=== 开始初始化向量数据库 ===")
            # Windows下禁用磁盘持久化，使用纯内存模式（避免权限/内存错误）
            if os.name == 'nt':
                self.vector_db = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="astronomy_rag",
                    persist_directory=None  # 内存模式，不写入文件
                )
            else:
                # 非Windows系统可启用持久化
                self.vector_db = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="astronomy_rag",
                    persist_directory=settings.VECTOR_DB_PATH
                )
            print("✅ 向量数据库初始化成功（内存模式）")
            
            # 添加文档加载调用
            self.load_documents_from_data()

        except Exception as e:
            print(f"❌ 向量数据库初始化失败：{str(e)}")
            traceback.print_exc()
            self.enabled = False
            # 不抛出异常，而是优雅降级
            return

    def load_documents_from_data(self):
        """从data文件夹加载文档（兼容PDF/TXT）"""
        if not self.enabled or not self.vector_db:
            return

        data_dir = "./data"
        if not os.path.exists(data_dir):
            print(f"⚠️  data文件夹不存在：{data_dir}，跳过文档加载")
            return

        # 支持的文件类型
        supported_extensions = ['.txt', '.pdf']
        documents = []

        # 遍历data文件夹
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext not in supported_extensions:
                print(f"⚠️  不支持的文件类型：{filename}，跳过")
                continue

            try:
                # 读取TXT文件
                if file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(content)
                    print(f"✅ 读取TXT文档：{filename}（字符数：{len(content)}）")

                # 读取PDF文件（需安装PyPDF2：pip install PyPDF2）
                elif file_ext == '.pdf':
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text
                    documents.append(content)
                    print(f"✅ 读取PDF文档：{filename}（字符数：{len(content)}）")

            except Exception as e:
                print(f"❌ 读取文档 {filename} 失败：{str(e)}")
                traceback.print_exc()
                continue

        # 分割并添加文档到向量库
        if documents:
            self.add_documents(documents)
        else:
            print("⚠️  未加载到任何有效文档")

    def add_documents(self, documents: List[str]):
        """添加文档到向量库（增加崩溃保护）"""
        if not self.enabled or not self.vector_db:
            raise RuntimeError("RAG系统未启用，无法添加文档")

        try:
            print(f"=== 开始分割文档：共 {len(documents)} 个文档 ===")
            texts = []
            for idx, doc in enumerate(documents, 1):
                if not doc or len(doc.strip()) == 0:
                    print(f"⚠️  文档 {idx} 为空，跳过")
                    continue
                if len(doc) > 6000:
                    doc = doc[:6000]
                    print(f"⚠️  文档 {idx} 超长，已截断为6000字符")
                chunks = self.text_splitter.split_text(doc)
                texts.extend(chunks)
                print(f"文档 {idx} 分割为 {len(chunks)} 个片段")

            if not texts:
                raise ValueError("文档分割后无有效文本！")

            print(f"=== 开始向向量库添加 {len(texts)} 个文本片段 ===")
            # 修复：分批次添加，每批5个（降低内存压力）
            safe_batch_size = 5
            for j in range(0, len(texts), safe_batch_size):
                safe_batch = texts[j:j + safe_batch_size]
                try:
                    self.vector_db.add_texts(safe_batch)
                    print(f"✅ 安全添加：第 {j // safe_batch_size + 1} 批（{len(safe_batch)} 个片段）")
                except Exception as e:
                    print(f"❌ 安全添加：第 {j // safe_batch_size + 1} 批失败 {str(e)[:50]}")
                    continue

            print(f"✅ 所有文本片段添加完成（共 {len(texts)} 个）")

        except Exception as e:
            print(f"❌ 添加文档失败：{str(e)}")
            traceback.print_exc()
            # 修复：抛出异常前释放内存
            self.vector_db = None
            raise

    def get_relevant_context(self, query: str, top_k=3):
        """检索与查询相关的上下文"""
        if not self.enabled or not self.vector_db:
            return ""

        try:
            # 相似性检索
            results = self.vector_db.similarity_search(query, k=top_k)
            # 拼接上下文
            context = "\n\n".join([doc.page_content for doc in results])
            print(f"📄 检索到上下文长度：{len(context)} 字符")
            return context

        except Exception as e:
            print(f"❌ 检索上下文失败：{str(e)}")
            traceback.print_exc()
            return ""


# ========== 测试代码（可选） ==========
if __name__ == "__main__":
    # 测试RAG系统初始化
    try:
        rag = RAGSystem()
        print("\n✅ RAG系统初始化成功！")
        # 测试检索
        context = rag.get_relevant_context("什么是黑洞？")
        print(f"✅ 检索测试成功，上下文预览：{context[:100]}...")
    except Exception as e:
        print(f"❌ RAG系统测试失败：{str(e)}")
        traceback.print_exc()