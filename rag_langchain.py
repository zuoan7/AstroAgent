from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from config import settings
from typing import List, Optional
import os
from logger import logger


class RAGSystem:
    """基于LangChain的RAG系统"""
    
    def __init__(self):
        self.enabled = True
        self.embeddings = None
        self.vector_db = None
        self.text_splitter = None
        self.document_hashes = set()
        self.hash_file = os.path.join(settings.VECTOR_DB_PATH, "document_hashes.txt")
        
        # 初始化组件
        self._init_embeddings()
        self._init_text_splitter()
        self._load_document_hashes()
        self.initialize_db()
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            self.embeddings = DashScopeEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                dashscope_api_key=settings.DASHSCOPE_API_KEY
            )
            logger.info(f"✅ 内置千问嵌入模型 {settings.EMBEDDING_MODEL_NAME} 初始化成功")
        except Exception as e:
            logger.error(f"❌ 嵌入模型初始化失败：{str(e)}")
            self.enabled = False
            raise
    
    def _init_text_splitter(self):
        """初始化文本分割器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def _load_document_hashes(self):
        """加载文档哈希值"""
        try:
            if os.path.exists(self.hash_file):
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.document_hashes.add(line)
                logger.info(f"✅ 加载了 {len(self.document_hashes)} 个已添加文档的哈希值")
            else:
                logger.warning("⚠️  未找到文档哈希值文件，将创建新的")
        except Exception as e:
            logger.error(f"❌ 加载文档哈希值失败：{str(e)}")
    
    def _save_document_hashes(self):
        """保存文档哈希值"""
        try:
            if not os.path.exists(settings.VECTOR_DB_PATH):
                os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                for hash_val in self.document_hashes:
                    f.write(f"{hash_val}\n")
            logger.info(f"✅ 保存了 {len(self.document_hashes)} 个文档哈希值到 {self.hash_file}")
        except Exception as e:
            logger.error(f"❌ 保存文档哈希值失败：{str(e)}")
    
    def initialize_db(self):
        """初始化向量数据库"""
        if not self.enabled or not self.embeddings:
            return
        
        try:
            logger.info("=== 开始初始化向量数据库 ===")
            
            if not os.path.exists(settings.VECTOR_DB_PATH):
                os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
                logger.info(f"✅ 创建向量库目录：{settings.VECTOR_DB_PATH}")
            
            self.vector_db = Chroma(
                embedding_function=self.embeddings,
                collection_name="astronomy_rag",
                persist_directory=settings.VECTOR_DB_PATH
            )
            logger.info(f"✅ 向量数据库初始化成功，保存路径：{settings.VECTOR_DB_PATH}")
            
            self.load_documents_from_data()
        except Exception as e:
            logger.error(f"❌ 向量数据库初始化失败：{str(e)}")
            self.enabled = False
    
    def load_documents_from_data(self):
        """从data文件夹加载文档"""
        if not self.enabled or not self.vector_db:
            return
        
        data_dir = "./data"
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️  data文件夹不存在：{data_dir}，跳过文档加载")
            return
        
        supported_extensions = ['.txt', '.pdf']
        documents = []
        
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in supported_extensions:
                logger.warning(f"⚠️  不支持的文件类型：{filename}，跳过")
                continue
            
            try:
                if file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                else:
                    continue
                
                loaded_docs = loader.load()
                if loaded_docs:
                    # 计算文档内容的哈希值
                    import hashlib
                    doc_content = ' '.join([doc.page_content for doc in loaded_docs])
                    doc_hash = hashlib.md5(doc_content.encode('utf-8')).hexdigest()
                    
                    if doc_hash in self.document_hashes:
                        logger.warning(f"⚠️  文档 {filename} 已存在，跳过")
                        continue
                    
                    documents.extend(loaded_docs)
                    self.document_hashes.add(doc_hash)
                    logger.info(f"✅ 读取{file_ext.upper()}文档：{filename}（页数：{len(loaded_docs)}）")
            except Exception as e:
                logger.error(f"❌ 读取文档 {filename} 失败：{str(e)}")
                continue
        
        if documents:
            self.add_documents(documents)
        else:
            logger.warning("⚠️  未加载到任何有效文档")
        
        self._save_document_hashes()
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
        if not self.enabled or not self.vector_db:
            raise RuntimeError("RAG系统未启用，无法添加文档")
        
        try:
            logger.info(f"=== 开始分割文档：共 {len(documents)} 个文档 ===")
            
            # 分割文档
            texts = []
            metadatas = []
            
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                texts.extend([chunk.page_content for chunk in chunks])
                metadatas.extend([chunk.metadata for chunk in chunks])
                logger.info(f"文档分割为 {len(chunks)} 个片段")
            
            if not texts:
                logger.warning("⚠️  无有效文本可添加")
                return
            
            logger.info(f"=== 开始向向量库添加 {len(texts)} 个文本片段 ===")
            
            # 分批次添加
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                try:
                    self.vector_db.add_texts(batch_texts, batch_metadatas)
                    logger.info(f"✅ 安全添加：第 {i//batch_size + 1} 批（{len(batch_texts)} 个片段）")
                except Exception as e:
                    logger.error(f"❌ 安全添加：第 {i//batch_size + 1} 批失败 {str(e)[:50]}")
                    continue
            
            logger.info(f"✅ 所有文本片段添加完成（共 {len(texts)} 个）")
        except Exception as e:
            logger.error(f"❌ 添加文档失败：{str(e)}")
            raise
    
    def get_relevant_context(self, query: str, top_k=3):
        """检索相关上下文"""
        if not self.enabled or not self.vector_db:
            return ""
        
        try:
            results = self.vector_db.similarity_search(query, k=top_k)
            context = "\n\n".join([doc.page_content for doc in results])
            logger.info(f"📄 检索到上下文长度：{len(context)} 字符")
            return context
        except Exception as e:
            logger.error(f"❌ 检索上下文失败：{str(e)}")
            return ""
