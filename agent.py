
# ========== 强制修复Windows编码问题（必须放在最顶部） ==========
import sys
import os
import locale

# 1. 设置Python标准输出/错误编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# 2. 设置系统环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_CTYPE'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'

# 3. 强制设置locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, 'C')
# ==========================================
# ========== 强制禁用GPU，使用纯CPU运行 ==========
import os
# 禁用onnxruntime GPU
os.environ['ORT_DISABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 屏蔽所有GPU设备
# 禁用Chroma的GPU加速
os.environ['CHROMA_DISABLE_GPU'] = '1'
# 禁用DashScope的GPU调用
os.environ['DASHSCOPE_DISABLE_GPU'] = '1'
# ==============================================



import dashscope
from memory import ShortTermMemory
from rag import RAGSystem
from config import settings
from typing import List, Dict, Any, Generator
import time
import traceback  # 新增：打印详细异常栈


class AstroAgent:
    """天文Agent"""
    
    def __init__(self):
        # 初始化DashScope（验证API Key是否配置）
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("❌ DashScope API Key未配置！请在settings中设置DASHSCOPE_API_KEY")
        dashscope.api_key = settings.DASHSCOPE_API_KEY
        print("✅ DashScope API Key配置成功")
        
        # 初始化记忆系统
        self.memory = ShortTermMemory()
        print("✅ 短期记忆系统初始化成功")
        
        # 初始化RAG系统（新版RAGSystem在__init__中已自动调用initialize_db()）
        self.rag = RAGSystem()
        print("✅ RAG系统初始化成功（已自动加载data文件夹文档）")
        
        # 系统提示（优化：加入RAG上下文使用要求）
        self.system_prompt = """你是一个专业的天文知识助手，能够回答关于天文学的各种问题。
        请严格基于提供的上下文信息回答问题，确保专业、准确、详细。
        如果上下文没有相关信息，使用你的自有知识回答；如果不确定，如实告知。
        回答语言请使用中文，避免使用过于晦涩的术语，必要时给出解释。"""
    
    def generate_response(self, query: str) -> Generator[str, None, None]:
        """生成流式响应（优化解析逻辑+增强日志）"""
        print(f"\n=== 处理用户查询：{query} ===")
        
        # 1. 获取RAG相关上下文
        context = self.rag.get_relevant_context(query)
        print(f"📄 RAG检索到上下文长度：{len(context)} 字符")
        
        # 2. 获取最近的对话历史
        recent_messages = self.memory.get_recent_messages()
        print(f"🧠 加载对话历史条数：{len(recent_messages)}")
        
        # 3. 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加对话历史
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前查询和上下文
        if context:
            query_with_context = f"### 上下文信息：\n{context}\n\n### 用户问题：{query}"
        else:
            query_with_context = query
        
        messages.append({"role": "user", "content": query_with_context})
        print("✅ 消息列表构建完成，准备调用大模型")
        
        # 4. 生成响应（优化流式解析逻辑）
        full_response = ""
        previous_length = 0
        try:
            # 调用千问大模型（流式）
            response = dashscope.Generation.call(
                model=settings.MODEL_NAME,
                messages=messages,
                stream=True,
                result_format='message',  # 新增：指定返回格式，避免解析错误
                temperature=0.1  # 降低随机性，保证回答准确
            )
            
            for chunk in response:
                # 严格的空值判断（避免KeyError/NoneType错误）
                if not chunk:
                    continue
                output = chunk.get('output', {})
                choices = output.get('choices', [])
                if not choices:
                    continue
                
                # 解析流式内容
                delta = choices[0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    # 只返回新增的内容，避免重复输出
                    full_response += content
                    yield content  # 直接yield新增内容，无需计算长度差
        
        except Exception as e:
            print(f"\n❌ 生成响应失败：{str(e)}")
            traceback.print_exc()  # 打印详细异常栈，方便调试
            # 提供友好的默认响应
            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            full_response = default_response
            yield default_response
        
        # 5. 将消息添加到记忆（确保无论是否报错都记录）
        self.memory.add_message("user", query, time.time())
        self.memory.add_message("assistant", full_response, time.time())
        print(f"✅ 对话已存入记忆 | 助手响应长度：{len(full_response)} 字符")
    
    def add_astronomy_knowledge(self, knowledge: List[str]):
        """添加天文知识到RAG系统（兼容原有测试脚本）"""
        if not knowledge:
            print("⚠️  无有效知识可添加")
            return
        try:
            self.rag.add_documents(knowledge)
            print(f"✅ 成功添加 {len(knowledge)} 条知识到RAG系统")
        except Exception as e:
            print(f"❌ 添加知识失败：{str(e)}")
            traceback.print_exc()
    
    def clear_memory(self):
        """清空记忆"""
        try:
            self.memory.clear()
            print("✅ 记忆已清空")
        except Exception as e:
            print(f"❌ 清空记忆失败：{str(e)}")
            traceback.print_exc()