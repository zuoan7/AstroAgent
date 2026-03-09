from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from config import settings
from rag_langchain import RAGSystem
from memory import ShortTermMemory
from astronomy_tools import AstronomyTools, AstronomyEventsPredictor
from typing import Generator, List, Dict, Any
import time
import traceback
from logger import logger


class AstroAgent:
    """基于LangChain的天文Agent"""
    
    def __init__(self):
        # 验证API Key
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("❌ DashScope API Key未配置！请在settings中设置DASHSCOPE_API_KEY")
        
        # 初始化组件
        self.rag = RAGSystem()
        self.memory = ShortTermMemory()
        self.llm = self._init_llm()
        self.astronomy_tools = AstronomyTools()
        self.events_predictor = AstronomyEventsPredictor()
        self.tools = self._init_tools()
        self.agent_executor = self._build_agent()
        
        logger.info("✅ AstroAgent初始化完成，基于LangChain框架和React模型")
    
    def _init_llm(self):
        """初始化语言模型"""
        try:
            llm = ChatTongyi(
                model=settings.MODEL_NAME,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                temperature=0.1
            )
            logger.info(f"✅ 语言模型 {settings.MODEL_NAME} 初始化成功")
            return llm
        except Exception as e:
            logger.error(f"❌ 语言模型初始化失败：{str(e)}")
            raise
    
    def _init_tools(self):
        """初始化工具"""
        tools = []
        
        # RAG检索工具
        def rag_retrieve(query):
            """使用RAG系统检索相关天文信息"""
            return self.rag.get_relevant_context(query)
        
        tools.append(Tool(
            name="RAGRetrieve",
            func=rag_retrieve,
            description="使用RAG系统检索相关天文信息，参数：query（查询语句）"
        ))
        
        # 行星位置计算工具
        tools.append(Tool(
            name="GetPlanetPosition",
            func=self.astronomy_tools.get_planet_position,
            description="获取行星位置，参数：planet_name（行星名称，如 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'），observation_time（观测时间，可选），latitude（观测点纬度，可选），longitude（观测点经度，可选）"
        ))
        
        # 天体坐标转换工具
        tools.append(Tool(
            name="CoordinateTransformation",
            func=self.astronomy_tools.coordinate_transformation,
            description="天体坐标转换，参数：ra（赤经，小时），dec（赤纬，度），epoch（历元，默认为J2000），target_system（目标坐标系，默认为fk5）"
        ))
        
        # 升起落下时间工具
        tools.append(Tool(
            name="GetRiseSetTimes",
            func=self.astronomy_tools.get_rise_set_times,
            description="获取天体升起和落下时间，参数：body_name（天体名称，如 'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn'），latitude（观测点纬度），longitude（观测点经度），date（日期，可选）"
        ))
        
        # 当前天空天体工具
        def get_current_sky_objects_wrapper(*args, **kwargs):
            """包装器函数，处理各种格式的参数"""
            # 检查参数格式
            if args:
                params = args[0]
                if isinstance(params, str):
                    import re
                    # 尝试从字符串中提取latitude和longitude值
                    lat_match = re.search(r"latitude=([-\d.]+)", params)
                    lon_match = re.search(r"longitude=([-\d.]+)", params)
                    if lat_match and lon_match:
                        latitude = float(lat_match.group(1))
                        longitude = float(lon_match.group(1))
                        return self.astronomy_tools.get_current_sky_objects(latitude, longitude)
                    else:
                        raise ValueError("无效的输入格式，需要包含latitude和longitude值")
                elif isinstance(params, dict):
                    # 处理字典格式的参数
                    latitude = params.get('latitude')
                    longitude = params.get('longitude')
                    if latitude and longitude:
                        return self.astronomy_tools.get_current_sky_objects(latitude, longitude)
                    else:
                        raise ValueError("无效的输入格式，需要包含latitude和longitude值")
            # 处理关键字参数
            elif kwargs:
                latitude = kwargs.get('latitude')
                longitude = kwargs.get('longitude')
                if latitude and longitude:
                    return self.astronomy_tools.get_current_sky_objects(latitude, longitude)
                else:
                    raise ValueError("无效的输入格式，需要包含latitude和longitude值")
            else:
                raise ValueError("无效的输入格式，需要提供参数")
        
        tools.append(Tool(
            name="GetCurrentSkyObjects",
            func=get_current_sky_objects_wrapper,
            description="获取当前天空中的主要天体，参数：latitude（观测点纬度），longitude（观测点经度），date（日期，可选）"
        ))
        
        # 天体基本信息工具
        tools.append(Tool(
            name="GetAstrophysicalObjectInfo",
            func=self.astronomy_tools.get_astrophysical_object_info,
            description="查询天体基本信息，参数：object_name（天体名称）"
        ))
        
        # 星系数据查询工具
        tools.append(Tool(
            name="GetGalaxyData",
            func=self.astronomy_tools.get_galaxy_data,
            description="星系数据查询，参数：galaxy_name（星系名称）"
        ))
        
        # NASA每日天文图工具
        tools.append(Tool(
            name="GetNASAAPOD",
            func=self.astronomy_tools.get_nasa_apod,
            description="获取NASA每日天文图，参数：date（日期，格式为YYYY-MM-DD，可选），hd（是否获取高清图像，可选）"
        ))
        
        # 近地天体数据工具
        tools.append(Tool(
            name="GetNEOData",
            func=self.astronomy_tools.get_neo_data,
            description="获取近地天体数据，参数：start_date（开始日期，格式为YYYY-MM-DD，可选），end_date（结束日期，格式为YYYY-MM-DD，可选），limit（返回结果数量限制，可选）"
        ))
        
        # 天象预测工具
        def get_tonight_best_wrapper(*args, **kwargs):
            """包装器函数，处理无参数调用"""
            return self.events_predictor.get_tonight_best()
        
        def get_weekly_events_wrapper(*args, **kwargs):
            """包装器函数，处理各种格式的参数"""
            if args:
                params = args[0]
                if isinstance(params, dict):
                    start_date = params.get('start_date')
                    return self.events_predictor.get_weekly_events(start_date)
            return self.events_predictor.get_weekly_events()
        
        def get_monthly_events_wrapper(*args, **kwargs):
            """包装器函数，处理各种格式的参数"""
            year = None
            month = None
            if args:
                params = args[0]
                if isinstance(params, dict):
                    year = params.get('year')
                    month = params.get('month')
            return self.events_predictor.get_monthly_events(year, month)
        
        tools.append(Tool(
            name="GetTonightBest",
            func=get_tonight_best_wrapper,
            description="获取今晚最佳观测目标，无需参数"
        ))
        
        tools.append(Tool(
            name="GetWeeklyEvents",
            func=get_weekly_events_wrapper,
            description="获取未来一周的天象，参数：start_date（起始日期，可选，默认为今天）"
        ))
        
        tools.append(Tool(
            name="GetMonthlyEvents",
            func=get_monthly_events_wrapper,
            description="获取未来一个月的天象，参数：year（年份，可选，默认为当前年），month（月份，可选，默认为下个月）"
        ))
        
        logger.info(f"✅ 成功注册 {len(tools)} 个天文工具")
        return tools
    
    def _build_agent(self):
        """构建React Agent"""
        # 创建React提示模板
        from langchain_core.prompts import PromptTemplate
        
        # 从外部文件读取prompt模板
        try:
            with open('prompt_template.txt', 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info("✅ 成功从外部文件读取prompt模板")
        except Exception as e:
            logger.error(f"❌ 读取prompt模板文件失败：{str(e)}")
            # 使用默认模板作为后备
            template = '''
你是一个专业的天文助手，帮助用户解答天文问题。
            
**可用工具列表**：
{tools}

使用以下格式：

Question: {input}
Thought: 我需要思考如何回答这个问题
Action: 选择一个工具
Action Input: 工具参数
Observation: 工具返回结果
Thought: 现在我知道答案了
Final Answer: 最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}
'''
            logger.info("⚠️  使用默认prompt模板")
        
        prompt = PromptTemplate.from_template(template)
        
        # 创建React agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # 创建agent执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
        
        logger.info("✅ React Agent构建完成")
        return agent_executor
    
    def _get_history_text(self):
        """获取对话历史文本"""
        history = self.memory.get_recent_messages()
        history_text = ""
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}：{msg['content']}\n"
        return history_text
    
    def generate_response(self, query: str) -> Generator[str, None, None]:
        """生成流式响应"""
        logger.info(f"\n=== 处理用户查询：{query} ===")
        
        try:
            # 调用Agent执行器获取响应
            response = self.agent_executor.invoke({
                "input": query
            })
            
            # 获取最终响应
            final_response = response.get("output", "")
            
            # 模拟流式输出
            for i in range(0, len(final_response), 50):
                chunk = final_response[i:i+50]
                yield chunk
                time.sleep(0.1)  # 模拟流式效果
            
            # 保存对话到记忆
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", final_response, time.time())
            logger.info(f"✅ 对话已存入记忆 | 助手响应长度：{len(final_response)} 字符")
            
        except Exception as e:
            logger.error(f"❌ 生成响应失败：{str(e)}")
            traceback.print_exc()
            # 提供默认响应
            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            # 保存对话到记忆
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", default_response, time.time())
    
    def add_astronomy_knowledge(self, knowledge: List[str]):
        """添加天文知识到RAG系统"""
        if not knowledge:
            logger.warning("⚠️  无有效知识可添加")
            return
        
        try:
            # 将字符串转换为Document对象
            from langchain.schema import Document
            documents = [Document(page_content=k) for k in knowledge]
            self.rag.add_documents(documents)
            logger.info(f"✅ 成功添加 {len(knowledge)} 条知识到RAG系统")
        except Exception as e:
            logger.error(f"❌ 添加知识失败：{str(e)}")
            traceback.print_exc()
    
    def clear_memory(self):
        """清空记忆"""
        try:
            self.memory.clear()
            logger.info("✅ 记忆已清空")
        except Exception as e:
            logger.error(f"❌ 清空记忆失败：{str(e)}")
            traceback.print_exc()
