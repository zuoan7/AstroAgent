from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from config import settings
from rag.online_retriever import OnlineRetriever
from memory import ShortTermMemory
from typing import Generator, List, Dict, Any, Optional, AsyncGenerator
import time
import traceback
import json
from logger import logger

# HTTP 客户端库
import httpx
import asyncio
import uuid
import threading

# MCP服务器配置 - 注意没有结尾斜杠！
# 默认使用 8001 端口，与 FastAPI 服务错开
MCP_SERVER_URL = "http://localhost:8001/mcp"


class AstroAgent:
    """基于LangChain的天文Agent - 支持HTTP调用MCP服务器（完整会话管理）"""
    
    def __init__(self):
        # 验证API Key
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("❌ DashScope API Key未配置！请在settings中设置DASHSCOPE_API_KEY")
        
        # 初始化组件
        self.rag = OnlineRetriever()
        self.memory = ShortTermMemory()
        self.llm = self._init_llm()
        
        # MCP会话管理
        self.mcp_session_id: Optional[str] = None
        self.mcp_initialized = False
        self.http_client: Optional[httpx.Client] = None
        # 工具调用跟踪
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._current_request_id: Optional[str] = None
        
        # 初始化MCP会话（使用同步方式）
        self._init_mcp_session_sync()
        
        # 初始化工具
        self.tools = self._init_tools()
        self.agent_executor = self._build_agent()
        
        logger.info("✅ AstroAgent初始化完成，使用HTTP方式调用MCP服务器工具（完整会话管理）")
    
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

    def _parse_sse_response(self, response_text: str) -> Optional[dict]:
        """
        解析 SSE 格式的响应
        
        Args:
            response_text: SSE 响应文本
            
        Returns:
            解析后的 JSON 字典，如果解析失败返回 None
        """
        try:
            # SSE 格式通常是 "event: message\ndata: {...}\n\n"
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith("data: "):
                    json_str = line[6:]  # 去掉 "data: "
                    return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"解析 SSE 响应失败: {e}")
            return None

    def _init_mcp_session_sync(self):
        """使用同步方式初始化MCP会话"""
        try:
            # 使用同步HTTP客户端
            client = httpx.Client(timeout=30.0)
            
            # 1. 建立SSE连接获取session ID
            logger.info("正在建立SSE连接...")
            sse_response = client.get(
                MCP_SERVER_URL,
                headers={"Accept": "text/event-stream"}
            )
            
            # 修正2：使用小写的 header 名称
            session_id = sse_response.headers.get("mcp-session-id")
            if not session_id:
                raise Exception("无法获取session ID")
            
            logger.info(f"✅ 获取到session ID: {session_id}")
            
            # 2. 发送初始化请求
            logger.info("发送初始化请求...")
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "AstroAgent",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }
            
            response = client.post(
                MCP_SERVER_URL,
                json=init_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"初始化失败: {response.status_code}")
            
            # 解析初始化响应（验证服务器返回）
            init_result = self._parse_sse_response(response.text)
            if init_result:
                logger.debug(f"初始化成功，服务器信息: {init_result.get('result', {}).get('serverInfo', {})}")
            else:
                logger.warning("无法解析初始化响应")
            
            # 3. 发送initialized通知
            logger.info("发送initialized通知...")
            notif_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            client.post(
                MCP_SERVER_URL,
                json=notif_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )
            
            # 4. 获取工具列表（可选，用于验证）
            logger.info("获取工具列表...")
            list_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
            
            response = client.post(
                MCP_SERVER_URL,
                json=list_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )
            
            # 解析工具列表
            tools_result = self._parse_sse_response(response.text)
            if tools_result:
                tools_list = tools_result.get("result", {}).get("tools", [])
                logger.info(f"✅ 从服务器获取到 {len(tools_list)} 个工具")
            else:
                logger.warning("无法解析工具列表响应")
            
            # 保存会话信息
            self.mcp_session_id = session_id
            self.mcp_initialized = True
            self.http_client = client
            
            logger.info(f"✅ MCP会话初始化成功，会话ID: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ MCP会话初始化失败: {e}")
            self.mcp_initialized = False
            self.http_client = None
    
    def _call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """调用MCP工具（同步方法）- 修复参数传递问题"""
        if not self.mcp_initialized or not self.mcp_session_id:
            logger.error("❌ MCP会话未初始化")
            return f"错误：MCP会话未初始化，请检查桥服务器是否运行"
        
        try:
            # 重要：确保参数类型正确
            # 对于数字参数，确保它们是整数类型
            processed_kwargs = {}
            for key, value in kwargs.items():
                if key in ['year', 'month', 'limit']:
                    try:
                        # 如果是字符串数字，转换为整数
                        if isinstance(value, str) and value.isdigit():
                            processed_kwargs[key] = int(value)
                        # 如果是其他字符串（如日期），保持原样
                        elif isinstance(value, str):
                            processed_kwargs[key] = value
                        # 如果是数字，直接使用
                        else:
                            processed_kwargs[key] = value
                    except:
                        processed_kwargs[key] = value
                else:
                    processed_kwargs[key] = value
            
            # 构建标准MCP工具调用请求
            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": processed_kwargs  # 使用处理后的参数
                },
                "id": int(time.time() * 1000)
            }
            
            logger.debug(f"调用工具 {tool_name}，处理后的参数: {processed_kwargs}")
            
            # 使用同步客户端发送请求
            response = self.http_client.post(
                MCP_SERVER_URL,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self.mcp_session_id
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                return f"HTTP错误: {response.status_code}"
            
            # 解析 SSE 格式的响应
            result = self._parse_sse_response(response.text)
            if not result:
                logger.error(f"无法解析响应: {response.text[:200]}")
                return f"解析响应失败"
            
            logger.debug(f"工具响应: {json.dumps(result, ensure_ascii=False)[:200]}")
            
            # 检查错误
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                error_code = result["error"].get("code", "")
                return f"工具调用错误 [{error_code}]: {error_msg}"
            
            # 从标准格式中提取文本内容
            if "result" in result:
                # 处理标准MCP格式：result.content[0].text
                if "content" in result["result"]:
                    content = result["result"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        for item in content:
                            if item.get("type") == "text":
                                return item.get("text", "")
                
                # 处理直接返回字符串的情况
                if isinstance(result["result"], str):
                    return result["result"]
                
                # 处理其他格式
                return str(result["result"])
            
            # 如果既没有error也没有result
            logger.warning(f"未知响应格式: {result}")
            return str(result)
            
        except httpx.TimeoutException:
            logger.error(f"❌ MCP工具调用超时: {tool_name}")
            return f"调用工具超时，请稍后重试"
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到MCP服务器: {MCP_SERVER_URL}")
            return f"错误：无法连接到MCP服务器"
        except Exception as e:
            logger.error(f"❌ 调用工具 {tool_name} 失败: {e}")
            return f"调用工具失败: {str(e)}"

    def _init_tools(self):
        """初始化工具"""
        tools = []
        
        # RAG检索工具（本地）
        def rag_retrieve(query):
            """使用RAG系统检索相关天文信息"""
            return self.rag.get_relevant_context(query)
        
        tools.append(Tool(
            name="RAGRetrieve",
            func=rag_retrieve,
            description="使用RAG系统检索相关天文信息，参数：query（查询语句）"
        ))
        
        # 行星位置计算工具
        def get_planet_position(planet_name, observation_time=None, latitude=None, longitude=None):
            """获取行星位置"""
            return self._call_mcp_tool(
                "get_planet_position", 
                planet_name=planet_name, 
                observation_time=observation_time, 
                latitude=latitude, 
                longitude=longitude
            )
        
        tools.append(Tool(
            name="GetPlanetPosition",
            func=get_planet_position,
            description="获取行星位置，参数：planet_name（行星名称），observation_time（可选），latitude（可选），longitude（可选）"
        ))
        
        # 天体坐标转换工具
        def coordinate_transformation(ra, dec, epoch="J2000", target_system="fk5"):
            """天体坐标转换"""
            return self._call_mcp_tool(
                "coordinate_transformation", 
                ra=ra, 
                dec=dec, 
                epoch=epoch, 
                target_system=target_system
            )
        
        tools.append(Tool(
            name="CoordinateTransformation",
            func=coordinate_transformation,
            description="天体坐标转换，参数：ra（赤经），dec（赤纬），epoch（历元），target_system（目标坐标系）"
        ))
        
        # 升起落下时间工具
        def get_rise_set_times(body_name, latitude, longitude, date=None):
            """获取天体升起和落下时间"""
            return self._call_mcp_tool(
                "get_rise_set_times", 
                body_name=body_name, 
                latitude=latitude, 
                longitude=longitude, 
                date=date
            )
        
        tools.append(Tool(
            name="GetRiseSetTimes",
            func=get_rise_set_times,
            description="获取天体升起和落下时间，参数：body_name（天体名称），latitude（纬度），longitude（经度），date（日期，可选）"
        ))
        
        # 当前天空天体工具
        def get_current_sky_objects(latitude, longitude=None, date=None):
            """获取当前天空中的主要天体"""
            return self._call_mcp_tool(
                "get_current_sky_objects",
                latitude=latitude,
                longitude=longitude,
                date=date,
            )
        
        tools.append(Tool(
            name="GetCurrentSkyObjects",
            func=get_current_sky_objects,
            description="获取当前天空中的主要天体，参数：latitude（纬度），longitude（经度），date（日期，可选）"
        ))
        
        # 天体基本信息工具
        def get_astrophysical_object_info(object_name):
            """查询天体基本信息"""
            return self._call_mcp_tool(
                "get_astrophysical_object_info", 
                object_name=object_name
            )
        
        tools.append(Tool(
            name="GetAstrophysicalObjectInfo",
            func=get_astrophysical_object_info,
            description="查询天体基本信息，参数：object_name（天体名称）"
        ))
        
        # 星系数据查询工具
        def get_galaxy_data(galaxy_name):
            """星系数据查询"""
            return self._call_mcp_tool(
                "get_galaxy_data", 
                galaxy_name=galaxy_name
            )
        
        tools.append(Tool(
            name="GetGalaxyData",
            func=get_galaxy_data,
            description="星系数据查询，参数：galaxy_name（星系名称）"
        ))
        
        # NASA每日天文图工具
        def get_nasa_apod(date=None, hd=False):
            """获取NASA每日天文图"""
            return self._call_mcp_tool(
                "get_nasa_apod", 
                date=date, 
                hd=hd
            )
        
        tools.append(Tool(
            name="GetNASAAPOD",
            func=get_nasa_apod,
            description="获取NASA每日天文图，参数：date（日期），hd（是否高清）"
        ))
        
        # 近地天体数据工具
        def get_neo_data(start_date=None, end_date=None, limit=10):
            """获取近地天体数据"""
            return self._call_mcp_tool(
                "get_neo_data", 
                start_date=start_date, 
                end_date=end_date, 
                limit=limit
            )
        
        tools.append(Tool(
            name="GetNEOData",
            func=get_neo_data,
            description="获取近地天体数据，参数：start_date（开始日期），end_date（结束日期），limit（数量限制）"
        ))
        
        # 今晚最佳观测目标工具
        def get_tonight_best(*args, **kwargs):
            """获取今晚最佳观测目标"""
            return self._call_mcp_tool("get_tonight_best")
        
        tools.append(Tool(
            name="GetTonightBest",
            func=get_tonight_best,
            description="获取今晚最佳观测目标"
        ))
        
        # 未来一周天象工具
        def get_weekly_events(start_date=None):
            """获取未来一周的天象"""
            return self._call_mcp_tool(
                "get_weekly_events", 
                start_date=start_date
            )
        
        tools.append(Tool(
            name="GetWeeklyEvents",
            func=get_weekly_events,
            description="获取未来一周的天象，参数：start_date（起始日期）"
        ))
        
        # 未来一个月天象工具 - 修复参数传递
        def get_monthly_events(year=None, month=None):
            """获取未来一个月的天象"""
            # 确保year和month是整数类型
            params = {}
            if year is not None:
                try:
                    params['year'] = int(year)
                except:
                    params['year'] = year
            if month is not None:
                try:
                    params['month'] = int(month)
                except:
                    params['month'] = month
            
            return self._call_mcp_tool(
                "get_monthly_events", 
                **params
            )
        
        tools.append(Tool(
            name="GetMonthlyEvents",
            func=get_monthly_events,
            description="获取未来一个月的天象，参数：year（年份），month（月份）"
        ))
        
        logger.info(f"✅ 成功注册 {len(tools)} 个天文工具")
        return tools
    
    def _build_agent(self):
        """构建React Agent"""
        from langchain_core.prompts import PromptTemplate
        
        try:
            with open('prompt_template.txt', 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info("✅ 成功从外部文件读取prompt模板")
        except Exception as e:
            logger.error(f"❌ 读取prompt模板文件失败：{str(e)}")
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
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True  # 添加错误处理
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
            response = self.agent_executor.invoke({
                "input": query
            })
            
            final_response = response.get("output", "")
            
            for i in range(0, len(final_response), 50):
                chunk = final_response[i:i+50]
                yield chunk
                time.sleep(0.1)
            
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", final_response, time.time())
            logger.info(f"✅ 对话已存入记忆 | 助手响应长度：{len(final_response)} 字符")
            
        except Exception as e:
            logger.error(f"❌ 生成响应失败：{str(e)}")
            traceback.print_exc()
            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", default_response, time.time())

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        使用 LangChain Agent 的事件流实现端到端流式输出，并带有工具调用 trace。
        """
        request_id = uuid.uuid4().hex[:8]
        self._current_request_id = request_id
        logger.info(f"[{request_id}] 开始处理流式查询：{query}")

        final_chunks: list[str] = []

        try:
            # 使用事件流接口捕获 LLM token 与工具事件
            async for event in self.agent_executor.astream_events(
                {"input": query},
                version="v1",
            ):
                event_type = event.get("event")
                data = event.get("data", {}) or {}
                run_id = event.get("run_id")

                # 工具开始：记录输入与开始时间
                if event_type == "on_tool_start":
                    tool_name = data.get("name") or data.get("tool")
                    tool_input = data.get("input")
                    self._tool_runs[run_id] = {
                        "name": tool_name,
                        "input": str(tool_input),
                        "start_time": time.time(),
                        "request_id": request_id,
                    }
                    logger.info(
                        json.dumps(
                            {
                                "type": "tool_start",
                                "request_id": request_id,
                                "run_id": run_id,
                                "tool_name": tool_name,
                                "input": str(tool_input),
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

                # 工具结束：记录输出与耗时
                if event_type == "on_tool_end":
                    meta = self._tool_runs.pop(run_id, {})
                    duration = None
                    if meta.get("start_time") is not None:
                        duration = time.time() - meta["start_time"]
                    tool_output = data.get("output")
                    logger.info(
                        json.dumps(
                            {
                                "type": "tool_end",
                                "request_id": request_id,
                                "run_id": run_id,
                                "tool_name": meta.get("name"),
                                "duration_sec": duration,
                                "output_preview": str(tool_output)[:200],
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

                # LLM token 级别流：on_chat_model_stream / on_llm_stream
                if event_type in ("on_chat_model_stream", "on_llm_stream"):
                    chunk = data.get("chunk")
                    if not chunk:
                        continue

                    text = getattr(chunk, "content", None) or getattr(
                        chunk, "text", None
                    )
                    if not text:
                        continue

                    final_chunks.append(text)
                    yield text

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 流式生成失败：{e}")
            traceback.print_exc()
            fallback = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield fallback
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", fallback, time.time())
        else:
            final_response = "".join(final_chunks)
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", final_response, time.time())
            logger.info(
                f"[{request_id}] ✅ 流式对话完成，响应长度：{len(final_response)} 字符"
            )
        finally:
            self._current_request_id = None
    
    def add_astronomy_knowledge(self, knowledge: List[str]):
        """添加天文知识到RAG系统"""
        if not knowledge:
            logger.warning("⚠️  无有效知识可添加")
            return
        
        try:
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
    
    def __del__(self):
        """析构函数，确保HTTP客户端被正确关闭"""
        if hasattr(self, 'http_client') and self.http_client:
            try:
                self.http_client.close()
                logger.info("✅ HTTP客户端已关闭")
            except:
                pass