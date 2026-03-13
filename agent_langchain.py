from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from config import settings
from rag.online_retriever import OnlineRetriever
from memory import ShortTermMemory
from skills import AstronomySkillRouter
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
import time
import traceback
import json
from logger import logger

# HTTP 客户端库
import httpx
import asyncio
import uuid
import threading
from pathlib import Path
import re

# DashScope 多模态
import dashscope
from dashscope import MultiModalConversation

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

        # Skill 路由层：对上暴露“技能”，对下通过 _call_mcp_tool 访问 MCP 工具
        self.skill_router = AstronomySkillRouter(self._call_mcp_tool)
        
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

    def describe_image(self, image_path: str, prompt: str) -> str:
        """
        使用 DashScope Qwen-VL 对图片做描述/问答。
        image_path: 本地文件路径，将以 file:// URI 传入。
        """
        try:
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            p = Path(image_path).resolve()
            image_uri = f"file://{p}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_uri},
                        {"text": prompt},
                    ],
                }
            ]
            resp = MultiModalConversation.call(
                model=settings.VISION_MODEL_NAME,
                messages=messages,
            )
            # 兼容不同返回结构
            if isinstance(resp, dict):
                # DashScope Python SDK 常见结构：output.choices[0].message.content
                out = resp.get("output") or {}
                choices = out.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    # content 可能是 string 或 list[dict]
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(str(item["text"]))
                        return "\n".join(texts).strip()
            # fallback
            return str(resp)
        except Exception as e:
            logger.error(f"❌ 图片理解失败: {e}")
            return f"图片理解失败：{e}"

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
        """初始化工具（改造为基于 Skill 的高层接口）"""
        tools: List[Tool] = []

        # RAG检索工具（本地，不依赖 MCP）
        def rag_retrieve(query: str):
            """使用RAG系统检索相关天文信息"""
            return self.rag.get_relevant_context(query)

        tools.append(
            Tool(
                name="RAGRetrieve",
                func=rag_retrieve,
                description="使用本地RAG知识库检索天文知识、概念解释、历史资料等。参数：query（查询语句，中文即可）。",
            )
        )

        # ===== Skill 层工具：上层Agent只看到这些“技能”，不直接看到底层 MCP 工具 =====

        def weather_lookup_skill(
            city: str = None,
            location: str = None,
            extensions: str = "all",
        ):
            """天气查询技能：包装 get_weather，用于单独查询观测相关天气"""
            # city 与 location 二选一，优先使用 city
            target = city or location
            return self.skill_router.call(
                "weather-lookup",
                city=target,
                location=target,
                extensions=extensions,
            )

        tools.append(
            Tool(
                name="WeatherLookup",
                func=weather_lookup_skill,
                description=(
                    "查询指定城市的观测相关天气信息（skill: weather-lookup，对应 MCP 工具 get_weather）。\n"
                    "参数：city（城市名称或adcode，可选），"
                    "location（城市名称，和 city 等价，可选），"
                    "extensions（\"base\" 实时 或 \"all\" 预报，默认 all）。"
                ),
            )
        )

        def observation_planner_skill(date: str = None, location: str = None, duration: str = None):
            """观测计划技能：封装天气、天象事件等多源信息"""
            return self.skill_router.call(
                "observation-planner",
                date=date,
                location=location,
                duration=duration,
            )

        tools.append(
            Tool(
                name="ObservationPlanner",
                func=observation_planner_skill,
                description=(
                    "生成指定日期和地点的天文观测计划（skill: observation-planner）。\n"
                    "参数：date（观测日期，可为“今天”“明天”或YYYY-MM-DD，可选），"
                    "location（观测地点，城市名或“纬度,经度”，必填），"
                    "duration（观测时段，如“整晚”“前半夜”“后半夜”，可选）。"
                ),
            )
        )

        def celestial_events_forecast_skill(start_date: str = None, end_date: str = None, event_type: str = None):
            """天象预报技能：封装一周/月事件查询"""
            return self.skill_router.call(
                "celestial-events-forecast",
                start_date=start_date,
                end_date=end_date,
                event_type=event_type,
            )

        tools.append(
            Tool(
                name="CelestialEventsForecast",
                func=celestial_events_forecast_skill,
                description=(
                    "查询指定时间段的天象事件（skill: celestial-events-forecast）。\n"
                    "参数：start_date（开始日期YYYY-MM-DD，可选），"
                    "end_date（结束日期YYYY-MM-DD，可选），"
                    "event_type（事件类型，如“流星雨”“行星合月”“月食”，可选，用于意图说明）。"
                ),
            )
        )

        def deep_sky_observing_guide_skill(
            target: str,
            observer_location: str = None,
            date: str = None,
            equipment: str = None,
        ):
            """深空观测指导技能"""
            return self.skill_router.call(
                "deep-sky-observing-guide",
                target=target,
                observer_location=observer_location,
                date=date,
                equipment=equipment,
            )

        tools.append(
            Tool(
                name="DeepSkyObservingGuide",
                func=deep_sky_observing_guide_skill,
                description=(
                    "为指定深空天体提供观测指导（skill: deep-sky-observing-guide）。\n"
                    "参数：target（天体名称，如“M31”“猎户座大星云”，必填），"
                    "observer_location（观测者位置，可选），"
                    "date（观测日期，可选），"
                    "equipment（设备描述，如“裸眼”“双筒”“8寸望远镜”，可选）。"
                ),
            )
        )

        def neo_tracker_skill(
            time_range: str = None,
            min_size: float = None,
            max_distance: float = None,
            observable_only: bool = None,
        ):
            """近地天体追踪技能"""
            return self.skill_router.call(
                "neo-tracker",
                time_range=time_range,
                min_size=min_size,
                max_distance=max_distance,
                observable_only=observable_only,
            )

        tools.append(
            Tool(
                name="NEOTracker",
                func=neo_tracker_skill,
                description=(
                    "追踪近地天体飞掠事件（skill: neo-tracker）。\n"
                    "参数：time_range（时间范围，如“未来30天”“本月”，可选），"
                    "min_size（最小直径，单位米，可选），"
                    "max_distance（最大距离，单位地月距离倍数，可选），"
                    "observable_only（是否只返回具有观测价值的目标，布尔值，可选）。"
                ),
            )
        )

        def astrophotography_calculator_skill(
            target: str,
            camera: str,
            telescope: str = None,
            mount: str = None,
            location: str = None,
            date: str = None,
        ):
            """天文摄影参数计算技能"""
            return self.skill_router.call(
                "astrophotography-calculator",
                target=target,
                camera=camera,
                telescope=telescope,
                mount=mount,
                location=location,
                date=date,
            )

        tools.append(
            Tool(
                name="AstrophotographyCalculator",
                func=astrophotography_calculator_skill,
                description=(
                    "计算天文摄影参数与拍摄建议（skill: astrophotography-calculator）。\n"
                    "参数：target（拍摄目标，必填），"
                    "camera（相机型号，必填），"
                    "telescope（望远镜型号或焦距，可选），"
                    "mount（赤道仪型号，可选），"
                    "location（拍摄地点，可选），"
                    "date（拍摄日期，可选）。"
                ),
            )
        )

        def celestial_position_calculator_skill(
            target: str,
            datetime: str,
            location: str,
            output_format: str = None,
        ):
            """天体位置计算技能"""
            return self.skill_router.call(
                "celestial-position-calculator",
                target=target,
                datetime=datetime,
                location=location,
                output_format=output_format,
            )

        tools.append(
            Tool(
                name="CelestialPositionCalculator",
                func=celestial_position_calculator_skill,
                description=(
                    "计算天体在指定时间的位置（skill: celestial-position-calculator）。\n"
                    "参数：target（目标名称，如“mars”“jupiter”等，必填），"
                    "datetime（观测时间，建议YYYY-MM-DD HH:MM 格式，必填），"
                    "location（观测地点，经纬度“纬度,经度”形式，必填），"
                    "output_format（输出坐标格式，如“altaz”“radec”，可选）。"
                ),
            )
        )

        logger.info(f"✅ 成功注册 {len(tools)} 个高层技能工具（含RAG）")
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

    def _try_web_search_fallback(self, query: str) -> str:
        """
        降级机制：当工具调用失败时，尝试使用联网搜索
        """
        logger.warning("检测到工具调用可能失败，尝试使用联网搜索...")
        try:
            search_result = self._call_mcp_tool("web_search", query=query, max_results=5)
            logger.info("联网搜索降级方案执行成功")
            return search_result
        except Exception as e:
            logger.error(f"联网搜索降级也失败: {e}")
            return json.dumps({"error": f"降级搜索失败: {str(e)}"}, ensure_ascii=False)

    def generate_response(self, query: str) -> Generator[str, None, None]:
        """生成流式响应"""
        logger.info(f"\n=== 处理用户查询：{query} ===")

        # 首先尝试正常流程
        tool_call_failed = False
        fallback_used = False

        try:
            response = self.agent_executor.invoke({
                "input": query
            })

            output = response.get("output", "")

            # 检查是否需要降级
            # 如果返回结果包含错误指示或结果为空，尝试降级
            if self._should_use_fallback(output):
                logger.warning("检测到工具调用可能未返回有效结果，尝试联网搜索...")
                tool_call_failed = True
                search_result = self._try_web_search_fallback(query)
                output = self._format_fallback_response(query, search_result)
                fallback_used = True
            else:
                output = response.get("output", "")

            final_response = output

            for i in range(0, len(final_response), 50):
                chunk = final_response[i:i+50]
                yield chunk
                time.sleep(0.1)

            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", final_response, time.time())

            if fallback_used:
                logger.info(f"✅ 使用联网搜索降级 | 助手响应长度：{len(final_response)} 字符")
            else:
                logger.info(f"✅ 对话已存入记忆 | 助手响应长度：{len(final_response)} 字符")

        except Exception as e:
            logger.error(f"❌ 生成响应失败：{str(e)}")
            traceback.print_exc()

            # 降级机制：工具调用失败时尝试联网搜索
            if not tool_call_failed:
                logger.warning("检测到异常，尝试使用联网搜索降级...")
                try:
                    search_result = self._try_web_search_fallback(query)
                    fallback_response = self._format_fallback_response(query, search_result)
                    fallback_used = True

                    for i in range(0, len(fallback_response), 50):
                        chunk = fallback_response[i:i+50]
                        yield chunk
                        time.sleep(0.1)

                    self.memory.add_message("user", query, time.time())
                    self.memory.add_message("assistant", fallback_response, time.time())
                    logger.info(f"✅ 降级搜索成功 | 助手响应长度：{len(fallback_response)} 字符")
                    return
                except Exception as fallback_error:
                    logger.error(f"降级搜索也失败: {fallback_error}")

            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", default_response, time.time())

    def _should_use_fallback(self, output: str) -> bool:
        """
        判断是否应该使用降级搜索
        """
        if not output:
            return True
        # 更精确地检查“工具调用失败”类错误，避免正常业务文案（如“本周没有特殊天象”）触发降级
        error_patterns = [
            "工具调用错误",
            "调用工具失败",
            "调用工具超时",
            "无法连接到MCP服务器",
            "MCP会话未初始化",
            "HTTP错误",
        ]
        low_confidence_phrases = [
            "当前模型服务暂时不可用",
            "无法回答你的问题",
        ]

        condensed = output.strip()
        if not condensed:
            return True
        # 检查明显的工具/基础设施错误
        for kw in error_patterns:
            if kw in condensed:
                return True
        # 极短且带有低置信度措辞的回复也视为失败
        if len(condensed) < 60:
            for kw in low_confidence_phrases:
                if kw in condensed:
                    return True
        return False

    def _format_fallback_response(self, query: str, search_result: str) -> str:
        """
        格式化降级搜索结果为自然语言回复
        """
        try:
            result_data = json.loads(search_result)

            if "error" in result_data:
                return f"抱歉，我在处理您的查询「{query}」时遇到了问题：{result_data['error']}。请稍后再试或尝试其他问题。"

            answer = result_data.get("answer", "")
            results = result_data.get("results", [])

            if answer:
                response = f"根据搜索结果：\n\n{answer}\n\n"
            else:
                response = f"关于「{query}」，我找到以下信息：\n\n"

            for i, item in enumerate(results[:3], 1):
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")
                if title:
                    response += f"{i}. {title}\n"
                    if content:
                        response += f"   {content[:150]}...\n"
                    response += f"   来源: {url}\n\n"

            return response

        except Exception as e:
            logger.error(f"格式化降级结果失败: {e}")
            return f"抱歉，处理搜索结果时出现问题。请稍后再试。"

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

    async def generate_events(
        self, query: str, image_path: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        事件流：同时输出 text/image 两类事件，便于 API 返回图片。

        - text: {"type": "text", "content": "..."}
        - image: {"type": "image", "url": "...", "meta": {...}}
        """
        # 如果用户上传了图片，先做一次视觉理解，把结果注入 query
        if image_path:
            yield {
                "type": "text",
                "content": "已收到图片，正在进行视觉分析并结合天文知识回答……",
            }
            vision_prompt = (
                "请详细描述这张图片的内容。若包含星空/天体/望远镜设备，请指出："
                "1) 可能的天体/星座/现象；2) 光害/天空质量线索；3) 设备与拍摄参数线索；"
                "4) 适合的后续观测或拍摄建议。"
            )
            vision_desc = self.describe_image(image_path=image_path, prompt=vision_prompt)
            query = f"{query}\n\n[用户上传图片的视觉信息]\n{vision_desc}"

        # 复用现有真流式 token 输出，但升级为事件格式
        request_id = uuid.uuid4().hex[:8]
        self._current_request_id = request_id
        logger.info(f"[{request_id}] 开始处理事件流查询：{query[:200]}")

        final_chunks: list[str] = []

        def _maybe_extract_image_url(text: str) -> Optional[str]:
            # 简单提取常见图片 URL
            m = re.search(r"(https?://\\S+\\.(?:png|jpg|jpeg|webp))", text, re.IGNORECASE)
            return m.group(1) if m else None

        try:
            async for event in self.agent_executor.astream_events(
                {"input": query},
                version="v1",
            ):
                event_type = event.get("event")
                data = event.get("data", {}) or {}
                run_id = event.get("run_id")

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

                if event_type == "on_tool_end":
                    meta = self._tool_runs.pop(run_id, {})
                    duration = None
                    if meta.get("start_time") is not None:
                        duration = time.time() - meta["start_time"]
                    tool_output = data.get("output")
                    tool_output_str = "" if tool_output is None else str(tool_output)

                    logger.info(
                        json.dumps(
                            {
                                "type": "tool_end",
                                "request_id": request_id,
                                "run_id": run_id,
                                "tool_name": meta.get("name"),
                                "duration_sec": duration,
                                "output_preview": tool_output_str[:200],
                            },
                            ensure_ascii=False,
                        )
                    )

                    # 从工具输出中尽量抽取图片 URL（例如 NASA APOD 的 url/hdurl）
                    extracted_url = None
                    # 1) 如果工具输出是 JSON 字符串，解析 url 字段
                    if tool_output_str.strip().startswith("{"):
                        try:
                            obj = json.loads(tool_output_str)
                            if isinstance(obj, dict):
                                extracted_url = obj.get("hdurl") or obj.get("url")
                        except Exception:
                            extracted_url = None
                    # 2) 兜底：正则从文本里抓图片链接
                    if not extracted_url:
                        extracted_url = _maybe_extract_image_url(tool_output_str)

                    if extracted_url:
                        yield {
                            "type": "image",
                            "url": extracted_url,
                            "meta": {
                                "request_id": request_id,
                                "tool": meta.get("name"),
                            },
                        }
                    continue

                if event_type in ("on_chat_model_stream", "on_llm_stream"):
                    chunk = data.get("chunk")
                    if not chunk:
                        continue
                    text = getattr(chunk, "content", None) or getattr(chunk, "text", None)
                    if not text:
                        continue
                    final_chunks.append(text)
                    yield {"type": "text", "content": text}

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 事件流生成失败：{e}")
            traceback.print_exc()
            fallback = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield {"type": "text", "content": fallback}
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", fallback, time.time())
        else:
            final_response = "".join(final_chunks)
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", final_response, time.time())
            logger.info(f"[{request_id}] ✅ 事件流完成，响应长度：{len(final_response)} 字符")
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