import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from logger import logger
from core.errors import ErrorHandler
from utils.helpers import extract_image_url


class StreamingService:
    def __init__(
        self,
        agent_executor: Any,
        memory: Any,
        long_term_memory: Any = None,
        user_id: str = "anonymous",
        fallback_service: Optional[Any] = None,
    ):
        self._agent_executor = agent_executor
        self._memory = memory
        self._long_term_memory = long_term_memory
        self._user_id = user_id
        self._fallback_service = fallback_service
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._current_request_id: Optional[str] = None
        self._action_history: Dict[str, list] = {}
        self._max_same_action_count = 2

    def _format_chat_history(self) -> str:
        messages = self._memory.get_recent_messages()
        if not messages:
            return "无历史对话"
        formatted = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def _format_user_profile(self) -> str:
        """格式化用户画像"""
        if not self._long_term_memory:
            return "暂无用户偏好信息"
        return self._long_term_memory.format_profile_for_prompt(self._user_id)

    def _extract_and_update_long_term_memory(self, user_message: str, assistant_message: str):
        """从对话中提取并更新长期记忆"""
        if not self._long_term_memory:
            return

        try:
            extracted = self._long_term_memory.extract_from_conversation(user_message, assistant_message)

            has_info = (
                extracted.get("preferences") or
                extracted.get("habits") or
                extracted.get("constraints")
            )

            if has_info:
                self._long_term_memory.merge_and_update(self._user_id, extracted)
                logger.info(f"✅ 已提取并更新长期记忆 (user_id: {self._user_id})")
        except Exception as e:
            logger.error(f"❌ 更新长期记忆失败: {e}")

    def _check_repeated_action(self, request_id: str, tool_name: str, tool_input: str) -> bool:
        if request_id not in self._action_history:
            self._action_history[request_id] = []
        
        action_key = f"{tool_name}:{tool_input}"
        history = self._action_history[request_id]
        
        history.append(action_key)
        same_count = sum(1 for h in history if h == action_key)
        
        if same_count >= self._max_same_action_count:
            logger.warning(
                f"[{request_id}] ⚠️ 检测到重复操作：{tool_name} 已执行 {same_count} 次，强制终止"
            )
            return True
        
        return False

    def _build_response_from_intermediate_steps(
        self, query: str, intermediate_steps: list
    ) -> str:
        if not intermediate_steps:
            return ""
        
        tool_results = []
        for step in intermediate_steps:
            if hasattr(step, '__iter__') and len(step) >= 2:
                action, observation = step[0], step[1]
                tool_name = getattr(action, 'tool', 'unknown')
                tool_input = getattr(action, 'tool_input', '')
                if observation and not ErrorHandler.is_error_response(observation):
                    try:
                        obs_data = json.loads(str(observation))
                        if isinstance(obs_data, dict) and obs_data.get("error"):
                            continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                    tool_results.append({
                        "tool": tool_name,
                        "input": str(tool_input)[:100],
                        "output": str(observation)[:500]
                    })
        
        if not tool_results:
            return ""
        
        response_parts = [f"根据已获取的信息，为您回答「{query}」：\n"]
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result["tool"]
            output = result["output"]
            
            try:
                output_data = json.loads(output)
                if isinstance(output_data, dict):
                    if "error" in output_data:
                        continue
                    if "answer" in output_data:
                        response_parts.append(f"\n{output_data['answer']}")
                        continue
                    if "name" in output_data:
                        response_parts.append(f"\n目标名称：{output_data.get('name', '未知')}")
                    if "ra" in output_data and "dec" in output_data:
                        response_parts.append(f"位置：赤经 {output_data['ra']}°，赤纬 {output_data['dec']}°")
                    for key, value in list(output_data.items())[:5]:
                        if key not in ["name", "ra", "dec", "error", "answer"]:
                            response_parts.append(f"\n{key}：{value}")
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            
            if len(output) > 50:
                response_parts.append(f"\n{output}")
        
        response_parts.append("\n\n（注：由于处理时间限制，以上是基于已获取数据整理的信息。）")
        return "".join(response_parts)

    def generate_response(self, query: str) -> Generator[str, None, None]:
        logger.info(f"\n=== 处理用户查询：{query} ===")

        tool_call_failed = False
        fallback_used = False

        try:
            chat_history = self._format_chat_history()
            user_profile = self._format_user_profile()
            response = self._agent_executor.invoke({
                "input": query,
                "chat_history": chat_history,
                "user_profile": user_profile
            })
            output = response.get("output", "")
            intermediate_steps = response.get("intermediate_steps", [])

            if self._fallback_service and self._fallback_service.should_use_fallback(output):
                logger.warning("检测到工具调用可能未返回有效结果，尝试从中间步骤生成答案...")
                tool_call_failed = True
                
                if intermediate_steps:
                    built_response = self._build_response_from_intermediate_steps(query, intermediate_steps)
                    if built_response:
                        output = built_response
                        logger.info("✅ 成功从中间步骤生成答案")
                    else:
                        search_result = self._fallback_service.try_web_search_fallback(query)
                        output = self._fallback_service.format_fallback_response(query, search_result)
                        fallback_used = True
                else:
                    search_result = self._fallback_service.try_web_search_fallback(query)
                    output = self._fallback_service.format_fallback_response(query, search_result)
                    fallback_used = True

            final_response = output

            for i in range(0, len(final_response), 50):
                chunk = final_response[i:i+50]
                yield chunk
                time.sleep(0.1)

            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", final_response, time.time())

            # 提取并更新长期记忆
            self._extract_and_update_long_term_memory(query, final_response)

            if fallback_used:
                logger.info(f"✅ 使用联网搜索降级 | 助手响应长度：{len(final_response)} 字符")
            else:
                logger.info(f"✅ 对话已存入记忆 | 助手响应长度：{len(final_response)} 字符")

        except Exception as e:
            logger.error(f"❌ 生成响应失败：{str(e)}")

            if self._fallback_service and not tool_call_failed:
                logger.warning("检测到异常，尝试使用联网搜索降级...")
                try:
                    search_result = self._fallback_service.try_web_search_fallback(query)
                    fallback_response = self._fallback_service.format_fallback_response(query, search_result)
                    fallback_used = True

                    for i in range(0, len(fallback_response), 50):
                        chunk = fallback_response[i:i+50]
                        yield chunk
                        time.sleep(0.1)

                    self._memory.add_message("user", query, time.time())
                    self._memory.add_message("assistant", fallback_response, time.time())

                    # 提取并更新长期记忆
                    self._extract_and_update_long_term_memory(query, fallback_response)

                    logger.info(f"✅ 降级搜索成功 | 助手响应长度：{len(fallback_response)} 字符")
                    return
                except Exception as fallback_error:
                    logger.error(f"降级搜索也失败: {fallback_error}")

            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", default_response, time.time())

            # 提取并更新长期记忆
            self._extract_and_update_long_term_memory(query, default_response)

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        request_id = uuid.uuid4().hex[:8]
        self._current_request_id = request_id
        logger.info(f"[{request_id}] 开始处理流式查询：{query}")

        final_chunks: list[str] = []
        should_stop = False

        try:
            chat_history = self._format_chat_history()
            user_profile = self._format_user_profile()
            async for event in self._agent_executor.astream_events(
                {
                    "input": query,
                    "chat_history": chat_history,
                    "user_profile": user_profile
                },
                version="v1",
            ):
                if should_stop:
                    break
                    
                event_type = event.get("event")
                data = event.get("data", {}) or {}
                run_id = event.get("run_id")

                if event_type == "on_tool_start":
                    tool_name = data.get("name") or data.get("tool")
                    tool_input = data.get("input")
                    tool_input_str = str(tool_input) if tool_input else ""
                    
                    if self._check_repeated_action(request_id, tool_name or "unknown_tool", tool_input_str):
                        should_stop = True
                        fallback_msg = "\n\n⚠️ 检测到重复操作，已自动终止循环。"
                        final_chunks.append(fallback_msg)
                        yield fallback_msg
                        break
                    
                    self._tool_runs[run_id] = {
                        "name": tool_name or "unknown_tool",
                        "input": tool_input_str,
                        "start_time": time.time(),
                        "request_id": request_id,
                    }
                    logger.info(
                        json.dumps(
                            {
                                "type": "tool_start",
                                "request_id": request_id,
                                "run_id": run_id,
                                "tool_name": tool_name or "unknown_tool",
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
                    logger.info(
                        json.dumps(
                            {
                                "type": "tool_end",
                                "request_id": request_id,
                                "run_id": run_id,
                                "tool_name": meta.get("name") or "unknown_tool",
                                "duration_sec": duration,
                                "output_preview": str(tool_output)[:200],
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

                if event_type in ("on_chat_model_stream", "on_llm_stream"):
                    chunk = data.get("chunk")
                    if not chunk:
                        continue

                    text = getattr(chunk, "content", None) or getattr(chunk, "text", None)
                    if not text:
                        continue

                    final_chunks.append(text)
                    yield text

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 流式生成失败：{e}")
            fallback = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield fallback
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", fallback, time.time())

            # 提取并更新长期记忆
            self._extract_and_update_long_term_memory(query, fallback)
        else:
            final_response = "".join(final_chunks)
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", final_response, time.time())

            # 提取并更新长期记忆
            self._extract_and_update_long_term_memory(query, final_response)

            logger.info(f"[{request_id}] ✅ 流式对话完成，响应长度：{len(final_response)} 字符")
        finally:
            self._current_request_id = None

    async def generate_events(
        self, query: str, image_path: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        request_id = uuid.uuid4().hex[:8]
        self._current_request_id = request_id
        logger.info(f"[{request_id}] 开始处理事件流查询：{query[:200]}")

        final_chunks: list[str] = []
        thinking_buffer: list[str] = []
        in_thinking = True
        final_answer_started = False
        final_answer_extracted = False
        thinking_logged = False
        should_stop = False

        try:
            chat_history = self._format_chat_history()
            user_profile = self._format_user_profile()
            async for event in self._agent_executor.astream_events(
                {
                    "input": query,
                    "chat_history": chat_history,
                    "user_profile": user_profile
                },
                version="v1",
            ):
                event_type = event.get("event")
                data = event.get("data", {}) or {}
                run_id = event.get("run_id")

                if event_type == "on_tool_start":
                    tool_name = data.get("name") or data.get("tool")
                    tool_input = data.get("input")
                    self._tool_runs[run_id] = {
                        "name": tool_name or "unknown_tool",
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
                                "tool_name": tool_name or "unknown_tool",
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

                    extracted_url = None
                    if tool_output_str.strip().startswith("{"):
                        try:
                            obj = json.loads(tool_output_str)
                            if isinstance(obj, dict):
                                extracted_url = obj.get("hdurl") or obj.get("url")
                        except Exception:
                            extracted_url = None
                    if not extracted_url and self._fallback_service:
                        extracted_url = self._fallback_service.extract_image_url(tool_output_str)
                    elif not extracted_url:
                        extracted_url = extract_image_url(tool_output_str)

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

                    thinking_buffer.append(text)
                    combined_thinking = "".join(thinking_buffer)

                    if not final_answer_started:
                        if re.search(r'(Thought:|Action:|Observation:)\s*$', combined_thinking, re.IGNORECASE):
                            in_thinking = True
                        elif re.search(r'Final Answer:\s*', combined_thinking, re.IGNORECASE):
                            final_answer_started = True
                            if not thinking_logged and combined_thinking:
                                thinking_logged = True
                                logger.info(f"[{request_id}] 🔍 Thinking: {combined_thinking}")
                            match = re.search(r'Final Answer:\s*(.*)', combined_thinking, re.IGNORECASE | re.DOTALL)
                            if match and not final_answer_extracted:
                                final_answer_extracted = True
                                final_answer_text = match.group(1).strip()
                                if final_answer_text:
                                    final_chunks.append(final_answer_text)
                                    yield {"type": "text", "content": final_answer_text}
                                    logger.info(f"[{request_id}] Final Answer: {final_answer_text[:100]}...")
                            thinking_buffer = []
                            continue

                    if final_answer_started:
                        final_chunks.append(text)
                        yield {"type": "text", "content": text}
                    elif in_thinking and not final_answer_started:
                        yield {"type": "thinking", "content": text}
                    continue

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 事件流生成失败：{e}")
            fallback = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield {"type": "text", "content": fallback}
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", fallback, time.time())

            # 提取并更新长期记忆
            self._extract_and_update_long_term_memory(query, fallback)
        else:
            final_response = "".join(final_chunks)
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", final_response, time.time())

            # 提取并更新长期记忆
            self._extract_and_update_long_term_memory(query, final_response)

            if not thinking_logged and thinking_buffer:
                logger.info(f"[{request_id}] 🔍 Thinking: {''.join(thinking_buffer)}")
            logger.info(f"[{request_id}] ✅ 事件流完成，响应长度：{len(final_response)} 字符")
        finally:
            self._current_request_id = None
