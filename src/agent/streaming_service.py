import json
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from src.core.logger import logger
from src.core.errors import ErrorHandler
from src.utils.helpers import extract_image_url

MAX_ACTION_HISTORY_ENTRIES = 100


class BaseStreamingGenerator:
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
        self._action_history: OrderedDict[str, list] = OrderedDict()
        self._max_same_action_count = 2

    def _cleanup_action_history(self, request_id: Optional[str] = None):
        if request_id and request_id in self._action_history:
            del self._action_history[request_id]
            logger.debug(f"已清理请求 {request_id} 的操作历史")

        while len(self._action_history) > MAX_ACTION_HISTORY_ENTRIES:
            oldest_key, _ = self._action_history.popitem(last=False)
            logger.debug(f"LRU淘汰最旧的操作历史: {oldest_key}")

    def _log_memory_usage(self):
        total_entries = len(self._action_history)
        total_actions = sum(len(v) for v in self._action_history.values())
        logger.debug(
            f"操作历史内存状态: {total_entries} 个请求, 共 {total_actions} 条动作记录"
        )

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
        if not self._long_term_memory:
            return "暂无用户偏好信息"
        return self._long_term_memory.format_profile_for_prompt(self._user_id)

    def _extract_and_update_long_term_memory(self, user_message: str, assistant_message: str):
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

    def _prepare_input(self, query: str) -> Dict[str, str]:
        chat_history = self._format_chat_history()
        user_profile = self._format_user_profile()
        return {
            "input": query,
            "chat_history": chat_history,
            "user_profile": user_profile
        }

    def _handle_tool_start(self, request_id: str, run_id: str, data: dict, check_repeated: bool = False) -> Optional[str]:
        tool_name = data.get("name") or data.get("tool")
        tool_input = data.get("input")
        tool_input_str = str(tool_input) if tool_input else ""

        if check_repeated and self._check_repeated_action(request_id, tool_name or "unknown_tool", tool_input_str):
            return "repeated"

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
        return None

    def _handle_tool_end(self, request_id: str, run_id: str, data: dict) -> Dict[str, Any]:
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

        return {
            "meta": meta,
            "duration": duration,
            "tool_output_str": tool_output_str,
            "extracted_url": extracted_url,
        }

    def _save_to_memory(self, query: str, response: str):
        self._memory.add_message("user", query, time.time())
        self._memory.add_message("assistant", response, time.time())
        self._extract_and_update_long_term_memory(query, response)

    def _finalize_request(self, request_id: Optional[str]):
        self._current_request_id = None
        self._cleanup_action_history(request_id)
        self._log_memory_usage()

    def _extract_stream_text(self, data: dict) -> Optional[str]:
        chunk = data.get("chunk")
        if not chunk:
            return None
        text = getattr(chunk, "content", None) or getattr(chunk, "text", None)
        return text

    def _parse_thinking_and_final_answer(
        self,
        text: str,
        thinking_buffer: list[str],
        final_answer_started: bool,
        final_answer_extracted: bool,
        thinking_logged: bool,
        request_id: str,
    ) -> Dict[str, Any]:
        thinking_buffer.append(text)
        combined_thinking = "".join(thinking_buffer)

        result = {
            "thinking_buffer": thinking_buffer,
            "final_answer_started": final_answer_started,
            "final_answer_extracted": final_answer_extracted,
            "thinking_logged": thinking_logged,
            "final_answer_text": None,
            "thinking_text": None,
            "is_thinking": False,
            "is_final_answer_chunk": False,
            "should_continue": True,
        }

        if not final_answer_started:
            if re.search(r'(Thought:|Action:|Observation:)\s*$', combined_thinking, re.IGNORECASE):
                result["is_thinking"] = True
                result["thinking_text"] = text
            elif re.search(r'Final Answer:\s*', combined_thinking, re.IGNORECASE):
                result["final_answer_started"] = True
                if not thinking_logged and combined_thinking:
                    result["thinking_logged"] = True
                    logger.info(f"[{request_id}] 🔍 Thinking: {combined_thinking}")
                match = re.search(r'Final Answer:\s*(.*)', combined_thinking, re.IGNORECASE | re.DOTALL)
                if match and not final_answer_extracted:
                    result["final_answer_extracted"] = True
                    final_answer_text = match.group(1).strip()
                    if final_answer_text:
                        result["final_answer_text"] = final_answer_text
                        logger.info(f"[{request_id}] Final Answer: {final_answer_text[:100]}...")
                result["thinking_buffer"] = []
                result["should_continue"] = False
                return result
            else:
                result["is_thinking"] = True
                result["thinking_text"] = text
        else:
            result["is_final_answer_chunk"] = True

        return result


class StreamingService(BaseStreamingGenerator):
    def generate_response(self, query: str) -> Generator[str, None, None]:
        logger.info(f"\n=== 处理用户查询：{query} ===")

        tool_call_failed = False
        fallback_used = False

        try:
            agent_input = self._prepare_input(query)
            response = self._agent_executor.invoke(agent_input)
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

            yield final_response

            self._save_to_memory(query, final_response)

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

                    yield fallback_response

                    self._save_to_memory(query, fallback_response)

                    logger.info(f"✅ 降级搜索成功 | 助手响应长度：{len(fallback_response)} 字符")
                    return
                except Exception as fallback_error:
                    logger.error(f"降级搜索也失败: {fallback_error}")

            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            self._save_to_memory(query, default_response)

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        request_id = uuid.uuid4().hex[:8]
        self._current_request_id = request_id
        logger.info(f"[{request_id}] 开始处理流式查询：{query}")

        final_chunks: list[str] = []
        should_stop = False

        try:
            agent_input = self._prepare_input(query)
            async for event in self._agent_executor.astream_events(
                agent_input,
                version="v1",
            ):
                if should_stop:
                    break

                event_type = event.get("event")
                data = event.get("data", {}) or {}
                run_id = event.get("run_id")

                if event_type == "on_tool_start":
                    result = self._handle_tool_start(request_id, run_id, data, check_repeated=True)
                    if result == "repeated":
                        should_stop = True
                        fallback_msg = "\n\n⚠️ 检测到重复操作，已自动终止循环。"
                        final_chunks.append(fallback_msg)
                        yield fallback_msg
                        break
                    continue

                if event_type == "on_tool_end":
                    self._handle_tool_end(request_id, run_id, data)
                    continue

                if event_type in ("on_chat_model_stream", "on_llm_stream"):
                    text = self._extract_stream_text(data)
                    if not text:
                        continue

                    final_chunks.append(text)
                    yield text

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 流式生成失败：{e}")
            fallback = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield fallback
            self._save_to_memory(query, fallback)
        else:
            final_response = "".join(final_chunks)
            self._save_to_memory(query, final_response)
            logger.info(f"[{request_id}] ✅ 流式对话完成，响应长度：{len(final_response)} 字符")
        finally:
            self._finalize_request(request_id)

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
            agent_input = self._prepare_input(query)
            async for event in self._agent_executor.astream_events(
                agent_input,
                version="v1",
            ):
                event_type = event.get("event")
                data = event.get("data", {}) or {}
                run_id = event.get("run_id")

                if event_type == "on_tool_start":
                    self._handle_tool_start(request_id, run_id, data, check_repeated=False)
                    continue

                if event_type == "on_tool_end":
                    tool_result = self._handle_tool_end(request_id, run_id, data)
                    extracted_url = tool_result.get("extracted_url")
                    if extracted_url:
                        yield {
                            "type": "image",
                            "url": extracted_url,
                            "meta": {
                                "request_id": request_id,
                                "tool": tool_result["meta"].get("name"),
                            },
                        }
                    continue

                if event_type in ("on_chat_model_stream", "on_llm_stream"):
                    text = self._extract_stream_text(data)
                    if not text:
                        continue

                    parse_result = self._parse_thinking_and_final_answer(
                        text, thinking_buffer, final_answer_started,
                        final_answer_extracted, thinking_logged, request_id,
                    )
                    thinking_buffer = parse_result["thinking_buffer"]
                    final_answer_started = parse_result["final_answer_started"]
                    final_answer_extracted = parse_result["final_answer_extracted"]
                    thinking_logged = parse_result["thinking_logged"]

                    if not parse_result["should_continue"]:
                        if parse_result["final_answer_text"]:
                            final_chunks.append(parse_result["final_answer_text"])
                            yield {"type": "text", "content": parse_result["final_answer_text"]}
                        continue

                    if parse_result["is_final_answer_chunk"]:
                        final_chunks.append(text)
                        yield {"type": "text", "content": text}
                    elif parse_result["is_thinking"]:
                        yield {"type": "thinking", "content": text}
                    continue

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 事件流生成失败：{e}")
            fallback = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield {"type": "text", "content": fallback}
            self._save_to_memory(query, fallback)
        else:
            final_response = "".join(final_chunks)
            self._save_to_memory(query, final_response)

            if not thinking_logged and thinking_buffer:
                logger.info(f"[{request_id}] 🔍 Thinking: {''.join(thinking_buffer)}")
            logger.info(f"[{request_id}] ✅ 事件流完成，响应长度：{len(final_response)} 字符")
        finally:
            self._finalize_request(request_id)
