import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from logger import logger


class StreamingService:
    def __init__(
        self,
        agent_executor: Any,
        memory: Any,
        fallback_service: Optional[Any] = None,
    ):
        self._agent_executor = agent_executor
        self._memory = memory
        self._fallback_service = fallback_service
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._current_request_id: Optional[str] = None

    def _format_chat_history(self) -> str:
        messages = self._memory.get_recent_messages()
        if not messages:
            return "无历史对话"
        formatted = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def generate_response(self, query: str) -> Generator[str, None, None]:
        logger.info(f"\n=== 处理用户查询：{query} ===")

        tool_call_failed = False
        fallback_used = False

        try:
            chat_history = self._format_chat_history()
            response = self._agent_executor.invoke({"input": query, "chat_history": chat_history})
            output = response.get("output", "")

            if self._fallback_service and self._fallback_service.should_use_fallback(output):
                logger.warning("检测到工具调用可能未返回有效结果，尝试联网搜索...")
                tool_call_failed = True
                search_result = self._fallback_service.try_web_search_fallback(query)
                output = self._fallback_service.format_fallback_response(query, search_result)
                fallback_used = True
            else:
                output = response.get("output", "")

            final_response = output

            for i in range(0, len(final_response), 50):
                chunk = final_response[i:i+50]
                yield chunk
                time.sleep(0.1)

            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", final_response, time.time())

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
                    logger.info(f"✅ 降级搜索成功 | 助手响应长度：{len(fallback_response)} 字符")
                    return
                except Exception as fallback_error:
                    logger.error(f"降级搜索也失败: {fallback_error}")

            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", default_response, time.time())

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        request_id = uuid.uuid4().hex[:8]
        self._current_request_id = request_id
        logger.info(f"[{request_id}] 开始处理流式查询：{query}")

        final_chunks: list[str] = []

        try:
            chat_history = self._format_chat_history()
            async for event in self._agent_executor.astream_events(
                {"input": query, "chat_history": chat_history},
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
        else:
            final_response = "".join(final_chunks)
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", final_response, time.time())
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

        try:
            chat_history = self._format_chat_history()
            async for event in self._agent_executor.astream_events(
                {"input": query, "chat_history": chat_history},
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
                        extracted_url = self._extract_image_url_fallback(tool_output_str)

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
        else:
            final_response = "".join(final_chunks)
            self._memory.add_message("user", query, time.time())
            self._memory.add_message("assistant", final_response, time.time())
            if not thinking_logged and thinking_buffer:
                logger.info(f"[{request_id}] 🔍 Thinking: {''.join(thinking_buffer)}")
            logger.info(f"[{request_id}] ✅ 事件流完成，响应长度：{len(final_response)} 字符")
        finally:
            self._current_request_id = None

    def _extract_image_url_fallback(self, text: str) -> Optional[str]:
        import re
        m = re.search(r"(https?://\S+\.(?:png|jpg|jpeg|webp))", text, re.IGNORECASE)
        return m.group(1) if m else None
