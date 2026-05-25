"""FastAPI 服务入口，负责会话管理、模型切换、文本/图片/语音问答和记忆接口，并把请求转交 AstroAgent 主链路。"""

import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.agent import AstroAgent
from src.agent.streaming_events import SSEEventAdapter
from src.agent.streaming_service import StreamingService
from src.core.config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler
from src.core.logger import logger
from src.core.model_catalog import (
    get_default_provider,
    list_model_providers,
    model_selection_payload,
)
from src.memory.api.memory_service import MemoryService
from src.memory.long_term_memory import (
    LongTermMemoryService,
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    SourceType,
)

app = FastAPI(
    title="天文Agent API", description="具有会话记忆、长期记忆和流式输出的天文知识助手"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

UPLOAD_DIR = os.path.abspath("./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def sanitize_filename(filename: str) -> str:
    """清理上传文件名，移除路径和危险字符，避免目录穿越或非法文件名。"""
    if not filename:
        return ""
    filename = filename.replace("\\", "/")
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w\s\-.]", "", filename)
    filename = re.sub(r"\.{2,}", ".", filename)
    return filename.strip(". ")


def validate_upload_path(save_path: str) -> bool:
    """校验保存路径是否仍位于上传目录内。"""
    resolved = os.path.abspath(save_path)
    upload_dir_resolved = os.path.abspath(UPLOAD_DIR)
    return (
        resolved.startswith(upload_dir_resolved + os.sep)
        or resolved == upload_dir_resolved
    )


def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """根据扩展名判断上传文件是否属于允许类型。"""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in allowed_extensions


def normalize_user_id(user_id: Optional[str]) -> str:
    """规范化 user_id，缺省时回退到系统默认用户。"""
    if user_id and user_id.strip():
        return user_id.strip()
    return settings.DEFAULT_USER_ID


def normalize_session_id(session_id: Optional[str]) -> str:
    """规范化 session_id，缺省时使用 default 会话。"""
    if session_id and session_id.strip():
        return session_id.strip()
    return "default"


def build_session_key(user_id: str, session_id: str) -> str:
    """把 user_id 和 session_id 合成为内部会话键。"""
    return f"{user_id}::{session_id}"


class _AgentHolder:
    """懒加载Agent持有器，支持初始化失败降级"""

    def __init__(self):
        """初始化懒加载状态和初始化错误记录。"""
        self._agent = None
        self._initialized = False
        self._init_error = None

    def get(self):
        """返回 Agent 实例，首次访问时触发初始化。"""
        if not self._initialized:
            self._initialize()
        return self._agent

    def _initialize(self):
        """创建 AstroAgent，并在失败时保存错误原因。"""
        self._initialized = True
        try:
            self._agent = AstroAgent()
            logger.info("✅ AstroAgent懒加载初始化成功")
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"⚠️ AstroAgent初始化失败: {e}")
            self._agent = None

    @property
    def is_available(self):
        """判断 Agent 是否已经成功初始化。"""
        if not self._initialized:
            self._initialize()
        return self._agent is not None

    @property
    def init_error(self):
        """返回最近一次 Agent 初始化错误信息。"""
        return self._init_error


_agent_holder = _AgentHolder()


def get_agent():
    """获取已初始化的 AstroAgent，不可用时抛出统一 AgentError。"""
    agent = _agent_holder.get()
    if agent is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"Agent服务暂时不可用: {_agent_holder.init_error or '初始化失败'}",
        )
    return agent


class SessionData:
    """保存单个用户会话的模型选择、记忆服务和流式运行时。"""

    def __init__(self, user_id: str, session_id: str, agent: AstroAgent):
        """初始化 SessionData 的依赖、配置和内部状态。"""
        self.user_id = user_id
        self.session_id = session_id
        self.session_key = build_session_key(user_id, session_id)
        self._base_agent = agent
        self.memory = MemoryService(
            db_path=settings.MEMORY_PERSISTENCE_PATH,
            session_id=f"mem_{self.session_key}",
            user_id=user_id,
        )
        self.model_provider = ""
        self.model_name = ""
        self.model_label = ""
        self.llm = None
        self.task_orchestrator = None
        self.streaming_service = None
        self.agent_executor = None
        default_provider = getattr(agent, "model_provider", None)
        default_model_name = getattr(agent, "model_name", None)
        self.apply_model_selection(
            model_provider=(
                default_provider
                if isinstance(default_provider, str)
                else get_default_provider()
            ),
            model_name=(
                default_model_name if isinstance(default_model_name, str) else None
            ),
        )
        self.last_access_time = time.time()

    def apply_model_selection(
        self,
        *,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """为当前会话应用模型选择并重建运行时组件。"""
        runtime_factory = getattr(self._base_agent, "create_session_runtime", None)
        runtime = None
        if callable(runtime_factory):
            candidate = runtime_factory(
                user_id=self.user_id,
                memory=self.memory,
                model_provider=model_provider,
                model_name=model_name,
            )
            if isinstance(candidate, dict) and "streaming_service" in candidate:
                runtime = candidate

        if runtime is None:
            payload = model_selection_payload(model_provider, model_name)
            self.llm = getattr(self._base_agent, "llm", None)
            self.task_orchestrator = getattr(
                self._base_agent, "task_orchestrator", None
            )
            self.streaming_service = StreamingService(
                agent_executor=getattr(self._base_agent, "_agent_executor", None),
                memory=self.memory,
                long_term_memory=getattr(self._base_agent, "long_term_memory", None),
                user_id=self.user_id,
                fallback_service=getattr(self._base_agent, "fallback_service", None),
                request_router=getattr(self._base_agent, "__dict__", {}).get(
                    "request_router"
                ),
                capability_kit=getattr(self._base_agent, "__dict__", {}).get(
                    "capability_kit"
                ),
                rag_retriever=getattr(self._base_agent, "__dict__", {}).get("rag"),
                execution_engine=getattr(self._base_agent, "__dict__", {}).get(
                    "execution_engine"
                ),
            )
            self.agent_executor = getattr(self._base_agent, "_agent_executor", None)
            self.model_provider = payload["model_provider"]
            self.model_name = payload["model_name"]
            self.model_label = payload["model_label"]
            return self.get_model_payload()

        self.llm = runtime["llm"]
        self.task_orchestrator = runtime["task_orchestrator"]
        self.streaming_service = runtime["streaming_service"]
        self.agent_executor = runtime["agent_executor"]
        self.model_provider = runtime["model_provider"]
        self.model_name = runtime["model_name"]
        self.model_label = runtime["model_label"]
        logger.info(
            "会话模型已更新: user_id=%s, session_id=%s, provider=%s, model=%s",
            self.user_id,
            self.session_id,
            self.model_provider,
            self.model_name,
        )
        return self.get_model_payload()

    def get_model_payload(self) -> Dict[str, str]:
        """返回当前会话对外展示的模型选择信息。"""
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "model_label": self.model_label,
        }


class SessionManager:
    """管理多用户会话生命周期，按访问时间清理过期会话。"""

    def __init__(self, max_age: int = None, cleanup_interval: int = None):
        """初始化 SessionManager 的依赖、配置和内部状态。"""
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._max_age = max_age or settings.SESSION_MAX_AGE_SECONDS
        self._cleanup_interval = (
            cleanup_interval or settings.SESSION_CLEANUP_INTERVAL_SECONDS
        )
        self._last_cleanup = time.time()

    def get_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        *,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> SessionData:
        """获取或创建指定用户会话，并刷新最后访问时间。"""
        with self._lock:
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale_sessions()

            effective_session_id = normalize_session_id(session_id)
            session_key = build_session_key(user_id, effective_session_id)

            if session_key not in self._sessions:
                agent = get_agent()
                logger.info(
                    f"创建新会话: user_id={user_id}, session_id={effective_session_id}"
                )
                self._sessions[session_key] = SessionData(
                    user_id, effective_session_id, agent
                )

            session = self._sessions[session_key]
            if model_provider or model_name:
                requested = model_selection_payload(model_provider, model_name)
                if (
                    requested["model_provider"] != session.model_provider
                    or requested["model_name"] != session.model_name
                ):
                    session.apply_model_selection(
                        model_provider=requested["model_provider"],
                        model_name=requested["model_name"],
                    )
            session.last_access_time = now
            return session

    def clear_session(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """清理指定用户会话并释放对应短期记忆实例。"""
        with self._lock:
            effective_session_id = normalize_session_id(session_id)
            session_key = build_session_key(user_id, effective_session_id)
            if session_key in self._sessions:
                self._sessions[session_key].memory.clear()
                del self._sessions[session_key]
                logger.info(
                    f"会话已清除: user_id={user_id}, session_id={effective_session_id}"
                )
                return True
            return False

    def clear_sessions_for_user(self, user_id: str) -> int:
        """清理某个用户名下的所有活跃会话。"""
        with self._lock:
            matched_keys = [
                key
                for key, session in self._sessions.items()
                if session.user_id == user_id
            ]
            for key in matched_keys:
                self._sessions[key].memory.clear()
                del self._sessions[key]
            if matched_keys:
                logger.info(
                    f"已清除用户全部会话: user_id={user_id}, count={len(matched_keys)}"
                )
            return len(matched_keys)

    def _cleanup_stale_sessions(self):
        """按最大空闲时间清理过期会话。"""
        now = time.time()
        stale_users = [
            uid
            for uid, session in self._sessions.items()
            if now - session.last_access_time > self._max_age
        ]
        for uid in stale_users:
            del self._sessions[uid]
            logger.info(f"清理过期会话: user_id={uid}")
        if stale_users:
            logger.info(f"清理了 {len(stale_users)} 个过期会话")
        self._last_cleanup = now

    def get_active_session_count(self) -> int:
        """返回当前保留的活跃会话数量。"""
        with self._lock:
            return len(self._sessions)


session_manager = SessionManager()
sse_adapter = SSEEventAdapter()


@app.on_event("startup")
async def startup_warmup():
    """应用启动后按配置预热 Agent 和 MCP 会话。"""
    try:
        agent = await asyncio.to_thread(get_agent)
        await asyncio.to_thread(agent.capability_kit.prewarm)
        logger.info("✅ 启动预热完成：Agent 与 MCP 已预热")
    except Exception as e:
        logger.warning(f"启动预热失败，将在请求时懒加载: {e}")


class QueryRequest(BaseModel):
    """文本问答接口的请求体模型。"""

    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    disable_long_term_memory: bool = False
    model_provider: Optional[str] = None
    model_name: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    """会话模型切换接口的请求体模型。"""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model_provider: str
    model_name: Optional[str] = None


class KnowledgeRequest(BaseModel):
    """人工知识写入接口的请求体模型。"""

    knowledge: list[str]
    user_id: Optional[str] = None


class UserProfileRequest(BaseModel):
    """用户画像更新接口的请求体模型。"""

    user_id: Optional[str] = None


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    """将 AgentError 转换为统一 JSON 错误响应。"""
    status_map = {
        ErrorCode.LLM_ERROR: 503,
        ErrorCode.API_ERROR: 502,
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.PARAM_PARSE_ERROR: 400,
        ErrorCode.FILE_NOT_FOUND: 404,
        ErrorCode.SECURITY_ERROR: 403,
        ErrorCode.RATE_LIMIT_ERROR: 429,
        ErrorCode.FILE_TOO_LARGE: 413,
        ErrorCode.FILE_TYPE_NOT_ALLOWED: 415,
        ErrorCode.PATH_TRAVERSAL_ERROR: 403,
    }
    status = status_map.get(exc.code, 500)
    return JSONResponse(status_code=status, content=exc.to_dict())


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """将未捕获异常转换为统一 JSON 错误响应。"""
    error = ErrorHandler.handle(exc, {"path": str(request.url)})
    return JSONResponse(status_code=500, content=error.to_dict())


@app.post("/query")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    """处理文本问答请求并返回 SSE 流式响应。"""
    user_id = normalize_user_id(body.user_id)
    session = session_manager.get_session(
        user_id,
        body.session_id,
        model_provider=body.model_provider,
        model_name=body.model_name,
    )

    async def generate():
        """生成当前接口所需的流式响应片段。"""
        async for chunk in session.streaming_service.generate_sse(
            body.query,
            use_long_term_memory=not body.disable_long_term_memory,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query_with_image")
@limiter.limit(f"{settings.UPLOAD_RATE_LIMIT_PER_MINUTE}/minute")
async def query_with_image_endpoint(
    request: Request,
    query: str = Form(...),
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    model_provider: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
):
    """保存上传图片并进入图文问答流式链路。"""
    sanitized_name = sanitize_filename(image.filename or "")
    ext = os.path.splitext(sanitized_name)[1].lower()

    if not ext or ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise AgentError(
            code=ErrorCode.FILE_TYPE_NOT_ALLOWED,
            message=f"不支持的图片文件类型: {ext}，允许的类型: {', '.join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS))}",
        )

    content = await image.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise AgentError(
            code=ErrorCode.FILE_TOO_LARGE,
            message=f"文件大小 {len(content)} 字节超过限制 {settings.MAX_UPLOAD_SIZE} 字节（{settings.MAX_UPLOAD_SIZE // (1024*1024)}MB）",
        )

    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    save_path = os.path.abspath(save_path)

    if not validate_upload_path(save_path):
        raise AgentError(
            code=ErrorCode.PATH_TRAVERSAL_ERROR,
            message="检测到非法文件路径",
        )

    with open(save_path, "wb") as f:
        f.write(content)

    image_url = f"/uploads/{file_id}{ext}"
    effective_user_id = normalize_user_id(user_id)
    session = session_manager.get_session(
        effective_user_id,
        session_id,
        model_provider=model_provider,
        model_name=model_name,
    )

    async def generate():
        """生成当前接口所需的流式响应片段。"""
        yield sse_adapter.serialize_payload(
            {"type": "image", "url": image_url, "meta": {"source": "user_upload"}}
        )
        async for chunk in session.streaming_service.generate_sse(
            query, image_path=save_path
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query_with_audio")
@limiter.limit(f"{settings.UPLOAD_RATE_LIMIT_PER_MINUTE}/minute")
async def query_with_audio_endpoint(
    request: Request,
    query: str = Form(""),
    audio: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    model_provider: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
):
    """保存上传音频、转写文本并进入问答流式链路。"""
    sanitized_name = sanitize_filename(audio.filename or "")
    ext = os.path.splitext(sanitized_name)[1].lower()

    if not ext or ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise AgentError(
            code=ErrorCode.FILE_TYPE_NOT_ALLOWED,
            message=f"不支持的音频文件类型: {ext}，允许的类型: {', '.join(sorted(settings.ALLOWED_AUDIO_EXTENSIONS))}",
        )

    content = await audio.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise AgentError(
            code=ErrorCode.FILE_TOO_LARGE,
            message=f"文件大小 {len(content)} 字节超过限制 {settings.MAX_UPLOAD_SIZE} 字节（{settings.MAX_UPLOAD_SIZE // (1024*1024)}MB）",
        )

    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    save_path = os.path.abspath(save_path)

    if not validate_upload_path(save_path):
        raise AgentError(
            code=ErrorCode.PATH_TRAVERSAL_ERROR,
            message="检测到非法文件路径",
        )

    with open(save_path, "wb") as f:
        f.write(content)

    effective_user_id = normalize_user_id(user_id)
    session = session_manager.get_session(
        effective_user_id,
        session_id,
        model_provider=model_provider,
        model_name=model_name,
    )

    augmented_query = await asyncio.to_thread(
        get_agent().speech_service.build_speech_query, query, save_path
    )

    async def generate():
        """生成当前接口所需的流式响应片段。"""
        yield sse_adapter.serialize_payload(
            {"type": "transcription", "text": augmented_query}
        )
        async for chunk in session.streaming_service.generate_sse(augmented_query):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/add_knowledge")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def add_knowledge_endpoint(request: Request, body: KnowledgeRequest):
    """把用户提交的天文知识写入 RAG 知识库。"""
    effective_user_id = normalize_user_id(body.user_id)
    result = get_agent().add_astronomy_knowledge(body.knowledge)
    return {
        "status": "success",
        "message": "知识添加成功",
        "user_id": effective_user_id,
        "result": result,
    }


@app.get("/profile")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_profile_endpoint(request: Request, user_id: Optional[str] = None):
    """查询指定用户的长期记忆画像。"""
    effective_user_id = normalize_user_id(user_id)
    ltm = get_agent().long_term_memory
    profile = ltm.get_profile(effective_user_id)
    if profile:
        return {
            "status": "success",
            "user_id": profile.get("user_id", effective_user_id),
            "preferences": profile.get("preferences", {}),
            "habits": profile.get("habits", {}),
            "constraints": profile.get("constraints", []),
            "background": profile.get("background", {}),
            "facts": profile.get("facts", []),
            "created_at": profile.get("created_at", ""),
            "updated_at": profile.get("updated_at", ""),
        }
    else:
        return {
            "status": "success",
            "message": "暂无用户画像信息",
            "user_id": effective_user_id,
            "preferences": {},
            "habits": {},
            "constraints": [],
            "background": {},
            "facts": [],
        }


@app.delete("/profile")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def delete_profile_endpoint(request: Request, user_id: Optional[str] = None):
    """删除指定用户的长期记忆画像。"""
    effective_user_id = normalize_user_id(user_id)
    deleted = get_agent().long_term_memory.delete_profile(effective_user_id)
    if deleted:
        return {
            "status": "success",
            "message": "用户画像已删除",
            "user_id": effective_user_id,
        }
    else:
        return {
            "status": "success",
            "message": "用户画像不存在或已被删除",
            "user_id": effective_user_id,
        }


@app.post("/clear_memory")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def clear_memory_endpoint(
    request: Request,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    scope: str = Query("all"),
):
    """清理指定会话的短期记忆。"""
    effective_user_id = normalize_user_id(user_id)
    effective_session_id = normalize_session_id(session_id)
    cleared = {"session": False, "all_sessions": False, "long_term": False}
    if scope == "session":
        cleared["session"] = session_manager.clear_session(
            effective_user_id, effective_session_id
        )
    else:
        cleared_count = session_manager.clear_sessions_for_user(effective_user_id)
        cleared["session"] = cleared_count > 0
        cleared["all_sessions"] = True
        _get_ltm().delete_profile(effective_user_id)
        cleared["long_term"] = True
    return {
        "status": "success",
        "message": "记忆已清空",
        "user_id": effective_user_id,
        "session_id": effective_session_id,
        "scope": scope,
        "cleared": cleared,
    }


@app.get("/models")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def list_models_endpoint(request: Request):
    """返回后端支持的模型供应商和模型列表。"""
    return {
        "status": "success",
        "default_provider": get_default_provider(),
        "providers": list_model_providers(),
    }


@app.get("/session/model")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_session_model_endpoint(
    request: Request,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """查询当前会话正在使用的模型配置。"""
    effective_user_id = normalize_user_id(user_id)
    effective_session_id = normalize_session_id(session_id)
    session = session_manager.get_session(effective_user_id, effective_session_id)
    return {
        "status": "success",
        "user_id": effective_user_id,
        "session_id": effective_session_id,
        **session.get_model_payload(),
    }


@app.post("/session/model")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def set_session_model_endpoint(request: Request, body: ModelSwitchRequest):
    """切换当前会话的模型供应商和模型名称。"""
    effective_user_id = normalize_user_id(body.user_id)
    effective_session_id = normalize_session_id(body.session_id)
    session = session_manager.get_session(effective_user_id, effective_session_id)
    payload = session.apply_model_selection(
        model_provider=body.model_provider,
        model_name=body.model_name,
    )
    return {
        "status": "success",
        "message": "会话模型切换成功，已保留历史消息，新消息将使用新模型生成",
        "user_id": effective_user_id,
        "session_id": effective_session_id,
        **payload,
    }


class MemoryCreateRequest(BaseModel):
    """长期记忆创建接口的请求体模型。"""

    user_id: Optional[str] = None
    memory_type: str
    category: str
    key: str
    value: Any
    confidence: Optional[float] = None
    source_type: Optional[str] = "manual"
    priority: Optional[int] = 0
    metadata: Optional[Dict[str, Any]] = None


class MemoryUpdateRequest(BaseModel):
    """长期记忆更新接口的请求体模型。"""

    user_id: Optional[str] = None
    value: Optional[Any] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    priority: Optional[int] = None
    confirmed_by_user: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryQueryRequest(BaseModel):
    """长期记忆检索接口的请求体模型。"""

    user_id: Optional[str] = None
    memory_type: Optional[str] = None
    category: Optional[str] = None
    key: Optional[str] = None
    status: Optional[str] = None
    source_type: Optional[str] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    keyword: Optional[str] = None
    limit: Optional[int] = 50
    offset: Optional[int] = 0


class ConfirmationResolveRequest(BaseModel):
    """长期记忆确认项处理接口的请求体模型。"""

    status: str
    confirmation_ids: Optional[List[str]] = None


class BatchConfirmRequest(BaseModel):
    """长期记忆确认项批量处理接口的请求体模型。"""

    user_id: Optional[str] = None
    confirmation_ids: List[str]
    status: str


def _get_ltm() -> LongTermMemoryService:
    """获取 AstroAgent 持有的长期记忆服务实例。"""
    agent = get_agent()
    ltm = agent.long_term_memory
    if isinstance(ltm, LongTermMemoryService):
        return ltm
    if hasattr(ltm, "delete_profile") and hasattr(ltm, "get_profile"):
        return ltm
    else:
        raise AgentError(
            code=ErrorCode.MEMORY_ERROR,
            message="长期记忆模块未正确初始化",
        )


@app.post("/memories")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def create_memory_endpoint(request: Request, body: MemoryCreateRequest):
    """创建一条长期记忆并返回序列化结果。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(body.user_id)
    item = ltm.add_memory(
        user_id=effective_user_id,
        memory_type=body.memory_type,
        category=body.category,
        key=body.key,
        value=body.value,
        confidence=body.confidence,
        source_type=body.source_type or "manual",
        priority=body.priority or 0,
        metadata=body.metadata,
    )
    if not item:
        raise AgentError(
            code=ErrorCode.MEMORY_VALIDATION_ERROR,
            message="记忆置信度过低或验证失败，未存储",
        )
    return {"status": "success", "memory": item.to_dict()}


@app.get("/memories/{memory_id}")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_memory_endpoint(
    request: Request, memory_id: str, user_id: Optional[str] = None
):
    """按记忆 ID 读取长期记忆详情。"""
    ltm = _get_ltm()
    item = ltm.get_memory(memory_id)
    if not item:
        raise AgentError(
            code=ErrorCode.MEMORY_NOT_FOUND, message=f"记忆不存在: {memory_id}"
        )
    effective_user_id = normalize_user_id(user_id)
    if item.user_id != effective_user_id:
        raise AgentError(code=ErrorCode.MEMORY_ACCESS_DENIED, message="无权访问该记忆")
    return {"status": "success", "memory": item.to_dict()}


@app.put("/memories/{memory_id}")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def update_memory_endpoint(
    request: Request, memory_id: str, body: MemoryUpdateRequest
):
    """更新指定长期记忆的内容、状态或元信息。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(body.user_id)
    item = ltm.update_memory(
        memory_id=memory_id,
        user_id=effective_user_id,
        value=body.value,
        confidence=body.confidence,
        status=body.status,
        priority=body.priority,
        confirmed_by_user=body.confirmed_by_user,
        metadata=body.metadata,
    )
    if not item:
        raise AgentError(
            code=ErrorCode.MEMORY_NOT_FOUND,
            message=f"记忆不存在或无权修改: {memory_id}",
        )
    return {"status": "success", "memory": item.to_dict()}


@app.delete("/memories/{memory_id}")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def delete_memory_endpoint(
    request: Request, memory_id: str, user_id: Optional[str] = None
):
    """删除指定长期记忆记录。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    deleted = ltm.delete_memory(memory_id, effective_user_id)
    if not deleted:
        raise AgentError(
            code=ErrorCode.MEMORY_NOT_FOUND,
            message=f"记忆不存在或无权删除: {memory_id}",
        )
    return {"status": "success", "message": "记忆已删除", "memory_id": memory_id}


@app.post("/memories/query")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def query_memories_endpoint(request: Request, body: MemoryQueryRequest):
    """按查询条件检索长期记忆列表。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(body.user_id)
    query = MemoryQuery(
        user_id=effective_user_id,
        memory_type=body.memory_type,
        category=body.category,
        key=body.key,
        status=body.status,
        source_type=body.source_type,
        min_confidence=body.min_confidence,
        max_confidence=body.max_confidence,
        created_after=body.created_after,
        created_before=body.created_before,
        keyword=body.keyword,
        limit=body.limit or 50,
        offset=body.offset or 0,
    )
    items = ltm.query_memories(query)
    total = ltm.count_memories(query)
    return {
        "status": "success",
        "total": total,
        "items": [item.to_dict() for item in items],
        "limit": query.limit,
        "offset": query.offset,
    }


@app.get("/memories/{memory_id}/versions")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_memory_versions_endpoint(
    request: Request, memory_id: str, limit: int = 20
):
    """查询长期记忆的历史版本。"""
    ltm = _get_ltm()
    versions = ltm.get_versions(memory_id, limit=limit)
    return {
        "status": "success",
        "memory_id": memory_id,
        "versions": [v.to_dict() for v in versions],
    }


@app.post("/memories/{memory_id}/restore_version")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def restore_memory_version_endpoint(
    request: Request, memory_id: str, version: int, user_id: Optional[str] = None
):
    """把长期记忆恢复到指定历史版本。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    item = ltm.restore_version(memory_id, version, effective_user_id)
    if not item:
        raise AgentError(
            code=ErrorCode.MEMORY_NOT_FOUND,
            message=f"记忆或版本不存在: {memory_id} v{version}",
        )
    return {"status": "success", "memory": item.to_dict()}


@app.get("/memories/{memory_id}/trace")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_memory_trace_endpoint(
    request: Request, memory_id: str, user_id: Optional[str] = None
):
    """查询长期记忆的来源和变更追踪信息。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    trace = ltm.get_memory_trace(effective_user_id, memory_id)
    return {
        "status": "success",
        "memory_id": memory_id,
        "trace": [entry.to_dict() for entry in trace],
    }


@app.get("/candidates")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def list_candidates_endpoint(
    request: Request, user_id: Optional[str] = None, limit: int = 50, offset: int = 0
):
    """列出待提升的长期记忆候选项。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    candidates = ltm.list_candidates(effective_user_id, limit=limit, offset=offset)
    return {
        "status": "success",
        "candidates": [c.to_dict() for c in candidates],
    }


@app.post("/candidates/{candidate_id}/promote")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def promote_candidate_endpoint(request: Request, candidate_id: str):
    """将候选项提升为正式长期记忆。"""
    ltm = _get_ltm()
    item = ltm.promote_candidate(candidate_id)
    if not item:
        raise AgentError(
            code=ErrorCode.MEMORY_CANDIDATE_ERROR,
            message=f"候选记忆提升失败: {candidate_id}",
        )
    return {"status": "success", "memory": item.to_dict()}


@app.post("/candidates/{candidate_id}/reject")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def reject_candidate_endpoint(
    request: Request, candidate_id: str, reason: str = ""
):
    """拒绝指定长期记忆候选项。"""
    ltm = _get_ltm()
    rejected = ltm.reject_candidate(candidate_id, reason)
    if not rejected:
        raise AgentError(
            code=ErrorCode.MEMORY_CANDIDATE_ERROR,
            message=f"候选记忆拒绝失败: {candidate_id}",
        )
    return {"status": "success", "message": "候选记忆已拒绝"}


@app.get("/confirmations")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def list_confirmations_endpoint(
    request: Request, user_id: Optional[str] = None, limit: int = 20
):
    """列出需要用户确认的长期记忆项。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    confirmations = ltm.list_pending_confirmations(effective_user_id, limit=limit)
    return {
        "status": "success",
        "confirmations": [c.to_dict() for c in confirmations],
    }


@app.post("/confirmations/{confirmation_id}/resolve")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def resolve_confirmation_endpoint(
    request: Request, confirmation_id: str, body: ConfirmationResolveRequest
):
    """处理单个长期记忆确认项。"""
    ltm = _get_ltm()
    result = ltm.resolve_confirmation(confirmation_id, body.status)
    if not result:
        raise AgentError(
            code=ErrorCode.MEMORY_CONFIRMATION_ERROR,
            message=f"确认请求处理失败: {confirmation_id}",
        )
    return {"status": "success", "confirmation": result.to_dict()}


@app.post("/confirmations/batch")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def batch_confirm_endpoint(request: Request, body: BatchConfirmRequest):
    """批量接受或拒绝长期记忆确认项。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(body.user_id)
    results = ltm.batch_confirm(effective_user_id, body.confirmation_ids, body.status)
    return {
        "status": "success",
        "resolved_count": len(results),
        "confirmations": [c.to_dict() for c in results],
    }


@app.get("/events")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def list_events_endpoint(
    request: Request,
    user_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """查询长期记忆事件流水。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    events = ltm.get_event_logs(
        effective_user_id,
        memory_id=memory_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )
    return {
        "status": "success",
        "events": [e.to_dict() for e in events],
    }


@app.get("/memory/stats")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def memory_stats_endpoint(request: Request, user_id: Optional[str] = None):
    """返回长期记忆统计信息。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    stats = ltm.get_stats(effective_user_id)
    return {"status": "success", "stats": stats}


@app.post("/memory/maintenance")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def memory_maintenance_endpoint(request: Request, user_id: Optional[str] = None):
    """触发长期记忆维护任务。"""
    ltm = _get_ltm()
    effective_user_id = normalize_user_id(user_id)
    result = ltm.run_maintenance(effective_user_id)
    return {"status": "success", "result": result}


@app.post("/memory/backup")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def create_backup_endpoint(request: Request, tag: Optional[str] = None):
    """创建长期记忆备份。"""
    ltm = _get_ltm()
    backup_path = ltm.create_backup(tag)
    if not backup_path:
        raise AgentError(code=ErrorCode.MEMORY_BACKUP_ERROR, message="备份创建失败")
    return {"status": "success", "backup_path": backup_path}


@app.get("/memory/backups")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def list_backups_endpoint(request: Request):
    """列出可用长期记忆备份。"""
    ltm = _get_ltm()
    backups = ltm.list_backups()
    return {"status": "success", "backups": backups}


@app.post("/memory/restore")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def restore_backup_endpoint(request: Request, backup_path: str):
    """从备份恢复长期记忆数据。"""
    ltm = _get_ltm()
    success = ltm.restore_from_backup(backup_path)
    if not success:
        raise AgentError(code=ErrorCode.MEMORY_BACKUP_ERROR, message="备份恢复失败")
    return {"status": "success", "message": "数据库恢复成功"}


@app.get("/session/context")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def session_context_endpoint(
    request: Request,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """返回短期记忆为当前会话构建的上下文。"""
    effective_user_id = normalize_user_id(user_id)
    effective_session_id = normalize_session_id(session_id)
    session = session_manager.get_session(effective_user_id, effective_session_id)
    return {
        "status": "success",
        "user_id": effective_user_id,
        "session_id": effective_session_id,
        "session": session.memory.get_debug_info(),
        "context": session.memory.get_context_debug_info(),
        "task_state": session.memory.get_task_state().to_dict(),
        "messages": session.memory.get_all_messages(),
        "tool_calls": session.memory.get_tool_calls(),
        "salient_facts": session.memory.get_salient_facts(),
    }


@app.get("/memory/overview")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def memory_overview_endpoint(
    request: Request,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
):
    """聚合返回短期记忆、长期记忆和会话上下文概览。"""
    effective_user_id = normalize_user_id(user_id)
    effective_session_id = normalize_session_id(session_id)
    ltm = _get_ltm()
    session = session_manager.get_session(effective_user_id, effective_session_id)
    profile = ltm.get_profile(effective_user_id) or {
        "user_id": effective_user_id,
        "preferences": {},
        "habits": {},
        "constraints": [],
        "background": {},
        "facts": [],
    }
    memories_query = MemoryQuery(
        user_id=effective_user_id,
        limit=limit,
        offset=0,
    )
    memories = ltm.query_memories(memories_query)
    candidates = ltm.list_candidates(effective_user_id, limit=limit, offset=0)
    confirmations = ltm.list_pending_confirmations(
        effective_user_id, limit=min(limit, 20)
    )
    stats = ltm.get_stats(effective_user_id)
    recent_events = ltm.get_event_logs(
        effective_user_id, limit=min(limit, 20), offset=0
    )
    return {
        "status": "success",
        "user_id": effective_user_id,
        "session_id": effective_session_id,
        "profile": profile,
        "stats": stats,
        "memories": [item.to_dict() for item in memories],
        "candidates": [item.to_dict() for item in candidates],
        "confirmations": [item.to_dict() for item in confirmations],
        "recent_events": [item.to_dict() for item in recent_events],
        "session_memory": {
            "summary": session.memory.get_summary(),
            "task_state": session.memory.get_task_state().to_dict(),
            "messages": session.memory.get_all_messages(),
            "tool_calls": session.memory.get_tool_calls(),
            "salient_facts": session.memory.get_salient_facts(),
            "debug": session.memory.get_debug_info(),
        },
    }


@app.get("/")
async def root():
    """返回 API 根路径的服务说明。"""
    return {
        "message": "天文Agent API",
        "version": "1.0.0",
        "agent_available": _agent_holder.is_available,
    }


@app.get("/health")
async def health_check():
    """返回服务健康状态、Agent 可用性和会话数量。"""
    return {
        "status": "running" if _agent_holder.is_available else "degraded",
        "agent_available": _agent_holder.is_available,
        "active_sessions": session_manager.get_active_session_count(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True
    )
