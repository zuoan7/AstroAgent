import os
import re
import time
import threading
from collections import OrderedDict
from typing import Optional, Dict

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.agent import AstroAgent
from src.agent.streaming_service import StreamingService
from src.memory.memory import ShortTermMemory
from src.core.errors import AgentError, ErrorHandler, ErrorCode
from src.core.config import settings
from src.core.logger import logger
import json
import uuid
import asyncio


app = FastAPI(title="天文Agent API", description="具有短期记忆、长期记忆和流式输出的天文知识助手")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

UPLOAD_DIR = os.path.abspath("./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def sanitize_filename(filename: str) -> str:
    if not filename:
        return ""
    filename = filename.replace("\\", "/")
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-.]', '', filename)
    filename = re.sub(r'\.{2,}', '.', filename)
    return filename.strip('. ')


def validate_upload_path(save_path: str) -> bool:
    resolved = os.path.abspath(save_path)
    upload_dir_resolved = os.path.abspath(UPLOAD_DIR)
    return resolved.startswith(upload_dir_resolved + os.sep) or resolved == upload_dir_resolved


def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in allowed_extensions


class _AgentHolder:
    """懒加载Agent持有器，支持初始化失败降级"""

    def __init__(self):
        self._agent = None
        self._initialized = False
        self._init_error = None

    def get(self):
        if not self._initialized:
            self._initialize()
        return self._agent

    def _initialize(self):
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
        if not self._initialized:
            self._initialize()
        return self._agent is not None

    @property
    def init_error(self):
        return self._init_error


_agent_holder = _AgentHolder()


def get_agent():
    agent = _agent_holder.get()
    if agent is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"Agent服务暂时不可用: {_agent_holder.init_error or '初始化失败'}",
        )
    return agent


class SessionData:
    def __init__(self, user_id: str, agent: AstroAgent):
        self.user_id = user_id
        self.memory = ShortTermMemory(session_id=f"stm_{user_id}", user_id=user_id)
        self.streaming_service = StreamingService(
            agent_executor=agent._agent_executor,
            memory=self.memory,
            long_term_memory=agent.long_term_memory,
            user_id=user_id,
            fallback_service=agent.fallback_service,
        )
        self.last_access_time = time.time()


class SessionManager:
    def __init__(self, max_age: int = None, cleanup_interval: int = None):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._max_age = max_age or settings.SESSION_MAX_AGE_SECONDS
        self._cleanup_interval = cleanup_interval or settings.SESSION_CLEANUP_INTERVAL_SECONDS
        self._last_cleanup = time.time()

    def get_session(self, user_id: str) -> SessionData:
        with self._lock:
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale_sessions()

            if user_id not in self._sessions:
                agent = get_agent()
                logger.info(f"创建新会话: user_id={user_id}")
                self._sessions[user_id] = SessionData(user_id, agent)

            session = self._sessions[user_id]
            session.last_access_time = now
            return session

    def clear_session(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._sessions:
                self._sessions[user_id].memory.clear()
                del self._sessions[user_id]
                logger.info(f"会话已清除: user_id={user_id}")
                return True
            return False

    def _cleanup_stale_sessions(self):
        now = time.time()
        stale_users = [
            uid for uid, session in self._sessions.items()
            if now - session.last_access_time > self._max_age
        ]
        for uid in stale_users:
            del self._sessions[uid]
            logger.info(f"清理过期会话: user_id={uid}")
        if stale_users:
            logger.info(f"清理了 {len(stale_users)} 个过期会话")
        self._last_cleanup = now

    def get_active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


session_manager = SessionManager()


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None


class KnowledgeRequest(BaseModel):
    knowledge: list[str]
    user_id: Optional[str] = None


class UserProfileRequest(BaseModel):
    user_id: Optional[str] = None


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    status_map = {
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
    error = ErrorHandler.handle(exc, {"path": str(request.url)})
    return JSONResponse(status_code=500, content=error.to_dict())


@app.post("/query")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    user_id = body.user_id or settings.DEFAULT_USER_ID
    session = session_manager.get_session(user_id)

    async def generate():
        async for event in session.streaming_service.generate_events(body.query):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query_with_image")
@limiter.limit(f"{settings.UPLOAD_RATE_LIMIT_PER_MINUTE}/minute")
async def query_with_image_endpoint(
    request: Request,
    query: str = Form(...),
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
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
    effective_user_id = user_id or settings.DEFAULT_USER_ID
    session = session_manager.get_session(effective_user_id)

    async def generate():
        yield f"data: {json.dumps({'type': 'image', 'url': image_url, 'meta': {'source': 'user_upload'}}, ensure_ascii=False)}\n\n"
        async for event in session.streaming_service.generate_events(query, image_path=save_path):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query_with_audio")
@limiter.limit(f"{settings.UPLOAD_RATE_LIMIT_PER_MINUTE}/minute")
async def query_with_audio_endpoint(
    request: Request,
    query: str = Form(""),
    audio: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
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

    effective_user_id = user_id or settings.DEFAULT_USER_ID
    session = session_manager.get_session(effective_user_id)

    augmented_query = await asyncio.to_thread(
        get_agent().speech_service.build_speech_query, query, save_path
    )

    async def generate():
        yield f"data: {json.dumps({'type': 'transcription', 'text': augmented_query}, ensure_ascii=False)}\n\n"
        async for event in session.streaming_service.generate_events(augmented_query):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/add_knowledge")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def add_knowledge_endpoint(request: Request, body: KnowledgeRequest):
    get_agent().add_astronomy_knowledge(body.knowledge)
    return {"status": "success", "message": "知识添加成功"}


@app.get("/profile")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_profile_endpoint(request: Request, user_id: Optional[str] = None):
    effective_user_id = user_id or settings.DEFAULT_USER_ID
    profile = get_agent().long_term_memory.load_profile(effective_user_id)
    if profile:
        return {
            "status": "success",
            "user_id": profile.user_id,
            "preferences": profile.preferences,
            "habits": profile.habits,
            "constraints": profile.constraints,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at
        }
    else:
        return {
            "status": "success",
            "message": "暂无用户画像信息",
            "user_id": effective_user_id,
            "preferences": {},
            "habits": {},
            "constraints": []
        }


@app.delete("/profile")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def delete_profile_endpoint(request: Request, user_id: Optional[str] = None):
    effective_user_id = user_id or settings.DEFAULT_USER_ID
    deleted = get_agent().long_term_memory.delete_profile(effective_user_id)
    if deleted:
        return {"status": "success", "message": "用户画像已删除", "user_id": effective_user_id}
    else:
        return {"status": "success", "message": "用户画像不存在或已被删除", "user_id": effective_user_id}


@app.post("/clear_memory")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def clear_memory_endpoint(request: Request, user_id: Optional[str] = None):
    effective_user_id = user_id or settings.DEFAULT_USER_ID
    session_manager.clear_session(effective_user_id)
    return {"status": "success", "message": "记忆已清空", "user_id": effective_user_id}


@app.get("/")
async def root():
    return {
        "message": "天文Agent API",
        "version": "1.0.0",
        "agent_available": _agent_holder.is_available,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "running" if _agent_holder.is_available else "degraded",
        "agent_available": _agent_holder.is_available,
        "active_sessions": session_manager.get_active_session_count(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
