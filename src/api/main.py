import os
import sys

# 确保src在Python路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from src.agent import AstroAgent
from src.core.errors import AgentError, ErrorHandler, ErrorCode
import json
import uuid
import asyncio

app = FastAPI(title="天文Agent API", description="具有短期记忆、长期记忆和流式输出的天文知识助手")

UPLOAD_DIR = os.path.abspath("./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

agent = AstroAgent()


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
    }
    status = status_map.get(exc.code, 500)
    return JSONResponse(status_code=status, content=exc.to_dict())


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    error = ErrorHandler.handle(exc, {"path": str(request.url)})
    return JSONResponse(status_code=500, content=error.to_dict())


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    user_id = request.user_id or agent.user_id

    async def generate():
        async for event in agent.generate_events(request.query):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query_with_image")
async def query_with_image_endpoint(
    query: str = Form(...),
    image: UploadFile = File(...),
):
    ext = os.path.splitext(image.filename or "")[1].lower() or ".png"
    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    save_path = os.path.abspath(save_path)
    content = await image.read()
    with open(save_path, "wb") as f:
        f.write(content)

    image_url = f"/uploads/{file_id}{ext}"

    async def generate():
        yield f"data: {json.dumps({'type': 'image', 'url': image_url, 'meta': {'source': 'user_upload'}}, ensure_ascii=False)}\n\n"
        async for event in agent.generate_events(query, image_path=save_path):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query_with_audio")
async def query_with_audio_endpoint(
    query: str = Form(""),
    audio: UploadFile = File(...),
):
    ext = os.path.splitext(audio.filename or "")[1].lower() or ".wav"
    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    save_path = os.path.abspath(save_path)
    content = await audio.read()
    with open(save_path, "wb") as f:
        f.write(content)

    augmented_query = await asyncio.to_thread(
        agent.speech_service.build_speech_query, query, save_path
    )

    async def generate():
        yield f"data: {json.dumps({'type': 'transcription', 'text': augmented_query}, ensure_ascii=False)}\n\n"
        async for event in agent.generate_events(augmented_query):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/add_knowledge")
async def add_knowledge_endpoint(request: KnowledgeRequest):
    agent.add_astronomy_knowledge(request.knowledge)
    return {"status": "success", "message": "知识添加成功"}


@app.get("/profile")
async def get_profile_endpoint(user_id: Optional[str] = None):
    user_id = user_id or agent.user_id
    profile = agent.long_term_memory.load_profile(user_id)
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
            "user_id": user_id,
            "preferences": {},
            "habits": {},
            "constraints": []
        }


@app.delete("/profile")
async def delete_profile_endpoint(user_id: Optional[str] = None):
    user_id = user_id or agent.user_id
    deleted = agent.long_term_memory.delete_profile(user_id)
    if deleted:
        return {"status": "success", "message": "用户画像已删除", "user_id": user_id}
    else:
        return {"status": "success", "message": "用户画像不存在或已被删除", "user_id": user_id}


@app.post("/clear_memory")
async def clear_memory_endpoint():
    agent.clear_memory()
    return {"status": "success", "message": "记忆已清空"}


@app.get("/")
async def root():
    return {"message": "天文Agent API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    from src.core.config import settings
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
