from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import AstroAgent
import json
import os
import uuid

# 创建FastAPI应用
app = FastAPI(title="天文Agent API", description="具有短期记忆和流式输出的天文知识助手")

# 上传目录（用于让回答能“返回图片URL”）
UPLOAD_DIR = os.path.abspath("./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 创建全局Agent实例
agent = AstroAgent()


class QueryRequest(BaseModel):
    """查询请求"""
    query: str


class KnowledgeRequest(BaseModel):
    """知识添加请求"""
    knowledge: list[str]


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """查询接口，支持流式输出"""
    try:
        async def generate():
            async for event in agent.generate_events(request.query):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_with_image")
async def query_with_image_endpoint(
    query: str = Form(...),
    image: UploadFile = File(...),
):
    """
    多模态查询：用户上传图片 + 文本 query，返回 text/image 的 SSE 事件流。
    """
    try:
        # 保存上传图片到本地，生成可被静态服务访问的 URL
        ext = os.path.splitext(image.filename or "")[1].lower() or ".png"
        file_id = uuid.uuid4().hex
        save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        save_path = os.path.abspath(save_path)
        content = await image.read()
        with open(save_path, "wb") as f:
            f.write(content)

        image_url = f"/uploads/{file_id}{ext}"

        async def generate():
            # 先把用户上传图片回显给前端（满足“回答可以返回图片”）
            yield f"data: {json.dumps({'type': 'image', 'url': image_url, 'meta': {'source': 'user_upload'}}, ensure_ascii=False)}\n\n"
            async for event in agent.generate_events(query, image_path=save_path):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_knowledge")
async def add_knowledge_endpoint(request: KnowledgeRequest):
    """添加天文知识到RAG系统"""
    try:
        agent.add_astronomy_knowledge(request.knowledge)
        return {"status": "success", "message": "知识添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_memory")
async def clear_memory_endpoint():
    """清空记忆"""
    try:
        agent.clear_memory()
        return {"status": "success", "message": "记忆已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """根路径"""
    return {"message": "天文Agent API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    from config import settings
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
