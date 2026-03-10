from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent_langchain import AstroAgent
import json

# 创建FastAPI应用
app = FastAPI(title="天文Agent API", description="具有短期记忆和流式输出的天文知识助手")

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
        def generate():
            for chunk in agent.generate_response(request.query):
                # 流式输出，每个chunk作为一个SSE事件
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        
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
