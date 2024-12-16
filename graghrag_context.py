from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import context_result
import numpy as np

app = FastAPI(
    title="GraghRAG API",
    description="API for GraghRAG.",
    version="v1.0.0",
)


class QueryResponse(BaseModel):
    """
    响应模型定义。
    """

    records: list  # 包含检索结果的列表，每个元素是一个字典


class QueryRequest(BaseModel):
    """
    请求模型定义。
    """

    knowledge_id: str = ""  # 知识库的唯一标识符
    query: str = ""  # 用户的查询内容
    top_k: int = 5  # 检索结果的最大数量，默认为 5
    score_threshold: float = 0.5  # 结果的相关性分数阈值，默认为 0.5


@app.post("/retrieval", response_model=QueryResponse)
async def query_dify(request: QueryRequest | None = None):
    context_data = await context_result.query_graphrag_context(request.query)
    records = np.array([])
    entities = np.array([])
    relationships = np.array([])

    for item in context_data.get("reports", []):
        records = np.append(
            records,
            {
                "metadata": {
                    "path": item["id"] + ".txt",
                    "description": item["title"],
                },
                "score": 0.98,
                "title": item["id"] + ".txt",
                "content": item["content"],
            },
        )
    for item in context_data.get("entities", []):
        entities = np.append(entities, item["entity"] + "-" + item["description"])

    for item in context_data.get("relationships", []):
        relationships = np.append(
            relationships,
            item["source"] + "-" + item["target"] + "-" + item["description"],
        )

    if entities.size > 0:
        records = np.append(
            records,
            {
                "metadata": {
                    "path": "entity" + ".txt",
                    "description": "entity" + ".txt",
                },
                "score": 0.98,
                "title": "entity" + ".txt",
                "content": "\n".join(entities),
            },
        )
    if relationships.size > 0:
        records = np.append(
            records,
            {
                "metadata": {
                    "path": "relationships" + ".txt",
                    "description": "relationships" + ".txt",
                },
                "score": 0.98,
                "title": "relationships" + ".txt",
                "content": "\n".join(relationships),
            },
        )
    for item in context_data.get("sources", []):
        records = np.append(
            records,
            {
                "metadata": {
                    "path": item["id"] + ".txt",
                    "description": "sources",
                },
                "score": 0.98,
                "title": item["id"] + ".txt",
                "content": item["text"],
            },
        )
    if len(context_data.get("chunks", [])) > 0:
        records = np.append(
            records,
            {
                "metadata": {
                    "path": "chunks.txt",
                    "description": "chunks",
                },
                "score": 0.98,
                "title": "chunks.txt",
                "content": "".join(context_data.get("chunks", [])),
            },
        )
    return {"records": records}


@app.on_event("startup")
async def on_startup():
    """FastAPI 启动事件"""
    await context_result.initialize_context_builder()
