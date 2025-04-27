from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from query_data import get_cached_context, query_rag

app = FastAPI(
    title="API RAG ChromaDB",
    description="API pour la recherche et la génération de réponses à partir de ChromaDB et OpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query_text: str
    goal_id: str = os.getenv("DEFAULT_GOAL_ID", "doc_1")
    top_k: int = 10
    similarity_threshold: float = 0.2

class QueryResponse(BaseModel):
    context: str
    response: str

class GenerateResponse(BaseModel):
    response: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    try:
        context = get_cached_context(req.query_text, req.goal_id, req.similarity_threshold, req.top_k)
        response = query_rag(req.query_text, req.goal_id, req.similarity_threshold, req.top_k)
        return QueryResponse(context=context, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: QueryRequest):
    try:
        response = query_rag(req.query_text, req.goal_id, req.similarity_threshold, req.top_k)
        return GenerateResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

