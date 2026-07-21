from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.retrieval.retriever import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.prompt import build_prompt
from app.generation.llm import generate_answer
from app.core.config import RERANK_SCORE_THRESHOLD

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What are the latest techniques used by APT29?"])
    top_k_retrieve: int = Field(default=10, ge=1, le=50)
    top_n_rerank: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    parent_id: str
    source_type: str
    rerank_score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    candidates = hybrid_search(request.question, top_k=request.top_k_retrieve)
    reranked = rerank(request.question, candidates, top_n=request.top_n_rerank)
    reranked = [r for r in reranked if r.rerank_score >= RERANK_SCORE_THRESHOLD]
    system_prompt, user_message = build_prompt(request.question, reranked)
    answer = generate_answer(system_prompt, user_message)

    return QueryResponse(
        answer=answer,
        sources=[
            SourceChunk(
                parent_id=r.chunk.parent_id,
                source_type=r.chunk.source_type,
                rerank_score=r.rerank_score,
                text=r.chunk.text,
            )
            for r in reranked
        ],
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}