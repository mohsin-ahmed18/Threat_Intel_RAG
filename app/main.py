from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload both models once at startup so the FIRST real request isn't the
    # one paying for BGE-M3 + reranker weight loading (~seconds of dead time
    # otherwise land on whichever user happens to hit the API first).
    from app.ingestion.embed import _get_model
    from app.retrieval.reranker import _get_reranker

    print("Preloading embedding model...")
    _get_model()
    print("Preloading reranker model...")
    _get_reranker()
    print("Models loaded. API ready.")

    yield  # app runs here


app = FastAPI(
    title="Threat Intel RAG Assistant",
    description="RAG-powered assistant over CVE and MITRE ATT&CK data",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)