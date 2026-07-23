from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    from app.ingestion.embed import _get_model
    from app.retrieval.reranker import _get_reranker

    print("Preloading embedding model...")
    _get_model()
    print("Preloading reranker model...")
    _get_reranker()
    print("Models loaded. API ready.")

    yield 


app = FastAPI(
    title="Threat Intel RAG Assistant",
    description="RAG-powered assistant over CVE and MITRE ATT&CK data",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")