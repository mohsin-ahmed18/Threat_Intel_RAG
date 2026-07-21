from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

# Frontend is a static file served from the same origin as the API, so this
# is mainly a safety net for local development with --reload on a different
# port, or if you ever split the frontend onto its own host later.
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