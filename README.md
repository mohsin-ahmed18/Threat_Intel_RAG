# Threat Intel RAG Assistant

A RAG system that answers natural-language questions about cybersecurity threats, grounded in real CVE and MITRE ATT&CK data, with every claim cited to its source record.

**[Live demo GIF / screenshot here]**

---

## What it does

Ask questions like:
- *"How does process injection work as a defense evasion technique?"*
- *"How do adversaries abuse the Windows Task Scheduler?"*
- *"What techniques are used for lateral movement via VNC?"*

Answers are generated strictly from retrieved MITRE ATT&CK and NVD CVE records — not general model knowledge. Every claim cites a technique or CVE ID, and the source chunks used are shown alongside the answer with relevance scores.

## Design decisions

| Decision | Why |
|---|---|
| **Structure-aware chunking** (not fixed-size or semantic) | CVE/ATT&CK records have natural boundaries — one CVE, one technique — so splitting on structure avoids cutting a record mid-sentence |
| **BGE-M3 hybrid embeddings** (dense + sparse) | Security text has exact identifiers (`CVE-2024-3094`, `T1055`) dense embeddings under-match; sparse retrieval catches the exact token |
| **Qdrant** over ChromaDB | Filters metadata *before* the ANN search, preserving recall under narrow filters (e.g. `CVSS > 9 AND date > 2025`) |
| **Reciprocal Rank Fusion** | Combines dense (cosine) and sparse (lexical) rankings without needing their scores to be on the same scale |
| **BGE-Reranker-v2 cross-encoder** | Scores query+chunk jointly for sharper relevance ordering, applied only to the top candidates from hybrid search |
| **Citation-enforcing prompt** | Explicitly instructs the model to cite sources and say when context is insufficient, rather than filling gaps from general knowledge |

## Architecture

```
INGESTION (offline)
  NVD API ──┐
            ├─→ structure-aware chunking ─→ BGE-M3 (dense+sparse) ─→ Qdrant
  MITRE ATT&CK STIX ──┘

QUERY (live)
  Question ─→ BGE-M3 embed ─→ Qdrant hybrid search (RRF)
           ─→ BGE-Reranker-v2 ─→ relevance threshold
           ─→ prompt assembly ─→ GPT-4o ─→ cited answer + sources
```

## Tech stack

- **Backend**: FastAPI
- **Embeddings**: BGE-M3 (`BAAI/bge-m3`), hybrid dense + sparse
- **Reranking**: BGE-Reranker-v2-m3
- **Vector store**: Qdrant (Cloud free tier)
- **LLM**: GPT-4o (OpenAI), Anthropic also supported
- **Frontend**: Vanilla HTML/CSS/JS query console

## Data sources

- **CVEs**: [NVD REST API v2.0](https://nvd.nist.gov/developers/vulnerability-data-api)
- **MITRE ATT&CK**: [official STIX 2.1 bundles](https://github.com/mitre/cti)
- **Threat reports/advisories**: planned, not yet implemented

## Project structure

```
threat-intel-rag/
├── app/
│   ├── main.py                # FastAPI entrypoint + frontend serving
│   ├── api/routes.py          # /query, /health
│   ├── static/index.html      # frontend UI
│   ├── ingestion/
│   │   ├── cve_loader.py      # NVD API loader
│   │   ├── attack_loader.py   # MITRE ATT&CK STIX loader
│   │   ├── chunking.py        # structure-aware chunker
│   │   └── embed.py           # BGE-M3 embedding pipeline
│   ├── retrieval/
│   │   ├── vector_store.py    # Qdrant collection + upsert
│   │   ├── retriever.py       # hybrid search + RRF fusion
│   │   └── reranker.py        # cross-encoder reranking
│   ├── generation/
│   │   ├── prompt.py          # citation-enforcing prompt
│   │   └── llm.py             # LLM call (OpenAI / Anthropic)
│   └── core/config.py
├── data/                      # gitignored, reproducible via loaders
├── requirements.txt
└── .env.example
```

## Running it locally

**1. Install**
```bash
git clone https://github.com/mohsin-ahmed18/Threat_Intel_RAG.git
cd Threat_Intel_RAG
python -m venv myenv
myenv\Scripts\activate      # Windows
pip install -r requirements.txt
```

**2. Configure** — copy `.env.example` to `.env`:
```
NVD_API_KEY=          # optional, raises rate limit 5→50 req/30s
QDRANT_URL=            # from cloud.qdrant.io
QDRANT_API_KEY=
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=
```

**3. Ingest**
```bash
python -m app.ingestion.cve_loader
python -m app.ingestion.attack_loader
python -m app.retrieval.vector_store
```

**4. Run**
```bash
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/` for the UI, `/docs` for the raw API.

## API

**`POST /query`**
```json
{ "question": "How does process injection work as a defense evasion technique?" }
```

Returns:
```json
{
  "answer": "Process injection works as a defense evasion technique by... (T1055.011)",
  "sources": [
    { "parent_id": "T1055.011", "source_type": "attack_technique", "rerank_score": 0.6759, "text": "..." }
  ]
}
```

**`GET /health`** — liveness check.

## Roadmap

- [x] NVD CVE + MITRE ATT&CK ingestion (live sources)
- [x] Structure-aware chunking, hybrid embedding, Qdrant storage
- [x] Hybrid retrieval (RRF) + cross-encoder reranking
- [x] Citation-grounded generation, FastAPI backend + UI
- [ ] Threat report/advisory ingestion (PDF/HTML)
- [ ] Full-corpus ingestion (currently a curated sample)
- [ ] Streaming responses (SSE) for live pipeline visibility
- [ ] Metadata-filtered queries
- [ ] Agentic multi-hop query planning
