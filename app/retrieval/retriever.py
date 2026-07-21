"""
Hybrid retrieval against the Qdrant collection.

Queries BOTH named vectors (dense, sparse) in one call and lets Qdrant fuse
the two rankings server-side via Reciprocal Rank Fusion (RRF) — rather than
running two separate searches and merging scores ourselves, which would mean
reimplementing fusion logic Qdrant already provides correctly.

RRF, briefly: instead of combining raw similarity scores (which live on
different, incomparable scales for dense cosine-similarity vs. sparse
lexical-weight matching), RRF combines RANKS — a chunk ranked #1 by both
dense and sparse search outranks one ranked #1 by only one of them. This
sidesteps the score-scale mismatch entirely.
"""
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client.http import models as qmodels

from app.core.config import QDRANT_COLLECTION
from app.retrieval.vector_store import get_client


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float          # RRF fusion score — useful for relative ranking, not an absolute confidence measure
    source_type: str
    parent_id: str
    metadata: dict[str, Any]


def _embed_query(query: str) -> tuple[list[float], dict[str, float]]:
    """Embed a query string the same way chunks were embedded — BGE-M3 doesn't
    require different encoding for queries vs. documents, unlike some models
    (e.g. E5) that need a 'query:' prefix, so this reuses embed_chunks directly."""
    from app.ingestion.chunking import Chunk
    from app.ingestion.embed import embed_chunks

    fake_chunk = Chunk(chunk_id="query", text=query, source_type="query", parent_id="query")
    embedded = embed_chunks([fake_chunk])[0]
    return embedded.dense_vector, embedded.sparse_vector


def hybrid_search(
    query: str,
    top_k: int = 20,
    filters: Optional[qmodels.Filter] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top_k chunks using hybrid dense+sparse search with RRF fusion.

    `filters` accepts a qdrant_client Filter — e.g. to restrict to only
    ATT&CK techniques, or only CVEs above a CVSS threshold. This is the
    pre-filtering behavior discussed earlier: Qdrant applies it during the
    ANN search itself, not after.
    """
    client = get_client()
    dense_vec, sparse_vec = _embed_query(query)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            qmodels.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 2,  # overfetch before fusion narrows back to top_k
                filter=filters,
            ),
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=[int(k) for k in sparse_vec.keys()],
                    values=list(sparse_vec.values()),
                ),
                using="sparse",
                limit=top_k * 2,
                filter=filters,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
    )

    return [
        RetrievedChunk(
            chunk_id=str(point.id),
            text=point.payload.get("text", ""),
            score=point.score,
            source_type=point.payload.get("source_type", ""),
            parent_id=point.payload.get("parent_id", ""),
            metadata={k: v for k, v in point.payload.items() if k not in ("text",)},
        )
        for point in results.points
    ]


if __name__ == "__main__":
    query = "How does process injection work as a defense evasion technique?"
    results = hybrid_search(query, top_k=5)

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for r in results:
        print(f"[{r.score:.4f}] {r.parent_id} ({r.source_type}) — {r.text[:100]}...")