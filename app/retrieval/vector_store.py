"""
Qdrant vector store wrapper.

Uses Qdrant's NAMED VECTORS feature to store both the dense and sparse
representations of each chunk in one point, enabling hybrid search in a
single query (rather than running two separate searches and merging
results yourself). This is the piece of the "why Qdrant" argument from
earlier that actually gets implemented here — native hybrid support.

Requires a running Qdrant instance. Easiest local setup:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_DIM,
)
from app.ingestion.embed import EmbeddedChunk

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def ensure_collection(recreate: bool = False) -> None:
    """
    Create the collection with two named vectors: 'dense' (BGE-M3's 1024-dim
    dense embedding, compared via cosine similarity) and 'sparse' (BGE-M3's
    lexical weights, Qdrant's native sparse vector type — no dimension needed,
    it's a sparse index keyed by token id).
    """
    client = get_client()

    if recreate and client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(),
            },
        )
        # Index the fields we'll actually filter on (from the earlier
        # "pre-filter, not post-filter" discussion) — without an explicit
        # payload index, filtering still works but is slower at scale.
        for field_name, field_schema in [
            ("cve_id", qmodels.PayloadSchemaType.KEYWORD),
            ("technique_id", qmodels.PayloadSchemaType.KEYWORD),
            ("cvss_score", qmodels.PayloadSchemaType.FLOAT),
            ("cvss_severity", qmodels.PayloadSchemaType.KEYWORD),
            ("tactics", qmodels.PayloadSchemaType.KEYWORD),
            ("source_type", qmodels.PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )


def upsert_embedded_chunks(embedded_chunks: list[EmbeddedChunk], batch_size: int = 100) -> None:
    client = get_client()

    points = []
    for e in embedded_chunks:
        payload = {
            "text": e.chunk.text,
            "source_type": e.chunk.source_type,
            "parent_id": e.chunk.parent_id,
            **e.chunk.metadata,
        }
        points.append(
            qmodels.PointStruct(
                id=e.chunk.chunk_id,
                vector={
                    "dense": e.dense_vector,
                    "sparse": qmodels.SparseVector(
                        indices=[int(k) for k in e.sparse_vector.keys()],
                        values=list(e.sparse_vector.values()),
                    ),
                },
                payload=payload,
            )
        )

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i : i + batch_size])


if __name__ == "__main__":
    from app.ingestion.attack_loader import fetch_attack_bundle, parse_attack_bundle
    from app.ingestion.chunking import chunk_attack_technique
    from app.ingestion.embed import embed_chunks

    ensure_collection(recreate=True)
    print(f"Collection '{QDRANT_COLLECTION}' ready with dense+sparse named vectors.")

    bundle = fetch_attack_bundle(save=False)
    techniques = parse_attack_bundle(bundle)[:5]  # small smoke test

    chunks = []
    for t in techniques:
        chunks.extend(chunk_attack_technique(t))

    embedded = embed_chunks(chunks)
    upsert_embedded_chunks(embedded)

    count = get_client().count(QDRANT_COLLECTION).count
    print(f"Upserted {len(embedded)} chunks. Collection now holds {count} points.")