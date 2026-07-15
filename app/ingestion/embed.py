"""
Embedding pipeline — turns Chunk objects into dense + sparse vectors using
BGE-M3's hybrid embedding output.

Uses BAAI's own `FlagEmbedding` library rather than raw `transformers`,
because BGE-M3 was trained to produce dense AND sparse (lexical weight)
vectors from a single forward pass, and FlagEmbedding exposes both directly.
Going through generic transformers would mean writing custom code to
extract the sparse weights ourselves — FlagEmbedding does this correctly
out of the box.

Note: model weights (~2.2GB) download from Hugging Face on first run, which
isn't reachable from Claude's build sandbox — this file is written and
structured correctly but validated by you locally, same as the CVE loader.
"""
from dataclasses import dataclass
from typing import Any

from app.ingestion.chunking import Chunk

_model = None  # lazy-loaded singleton — avoid reloading a 2GB model per call


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        # use_fp16=True halves memory/speeds up inference on GPU with negligible
        # accuracy loss; harmless fallback to fp32 behavior on CPU-only machines
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    dense_vector: list[float]        # 1024-dim
    sparse_vector: dict[str, float]  # token_id (as str) -> weight


def embed_chunks(chunks: list[Chunk], batch_size: int = 12) -> list[EmbeddedChunk]:
    """
    Batch-embed a list of Chunks. Batching matters here specifically because
    BGE-M3 is a ~560M param model — embedding one chunk at a time on CPU is
    painfully slow; batches of 8-16 give a large throughput win.
    """
    if not chunks:
        return []

    model = _get_model()
    texts = [c.text for c in chunks]

    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512,          # matches our ~400-token chunk ceiling with headroom
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # ColBERT multi-vector output unused here — skip for speed
    )

    dense_vecs = output["dense_vecs"]
    sparse_vecs = output["lexical_weights"]  # list[dict[int, float]] keyed by token id

    embedded = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        embedded.append(
            EmbeddedChunk(
                chunk=chunk,
                dense_vector=dense.tolist(),
                sparse_vector={str(k): float(v) for k, v in sparse.items()},
            )
        )
    return embedded


if __name__ == "__main__":
    from app.ingestion.attack_loader import fetch_attack_bundle, parse_attack_bundle
    from app.ingestion.chunking import chunk_attack_technique

    bundle = fetch_attack_bundle(save=False)
    techniques = parse_attack_bundle(bundle)[:3]  # small smoke test — 3 techniques only

    all_chunks: list[Chunk] = []
    for t in techniques:
        all_chunks.extend(chunk_attack_technique(t))

    print(f"Embedding {len(all_chunks)} chunks from {len(techniques)} techniques...")
    embedded = embed_chunks(all_chunks)

    for e in embedded[:2]:
        print(f"\n{e.chunk.parent_id} (part {e.chunk.metadata['part']}):")
        print(f"  dense_vector: len={len(e.dense_vector)}, sample={e.dense_vector[:5]}")
        print(f"  sparse_vector: {len(e.sparse_vector)} nonzero terms, "
              f"sample={dict(list(e.sparse_vector.items())[:3])}")