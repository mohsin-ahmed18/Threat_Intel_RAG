"""
Reranking layer — takes the hybrid_search candidate pool and re-scores it
with BGE-reranker-v2-m3, a cross-encoder.

Cross-encoder vs. the bi-encoder used in retrieval: BGE-M3 (retrieval) embeds
query and chunk SEPARATELY, then compares vectors — fast enough to run
against thousands of chunks, but the model never sees query and chunk
together. The reranker feeds [query, chunk] into the model AS A PAIR in one
forward pass, so it can pick up on interactions between them that separate
embeddings lose — at the cost of being far too slow to run on a whole
collection. Hence the two-stage design: cheap hybrid search narrows
thousands of chunks to ~20, then the expensive reranker only touches those 20.
"""
from dataclasses import dataclass

from app.retrieval.retriever import RetrievedChunk

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


@dataclass
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float  # raw cross-encoder relevance logit — higher = more relevant


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RerankedChunk]:
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)  # normalize=True -> scores in [0,1]

    # compute_score returns a bare float instead of a list when given exactly one pair
    if isinstance(scores, float):
        scores = [scores]

    reranked = [
        RerankedChunk(chunk=c, rerank_score=s) for c, s in zip(candidates, scores)
    ]
    reranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return reranked[:top_n]


if __name__ == "__main__":
    from app.retrieval.retriever import hybrid_search

    query = "How does process injection work as a defense evasion technique?"
    candidates = hybrid_search(query, top_k=10)

    print(f"Query: {query}")
    print(f"\n--- Before reranking (hybrid RRF order) ---")
    for c in candidates:
        print(f"[{c.score:.4f}] {c.parent_id} — {c.text[:80]}...")

    reranked = rerank(query, candidates, top_n=5)
    print(f"\n--- After reranking (top 5) ---")
    for r in reranked:
        print(f"[{r.rerank_score:.4f}] {r.chunk.parent_id} — {r.chunk.text[:80]}...")