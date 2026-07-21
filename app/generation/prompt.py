"""
Prompt assembly for the generation step.

Design choices worth noting:
- System prompt explicitly instructs citation by ID (CVE-xxxx-xxxx / Txxxx),
  not just "cite your sources" — vague instructions produce vague citations.
  Forcing ID-level citation means a user can go verify any claim against the
  actual NVD/ATT&CK record, which matters a lot for a security tool where
  wrong information has real consequences.
- Explicitly instructs the model to say when context is insufficient, rather
  than filling gaps from its own training knowledge — this is the core
  grounding guarantee RAG is supposed to provide. Without this instruction,
  models default to blending retrieved context with parametric knowledge,
  which defeats the point of building retrieval at all.
"""
from app.retrieval.reranker import RerankedChunk

SYSTEM_PROMPT = """You are a threat intelligence assistant. Answer the user's question using ONLY the provided context chunks below — do not use outside knowledge, even if you know more about the topic.

Rules:
1. Every factual claim must cite its source using the ID shown in the chunk (e.g. "(T1055.011)" or "(CVE-2024-3094)").
2. If the provided context does not fully answer the question, say so explicitly — name what's missing rather than filling the gap from general knowledge.
3. If multiple chunks disagree or provide partial information, present that clearly rather than picking one.
4. Keep answers precise and technical — the user is a security practitioner, not a general audience."""


def build_prompt(query: str, chunks: list[RerankedChunk]) -> tuple[str, str]:
    """Returns (system_prompt, user_message) ready to send to the LLM."""
    if not chunks:
        context_block = "(No relevant context was retrieved for this query.)"
    else:
        parts = []
        for c in chunks:
            source_id = c.chunk.parent_id
            parts.append(f"[Source: {source_id}]\n{c.chunk.text}")
        context_block = "\n\n---\n\n".join(parts)

    user_message = f"""Context:
{context_block}

Question: {query}"""

    return SYSTEM_PROMPT, user_message