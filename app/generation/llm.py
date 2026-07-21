"""
LLM generation call.

Provider is picked from config (LLM_PROVIDER) — both 'anthropic' and 'openai'
are implemented. Kept as a thin wrapper — one function in, one string out —
so the FastAPI route and any future swap to a local model (Ollama/vLLM)
don't need to change anything except this file.
"""
from app.core.config import LLM_PROVIDER, LLM_MODEL

_anthropic_client = None
_openai_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    return _openai_client


def generate_answer(system_prompt: str, user_message: str, max_tokens: int = 1024) -> str:
    if LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    if LLM_PROVIDER == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    raise NotImplementedError(
        f"LLM_PROVIDER='{LLM_PROVIDER}' not implemented yet. "
        f"Supported: 'anthropic', 'openai'."
    )


if __name__ == "__main__":
    from app.retrieval.retriever import hybrid_search
    from app.retrieval.reranker import rerank
    from app.generation.prompt import build_prompt

    query = "How does process injection work as a defense evasion technique?"
    candidates = hybrid_search(query, top_k=10)
    reranked = rerank(query, candidates, top_n=5)
    system_prompt, user_message = build_prompt(query, reranked)

    print("Calling LLM...\n")
    answer = generate_answer(system_prompt, user_message)
    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}")