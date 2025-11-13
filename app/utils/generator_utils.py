from typing import List, Tuple
import os

OPENAI_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


def _build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts[:3])
    return (
        "You are a helpful HR policy assistant. Answer using ONLY the provided context. "
        "If the answer is not present, say you cannot answer from the provided documents.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
    )


def _openai_answer(query: str, contexts: List[str]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(query, contexts)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def generate_answer(query: str, contexts: List[str]) -> Tuple[str, List[str]]:
    if not contexts:
        return ("I couldn't find relevant information in the provided documents.", [])

    # Try OpenAI if configured
    llm_answer = _openai_answer(query, contexts)
    if llm_answer:
        return llm_answer, contexts[:3]

    # Fallback: return concatenated snippets
    snippet = "\n\n".join(contexts[:3])
    answer = (
        "Here is a response based on the most relevant policy snippets I found: \n\n" + snippet
    )
    return answer, contexts[:3]
