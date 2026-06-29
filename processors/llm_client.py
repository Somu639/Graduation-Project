"""Lightweight LLM client for review analysis (no LangChain required).

Supports OpenAI, Anthropic, and Groq via their official SDKs. Used by the live
review pipeline, sentiment/theme extractors, and the RAG query engine.
"""

from __future__ import annotations

import os


def llm_configured(provider: str | None = None) -> bool:
    """True if an API key is set for the given or configured provider."""
    p = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if p == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if p == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def _default_model(provider: str) -> str:
    defaults = {
        "groq": "llama-3.3-70b-versatile",
        "anthropic": "claude-3-5-sonnet-latest",
        "openai": "gpt-4o-mini",
    }
    return os.getenv("LLM_MODEL") or defaults.get(provider, "gpt-4o-mini")


def chat_complete(prompt: str, *, temperature: float = 0, provider: str | None = None) -> str:
    """Run a single-turn chat completion and return the assistant text."""
    p = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    model = _default_model(p)

    if p == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY must be set when LLM_PROVIDER=groq.")
        from groq import Groq

        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    if p == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic.")
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(parts)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai.")
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
