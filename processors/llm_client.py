"""Lightweight LLM client for review analysis (no LangChain required).

Supports OpenAI, Anthropic, and Groq via their official SDKs. Used by the live
review pipeline, sentiment/theme extractors, and the RAG query engine.
"""

from __future__ import annotations

import os
import re

_PLACEHOLDER_RE = re.compile(
    r"your-|sk-your|gsk-your|sk-ant-your|changeme|xxx|placeholder|example",
    re.I,
)

_ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "VECTOR_STORE",
)


def _clean_secret(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'")


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(value))


def _valid_api_key(value: str | None) -> bool:
    cleaned = _clean_secret(value)
    return len(cleaned) >= 12 and not _is_placeholder(cleaned)


def bootstrap_env() -> None:
    """Normalize env vars; hydrate from Streamlit secrets when running on Cloud."""
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.getenv(key):
            os.environ[key] = _clean_secret(os.getenv(key))

    try:
        import streamlit as st

        for key in _ENV_KEYS:
            if key not in st.secrets:
                continue
            secret_val = _clean_secret(str(st.secrets[key]))
            if not secret_val:
                continue
            # Secrets always win for provider + API keys (fixes stale/wrong env on Cloud).
            if key.endswith("_API_KEY") or key in ("LLM_PROVIDER", "LLM_MODEL"):
                os.environ[key] = secret_val
                continue
            current = _clean_secret(os.getenv(key))
            if not current or _is_placeholder(current):
                os.environ[key] = secret_val
    except Exception:
        pass

    # Default to local embeddings when unset (avoids OpenAI 401 on embeddings/search).
    os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
    os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    os.environ.setdefault("VECTOR_STORE", "memory")


def auto_llm_provider() -> str | None:
    """Return the first provider with a valid-looking API key, respecting LLM_PROVIDER."""
    bootstrap_env()
    preferred = (os.getenv("LLM_PROVIDER") or "").lower()
    order: list[str] = []
    if preferred in ("groq", "openai", "anthropic"):
        order.append(preferred)
    for p in ("groq", "openai", "anthropic"):
        if p not in order:
            order.append(p)
    for p in order:
        if llm_configured(p):
            return p
    return None


def _is_auth_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("401", "invalid api key", "invalid_api_key", "authentication", "unauthorized")
    )


def chat_complete_safe(
    prompt: str, *, temperature: float = 0, provider: str | None = None
) -> tuple[str | None, str | None]:
    """Like chat_complete but never raises — returns (text, error)."""
    try:
        return chat_complete(prompt, temperature=temperature, provider=provider), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def llm_configured(provider: str | None = None) -> bool:
    """True if a non-placeholder API key is set for the given or configured provider."""
    bootstrap_env()
    p = (provider or os.getenv("LLM_PROVIDER") or "groq").lower()
    if p == "groq":
        return _valid_api_key(os.getenv("GROQ_API_KEY"))
    if p == "anthropic":
        return _valid_api_key(os.getenv("ANTHROPIC_API_KEY"))
    return _valid_api_key(os.getenv("OPENAI_API_KEY"))


def _default_model(provider: str) -> str:
    defaults = {
        "groq": "llama-3.3-70b-versatile",
        "anthropic": "claude-3-5-sonnet-latest",
        "openai": "gpt-4o-mini",
    }
    return os.getenv("LLM_MODEL") or defaults.get(provider, "gpt-4o-mini")


def chat_complete(
    prompt: str,
    *,
    temperature: float = 0,
    provider: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """Run a single-turn chat completion and return the assistant text."""
    bootstrap_env()
    p = (provider or auto_llm_provider() or "groq").lower()
    model = _default_model(p)

    if p == "groq":
        api_key = _clean_secret(os.getenv("GROQ_API_KEY"))
        if not _valid_api_key(api_key):
            raise RuntimeError(
                "GROQ_API_KEY is missing or invalid. Set a valid key in `.env` "
                "or Streamlit Secrets (LLM_PROVIDER=groq)."
            )
        os.environ["GROQ_API_KEY"] = api_key
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    if p == "anthropic":
        api_key = _clean_secret(os.getenv("ANTHROPIC_API_KEY"))
        if not _valid_api_key(api_key):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing or invalid. Set a valid key in `.env "
                "or Streamlit Secrets."
            )
        os.environ["ANTHROPIC_API_KEY"] = api_key
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

    api_key = _clean_secret(os.getenv("OPENAI_API_KEY"))
    if not _valid_api_key(api_key):
        raise RuntimeError(
            "OPENAI_API_KEY is missing or invalid. Set a valid key in `.env` "
            "or Streamlit Secrets."
        )
    os.environ["OPENAI_API_KEY"] = api_key
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
