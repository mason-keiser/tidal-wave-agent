# rag.py
from __future__ import annotations

import os
import time
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
import tiktoken


# -------------- Config (env) --------------
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# Dev/runtime toggles
USE_LLM = os.getenv("USE_LLM", "1") != "0"              # set USE_LLM=0 to skip LLM (dev mode)
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))

# OpenAI network settings
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))


# -------------- Utilities --------------
def _require_api_key(for_what: str) -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set but required for {for_what}. "
            "Add it to your environment or .env file."
        )

def _get_encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text or ""))

def _retry_sleep(attempt: int) -> None:
    # Exponential backoff with tiny jitter
    time.sleep(min(2 ** attempt, 8) + 0.1 * attempt)

def _to_plain(text: str) -> str:
    """Strip common markdown/list artifacts and collapse whitespace."""
    # remove emphasis and headings
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # remove list markers / numbering
    text = re.sub(r'^\s*([-*+]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    # collapse whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@dataclass
class RetrievedDoc:
    id: str
    text: str
    meta: Dict[str, Any]
    score: Optional[float] = None


# -------------- Chroma helpers --------------
def get_chroma() -> Client:
    """Return a Chroma client persisted to CHROMA_DIR."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Client(Settings(persist_directory=CHROMA_DIR))

def get_or_create_collection(chroma: Client, name: str = "mason"):
    """Create or fetch a Chroma collection using OpenAI embeddings."""
    api_key = os.getenv("OPENAI_API_KEY")  # <-- get fresh value
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set but required for embeddings (Chroma OpenAIEmbeddingFunction). "
            "Add it to your environment or .env file."
        )
    emb_fn = OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBED_MODEL)
    try:
        return chroma.get_collection(name, embedding_function=emb_fn)
    except Exception:
        return chroma.create_collection(name, embedding_function=emb_fn)

def retrieve(collection, question: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Query Chroma for top-k documents. Returns list of dicts: {id, text, meta}.
    """
    res = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas"]  # <-- removed "ids"
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    # Chroma may not return ids anymore; use index or meta info
    return [{"id": metas[i].get("id", f"doc_{i}"), "text": d or "", "meta": (m or {})} for i, (d, m) in enumerate(zip(docs, metas))]


# -------------- Prompt building & generation --------------
def _build_context_snippets(
    retrieved: List[RetrievedDoc],
    model_for_tokens: str,
    max_tokens: int,
) -> Tuple[str, List[RetrievedDoc]]:
    """
    Build a context string from retrieved docs that fits within the token budget.
    Returns (context_text, used_docs).
    """
    enc = _get_encoder(model_for_tokens)
    used: List[RetrievedDoc] = []
    pieces: List[str] = []
    budget = max_tokens

    for doc in retrieved:
        source = doc.meta.get("path") or doc.meta.get("source") or doc.meta.get("title") or doc.id
        header = f"[Source: {source}]\n"
        body = (doc.text or "").strip()
        chunk = header + body + "\n\n"

        tokens = _count_tokens(chunk, enc)
        if tokens > budget:
            # Try to fit a truncated body
            header_tokens = _count_tokens(header, enc)
            remain = max(0, budget - header_tokens)
            if remain <= 0:
                break
            approx_chars_per_token = 4
            truncated_body = body[: remain * approx_chars_per_token]
            truncated = header + truncated_body + "\n\n"
            pieces.append(truncated)
            used.append(doc)
            break
        pieces.append(chunk)
        used.append(doc)
        budget -= tokens
        if budget <= 0:
            break

    return "".join(pieces).strip(), used

def _openai_client() -> OpenAI:
    _require_api_key("chat completions")
    return OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)

def _chat_with_retries(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_msg: str,
    temperature: float = TEMPERATURE,
    max_retries: int = OPENAI_MAX_RETRIES,
) -> str:
    """OpenAI Chat Completions with simple retry/backoff on transient errors."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            _retry_sleep(attempt)
        except Exception:
            raise
    raise RuntimeError(f"OpenAI chat completion failed after {max_retries} attempts: {last_err}")


def generate(system_prompt: str, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Generate an answer in 2–4 plain-English sentences (no bullets/markdown).
    If LLM is disabled or not configured, returns a single-sentence fallback.
    """
    # Normalize inputs -> RetrievedDoc objects
    docs: List[RetrievedDoc] = [
        RetrievedDoc(
            id=str(d.get("id", "")),
            text=str(d.get("text", "")),
            meta=d.get("meta") or {},
            score=None,
        )
        for d in retrieved_docs
    ]

    # Dev fallback (no LLM)
    if not USE_LLM or not OPENAI_API_KEY:
        if not docs:
            return ("I don’t have enough indexed context to answer that yet. "
                    "Add docs to the knowledge base and re-ingest.")
        top_source = docs[0].meta.get("path") or docs[0].meta.get("source") or docs[0].id
        return (f"Based on the currently indexed material (e.g., {top_source}), "
                f"I need LLM synthesis enabled to provide a proper paragraph answer. "
                f"Please enable the LLM or add more context and re-try.")

    # Build a token-budgeted context
    context, used_docs = _build_context_snippets(docs, CHAT_MODEL, MAX_CONTEXT_TOKENS)

    # Instruction: plain sentences, no lists/markdown, no in-text citations
    user_msg = f"""Context:
{context}

Question:
{question}

Write the answer as 2–4 full sentences in plain English.
Do NOT use bullets, numbering, markdown, code blocks, or section headers.
Do NOT include citations inside the text; the API will attach citations separately.
If the context is insufficient, say you don't know in one sentence.
"""

    client = _openai_client()
    raw = _chat_with_retries(
        client=client,
        model=CHAT_MODEL,
        system_prompt=system_prompt,
        user_msg=user_msg,
        temperature=TEMPERATURE,
    )
    return _to_plain(raw)
