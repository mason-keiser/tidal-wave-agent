import os, pathlib
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from app.kb_loader import load_kb_paths, read_text
from app.rag import get_chroma, get_or_create_collection, retrieve, generate

# Load .env from repo root
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / '.env', override=False)
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# Defaults (relative to repo root) can be overridden by env
KNOWLEDGE_DIR = os.getenv('KNOWLEDGE_DIR', str(ROOT / 'knowledge' / 'kb'))
SYSTEM_PROMPT_PATH = os.getenv('SYSTEM_PROMPT_PATH', str(ROOT / 'knowledge' / 'system_prompt.md'))
GOLDEN_PATH = os.getenv('GOLDEN_PATH', str(ROOT / 'knowledge' / 'golden_set.jsonl'))

app = FastAPI(title="Mason Agent API", version="0.2.0" )

@app.on_event("startup")
def startup_index():
    chroma = get_chroma()
    collection = get_or_create_collection(chroma)
    try:
        if collection.count() > 0:
            return
    except Exception:
        pass

    paths = load_kb_paths(KNOWLEDGE_DIR)
    if not paths:
        print(f"[warn] No KB files found under {KNOWLEDGE_DIR}. Did you unzip the seed pack?")
        return

    ids, docs, metas = [], [], []
    for p in paths:
        ids.append(p)
        docs.append(read_text(p))
        metas.append({"path": p})
    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"[ok] Indexed {len(ids)} docs into Chroma.")

class AskPayload(BaseModel):
    question: str
    k: int | None = 4

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(payload: AskPayload):
    chroma = get_chroma()
    collection = get_or_create_collection(chroma)
    system_prompt = read_text(SYSTEM_PROMPT_PATH)
    hits = retrieve(collection, payload.question, k=payload.k or 4)
    answer = generate(system_prompt, payload.question, hits)
    return {
        "answer": answer,
        "sources": [{"path": h.get("meta", {}).get("path", "")} for h in hits]
    }
