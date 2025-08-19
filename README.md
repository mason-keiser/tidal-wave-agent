# Mason Agent — FastAPI RAG (No Docker)

A minimal Retrieval-Augmented Generation API that serves answers from your **Mason KB**.  
This version is **non‑Docker**, auto-loads a `.env`, and uses Windows-friendly paths.

## Quick start ###

1) **Unzip your seed pack** (from earlier) into `./knowledge/`
```
mason_agent_fastapi_nodocker/
  app/
  tests/
  knowledge/
    system_prompt.md
    golden_set.jsonl
    kb/
      playbooks/
      heuristics/
      faq/
```

2) **Create `.env`** (copy from `.env.example`) and set variables:
```
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
# Optional overrides (defaults below)
CHROMA_DIR=.chroma
KNOWLEDGE_DIR=knowledge/kb
SYSTEM_PROMPT_PATH=knowledge/system_prompt.md
GOLDEN_PATH=knowledge/golden_set.jsonl
```

3) **Install & run**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

4) **Test**
```powershell
# health
curl http://127.0.0.1:8000/health

# ask a question
curl -s -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How do I add a new questionnaire variant?\"}"
```

## Golden set (local check)
```powershell
python tests/eval_golden.py
```

## Notes
- Indexing happens once on startup if the collection is empty.
- Paths resolve relative to the **repo root** by default and can be overridden in `.env`.
- Chroma persists embeddings in `./.chroma` (create/delete safely).

— Generated 2025-08-16
