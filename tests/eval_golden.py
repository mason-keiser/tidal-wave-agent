import os, pathlib, json
from dotenv import load_dotenv
from app.kb_loader import read_text
from app.rag import get_chroma, get_or_create_collection, retrieve, generate

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / '.env', override=False)

KNOWLEDGE_DIR = os.getenv('KNOWLEDGE_DIR', str(ROOT / 'knowledge' / 'kb'))
SYSTEM_PROMPT_PATH = os.getenv('SYSTEM_PROMPT_PATH', str(ROOT / 'knowledge' / 'system_prompt.md'))
GOLDEN_PATH = os.getenv('GOLDEN_PATH', str(ROOT / 'knowledge' / 'golden_set.jsonl'))

def main():
    chroma = get_chroma()
    collection = get_or_create_collection(chroma)
    system_prompt = read_text(SYSTEM_PROMPT_PATH)

    total = 0
    must_ok = 0

    with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            q = obj['question']
            must = set(obj.get('must_cite', []))

            hits = retrieve(collection, q, k=4)
            _ = generate(system_prompt, q, hits)

            retrieved = {h['meta'].get('path','') for h in hits}
            ok = all(any(req in p for p in retrieved) for req in must)
            print(f"Q{total}: {q}\n - must-cite satisfied: {ok}\n - retrieved: {retrieved}\n")
            if ok:
                must_ok += 1

    print(f"Must-cite satisfied: {must_ok}/{total}")
    return 0 if must_ok == total else 1

if __name__ == '__main__':
    raise SystemExit(main())
