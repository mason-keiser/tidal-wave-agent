import os, glob

def load_kb_paths(knowledge_dir: str):
    patterns = ["**/*.yaml", "**/*.yml", "**/*.jsonl", "**/*.md"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(knowledge_dir, pat), recursive=True))
    return [p for p in paths if os.path.isfile(p)]

def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
