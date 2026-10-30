"""Check the source package without importing the app or contacting Ollama."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ("app.py", "hybrid_engine.py", "rag_pi4.py", "rag_pi4_chat_v3_1.py",
                 "requirements.txt", "templates/index.html", "docs_aai/sources.json"):
        if not (ROOT / name).is_file():
            raise ValueError(f"Missing required file: {name}")
    python_files = (list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                    + list((ROOT / "eval").glob("*.py")))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    json_files = (list((ROOT / "data").glob("*.json")) + list((ROOT / "docs").glob("*.json"))
                  + [ROOT / "docs_aai/sources.json", ROOT / "eval/eval_questions.json"])
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8-sig"))

    cards = json.loads((ROOT / "data/fact_cards.json").read_text(encoding="utf-8"))
    embedded = json.loads((ROOT / "data/fact_cards_embedded.json").read_text(encoding="utf-8"))
    canonical = {card["id"]: card for card in cards}
    ids = [card["id"] for card in embedded]
    if len(canonical) != len(cards) or len(set(ids)) != len(ids) or set(ids) != set(canonical):
        raise ValueError("Duplicate IDs or mismatched canonical/embedded card IDs")
    dimensions = set()
    vectors = 0
    for card in embedded:
        original = canonical[card["id"]]
        for key in ("category", "title", "canned_answer", "context_text", "sources"):
            if card.get(key) != original.get(key):
                raise ValueError(f"Stale embedded field: {card['id']}/{key}")
        for vector in [card["card_embedding"]] + [q["embedding"] for q in card["query_embeddings"]]:
            if not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise ValueError(f"Invalid embedding: {card['id']}")
            if not any(vector):
                raise ValueError(f"Zero embedding: {card['id']}")
            dimensions.add(len(vector))
            vectors += 1
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dimensions}")

    manifest = json.loads((ROOT / "docs/snapshot-sha256.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    print(f"OK: {len(python_files)} Python files; {len(cards)} cards; {vectors} vectors; dimensions={dimensions}")
    print("Source snapshot checksums match. No network requests were made.")
    print("SQLite index: " + ("present" if (ROOT / "aai_rag.sqlite3").is_file() else "not generated; run README setup before app.py"))


if __name__ == "__main__":
    main()
