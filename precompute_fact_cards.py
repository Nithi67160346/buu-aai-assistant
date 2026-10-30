# -*- coding: utf-8 -*-
import json
import time
import requests
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
MODEL = "embeddinggemma:300m-qat-q4_0"

def get_embedding(text):
    for _ in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text}, timeout=15)
            if resp.status_code == 200:
                return resp.json()["embedding"]
        except Exception as e:
            time.sleep(0.5)
    raise RuntimeError(f"Failed to get embedding for: {text[:40]}")

def main():
    cards_path = Path("data/fact_cards.json")
    out_path = Path("data/fact_cards_embedded.json")
    with open(cards_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    print(f"Precomputing embeddings for {len(cards)} cards using {MODEL}...")
    t0 = time.time()
    embedded_cards = []

    for idx, card in enumerate(cards, 1):
        c_text = f"{card['title']}. {card['context_text']}"
        c_emb = get_embedding(c_text)
        q_embs = []
        for q in card.get("sample_queries", []):
            q_emb = get_embedding(q)
            q_embs.append({"query": q, "embedding": q_emb})

        embedded_card = {
            "id": card["id"],
            "category": card.get("category", "general"),
            "title": card["title"],
            "canned_answer": card["canned_answer"],
            "context_text": card["context_text"],
            "sources": card.get("sources", []),
            "card_embedding": c_emb,
            "query_embeddings": q_embs,
        }
        embedded_cards.append(embedded_card)
        if idx % 5 == 0 or idx == len(cards):
            print(f"[{idx}/{len(cards)}] Processed in {time.time()-t0:.1f}s")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embedded_cards, f, ensure_ascii=False)

    total_time = time.time() - t0
    print(f"Finished in {total_time:.1f}s! Saved {len(embedded_cards)} embedded cards to {out_path}")

if __name__ == "__main__":
    main()
