import json, time, sys, requests

BASE_URL = "http://127.0.0.1:8080"

def query_stream(session_id, question):
    url = f"{BASE_URL}/api/chat/stream"
    payload = {"question": question, "session_id": session_id}
    answer_parts = []
    sources = []
    timings = {}
    
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                try:
                    data = json.loads(data_str)
                    dtype = data.get("type")
                    if dtype == "token":
                        answer_parts.append(data.get("content", ""))
                    elif dtype == "sources":
                        sources = data.get("sources", [])
                    elif dtype == "done":
                        timings = data.get("timings", {})
                except Exception as e:
                    pass
    except Exception as e:
        answer_parts.append(f"[Error: {e}]")
    
    total_client_sec = round(time.time() - t0, 3)
    if "total" not in timings:
        timings["total"] = total_client_sec

    answer = "".join(answer_parts).strip()
    return answer, sources, timings

def main():
    print("[Eval] Loading eval/eval_questions.json ...")
    with open("eval/eval_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[Eval] Loaded {len(questions)} questions. Target server: {BASE_URL}")

    results = []
    t_start = time.time()

    for item in questions:
        qid = item["id"]
        group = item["group"]
        q = item["q"]
        seed = item.get("seed")
        sid = f"eval_{group}_{qid}_{int(time.time())}"

        # If follow-up, seed first
        if seed:
            query_stream(sid, seed)

        print(f"[{qid:02d}/40] [{group.upper():<8}] Q: {q} ... ", end="", flush=True)
        ans, srcs, timings = query_stream(sid, q)
        
        tot = timings.get("total", 0.0)
        src_cnt = len(srcs)
        ans_preview = (ans[:45] + "...") if len(ans) > 45 else ans
        ans_preview = ans_preview.replace("\n", " ")
        print(f"Done ({tot}s, {src_cnt} srcs) -> {ans_preview}")

        results.append({
            "id": qid,
            "group": group,
            "question": q,
            "seed": seed,
            "answer": ans,
            "timings": timings,
            "sources": srcs,
            "char_count": len(ans)
        })

    total_eval_time = round(time.time() - t_start, 2)
    output_path = sys.argv[1] if len(sys.argv) > 1 else "eval/results_hybrid.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"[Eval Complete] Saved {len(results)} results to {output_path}")
    print(f"Total time elapsed: {total_eval_time}s")
    print("="*60)

    # Summary by group
    groups = {}
    for r in results:
        g = r["group"]
        groups.setdefault(g, []).append(r)

    print(f"{'Group':<12} | {'Count':<5} | {'Avg Total (s)':<14} | {'Avg Search (s)':<15} | {'Avg Sources':<12}")
    print("-" * 68)
    for g, items in groups.items():
        avg_tot = sum(it["timings"].get("total", 0) for it in items) / len(items)
        avg_src = sum(it["timings"].get("search", 0) for it in items) / len(items)
        avg_cnt = sum(len(it["sources"]) for it in items) / len(items)
        print(f"{g:<12} | {len(items):<5} | {avg_tot:<14.2f} | {avg_src:<15.2f} | {avg_cnt:<12.1f}")
    print("="*60)

if __name__ == "__main__":
    main()
