# Known issues — snapshot 2026-10-30

This package preserves the existing application behavior. These issues were documented rather than patched.

1. **Repeated questions can select a different fact card.** The anti-loop block in
   `HybridEngine.stream_reply()` switches to the second card when the top card repeats.
   A live test asking “ค่าเทอม AAI รหัส 67” twice in one session returned pre-67 tuition
   on the second request. Remove this switching rule before relying on repeat-answer consistency.
2. **The active hybrid path is semantic FAQ plus general LLM chat.** It returns
   `canned_answer` for accepted cards. Its general LLM prompt does not include retrieved
   passages. The older passage-based RAG path is bypassed when HybridEngine loads.
3. **Source coverage is incomplete.** 17 of 57 cards have no sources, including
   `edge_dorm` and `edge_phone`; the remainder are mostly general chat or edge cases.
   A source field alone does not establish that every statement was verified.
4. **Session storage is in memory.** Missing IDs share `default`; there is no expiry
   or per-session synchronization in the Flask application. Restarting loses history.
5. **Evaluation is not an accuracy score.** Existing evaluation tools record responses
   and timings. The prior 50-question web benchmark had 40 questions present in the
   precomputed query cache; cached latency does not represent all new queries.
6. **Startup requires a SQLite index and a separately supplied chat model.**
   Neither generated database nor model weights are included. See the README setup steps.
7. **The original UI uses external CDNs.** Local inference does not make the entire
   browser interface independent of the internet. Flask binds to all interfaces and
   has no application authentication in this snapshot.
