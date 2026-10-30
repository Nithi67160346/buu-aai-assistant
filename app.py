import os
import sys
import json
import time
from pathlib import Path
from flask import Flask, request, Response, render_template, jsonify, stream_with_context

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import RAG engine from rag_pi4_chat_v3_1
from rag_pi4_chat_v3_1 import Knowledge, Session
import rag_pi4 as rag
LLM_MODEL = getattr(rag, "LLM_MODEL", "qwen3:1.7b")

app = Flask(__name__, template_folder="templates", static_folder="static")

print("[Init] Initializing Knowledge Base...")
kb = Knowledge()
print(f"[Init] Knowledge Base loaded successfully with {len(kb.records)} passages.")

# Store active sessions in memory: session_id -> Session
sessions = {}

def get_session(session_id: str) -> Session:
    sid = session_id.strip() if session_id else "default"
    if sid not in sessions:
        sessions[sid] = Session(kb)
    return sessions[sid]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "online",
        "passages": len(kb.records),
        "roster": list(kb.roster()),
        "sqlite_path": str(kb.dbpath)
    })

@app.route("/api/chat/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default").strip()
    sessions[session_id] = Session(kb)
    return jsonify({"success": True, "session_id": session_id})

@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "default").strip()
    now_str = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now_str}] [User Question] (session={session_id}): {question}", flush=True)

    if not question:
        def empty_gen():
            yield f"data: {json.dumps({'type': 'token', 'content': 'กรุณาระบุคำถามครับ'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'timings': {}}, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(empty_gen()), mimetype="text/event-stream")

    sess = get_session(session_id)

    def generate():
        try:
            for event in sess.ask_stream(question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            err_msg = f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
            yield f"data: {json.dumps({'type': 'token', 'content': err_msg}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'timings': {}}, ensure_ascii=False)}\n\n"

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache, no-transform"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[Server] AAI BUU RAG Web UI listening on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
