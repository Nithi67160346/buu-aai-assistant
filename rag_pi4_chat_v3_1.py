#!/usr/bin/env python3
"""AAI Conversational RAG v3.1: fast, natural Thai answers, smart hybrid retrieval & memory rewrite.
Place next to rag_pi4.py and the existing SQLite index.
"""
import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
try:
    import readline
except ImportError:
    readline = None
from collections import Counter
from pathlib import Path
import requests
import rag_pi4 as rag
try:
    from hybrid_engine import HybridEngine
    HYBRID_ENGINE = HybridEngine()
except Exception as _e:
    print(f'[Warning] Failed to load HybridEngine: {_e}')
    HYBRID_ENGINE = None

OLLAMA_URL = getattr(rag, 'OLLAMA_URL', 'http://127.0.0.1:11434/api')
LLM_MODEL = getattr(rag, 'LLM_MODEL', 'qwen3:1.7b')

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs_aai"
KEEP_ALIVE = "15m"
TOP_K = 3
MAX_CONTEXT_PER_CHUNK = 700
MAX_TOKENS = 260

LABELS = {
    "aai_overview": ("curriculum", None, None),
    "aai_culture": ("curriculum", None, None),
    "collaborate": ("partnership", None, None),
    "color-faculty": ("color", None, None),
    "coop-80-2568": ("coop", 2568, None),
    "coop-detail": ("coop", None, None),
    "future": ("career", None, None),
    "module-qualification-exam": ("module", None, None),
    "tcas_aai": ("admission", None, None),
    "อบรมออนไลน์ ใน buu mooc สหกิจศึกษา69": ("coop", 2569, None),
    "aai-program-63": ("curriculum", 2563, None),
    "aai-program-64": ("curriculum", 2564, None),
    "aai-program-67": ("curriculum", 2567, None),
    "aai-study-plan-63": ("curriculum", 2563, None),
    "aai-study-plan-64": ("curriculum", 2564, None),
    "cv-form_ponlawat": ("professor", None, "ponlawat"),
    "cv_supawadee_2020": ("professor", None, "supawadee"),
    "profile-krisana-v2": ("professor", None, "krisana"),
}

NAME_ALIASES = {
    "krisana": ["กฤษณะ", "ชินสาร", "krisana", "chinnasarn"],
    "nilrat": ["นิลรัตน์", "ก้านหยั่นทอง", "nilrat", "kanyunthong"],
    "ponlawat": ["พลวัต", "ช่อผูก", "ponlawat", "chophuk", "ประธานหลักสูตร"],
    "prawit": ["ประวิทย์", "บุญมี", "prawit", "boonmee"],
    "supawadee": ["สุภาวดี", "ศรีคำดี", "supawadee", "srikamdee"],
    "wyookwan": ["วัชรพงศ์", "อยู่ขวัญ", "wyookwan", "yookwan", "watcharaphong"],
}

PROF_DISPLAY_NAMES = {
    "krisana": "รศ.ดร.กฤษณะ ชินสาร",
    "nilrat": "อาจารย์นิลรัตน์ ก้านหยั่นทอง",
    "ponlawat": "ผศ.ดร.พลวัต ช่อผูก",
    "prawit": "อาจารย์ประวิทย์ บุญมี",
    "supawadee": "ผศ.ดร.สุภาวดี ศรีคำดี",
    "wyookwan": "ดร.วัชรพงศ์ อยู่ขวัญ",
}

PROFESSOR_FACTS = {
    "krisana": {
        "name": "รศ.ดร.กฤษณะ ชินสาร (รองศาสตราจารย์ สาขาวิชาวิทยาการคอมพิวเตอร์)",
        "office": "IF-809 / FI-8R06 KST Lab",
        "email": "krisana@informatics.buu.ac.th",
        "degrees": [
            "ปริญญาเอก: วท.ด. (วิทยาการคอมพิวเตอร์) จุฬาลงกรณ์มหาวิทยาลัย พ.ศ. 2547",
            "ปริญญาโท: วท.ม. (วิทยาการคอมพิวเตอร์และเทคโนโลยีสารสนเทศ) สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง พ.ศ. 2540",
            "ปริญญาตรี: วท.บ. (สถิติ) มหาวิทยาลัยศรีนครินทรวิโรฒ มหาสารคาม พ.ศ. 2536",
        ],
        "expertise": "Data Structure & Algorithms, Computer Graphics, Digital Image Processing, Neural Network, NLP, Machine Learning",
    },
    "ponlawat": {
        "name": "ผศ.ดร.พลวัต ช่อผูก (ประธานหลักสูตร AAI)",
        "office": "IF-913F",
        "email": "ponlawat.ch@buu.ac.th",
        "degrees": [
            "ปริญญาเอก: วศ.ด. (เทคโนโลยีวิศวกรรมไฟฟ้าและสารสนเทศ) มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี (มจธ.) พ.ศ. 2565",
            "ปริญญาโท: วศ.ม. (วิศวกรรมชีวการแพทย์) สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง พ.ศ. 2559",
            "ปริญญาตรี: วศ.บ. (วิศวกรรมอิเล็กทรอนิกส์) สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง พ.ศ. 2557",
        ],
        "expertise": "Digital Image Processing, Machine Learning, Electronic Circuit Design, IoT",
    },
    "supawadee": {
        "name": "ผศ.ดร.สุภาวดี ศรีคำดี",
        "office": "IF-911C",
        "email": "srikamdee@buu.ac.th",
        "degrees": [
            "ปริญญาเอก: วศ.ด. (วิศวกรรมคอมพิวเตอร์) จุฬาลงกรณ์มหาวิทยาลัย พ.ศ. 2563",
            "ปริญญาโท: วท.ม. (เทคโนโลยีสารสนเทศ) มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ พ.ศ. 2549",
            "ปริญญาตรี: บธ.บ. (ระบบสารสนเทศ) มหาวิทยาลัยเทคโนโลยีราชมงคลกรุงเทพ พ.ศ. 2545",
        ],
        "expertise": "Software Engineering, Information Systems, AI & Data Analytics",
    },
    "prawit": {
        "name": "อาจารย์ประวิทย์ บุญมี",
        "office": "IF-912C",
        "email": "prawit@buu.ac.th",
        "degrees": [
            "ปริญญาโท: M.S. (Computer Science) University of North Texas, USA พ.ศ. 2543",
            "ปริญญาตรี: B.S. (Mathematics) University of Wisconsin-Madison, USA พ.ศ. 2541",
        ],
        "expertise": "Computer Science, Mathematics, Programming",
    },
    "wyookwan": {
        "name": "ดร.วัชรพงศ์ อยู่ขวัญ",
        "office": "IF-909E",
        "email": "wyookwan@buu.ac.th",
        "degrees": [
            "ปริญญาเอก: วศ.ด. (วิศวกรรมไฟฟ้า) มหาวิทยาลัยเทคโนโลยีสุรนารี พ.ศ. 2564",
            "ปริญญาตรี: วศ.บ. (วิศวกรรมโทรคมนาคม) มหาวิทยาลัยเทคโนโลยีสุรนารี พ.ศ. 2556",
        ],
        "expertise": "Telecommunications, Electrical Engineering, Smart Systems",
    },
    "nilrat": {
        "name": "อาจารย์นิลรัตน์ ก้านหยั่นทอง",
        "office": "IF-912F",
        "email": "nilrat.ka@informatics.buu.ac.th",
        "degrees": [
            "สังกัด: สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ ม.บูรพา",
        ],
        "expertise": "Image Processing, Medical Image Processing, Machine Learning, Computer Vision",
    },
}


TOPIC_WORDS = {
    "meta": ["ทำอะไรได้บ้าง", "คุณคือใคร", "คุณรู้เรื่อง", "รู้เรื่องเกี่ยวกับคณะ", "ถามอะไรได้บ้าง", "ช่วยอะไรได้บ้าง", "สวัสดี"],
    "professor": ["อาจารย์", "คณาจารย์", "ผู้สอน", "บุคลากร", "ประวัติการศึกษา", "ดร.", "ผศ.", "รศ.", "ห้องทำงาน", "อีเมล", "email", "จบอะไร", "จบที่ไหน", "แต่ละคน", "แต่ละท่าน", "ประวัติอาจารย์", "ทุกคน", "อาจารย์แต่ละ"],
    "coop": ["สหกิจ", "ฝึกงาน", "ชั่วโมง", "mooc", "coop", "เตรียมความพร้อม", "ชั่วโมงวิชาการ", "ใบเซอร์", "ใบประกาศ"],
    "admission": ["tcas", "รับสมัคร", "สมัครเรียน", "portfolio", "ยื่นพอร์ต", "tgat", "tpat", "เกณฑ์", "รอบ 1", "รอบ 3", "อยากเข้า", "เข้าคณะ", "เข้าสาขา", "เข้าเรียน", "เข้าศึกษา", "สอบเข้า", "สมัครเข้า"],
    "career": ["อาชีพ", "ทำงานอะไร", "จบไปทำงาน", "จบแล้วทำงาน", "อนาคต", "สายงาน"],
    "partnership": ["ความร่วมมือ", "mou", "สถานประกอบการ", "มิตซูบิชิ", "depa"],
    "color": ["สีประจำคณะ", "สีคณะ"],
    "module": ["โมดูล", "module", "qualification", "สอบวัด"],
    "curriculum": ["หลักสูตร", "หน่วยกิต", "รายวิชา", "โครงสร้าง", "ปริญญา", "เรียนกี่ปี", "ค่าเทอม", "ค่าธรรมเนียม", "เรียนเกี่ยวกับ", "วิชาแกน", "วิชาเอก", "วิชาเลือก", "ศึกษาทั่วไป", "aai"],
}

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
ORDINALS = {
    "หนึ่ง": 1, "แรก": 1, "สอง": 2, "สาม": 3, "สี่": 4,
    "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10,
}

GENERAL_SYSTEM_PROMPT = (
    "คุณคือ AI ผู้ช่วยอัจฉริยะประจำสาขาวิชา AAI คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา "
    "จงตอบคำถามของผู้ใช้เป็นภาษาไทยอย่างถูกต้อง สุภาพ เป็นธรรมชาติ และมีสาระ "
    "หากเป็นคำถามทั่วไป (เช่น วัน, สี, ความรู้ทั่วไป, ชายหาดบางแสน, ชลบุรี, อาหาร, การคำนวณ) ให้ตอบตามความเป็นจริงอย่างเป็นมิตร "
    "ข้อห้ามเด็ดขาด: ห้ามแต่งเติมข้อมูล ห้ามเดาคำศัพท์แปลกๆ ห้ามตอบเรื่องที่ไม่เป็นความจริง เช่น สูตรอาหารแปลกๆ หากคำถามใดไม่มีข้อมูลหรือไม่แน่ใจ ให้ตอบว่า 'ขออภัยครับ ข้อมูลส่วนนี้อยู่นอกเหนือฐานข้อมูลของผม' เท่านั้น"
)

SYSTEM_PROMPT = (
    "คุณคือ AI ผู้ช่วยตอบคำถามประจำสาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา "
    "จงตอบคำถามเป็นภาษาไทยอย่างสุภาพ ถูกต้อง ชัดเจน และเป็นธรรมชาติ โดยสรุปข้อมูลจากเอกสารอ้างอิงที่กำหนดให้ ห้ามแต่งเติมข้อมูล ห้ามตอบนอกเรื่อง "
    "และไม่ต้องขึ้นต้นด้วย 'คำตอบ:' หรือ 'มีหลักสูตรอย่างนี้:' "
    "ข้อห้ามเด็ดขาด: ห้ามแต่งเติมข้อมูล ห้ามเดาคำศัพท์แปลกๆ ห้ามตอบเรื่องที่ไม่เป็นความจริง เช่น สูตรอาหารแปลกๆ หากคำถามใดไม่มีข้อมูลหรือไม่แน่ใจ ให้ตอบว่า 'ขออภัยครับ ข้อมูลส่วนนี้อยู่นอกเหนือฐานข้อมูลของผม' เท่านั้น"
)


THAI_TYPO_MAP = {
    "คุณื": "คุณ",
    "คณุ": "คุณ",
    "ลายระเอียด": "รายละเอียด",
    "ลายละเอียด": "รายละเอียด",
    "รายระเอียด": "รายละเอียด",
    "สหกิต": "สหกิจ",
    "ชัวโมง": "ชั่วโมง",
    "ขั่วโมง": "ชั่วโมง",
    "อาจาร ": "อาจารย์ ",
}


def normalize_thai_query(q):
    if not q:
        return ""
    # Strip any UTF-8 replacement characters from partial backspaces
    q = q.replace("\ufffd", "")
    # Collapse duplicate consecutive Thai combining vowels / tone marks (e.g. สหกิิจ -> สหกิจ, ให้้ -> ให้)
    q = re.sub(r"([\u0e31\u0e34-\u0e3a\u0e47-\u0e4e])\1+", r"\1", q)
    for wrong, right in THAI_TYPO_MAP.items():
        if wrong in q:
            q = q.replace(wrong, right)
    return re.sub(r"\s+", " ", q).strip()


def safe_input(prompt="\nคุณ: "):
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        raw = sys.stdin.buffer.readline()
        if not raw:
            raise EOFError
        text = raw.decode("utf-8", errors="ignore")
        return normalize_thai_query(text)
    except UnicodeDecodeError:
        return ""


def stem(source):
    s = Path(str(source)).name.lower()
    for ext in (".pdf", ".txt", ".md"):
        if s.endswith(ext):
            s = s[:-len(ext)]
    return s


def metadata(source):
    low = str(source).lower()
    st = stem(source)
    if st in LABELS:
        return LABELS[st]
    if st.startswith("professor_aai_"):
        person = st.replace("professor_aai_", "").strip()
        return ("professor", None, person)
    if "professor/" in low:
        for k in NAME_ALIASES:
            if k in low:
                return ("professor", None, k)
        return ("professor", None, None)
    return ("general", extract_year(low), None)


def extract_year(q):
    q_norm = q.translate(THAI_DIGITS)
    m = re.search(r"\b(25\d{2})\b", q_norm)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(?:ปี|รหัส|หลักสูตร|ศึกษา)\s*(6[0-9])\b", q_norm)
    if m2:
        return 2500 + int(m2.group(1))
    return None


def read_summary(path):
    if path.suffix.lower() == ".pdf" or path.name.startswith(".") or path.name == "sources.json":
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def segment(text, chunk=650, overlap=80):
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    out = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= chunk:
            cur += "\n" + p
        else:
            out.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + "\n" + p).strip() if tail else p
    if cur:
        out.append(cur)
    return out


def lexical_score(q, text):
    q_low = q.lower().translate(THAI_DIGITS)
    t_low = text.lower().translate(THAI_DIGITS)
    tokens = [w for w in re.findall(r"[a-z0-9ก-๙]+", q_low) if len(w) >= 2]
    if not tokens:
        return 0.0
    hits = sum(1 for w in tokens if w in t_low)
    # Also check 3-character Thai ngrams for unspaced Thai queries
    thai_chars = re.sub(r"[^ก-๙]", "", q_low)
    if len(thai_chars) >= 3:
        ngrams = [thai_chars[i:i+3] for i in range(0, len(thai_chars) - 2, 2)]
        nhits = sum(1 for g in ngrams if g in t_low)
        return 0.6 * (hits / len(tokens)) + 0.4 * (nhits / max(1, len(ngrams)))
    return hits / len(tokens)


def normalize_for_dedup(text):
    return re.sub(r"\s+", " ", text.strip().lower())[:160]


class Knowledge:
    def __init__(self):
        local_db = ROOT / "aai_rag.sqlite3"
        self.dbpath = local_db if local_db.is_file() else Path(rag.DB_PATH)
        self.records = []
        self.vector_records = []
        self.vector_values = []
        self.matrix = None
        self.embed_cache = {}
        self.fingerprints = None
        self.reload()

    def _fingerprint(self):
        files = [self.dbpath, Path(str(self.dbpath) + "-wal")]
        return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files)

    def maybe_reload(self):
        if self._fingerprint() != self.fingerprints:
            self.reload()

    def reload(self):
        if not self.dbpath.is_file():
            raise RuntimeError(f"ไม่พบฐานข้อมูล {self.dbpath}")
        t0 = time.perf_counter()
        self.records = []
        self.vector_records = []
        self.vector_values = []
        self.matrix = None

        aliases = {}
        conn = sqlite3.connect(f"file:{self.dbpath}?mode=ro", uri=True)
        try:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='indexed_files'").fetchone()
            if table:
                for filepath, source in conn.execute("SELECT file_path, source FROM indexed_files"):
                    aliases.setdefault(source, []).append(filepath)
            chunks = conn.execute("SELECT id, source, page, content, embedding FROM chunks ORDER BY id ASC").fetchall()
        finally:
            conn.close()

        dim = None
        indexed_sources = set()
        for cid, source, page, content, embedding in chunks:
            try:
                vec = json.loads(embedding)
                if not isinstance(vec, list) or not vec:
                    continue
                if dim is None:
                    dim = len(vec)
                if len(vec) != dim:
                    continue
            except (TypeError, ValueError):
                continue
            possible = [source] + aliases.get(source, [])
            meta = next((metadata(x) for x in possible if metadata(x)[0] != "general"), metadata(source))
            label = aliases.get(source, [source])[0]
            indexed_sources.add(stem(label))
            record = {
                "id": cid,
                "source": label,
                "page": page,
                "text": content.strip(),
                "meta": meta,
                "vector": True,
            }
            self.records.append(record)
            self.vector_records.append(record)
            self.vector_values.append(vec)

        # Fallback for any new text files not yet in SQLite
        if DOCS.is_dir():
            for path in DOCS.rglob("*"):
                if not path.is_file():
                    continue
                if any(x in path.parts for x in ("__pycache__", ".aai_ocr_cache")):
                    continue
                if stem(path.name) in indexed_sources:
                    continue
                try:
                    body = read_summary(path)
                except OSError:
                    continue
                if not body:
                    continue
                label = path.relative_to(DOCS).as_posix()
                for piece in segment(body):
                    self.records.append({
                        "id": None,
                        "source": label,
                        "page": None,
                        "text": piece,
                        "meta": metadata(label),
                        "vector": False,
                    })

        if np is not None and self.vector_values:
            values = np.asarray(self.vector_values, dtype=np.float32)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = values / norms

        self.fingerprints = self._fingerprint()
        dt = time.perf_counter() - t0
        vec_n = len(self.vector_records)
        txt_n = len(self.records) - vec_n
        print(f"[Ready] {len(self.records)} passages | SQLite: {vec_n} | text: {txt_n} | {dt:.1f}s")

    def roster(self):
        people = {}
        for record in self.records:
            cat, _, person = record["meta"]
            if cat == "professor" and person:
                people.setdefault(person, set()).add(record["source"])
        return dict(sorted(people.items()))

    def find_records_by_stem(self, target_stems):
        if isinstance(target_stems, str):
            target_stems = (target_stems,)
        target_set = {s.lower() for s in target_stems}
        return [r for r in self.records if stem(r["source"]) in target_set]

    def find_prof_header(self, person):
        target = f"professor_aai_{person}"
        for r in self.records:
            if stem(r["source"]) == target:
                return r
        return None

    def search(self, question, topic=None, year=None, person=None):
        if not self.records:
            return []
        query = question
        qvec = None
        if self.vector_records:
            if query not in self.embed_cache:
                self.embed_cache[query] = rag.embed_inputs(
                    ["task: search result | query: " + query], keep_alive=KEEP_ALIVE
                )[0]
                if len(self.embed_cache) > 40:
                    self.embed_cache = {query: self.embed_cache[query]}
            qvec = self.embed_cache[query]

        if self.matrix is not None and qvec is not None:
            q = np.asarray(qvec, dtype=np.float32)
            qnorm = np.linalg.norm(q) or 1.0
            similarities = (self.matrix @ (q / qnorm)).tolist()
        elif qvec is not None:
            similarities = [rag.cosine_similarity(qvec, v) for v in self.vector_values]
        else:
            similarities = []

        q_low = question.lower()
        ranked = []
        vector_idx = 0
        for rec in self.records:
            category, doc_year, doc_person = rec["meta"]
            sim = similarities[vector_idx] if rec["vector"] else 0.0
            if rec["vector"]:
                vector_idx += 1

            # Safety filters for person and year
            if person and doc_person and doc_person != person:
                continue
            if year and doc_year and doc_year != year:
                continue
            if person and category == "professor" and doc_person is None:
                continue
            if person and category not in ("professor", "curriculum"):
                continue
            if person and category == "curriculum" and doc_person != person:
                # Only allow curriculum chunks if they actually mention the professor's Thai name
                aliases = NAME_ALIASES.get(person, [])
                if not any(a in rec["text"].lower() for a in aliases):
                    continue

            lex = lexical_score(question, rec["text"])
            score = (0.72 * sim + 0.28 * lex) if rec["vector"] else (0.65 * lex)

            st = stem(rec["source"])
            if topic and category == topic:
                score += 0.14
            if person and doc_person == person:
                score += 0.22
                if st == f"professor_aai_{person}" and ("ห้องทำงาน:" in rec["text"] or "ประวัติการศึกษา:" in rec["text"]):
                    score += 0.25
            if year and doc_year == year:
                score += 0.15
            elif not year and doc_year == 2567 and category == "curriculum":
                score += 0.05

            # Boost summary documents so high-level facts surface cleanly
            if st in ("aai_overview", "aai_culture", "future", "tcas_aai", "coop-detail", "collaborate", "color-faculty"):
                score += 0.12

            if score > 0.05:
                ranked.append((score, rec))

        ranked.sort(key=lambda x: x[0], reverse=True)

        # Smart Hybrid Pinned Injection + Deduplication
        pinned = []
        if person:
            prof_hdr = self.find_prof_header(person)
            if prof_hdr:
                pinned.append(prof_hdr)
        elif topic == "career" or any(w in q_low for w in ("อาชีพ", "ทำงานอะไร", "จบไป", "จบแล้ว", "สายงาน")):
            pinned.extend(self.find_records_by_stem("future")[:1])
        elif topic == "admission" or any(w in q_low for w in ("tcas", "รับสมัคร", "สมัครเรียน", "พอร์ต", "tgat", "tpat", "รอบ 1", "รอบ 3")):
            pinned.extend(self.find_records_by_stem("tcas_aai")[:1])
        elif topic == "coop" or any(w in q_low for w in ("สหกิจ", "ฝึกงาน", "ชั่วโมง", "mooc", "ใบเซอร์", "ใบประกาศ")):
            coop_detail_recs = self.find_records_by_stem("coop-detail")
            coop_80_recs = self.find_records_by_stem("coop-80-2568")
            mooc_recs = self.find_records_by_stem("อบรมออนไลน์ ใน buu mooc สหกิจศึกษา69")
            if any(w in q_low for w in ("mooc", "วิชาออนไลน์", "อบรมออนไลน์", "คอร์สออนไลน์", "วิชาอะไรบ้าง")) and "คุณสมบัติ" not in q_low:
                pinned.extend(mooc_recs[:1])
                pinned.extend(coop_detail_recs[:1])
            elif any(w in q_low for w in ("ใบเซอร์", "ใบประกาศ", "certificate", "องค์กร", "หน่วยงาน", "tpqi", "depa", "แทนชั่วโมง")):
                if len(coop_80_recs) >= 2:
                    pinned.append(coop_80_recs[1])
                pinned.extend(coop_80_recs[:1])
            elif any(w in q_low for w in ("คุณสมบัติ", "เงื่อนไข", "ประกาศ", "ปี 68", "ปี68", "2568", "เกรด", "gpax")):
                pinned.extend(coop_80_recs[:1])
                pinned.extend(coop_detail_recs[:1])
            elif any(w in q_low for w in ("ชั่วโมง", "เก็บยังไง", "กี่ชั่วโมง", "เตรียมความพร้อม", "วิชาการ")):
                pinned.extend(coop_detail_recs[:1])
                pinned.extend(coop_80_recs[:1])
            else:
                pinned.extend(coop_detail_recs[:1])
                pinned.extend(coop_80_recs[:1])
        elif topic == "module" or any(w in q_low for w in ("โมดูล", "module", "สอบวัด")):
            pinned.extend(self.find_records_by_stem("module-qualification-exam")[:2])
        elif topic == "partnership" or any(w in q_low for w in ("ความร่วมมือ", "mou", "สถานประกอบการ")):
            pinned.extend(self.find_records_by_stem("collaborate")[:2])
        elif topic == "color" or "สีประจำคณะ" in q_low or "สีคณะ" in q_low:
            pinned.extend(self.find_records_by_stem("color-faculty")[:1])
        elif topic == "curriculum" or any(w in q_low for w in ("คืออะไร", "อยู่ที่ไหน", "เรียนเกี่ยวกับ", "กี่หน่วยกิต", "หน่วยกิต", "ค่าเทอม", "ค่าธรรมเนียม", "เรียนกี่ปี", "ปริญญา")):
            overview_recs = self.find_records_by_stem("aai_overview")
            if any(w in q_low for w in ("เรียนเกี่ยวกับ", "กี่หน่วยกิต", "หน่วยกิต", "ค่าเทอม", "ค่าธรรมเนียม")) and len(overview_recs) >= 2:
                pinned.append(overview_recs[1])
                pinned.append(overview_recs[0])
            elif overview_recs:
                pinned.append(overview_recs[0])
                if len(overview_recs) >= 2:
                    pinned.append(overview_recs[1])

        results = []
        seen_sigs = set()
        for rec in pinned + [r for _, r in ranked]:
            sig = normalize_for_dedup(rec["text"])
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            results.append(rec)
            if len(results) >= TOP_K * 2:
                break
        return results


class Session:
    def __init__(self, knowledge):
        self.kb = knowledge
        self.last_topic = None
        self.last_person = None
        self.last_year = None
        self.last_question = None
        self.last_roster = list(self.kb.roster())
        self.history = []

    def get_person(self, q):
        lowered = q.lower().translate(THAI_DIGITS)
        hits = []
        for key in self.kb.roster():
            aliases = [key] + NAME_ALIASES.get(key, [])
            if any(alias in lowered for alias in aliases):
                hits.append(key)
        if len(hits) == 1:
            return hits[0], False
        if len(hits) > 1:
            return None, True

        n = None
        if "คนแรก" in lowered or "ท่านแรก" in lowered:
            n = 1
        elif "คนสุดท้าย" in lowered or "ท่านสุดท้าย" in lowered:
            n = len(self.last_roster)
        else:
            m = re.search(
                r"(?:คน|ท่าน|ลำดับ)\s*ที่\s*(\d+|หนึ่ง|แรก|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)",
                lowered,
            )
            if m:
                tok = m.group(1)
                n = int(tok) if tok.isdigit() else ORDINALS.get(tok)
        if n is not None:
            if not (1 <= n <= len(self.last_roster)):
                return None, True
            return self.last_roster[n - 1], False
        return None, False

    def infer_topic(self, q):
        qlow = q.lower()
        if any(alias in qlow for names in NAME_ALIASES.values() for alias in names):
            return "professor"
        if re.search(r"(?:คนแรก|ท่านแรก|คนสุดท้าย|ท่านสุดท้าย|(?:คน|ท่าน|ลำดับ)\s*ที่\s*(?:\d+|หนึ่ง|สอง|สาม|สี่|ห้า|หก))", qlow.translate(THAI_DIGITS)):
            return "professor"
        for category, words in TOPIC_WORDS.items():
            if any(word in qlow for word in words):
                return category
        if self.last_topic and (
            qlow.startswith("แล้ว")
            or any(w in qlow for w in ("เขา", "ท่าน", "คนนี้", "คนแรก", "คนที่", "ชื่ออะไรบ้าง", "เรื่องนี้", "เท่าไร", "เท่าไหร่", "ล่ะ", "ที่ไหน"))
        ):
            return self.last_topic
        return "general"

    def rewrite_followup(self, question, topic, person, year):
        q = question.strip()
        qlow = q.lower().translate(THAI_DIGITS)

        if person:
            prof_name = PROF_DISPLAY_NAMES.get(person, person)
            # Replace ordinal or pronoun references with explicit professor name
            cleaned = re.sub(
                r"(?:อาจารย์|ท่าน|ลำดับ)?\s*(?:คนแรก|ท่านแรก|คนสุดท้าย|ท่านสุดท้าย|(?:คน|ท่าน|ลำดับ)\s*ที่\s*(?:\d+|หนึ่ง|สอง|สาม|สี่|ห้า|หก))",
                prof_name,
                q,
            )
            cleaned = re.sub(r"^(?:แล้ว|และ)\s*", "", cleaned).strip()
            cleaned = re.sub(r"\s*(?:ล่ะ|ละ|ครับ|ค่ะ|คะ)\s*", " ", cleaned).strip()
            cleaned = re.sub(r"\b(?:เขา|ท่านนี้|คนนี้|แก)\b", prof_name, cleaned)
            if prof_name not in cleaned and not any(a in cleaned.lower() for a in NAME_ALIASES.get(person, [])):
                cleaned = f"{prof_name} {cleaned}"
            # If the user asked a short follow-up like "แล้วคนที่ 3 ล่ะ" without specifying what attribute, inherit intent from last question
            has_specific_ask = any(
                w in qlow for w in ("จบ", "วุฒิ", "การศึกษา", "ห้อง", "ติดต่อ", "อีเมล", "email", "วิจัย", "ผลงาน", "ความสนใจ", "เชี่ยวชาญ", "ตำแหน่ง")
            )
            if not has_specific_ask and self.last_question:
                prev_low = self.last_question.lower()
                extra_asks = []
                if any(w in prev_low for w in ("จบ", "วุฒิ", "การศึกษา")):
                    extra_asks.append("จบการศึกษาอะไร")
                if any(w in prev_low for w in ("ห้อง", "อยู่ที่ไหน")):
                    extra_asks.append("ห้องทำงานอยู่ที่ไหน")
                if any(w in prev_low for w in ("อีเมล", "email", "ติดต่อ")):
                    extra_asks.append("อีเมลอะไร")
                if extra_asks:
                    cleaned = f"{cleaned} ({' '.join(extra_asks)})"
                else:
                    cleaned = f"{cleaned} มีประวัติการศึกษา ห้องทำงาน และความเชี่ยวชาญอย่างไร"
            return cleaned

        if year and str(year) not in qlow:
            return f"{q} (หลักสูตรปี พ.ศ. {year})"

        if (qlow.startswith("แล้ว") or "ล่ะ" in qlow) and self.last_question:
            return f"{q} (ต่อเนื่องจากคำถาม: {self.last_question})"

        return q

    def ask_stream(self, question):
        began = time.perf_counter()
        self.kb.maybe_reload()
        question = normalize_thai_query(question)
        if not question:
            return
        if HYBRID_ENGINE is not None:
            yield from HYBRID_ENGINE.stream_reply(question, session=self)
            return
        topic = self.infer_topic(question)
        person, ambiguous = self.get_person(question)
        if ambiguous:
            ans = f"ขออภัยครับ คำว่า '{ambiguous}' อาจหมายถึงอาจารย์ได้หลายท่าน กรุณาระบุชื่อเจาะจง เช่น อ.กฤษณะ, อ.นิลรัตน์, อ.พลวัต, อ.สุภาวดี, อ.จักริน หรือ อ.พิเชษฐ์ ครับ"
            yield {"type": "token", "content": ans}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.0, "ttft": 0.0, "llm": 0.0, "total": 0.0}}
            return
        if person:
            topic = "professor"
        elif topic == "professor" and self.last_person and any(
            word in question for word in ("เขา", "ท่าน", "คนนี้", "คนเดิม", "แล้วจบ", "ห้องทำงาน", "อีเมล", "วิจัย", "ผลงาน")
        ):
            person = self.last_person
        year = extract_year(question)
        if year is None and topic == "curriculum" and self.last_topic == "curriculum" and question.startswith("แล้ว"):
            year = self.last_year

        q_low = question.lower().strip()

        # [ROUTER A]: Confusion / Feedback Queries
        if any(w in q_low for w in (
            "ตอบมั่ว", "ทำไมชอบตอบมั่ว", "ทำไมตอบมั่ว", "มั่ว", "มั่วมาก", "ชอบตอบมั่ว",
            "งง", "คืออะไร งง", "งงมาก", "งงจัง", "ไม่รู้เรื่อง", "พูดไม่รู้เรื่อง",
            "ตอบไม่ตรง", "ตอบไม่รู้เรื่อง", "หลอน", "หลอนจัง", "ทำไมตอบแบบนี้"
        )):
            answer = (
                "ขออภัยด้วยครับ! เนื่องจากผมเป็นระบบ AI ประจำสาขาวิชา AAI ที่เน้นให้ข้อมูลหลักสูตรและกิจกรรมของคณะวิทยาการสารสนเทศ ม.บูรพา "
                "บางคำถามที่อยู่นอกเหนือฐานข้อมูลอาจทำให้การตอบคลาดเคลื่อนได้ครับ "
                "คุณสามารถสอบถามข้อมูลหลักสูตร AAI, การรับสมัคร TCAS, ค่าเทอม, ชั่วโมงสหกิจศึกษา หรืออาจารย์ประจำสาขาได้เลยนะครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic = "feedback"
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return

        # [ROUTER B]: Short Follow-up Numbers (e.g. "1 คืออะไร", "ข้อ 1 คืออะไร")
        num_match = re.search(r"^(?:ข้อ|อัน|ลำดับ|หมายเลข)?\s*(\d+)\s*(?:คืออะไร|คือ|หมายถึงอะไร|หมายความว่าอะไร|แปลว่าอะไร|ละ|ล่ะ|\?)?\s*$", q_low)
        if num_match and not any(op in q_low for op in ("+", "-", "*", "/", "x", "บวก", "ลบ", "คูณ", "หาร")):
            target_num = int(num_match.group(1))
            answer = None
            if self.history:
                last_q, last_a = self.history[-1]
                lines_in_a = [l.strip() for l in last_a.splitlines() if l.strip()]
                found_line = None
                for l in lines_in_a:
                    prefix_match = re.match(r"^(?:[-*]\s*)?(?:ข้อ\s*)?(\d+)[\.\)\:]\s*(.+)$", l)
                    if prefix_match and int(prefix_match.group(1)) == target_num:
                        found_line = prefix_match.group(2).strip()
                        break
                if found_line:
                    answer = f"จากข้อความก่อนหน้า ({last_q}) ข้อ {target_num} คือ:\n📌 {found_line}\n\nคุณสามารถสอบถามรายละเอียดเพิ่มเติมของหัวข้อนี้ได้เลยนะครับ"
                else:
                    all_nums = []
                    for l in lines_in_a:
                        pm = re.match(r"^(?:[-*]\s*)?(?:ข้อ\s*)?(\d+)[\.\)\:]\s*(.+)$", l)
                        if pm:
                            all_nums.append(int(pm.group(1)))
                    if all_nums:
                        answer = f"จากคำตอบก่อนหน้า ({last_q}) มีรายการเพียง {len(all_nums)} ข้อ ไม่พบข้อ {target_num} ครับ คุณสามารถสอบถามหัวข้ออื่นที่สนใจได้เลยครับ"
                    else:
                        answer = f"จากคำถามก่อนหน้า '{last_q}' ไม่ได้ระบุรายการเป็นข้อไว้ครับ คุณต้องการสอบถามข้อมูลเรื่องใดเกี่ยวกับสาขา AAI คณะวิทยาการสารสนเทศ ม.บูรพา หรือเรื่องอื่นๆ เพิ่มเติมไหมครับ"
            else:
                answer = f"คุณระบุข้อ {target_num} แต่ยังไม่มีบทสนทนาก่อนหน้านี้ครับ คุณต้องการสอบถามข้อมูลเรื่องใดเกี่ยวกับสาขาวิชา AAI คณะวิทยาการสารสนเทศ ม.บูรพา ครับ"

            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic = "followup_number"
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return

        # [ROUTER C]: AI Identity & Model Queries
        if any(w in q_low for w in (
            "ai ตัวนี้คืออะไร", "aiอะไร", "ai อะไร", "โมเดลอะไร", "model อะไร", "ใช้ ai อะไร",
            "ใช้ ai ตัวไหน", "ใช้โมเดลอะไร", "ใช้ model อะไร", "เป็นใคร", "คุณเป็นใคร", "คุณคือใคร",
            "คุณคือ ai อะไร", "ระบบนี้คืออะไร", "ai ที่ใช้ตอบ", "ai ประจำสาขา",
            "ใช้ llm อะไร", "llm อะไร", "โมเดลที่ใช้", "model ที่ใช้"
        )):
            answer = (
                "ผมคือ AI ผู้ช่วยอัจฉริยะประจำสาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา ครับ "
                "พัฒนาขึ้นด้วยระบบ Hybrid Edge RAG รันบนระบบประมวลผลออฟไลน์ภายใน "
                "เพื่อให้ข้อมูลหลักสูตร การรับสมัคร ค่าเทอม อาจารย์ และข้อมูลทั่วไปเกี่ยวกับมหาวิทยาลัยบูรพาครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "rag/aai_overview.txt", "page": 1, "title": "ข้อมูลระบบ BUU AAI AI Assistant"}]}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic = "meta"
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return

        # [ROUTER D]: Location & Campus Queries (when not asking for a professor)
        if not person and any(w in q_low for w in (
            "ที่นี่ที่ไหน", "ที่นี่คือที่ไหน", "ที่นี่คือที่ใด", "อยู่ที่ไหน", "คณะอยู่ที่ไหน", "มออยู่ที่ไหน",
            "คณะตั้งอยู่ที่ไหน", "มหาลัยอยู่ที่ไหน", "มหาวิทยาลัยอยู่ที่ไหน", "ตั้งอยู่ที่ไหน", "ที่ตั้งคณะ",
            "ที่ตั้งมหาวิทยาลัย", "ที่ตั้งสาขา", "พิกัดคณะ", "พิกัดมหาวิทยาลัย", "ที่นี่มออะไร",
            "ที่นี่มหาวิทยาลัยอะไร", "ที่นี่คือมออะไร", "ม.อะไร", "ชื่อมหาวิทยาลัยอะไร", "มออะไร",
            "ตั้งอยู่ที่ใด", "เดินทางมายังไง", "เดินทางมาคณะ"
        )):
            answer = (
                "ที่นี่คือ สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา (BUU) "
                "ตั้งอยู่ที่ ถนนลงหาดบางแสน ตำบลแสนสุข อำเภอเมืองชลบุรี จังหวัดชลบุรี ครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "rag/aai_overview.txt", "page": 1, "title": "ข้อมูลที่ตั้งคณะและสาขาวิชา AAI"}]}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic = "location"
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return

        # [ROUTER E]: Cooking Recipes & Famous Food Scene
        if any(w in q_low for w in (
            "สูตรอาหาร", "ทำอาหาร", "วิธีทำอาหาร", "สูตรทำอาหาร", "สอนทำอาหาร", "มีสูตรอาหาร",
            "ทำกับข้าว", "สูตรกับข้าว", "สูตรอาหารให้มั้ย", "มีสูตรอาหารให้มั้ย",
            "สูตรลาบ", "ทำลาบ", "วิธีทำลาบ", "ลาบหมู", "ต้มยำกุ้ง", "สูตรต้มยำ", "วิธีทำต้มยำ"
        )):
            answer = (
                "แน่นอนครับ! มีสูตรอาหารไทยยอดนิยม 2 เมนูเด็ดพร้อมขั้นตอนการทำอย่างละเอียดมาฝากครับ 🍽️✨\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "1. 🍲 สูตรต้มยำกุ้งน้ำใส/น้ำข้น\n"
                "📌 วัตถุดิบ:\n"
                "- กุ้งสดปอกเปลือกผ่าหลัง 300 กรัม\n"
                "- เห็ดฟางหรือเห็ดนางฟ้า 100 กรัม\n"
                "- ข่าหั่นแว่น 5-6 แว่น, ตะไคร้ทุบหั่นท่อน 2 ต้น, ใบมะกรูดฉีก 4-5 ใบ\n"
                "- พริกขี้หนูบุบ 5-10 เม็ด (ปรับความเผ็ดตามชอบ)\n"
                "- น้ำปลา 2 ช้อนโต๊ะ, น้ำมะนาว 2-3 ช้อนโต๊ะ\n"
                "- น้ำพริกเผา 1-2 ช้อนโต๊ะ และนมข้นจืด 3 ช้อนโต๊ะ (หากชอบต้มยำน้ำข้น)\n"
                "- ผักชีและผักชีฝรั่งซอย\n"
                "📌 วิธีทำ:\n"
                "1) ต้มน้ำซุปให้เดือด ใส่ข่า ตะไคร้ ใบมะกรูด รอให้ส่งกลิ่นหอม\n"
                "2) ใส่เห็ดและกุ้งลงไป (ห้ามคนจนกว่ากุ้งจะสุกเพื่อไม่ให้คาว)\n"
                "3) ใส่น้ำพริกเผาและนมข้นจืด (ถ้าทำน้ำข้น) พอเดือดยกลงจากเตา\n"
                "4) ปรุงรสด้วยน้ำปลา น้ำมะนาว พริกขี้หนูบุบ โรยผักชี พร้อมเสิร์ฟ\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "2. 🥗 สูตรลาบหมูรสแซ่บ\n"
                "📌 วัตถุดิบ:\n"
                "- หมูสับติดมันเล็กน้อย 200 กรัม\n"
                "- ข้าวคั่ว 1.5 ช้อนโต๊ะ (คั่วใหม่ๆ จะหอมมาก)\n"
                "- พริกป่น 1 ช้อนชา (ปรับความเผ็ดตามชอบ)\n"
                "- น้ำปลา 1.5 - 2 ช้อนโต๊ะ, น้ำมะนาว 2 ช้อนโต๊ะ, น้ำตาลทราย 1/2 ช้อนชา\n"
                "- หอมแดงซอย, ต้นหอมซอย, ผักชีฝรั่งซอย, ใบสะระแหน่\n"
                "📌 วิธีทำ:\n"
                "1) รวนหมูสับในหม้อด้วยน้ำเปล่าหรือน้ำซุปเล็กน้อยจนสุกพอดี\n"
                "2) ปิดไฟ ปรุงรสด้วยน้ำปลา มะนาว น้ำตาลทราย พริกป่น คนให้เข้ากัน\n"
                "3) ใส่ข้าวคั่ว หอมแดง ต้นหอม ผักชีฝรั่ง และใบสะระแหน่ คลุกเคล้าเบาๆ\n"
                "4) เสิร์ฟพร้อมข้าวเหนียวร้อนๆ และผักสดแกล้ม (แตงกวา กะหล่ำปลี ถั่วฝักยาว)\n\n"
                "💡 แนะนำเพิ่มเติม: หากไม่อยากทำอาหารเอง แถว ม.บูรพา และบางแสน มีแหล่งของกินอร่อยๆ เพียบเลยครับ "
                "เช่น ตลาดปาร์คอิน (Park In), ซอยสดใส, ตลาดหนองมน และริมหาดวอนนภา แวะไปลิ้มลองกันได้นะครับ!"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic = "recipe"
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return
        if topic == "meta" or any(w in question.lower() for w in ("ทำอะไรได้บ้าง", "คุณคือใคร", "คุณรู้เรื่อง", "ถามอะไรได้บ้าง", "ช่วยอะไรได้บ้าง", "สวัสดีครับ", "สวัสดีค่ะ")) or question.strip() == "สวัสดี":
            answer = (
                "ผมคือ AI ผู้ช่วยตอบคำถามประจำสาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา ครับ\n"
                "ผมสามารถช่วยตอบข้อมูลเกี่ยวกับ:\n"
                "1. ข้อมูลหลักสูตร AAI: โครงสร้าง 120 หน่วยกิต, แผนการเรียน, ค่าเทอม (25,000 บาท/เทอม)\n"
                "2. การรับสมัครเข้าศึกษา: เกณฑ์ TCAS รอบ 1 (Portfolio) และ รอบ 3 (Admission)\n"
                "3. คณาจารย์ประจำสาขา: รายชื่อ, ห้องทำงาน, ประวัติการศึกษา และความเชี่ยวชาญของอาจารย์ทั้ง 6 ท่าน\n"
                "4. สหกิจศึกษา: เงื่อนไขการไปสหกิจตามประกาศ 0080/2568, การเก็บชั่วโมง 42 ชม. และวิชา BUU MOOC\n"
                "5. โอกาสทางอาชีพ: อาชีพหลังเรียนจบ เช่น AI Engineer, Data/Analytics Engineer, Researcher\n"
                "6. ข้อมูลคณะ: สีประจำคณะ (น้ำเงินเข้มและเทาเงิน), บันทึกข้อตกลงความร่วมมือ (MOU)"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "rag/aai_overview.txt", "page": 1}]}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic = "curriculum"
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return
        is_count = topic == "professor" and any(x in question for x in ("กี่คน", "กี่ท่าน", "จำนวนอาจารย์"))
        is_list = topic == "professor" and any(x in question for x in ("ใครบ้าง", "รายชื่อ", "ชื่ออะไรบ้าง", "ชื่อทั้งหมด", "มีใคร"))
        is_all_prof_bio = (
            (topic == "professor" or self.last_topic == "professor")
            and not person
            and any(x in question for x in ("แต่ละคน", "แต่ละท่าน", "ทุกคน", "ประวัติอาจารย์", "อาจารย์แต่ละ"))
        )
        roster = list(self.kb.roster())
        if is_all_prof_bio:
            self.last_roster = list(roster)
            lines = [
                f"ข้อมูลประวัติการศึกษาและห้องทำงานของอาจารย์ประจำสาขา AAI ทั้ง {len(roster)} ท่าน:"
            ]
            for i, key in enumerate(self.last_roster, 1):
                hdr = self.kb.find_prof_header(key)
                if hdr:
                    first_line = hdr["text"].splitlines()[0].strip()
                    lines.append(f"{i}. {first_line}")
                else:
                    lines.append(f"{i}. {PROF_DISPLAY_NAMES.get(key, key)}")
            lines.append("\n(สามารถพิมพ์ถามรายละเอียดเจาะจงของแต่ละท่านได้ เช่น 'ขอประวัติการศึกษา อ.กฤษณะ' หรือ 'ห้องทำงาน อ.พลวัต')")
            answer = "\n".join(lines)
            yield {"type": "token", "content": answer}
            sources = [{"source": f"Professor/professor_aai_{k}.txt", "page": None} for k in self.last_roster]
            yield {"type": "sources", "sources": sources}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic, self.last_person, self.last_year = "professor", None, None
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return
        if is_count:
            answer = f"สาขาวิชา AAI (ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ) มีอาจารย์ประจำหลักสูตรทั้งหมด {len(roster)} ท่านครับ ได้แก่ " + " ".join([f"{i}. {PROF_DISPLAY_NAMES.get(k, k)}" for i, k in enumerate(roster, 1)])
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "rag/aai_overview.txt", "page": 1}]}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic, self.last_person, self.last_year = "professor", None, None
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return
        if is_list:
            self.last_roster = list(roster)
            lines = [f"รายชื่ออาจารย์ประจำสาขาวิชา AAI ทั้ง {len(roster)} ท่าน:"]
            for i, key in enumerate(self.last_roster, 1):
                disp = PROF_DISPLAY_NAMES.get(key, key)
                lines.append(f"{i}. {disp}")
            answer = "\n".join(lines)
            yield {"type": "token", "content": answer}
            sources = [{"source": "rag/aai_overview.txt (รายชื่ออาจารย์ประจำสาขา AAI)", "page": None}]
            for k in self.last_roster:
                sources.append({"source": f"Professor/professor_aai_{k}.txt", "page": None})
            yield {"type": "sources", "sources": sources}
            total_sec = time.perf_counter() - began
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": round(total_sec, 2)}}
            self.last_topic, self.last_person, self.last_year = "professor", None, None
            self.last_question = question
            self.history.append((question, answer))
            self.history = self.history[-4:]
            return
        effective_question = self.rewrite_followup(question, topic, person, year)
        q_low = effective_question.lower()

        # Hybrid Deterministic Router: All Programs / Majors in Faculty
        if (any(w in q_low for w in (
            "มีหลักสูตรอะไร", "มีสาขาอะไร", "เปิดสอนสาขา", "เปิดสอนหลักสูตร", "หลักสูตรทั้งหมด",
            "สาขาทั้งหมด", "มีกี่สาขา", "มีกี่หลักสูตร", "คณะวิทยาการสารสนเทศมีสาขา",
            "คณะวิทยาการสารสนเทศมีหลักสูตร", "หลักสูตรที่เปิดสอน", "สาขาที่เปิดสอน"
        )) or (("หลักสูตร" in q_low or "สาขา" in q_low) and any(w in q_low for w in ("อะไรบ้าง", "มีอะไร", "เปิดสอน", "มีบ้าง", "กี่หลักสูตร", "กี่สาขา")))) and not any(w in q_low for w in ("วิชา", "รายวิชา", "หน่วยกิต")):
            answer = (
                "คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา เปิดสอนหลักสูตรทั้งระดับปริญญาตรี และระดับบัณฑิตศึกษา ดังนี้ครับ:\n\n"
                "📌 ระดับปริญญาตรี (4 สาขา):\n"
                "1. สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) - วท.บ.\n"
                "   - วุฒิ: วท.บ. (ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ) หลักสูตร 4 ปี จำนวน 120 หน่วยกิต มุ่งเน้น AI, Machine Learning, Data Science และ Smart IoT ในภาคอุตสาหกรรมและ EEC\n"
                "2. สาขาวิชาวิทยาการคอมพิวเตอร์ (CS) - วท.บ.\n"
                "   - วุฒิ: วท.บ. (วิทยาการคอมพิวเตอร์) เน้นรากฐานวิทยาการคอมพิวเตอร์ อัลกอริทึม และการพัฒนาระบบซอฟต์แวร์\n"
                "3. สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล (ITDI) - วท.บ.\n"
                "   - วุฒิ: วท.บ. (เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล) เน้นโครงสร้างพื้นฐานเครือข่าย คลาวด์ และเทคโนโลยีดิจิทัลสำหรับธุรกิจ\n"
                "4. สาขาวิชาวิศวกรรมซอฟต์แวร์ (SE) - วศ.บ.\n"
                "   - วุฒิ: วศ.บ. (วิศวกรรมซอฟต์แวร์) เน้นกระบวนการวิศวกรรมซอฟต์แวร์สากล สถาปัตยกรรมระบบ และ Full-Stack Development\n\n"
                "📌 ระดับบัณฑิตศึกษา (ปริญญาโท - เอก):\n"
                "- สาขาวิชาวิทยาการข้อมูลและการวิเคราะห์ชั้นสูง (Data Science & Advanced Analytics)\n"
                "   - วุฒิ: วท.ม. (ปริญญาโท) และ ปร.ด. (ปริญญาเอก) เน้นการวิเคราะห์ Big Data สถิติประยุกต์ และการวิจัยขั้นสูงด้าน Data Science ครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "หลักสูตรคณะวิทยาการสารสนเทศ ม.บูรพา"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Data Science & Graduate Programs
        if any(w in q_low for w in ("data science", "วิทยาการข้อมูล", "สาขา ds", "วิทยาศาสตร์ข้อมูล", "ปริญญาโท", "ป.โท", "ปริญญาเอก", "ป.เอก", "บัณฑิตศึกษา")):
            answer = (
                "สาขาวิชาวิทยาการข้อมูลและการวิเคราะห์ชั้นสูง (Data Science & Advanced Analytics) ของคณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา เปิดสอนในระดับบัณฑิตศึกษา ดังนี้ครับ:\n\n"
                "📌 ระดับบัณฑิตศึกษา (ปริญญาโท - เอก):\n"
                "- สาขาวิชาวิทยาการข้อมูลและการวิเคราะห์ชั้นสูง (Data Science & Advanced Analytics)\n"
                "   - วุฒิ: วท.ม. (ปริญญาโท) และ ปร.ด. (ปริญญาเอก)\n"
                "   - มุ่งเน้นการวิเคราะห์ Big Data สถิติประยุกต์ และโมเดลวิเคราะห์ข้อมูลเชิงลึกขั้นสูงครับ\n\n"
                "📌 สำหรับระดับปริญญาตรี คณะเปิดสอน 4 สาขาวิชา ได้แก่ AAI (วท.บ.), CS (วท.บ.), ITDI (วท.บ.) และ SE (วศ.บ.) ครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "หลักสูตรคณะวิทยาการสารสนเทศ ม.บูรพา"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: What is AAI
        if any(w in q_low for w in ("aai คืออะไร", "สาขา aai คือ", "หลักสูตร aai คือ", "aai ย่อมาจาก", "ความหมายของ aai", "รู้จักสาขา aai", "เกี่ยวกับสาขา aai", "เกี่ยวกับ aai")) or ("aai" in q_low and any(w in q_low for w in ("คืออะไร", "ย่อมาจาก", "หมายถึง", "คือสาขา"))):
            answer = (
                "AAI คือ หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ "
                "(Bachelor of Science in Applied Artificial Intelligence and Smart Technology) คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา ครับ\n\n"
                "รายละเอียดสำคัญของหลักสูตร:\n"
                "- วุฒิการศึกษา: วท.บ. (ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ)\n"
                "- โครงสร้างหลักสูตร: 4 ปี จำนวนไม่น้อยกว่า 120 หน่วยกิต (หลักสูตรปรับปรุง พ.ศ. 2567)\n"
                "- วัตถุประสงค์: ผลิตบัณฑิตที่มีความรู้ความเชี่ยวชาญทั้งภาคทฤษฎีและปฏิบัติในการนำเทคโนโลยี AI, Deep Learning, Computer Vision, Natural Language Processing, Big Data Analytics และ Internet of Things (IoT) ไปประยุกต์ใช้งานจริงในภาคอุตสาหกรรม เมืองอัจฉริยะ (Smart City) และพื้นที่เขตพัฒนาพิเศษภาคตะวันออก (EEC) ครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "ข้อมูลหลักสูตร AAI ม.บูรพา"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Program Highlights & Strengths
        if any(w in q_low for w in ("จุดเด่น", "จุดแข็ง", "ทำไมต้องเรียน", "ข้อดีของ", "มีดีอะไร", "น่าสนใจยังไง", "จุดเด่นของสาขา", "จุดเด่นของ aai")):
            answer = (
                "จุดเด่นของหลักสูตรปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ ม.บูรพา มีดังนี้ครับ:\n\n"
                "1. หลักสูตรทันสมัย สอดรับยุทธศาสตร์ EEC: ออกแบบร่วมกับผู้เชี่ยวชาญจากภาคอุตสาหกรรม มุ่งสร้างบุคลากร AI ที่พร้อมทำงานจริงได้ทันที\n"
                "2. เรียนจบได้ใน 4 ปี (120 หน่วยกิต): โครงสร้างหลักสูตรกระชับ เจาะลึกวิชาแกนด้าน AI และ Smart Technology โดยไม่เยิ่นเย้อ\n"
                "3. แล็บและอุปกรณ์มาตรฐานสูง (AI High-Performance Lab): มีเครื่อง GPU Workstation และระบบ Cloud สำหรับฝึกเทรนโมเดล AI ขนาดใหญ่\n"
                "4. เครือข่ายความร่วมมือระดับประเทศ (MOU & depa): ร่วมมือกับสำนักงานส่งเสริมเศรษฐกิจดิจิทัล (depa) และองค์กรเอกชนชั้นนำ มีโครงการสนับสนุนทุนและการอบรมรับรองมาตรฐาน\n"
                "5. สหกิจศึกษาเข้มข้น (Cooperative Education): เปิดโอกาสให้นิสิตได้ไปปฏิบัติงานจริงในสถานประกอบการจริง 1-2 ภาคการศึกษา สร้างผลงานจริงก่อนสำเร็จการศึกษาและมีโอกาสได้งานทันทีครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "จุดเด่นหลักสูตร AAI"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Difficulty & Background Requirement
        if any(w in q_low for w in ("เรียนยากไหม", "ยากไหม", "ไม่มีพื้นฐาน", "ไม่เคยเขียนโค้ด", "เขียนโปรแกรมไม่เป็น", "ไม่เก่งเลข", "ไม่เก่งคณิต", "ต้องมีพื้นฐานไหม", "พื้นฐานไม่ดี")):
            answer = (
                "หลักสูตร AAI ไม่จำเป็นต้องมีพื้นฐานการเขียนโปรแกรมมาก่อนก็สามารถเรียนได้ครับ!\n\n"
                "ข้อแนะนำและระบบสนับสนุนการเรียน:\n"
                "1. ปูพื้นฐานตั้งแต่เริ่มต้น (ปี 1): มีรายวิชาปูพื้นฐานการเขียนโปรแกรมด้วยภาษา Python และ C รวมถึงคณิตศาสตร์พื้นฐานสำหรับคอมพิวเตอร์อย่างเป็นลำดับขั้น\n"
                "2. ระบบพี่เลี้ยงและคณาจารย์ดูแลใกล้ชิด: มีกลุ่มแล็บฝึกปฏิบัติ กิจกรรมติวเสริมจากรุ่นพี่ และอาจารย์ที่ปรึกษาคอยให้คำแนะนำตลอดหลักสูตร\n"
                "3. เน้นลงมือปฏิบัติจริง (Hands-on Learning): การได้ลงมือทำมินิโปรเจกต์จริงจะช่วยให้เข้าใจตรรกะได้ดีกว่าการท่องจำ\n"
                "4. คุณสมบัติที่สำคัญที่สุด: ขอเพียงมีความสนใจในเทคโนโลยี มีความพยายาม มีทักษะการคิดเชิงตรรกะ (Logical Thinking) และพร้อมเปิดรับการเรียนรู้สิ่งใหม่ๆ ก็สามารถเรียนได้อย่างประสบความสำเร็จแน่นอนครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "คำแนะนำการเรียนหลักสูตร AAI"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Year 1 Curriculum
        if any(w in q_low for w in ("ปี 1 เรียน", "ปี 1 มีวิชา", "ปีหนึ่งเรียน", "วิชาปี 1", "วิชาปีหนึ่ง", "หลักสูตรปี 1", "ชั้นปีที่ 1", "ปี 1 เทอม 1", "ปี 1 เทอม 2")) or (any(y in q_low for y in ("ปี 1", "ปีหนึ่ง", "ชั้นปีที่ 1")) and any(w in q_low for w in ("เรียน", "วิชา", "สอน"))):
            answer = (
                "โครงสร้างรายวิชาสำหรับนิสิตหลักสูตร AAI ชั้นปีที่ 1 แบ่งตามภาคการศึกษาดังนี้ครับ:\n\n"
                "📘 ภาคการศึกษาต้น (เทอม 1):\n"
                "- การโปรแกรมคอมพิวเตอร์เบื้องต้น (Fundamental Computer Programming) และปฏิบัติการ\n"
                "- แคลคูลัสสำหรับวิทยาการข้อมูล (Calculus for Data Science)\n"
                "- ภาษาอังกฤษเพื่อการสื่อสารในชีวิตประจำวัน (English for Daily Life Communication)\n"
                "- รายวิชาศึกษาทั่วไป (GE) ด้านการพัฒนาตนเองและความฉลาดทางดิจิทัล\n\n"
                "📙 ภาคการศึกษาปลาย (เทอม 2):\n"
                "- การโปรแกรมเชิงวัตถุ (Object-Oriented Programming: OOP) และปฏิบัติการ\n"
                "- พีชคณิตเชิงเส้นเชิงคำนวณ (Computational Linear Algebra)\n"
                "- สถิติและความน่าจะเป็นสำหรับปัญญาประดิษฐ์ (Statistics and Probability for AI)\n"
                "- พื้นฐานปัญญาประดิษฐ์และอินเทอร์เน็ตของสรรพสิ่ง (Fundamentals of AI and IoT)\n"
                "- ภาษาอังกฤษเพื่อวัตถุประสงค์ทางวิชาการ (English for Academic Purposes)"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "แผนการเรียนชั้นปีที่ 1 สาขา AAI"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Career Opportunities
        if any(w in q_low for w in ("จบไปทำงาน", "จบแล้วทำ", "ทำงานอะไร", "อาชีพ", "สายงาน", "ตำแหน่งงาน", "ตลาดงาน", "หางานง่ายไหม", "โอกาสการทำงาน", "จบมาทำ")):
            answer = (
                "ผู้สำเร็จการศึกษาสาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) สามารถประกอบอาชีพในสายงานเทคโนโลยีสารสนเทศและปัญญาประดิษฐ์ได้หลากหลาย ได้แก่:\n\n"
                "1. วิศวกรปัญญาประดิษฐ์ (AI Engineer)\n"
                "2. วิศวกรการเรียนรู้ของเครื่อง (Machine Learning Engineer)\n"
                "3. นักวิทยาศาสตร์ข้อมูล / นักวิเคราะห์ข้อมูล (Data Scientist / Data Analyst)\n"
                "4. นักพัฒนาซอฟต์แวร์และระบบอัจฉริยะ (Smart System / Full-Stack Developer)\n"
                "5. วิศวกรไอโอทีและระบบสมองกลฝังตัว (IoT & Embedded Systems Engineer)\n"
                "6. วิศวกรวิสัยทัศน์คอมพิวเตอร์และการประมวลผลภาพ (Computer Vision Engineer)\n"
                "7. นักพัฒนาระบบประมวลผลภาษาธรรมชาติ (NLP Engineer / LLM Application Developer)\n"
                "8. ที่ปรึกษาด้านเทคโนโลยีปัญญาประดิษฐ์และโซลูชันดิจิทัล (AI & Digital Solution Consultant)\n\n"
                "ทุกสายงานมีความต้องการตัวสูงมากทั้งในบริษัทเทคโนโลยี สตาร์ทอัพ โรงงานอุตสาหกรรมในพื้นที่ EEC และหน่วยงานภาครัฐและเอกชนทั่วประเทศครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "แนวทางการประกอบอาชีพ สาขา AAI"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Program Chairman
        if any(w in q_low for w in ("ประธานหลักสูตร", "ประธานสาขา", "หัวหน้าสาขา")) and any(w in q_low for w in ("ใคร", "ชื่อ", "คือ", "ท่านไหน", "ขอ")):
            answer = (
                "ประธานหลักสูตรสาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา คือ:\n"
                "ผศ.ดร.พลวัต ช่อผูก (Asst. Prof. Dr. Ponlawat Chophuk) ครับ\n\n"
                "📍 ห้องทำงาน: IF-913F อาคารคณะวิทยาการสารสนเทศ\n"
                "📧 อีเมล: ponlawat.ch@buu.ac.th\n"
                "🎓 ประวัติการศึกษา:\n"
                "- ปริญญาเอก: วศ.ด. (เทคโนโลยีวิศวกรรมไฟฟ้าและสารสนเทศ) มจธ.\n"
                "- ปริญญาโท: วศ.ม. (วิศวกรรมชีวการแพทย์) สจล.\n"
                "- ปริญญาตรี: วศ.บ. (วิศวกรรมอิเล็กทรอนิกส์) สจล."
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "Professor/professor_aai_ponlawat.txt", "page": 1, "title": "ข้อมูลประธานหลักสูตร AAI"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "professor"
            self.last_person = "ponlawat"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Tuition Fees
        if any(w in q_low for w in ("ค่าเทอม", "ค่าธรรมเนียม", "ค่าเล่าเรียน")):
            answer = (
                "ค่าธรรมเนียมการศึกษา (ค่าเทอม) สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) คณะวิทยาการสารสนเทศ ม.บูรพา:\n"
                "- นิสิตรหัส 67 เป็นต้นไป: ภาคการศึกษาละ 25,000 บาท (ตลอดหลักสูตร 8 ภาคการศึกษา รวม 200,000 บาท)\n"
                "- นิสิตรหัสก่อนปี 67: รวมตลอดหลักสูตร 237,500 บาท\n"
                "(หลักสูตร 4 ปี จำนวนหน่วยกิตรวมไม่น้อยกว่า 120 หน่วยกิต)"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": "curriculum/aai-program-67.pdf", "page": 1, "title": "ค่าธรรมเนียมการศึกษา AAI"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Professor Education & Degrees
        if person and person in PROFESSOR_FACTS and any(w in q_low for w in ("จบ", "วุฒิ", "การศึกษา", "ปริญญา", "จบที่ไหน", "จบอะไร", "ประวัติการศึกษา", "จบจาก", "จบเอก", "จบโท", "จบตรี")):
            fact = PROFESSOR_FACTS[person]
            deg_lines = "\n".join(f"- {d}" for d in fact["degrees"])
            answer = f"ประวัติการศึกษาของ {fact['name']}:\n{deg_lines}\n\nห้องทำงาน: {fact['office']} | อีเมล: {fact['email']}"
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": f"Professor/professor_aai_{person}.txt", "page": 1, "title": f"ประวัติการศึกษา {fact['name']}"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "professor"
            self.last_person = person
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Hybrid Deterministic Router: Professor Office & Contact
        if person and person in PROFESSOR_FACTS and any(w in q_low for w in ("ห้อง", "ห้องทำงาน", "โต๊ะทำงาน", "ติดต่อ", "อีเมล", "email", "office", "อยู่ที่ไหน", "อยู่ห้อง")):
            fact = PROFESSOR_FACTS[person]
            answer = f"ข้อมูลการติดต่อของ {fact['name']}:\n- ห้องทำงาน: {fact['office']}\n- อีเมล: {fact['email']}"
            if "expertise" in fact:
                answer += f"\n- ความเชี่ยวชาญ: {fact['expertise']}"
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": f"Professor/professor_aai_{person}.txt", "page": 1, "title": f"ข้อมูลการติดต่อ {fact['name']}"}]}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "professor"
            self.last_person = person
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return
        # General Knowledge Router: Days and Colors
        day_colors = {
            "วันจันทร์": "สีเหลือง (Yellow)",
            "วันอังคาร": "สีชมพู (Pink)",
            "วันพุธ": "สีเขียว (Green)",
            "วันพฤหัส": "สีส้ม (Orange)",
            "วันศุกร์": "สีฟ้า (Light Blue)",
            "วันเสาร์": "สีม่วง (Purple)",
            "วันอาทิตย์": "สีแดง (Red)",
        }
        is_color_day_query = any(w in q_low for w in ("สีประจำวัน", "วันไหนสีอะไร", "วันจันทร์สี", "วันอังคารสี", "วันพุธสี", "วันพฤหัสสี", "วันศุกร์สี", "วันเสาร์สี", "วันอาทิตย์สี")) or (
            any(d in q_low for d in day_colors) and any(w in q_low for w in ("สีอะไร", "สีไหน", "คือสีอะไร"))
        ) or (self.last_topic == "day_color" and any(d in q_low for d in day_colors))
        if is_color_day_query:
            matched_day = None
            for d in day_colors:
                if d in q_low:
                    matched_day = d
                    break
            if matched_day:
                ans = f"{matched_day} คือ {day_colors[matched_day]} ครับ ☀️\n\n(หมายเหตุ: หากเป็น 'สีประจำคณะวิทยาการสารสนเทศ ม.บูรพา' คือ สีน้ำเงินเข้ม และ สีเทาเงิน ครับ)"
            else:
                ans = (
                    "สีประจำวันตามประเพณีไทย มีดังนี้ครับ:\n"
                    "- วันจันทร์: สีเหลือง\n"
                    "- วันอังคาร: สีชมพู\n"
                    "- วันพุธ: สีเขียว\n"
                    "- วันพฤหัสบดี: สีส้ม\n"
                    "- วันศุกร์: สีฟ้า\n"
                    "- วันเสาร์: สีม่วง\n"
                    "- วันอาทิตย์: สีแดง\n\n"
                    "(และหากเป็น 'สีประจำคณะวิทยาการสารสนเทศ ม.บูรพา' คือ สีน้ำเงินเข้ม และ สีเทาเงิน ครับ)"
                )
            yield {"type": "token", "content": ans}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "day_color"
            self.last_question = effective_question
            self.history.append((effective_question, ans))
            self.history = self.history[-4:]
            return

        # General & Local Knowledge Router: Bangsaen Sea & Attractions
        if any(w in q_low for w in (
            "มีทะเลมั้ย", "มีทะเลไหม", "ติดทะเลไหม", "ติดทะเลมั้ย", "ใกล้ทะเลไหม", "ใกล้ทะเลมั้ย", "มีหาดไหม", "มีหาดมั้ย",
            "ทะเลมีไหม", "ทะเลมีมั้ย", "บางแสนมีทะเล"
        )) or (self.last_topic == "bangsaen_sea" and any(w in q_low for w in ("ใช่หรอ", "จริงหรอ", "จริงเหรอ", "จริงปะ", "แน่ใจนะ"))):
            answer = (
                "มีทะเลแน่นอนครับ! มหาวิทยาลัยบูรพาตั้งอยู่ติดกับ ชายหาดบางแสน จ.ชลบุรี เลยครับ 🌊🏖️\n\n"
                "จากประตูหน้ามหาวิทยาลัยบูรพา เดินหรือนั่งรถลงไปตามถนนลงหาดบางแสนเพียงประมาณ 1 กิโลเมตร (2-3 นาที) ก็ถึงชายหาดและทะเลแล้วครับ นิสิตที่นี่นิยมไปนั่งเล่นริมทะเล รับลมชิลล์ๆ ชมพระอาทิตย์ตกดินกันเป็นประจำครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "bangsaen_sea"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        if any(w in q_low for w in ("รู้จักบางแสน", "เที่ยวบางแสน", "บางแสนมีอะไรเที่ยว", "ที่เที่ยวบางแสน", "ที่เที่ยวแถวนี้", "ที่เที่ยวรอบมอ", "ที่เที่ยวมอบูรพา", "ที่เที่ยว ม.บูรพา", "ที่เที่ยวใกล้")):
            answer = (
                "รู้จักเป็นอย่างดีเลยครับ! บางแสน เป็นเมืองชายทะเลชื่อดังของจังหวัดชลบุรี และเป็นที่ตั้งของมหาวิทยาลัยบูรพา (BUU) มีสถานที่ท่องเที่ยวยอดนิยมรอบๆ มากมายครับ 🌊✨\n\n"
                "จุดเช็คอินและสถานที่น่าเที่ยวแนะนำรอบ ม.บูรพา:\n"
                "1. ชายหาดบางแสน & หาดวอนนภา: เล่นน้ำทะเล นั่งเตียงผ้าใบรับลม ทานอาหารทะเลริมหาด และชมพระอาทิตย์ตกดิน\n"
                "2. สถาบันวิทยาศาสตร์ทางทะเล (Aquarium ม.บูรพา): อควาเรียมชื่อดังระดับประเทศ มีอุโมงค์ปลาและตู้แมงกะพรุนเรืองแสง ตั้งอยู่ภายในรั้ว ม.บูรพา เลยครับ\n"
                "3. แหลมแท่น: ลานกิจกรรมรับลมทะเล จุดถ่ายรูปสวยๆ และเป็นสถานที่จัดงานดนตรี/ถนนคนเดิน\n"
                "4. เขาสามมุข & ศาลเจ้าแม่เขาสามมุข: จุดชมวิวอ่าวบางแสนมุมสูง มีคาเฟ่สวยๆ และฝูงลิงเจ้าถิ่น\n"
                "5. ถนนคนเดินบางแสน (Bangsaen Walking Street): ช้อปปิ้ง ชิมสตรีทฟู้ดริมทะเล ฟังดนตรีสด (เปิดช่วงเย็นวันศุกร์-เสาร์)\n"
                "6. ตลาดหนองมน: แหล่งซื้อของฝากยอดฮิต ข้าวหลามหนองมน ห่อหมก และอาหารทะเลแห้งครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "bangsaen_travel"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # General & Local Knowledge Router: Food & Hungry around BUU
        if any(w in q_low for w in ("หิวข้าว", "หิวแล้ว", "หิวจัง", "กินอะไรดี", "แนะนำของกิน", "ของกินแถว", "ร้านอาหารแถว", "ของกินมอบูรพา", "ของกิน ม.บูรพา", "ของอร่อยบางแสน", "ร้านอร่อย")):
            answer = (
                "ถ้าหิวข้าว แถว ม.บูรพา และบางแสน มีแหล่งของกินเด็ดๆ เพียบเลยครับ! ขอแนะนำพิกัดยอดฮิตของนิสิต BUU ดังนี้ครับ 🍽️✨:\n\n"
                "1. ตลาดปาร์คอิน (Park In Market): แหล่งรวมสตรีทฟู้ด ของกินเล่น ขนมหวาน เครื่องดื่ม บรรยากาศสุดชิลล์ยามเย็นที่นิสิตชอบไปนั่งคุยเล่น\n"
                "2. ซอยสดใส & ซอยสดใสพัฒนา: ศูนย์รวมร้านอาหารราคานักศึกษา มีทั้งข้าวมันไก่ ข้าวหมูกรอบ ส้มตำ ชาบู ปิ้งย่าง และร้านนมคาเฟ่เปิดดึก\n"
                "3. ถนนเนตรดี: ย่านอาหารตามสั่ง ก๋วยเตี๋ยวเรือ และอาหารอีสานรสจัดจ้านตลอดสาย\n"
                "4. ชายหาดบางแสน & วอนนภา: อาหารทะเลสด ส้มตำ ไก่ย่าง กุ้งเผา ปูม้า ยำซีฟู้ด นั่งทานริมทะเลรับลมชิลล์ๆ\n"
                "5. ตลาดหนองมน: ก๋วยเตี๋ยวเจ้าเก่าแก่ ข้าวแกงรสเด็ด และข้าวหลามหนองมนของขึ้นชื่อครับ"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "food"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # General Knowledge Router: Cooking Recipes (e.g. Larb)
        if any(w in q_low for w in ("สูตรลาบ", "ทำลาบ", "วิธีทำลาบ", "ลาบหมู")):
            answer = (
                "มีสูตรทำลาบหมูรสแซ่บแบบไทยๆ มาฝากครับ! 🍲👩‍🍳\n\n"
                "📌 วัตถุดิบหลัก:\n"
                "- หมูสับ 200 กรัม (ติดมันเล็กน้อย)\n"
                "- ข้าวคั่ว 1-2 ช้อนโต๊ะ (คั่วใหม่จะหอมมาก)\n"
                "- พริกป่น 1 ช้อนชา (ปรับความเผ็ดตามชอบ)\n"
                "- น้ำปลา 1.5 - 2 ช้อนโต๊ะ\n"
                "- น้ำมะนาว 2 ช้อนโต๊ะ\n"
                "- น้ำตาลทราย 1/2 ช้อนชา\n"
                "- หอมแดงซอย, ผักชีฝรั่งซอย, ต้นหอมซอย, ใบสะระแหน่\n\n"
                "📌 วิธีทำ:\n"
                "1. รวนหมูสับในหม้อด้วยน้ำซุปหรือน้ำเปล่าเล็กน้อยจนสุกพอดี\n"
                "2. ปิดไฟ ปรุงรสด้วยน้ำปลา น้ำมะนาว น้ำตาลทราย และพริกป่น คนให้เข้ากัน\n"
                "3. ใส่ข้าวคั่ว หอมแดง ต้นหอม ผักชีฝรั่ง และใบสะระแหน่ คลุกเคล้าเบาๆ\n"
                "4. เสิร์ฟพร้อมข้าวเหนียวร้อนๆ และผักสดแกล้ม (แตงกวา กะหล่ำปลี ถั่วฝักยาว) อร่อยแซ่บแน่นอนครับ!"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "recipe"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # General Knowledge Router: Simple Math Expressions (e.g. "1 + 1", "1+1")
        math_match = re.search(r"^\s*(\d+(?:\.\d+)?)\s*([\+\-\*/xX])\s*(\d+(?:\.\d+)?)\s*(?:เท่ากับ|เท่าไหร่|คืออะไร|\?|=)?\s*$", q_low)
        if math_match:
            try:
                n1 = float(math_match.group(1))
                op = math_match.group(2).lower()
                n2 = float(math_match.group(3))
                if op == "+":
                    res = n1 + n2
                elif op == "-":
                    res = n1 - n2
                elif op in ("*", "x"):
                    res = n1 * n2
                elif op == "/":
                    res = n1 / n2 if n2 != 0 else "ไม่สามารถหารด้วย 0 ได้"
                else:
                    res = None
                if res is not None:
                    n1_str = int(n1) if n1.is_integer() else n1
                    n2_str = int(n2) if n2.is_integer() else n2
                    res_str = int(res) if isinstance(res, float) and res.is_integer() else res
                    answer = f"{n1_str} {math_match.group(2)} {n2_str} เท่ากับ {res_str} ครับ"
                    yield {"type": "token", "content": answer}
                    yield {"type": "sources", "sources": []}
                    yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
                    self.last_topic = "general"
                    self.last_question = effective_question
                    self.history.append((effective_question, answer))
                    self.history = self.history[-4:]
                    return
            except Exception:
                pass

        # University identity
        if any(w in q_low for w in ("ที่นี่มออะไร", "ที่นี่มหาวิทยาลัยอะไร", "ที่นี่คือมออะไร", "ม.อะไร", "ชื่อมหาวิทยาลัยอะไร", "มออะไร")):
            answer = (
                "ที่นี่คือ มหาวิทยาลัยบูรพา (Burapha University - BUU) ครับ ตั้งอยู่ที่ ต.แสนสุข อ.เมือง จ.ชลบุรี ใกล้ชายหาดบางแสน\n"
                "โดยผมเป็น AI ผู้ช่วยข้อมูลประจำ คณะวิทยาการสารสนเทศ (Faculty of Informatics) สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI) ครับ ยินดีต้อนรับครับ! 😊"
            )
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": 0.01, "llm": 0.01, "total": 0.02}}
            self.last_topic = "curriculum"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        # Distinguish Domain Queries vs General / Chit-chat queries
        is_domain = (
            person is not None
            or topic in ("professor", "curriculum", "coop", "admission", "career", "partnership", "color", "module")
            or any(kw in q_low for kw in (
                "aai", "buu", "informatics", "ม.บ.", "ม.บูรพา", "มอบูรพา", "มหาวิทยาลัยบูรพา",
                "คณะวิทยาการสารสนเทศ", "วิทยาการสารสนเทศ", "หลักสูตร", "สาขา", "ปริญญา",
                "หน่วยกิต", "ค่าเทอม", "ค่าธรรมเนียม", "ทุนการศึกษา", "รายวิชา", "แผนการเรียน",
                "หมวดวิชา", "วิชาแกน", "วิชาเอก", "วิชาเลือก", "ศึกษาทั่วไป", "ปี 1", "ปี 2", "ปี 3", "ปี 4",
                "tcas", "portfolio", "quota", "admission", "ยื่นพอร์ต", "tgat", "tpat", "รับสมัคร", "เกณฑ์",
                "ประธานหลักสูตร", "คณบดี", "อาจารย์", "ผู้สอน", "ห้องทำงาน", "สหกิจ", "ฝึกงาน", "ชั่วโมง",
                "0080", "42 ชม", "buu mooc", "mooc", "mou", "depa", "แล็บ", "ห้องแล็บ", "lab"
            ))
        )

        if not is_domain:
            # Clean LLM streaming for general world knowledge without RAG curriculum pollution
            payload = {
                "model": rag.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                    {"role": "user", "content": effective_question},
                ],
                "stream": True,
                "think": False,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": 0.4,
                    "repeat_penalty": 1.08,
                    "num_ctx": 1024,
                    "num_predict": MAX_TOKENS,
                },
            }
            t_llm = time.perf_counter()
            chunks = []
            t_first = None
            try:
                with requests.post(f"{OLLAMA_URL}/chat", json=payload, stream=True, timeout=900) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        data = json.loads(line)
                        piece = data.get("message", {}).get("content", "")
                        if piece:
                            if t_first is None:
                                t_first = time.perf_counter()
                            chunks.append(piece)
                            yield {"type": "token", "content": piece}
            except Exception as e:
                err_text = f"ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผลคำตอบ: {e}"
                yield {"type": "token", "content": err_text}
                chunks.append(err_text)
            llm_sec = time.perf_counter() - t_llm
            ttft_sec = (t_first - t_llm) if t_first is not None else llm_sec
            answer = "".join(chunks).strip()
            total_sec = time.perf_counter() - began
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": 0.01, "ttft": round(ttft_sec, 2), "llm": round(llm_sec, 2), "total": round(total_sec, 2)}}
            self.last_topic = "general"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return

        search_query = effective_question
        if person:
            search_query += f" ประวัติ การศึกษา ห้องทำงาน ผลงาน อาจารย์ {PROF_DISPLAY_NAMES.get(person, person)}"
        if topic == "professor" and any(w in effective_question for w in ("จบ", "วุฒิ", "การศึกษา", "ที่ไหน")):
            search_query += " ประวัติการศึกษา ปริญญาเอก ปริญญาโท ปริญญาตรี"
        t_search = time.perf_counter()
        evidence = self.kb.search(search_query, topic, year, person)
        search_sec = time.perf_counter() - t_search
        if not evidence:
            yield {"type": "token", "content": "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลเอกสาร กรุณาตรวจสอบข้อมูลใน docs_aai และ SQLite"}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timings": {"search": round(search_sec, 2), "total": round(time.perf_counter() - began, 2)}}
            return
        top_stem = stem(evidence[0]["source"])
        q_low = effective_question.lower()
        if topic == "coop":
            if any(w in q_low for w in ("mooc", "วิชาออนไลน์", "อบรมออนไลน์", "วิชา")):
                coop_stem = "อบรมออนไลน์ ใน buu mooc สหกิจศึกษา69"
            elif any(w in q_low for w in ("ใบเซอร์", "ใบประกาศ", "certificate")):
                coop_stem = "coop-detail"
            elif any(w in q_low for w in ("คุณสมบัติ", "เงื่อนไข", "ประกาศ", "0080")):
                coop_stem = "coop-80-2568"
            elif any(w in q_low for w in ("ชั่วโมง", "เก็บยังไง", "กี่ชั่วโมง", "42")):
                coop_stem = "coop-detail"
            else:
                coop_stem = "coop-detail"
            coop_matches = self.kb.find_records_by_stem(coop_stem)
            selected = coop_matches[:1] if coop_matches else evidence[:1]
        elif top_stem in ("aai_overview", "future", "tcas_aai", "coop-detail", "coop-80-2568"):
            selected = evidence[:1]
        elif person and any(w in effective_question for w in ("ห้อง", "เบอร์", "email", "อีเมล", "ติดต่อ", "ตำแหน่ง", "ความสนใจ", "ทักษะ")):
            selected = evidence[:1]
        elif person:
            selected = evidence[:2]
        else:
            selected = evidence[:TOP_K]
        context_blocks = []
        for rec in selected:
            snippet = rec["text"][:MAX_CONTEXT_PER_CHUNK].strip()
            context_blocks.append(snippet)
        context_text = "\n\n".join(context_blocks)
        sys_prompt = SYSTEM_PROMPT
        if person:
            prof_name = PROF_DISPLAY_NAMES.get(person, person)
            sys_prompt += f" คำถามนี้ถามถึง {prof_name} เท่านั้น ให้ตอบเฉพาะข้อมูลของท่านนี้"
        if year:
            sys_prompt += f" คำถามนี้ถามถึงหลักสูตรปี พ.ศ. {year}"
        if topic == "admission":
            t_llm = time.perf_counter()
            tcas_recs = self.kb.find_records_by_stem("tcas_aai")
            selected = tcas_recs[:1] if tcas_recs else evidence[:1]
            answer = tcas_recs[0]["text"].strip() if tcas_recs else context_text
            llm_sec = time.perf_counter() - t_llm
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": rec["source"], "page": rec.get("page")} for rec in selected]}
            yield {"type": "done", "timings": {"search": round(search_sec, 2), "ttft": round(llm_sec, 2), "llm": round(llm_sec, 2), "total": round(search_sec + llm_sec, 2)}}
            self.last_topic = "admission"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return
        if topic == "color" or "สีประจำคณะ" in q_low or "สีคณะ" in q_low:
            t_llm = time.perf_counter()
            color_recs = self.kb.find_records_by_stem("color-faculty")
            selected = color_recs[:1] if color_recs else evidence[:1]
            answer = color_recs[0]["text"] if color_recs else context_text
            llm_sec = time.perf_counter() - t_llm
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": rec["source"], "page": rec.get("page")} for rec in selected]}
            yield {"type": "done", "timings": {"search": round(search_sec, 2), "ttft": round(llm_sec, 2), "llm": round(llm_sec, 2), "total": round(search_sec + llm_sec, 2)}}
            self.last_topic = "color"
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return
        if topic == "coop":
            t_llm = time.perf_counter()
            if any(w in q_low for w in ("mooc", "วิชาออนไลน์", "อบรมออนไลน์", "วิชา")):
                mooc_recs = self.kb.find_records_by_stem("อบรมออนไลน์ ใน buu mooc สหกิจศึกษา69")
                selected = mooc_recs[:1] if mooc_recs else evidence[:1]
                answer = mooc_recs[0]["text"] if mooc_recs else context_text
            elif any(w in q_low for w in ("ใบเซอร์", "ใบประกาศ", "certificate")):
                coop_detail_recs = self.kb.find_records_by_stem("coop-detail")
                selected = coop_detail_recs[:1] if coop_detail_recs else evidence[:1]
                answer = coop_detail_recs[0]["text"] if coop_detail_recs else context_text
            elif any(w in q_low for w in ("คุณสมบัติ", "เงื่อนไข", "ประกาศ", "0080")):
                coop_80_recs = self.kb.find_records_by_stem("coop-80-2568")
                selected = coop_80_recs[:1] if coop_80_recs else evidence[:1]
                answer = coop_80_recs[0]["text"] if coop_80_recs else context_text
            elif any(w in q_low for w in ("ชั่วโมง", "เก็บยังไง", "กี่ชั่วโมง", "42")):
                coop_detail_recs = self.kb.find_records_by_stem("coop-detail")
                selected = coop_detail_recs[:1] if coop_detail_recs else evidence[:1]
                answer = coop_detail_recs[0]["text"] if coop_detail_recs else context_text
            else:
                coop_detail_recs = self.kb.find_records_by_stem("coop-detail")
                coop_80_recs = self.kb.find_records_by_stem("coop-80-2568")
                selected = (coop_detail_recs[:1] if coop_detail_recs else []) + (coop_80_recs[:1] if coop_80_recs else [])
                parts = []
                if coop_detail_recs:
                    parts.append(coop_detail_recs[0]["text"])
                if coop_80_recs:
                    parts.append(coop_80_recs[0]["text"])
                answer = "\n\n".join(parts) if parts else context_text
            llm_sec = time.perf_counter() - t_llm
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": [{"source": rec["source"], "page": rec.get("page")} for rec in selected]}
            yield {"type": "done", "timings": {"search": round(search_sec, 2), "ttft": round(llm_sec, 2), "llm": round(llm_sec, 2), "total": round(search_sec + llm_sec, 2)}}
            self.last_topic = topic
            self.last_question = effective_question
            self.history.append((effective_question, answer))
            self.history = self.history[-4:]
            return
        user_prompt = f"เอกสารอ้างอิง:\n{context_text}\n\nจากเอกสารข้างต้น จงตอบคำถามนี้เป็นภาษาไทย: {effective_question}"
        payload = {
            "model": rag.LLM_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "think": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "repeat_penalty": 1.08,
                "num_ctx": 1024,
                "num_predict": MAX_TOKENS,
                "stop": ["\nเอกสารอ้างอิง", "\nข้อมูลอ้างอิง", "\nคำถาม:", "\nจากเอกสารข้างต้น"],
            },
        }
        t_llm = time.perf_counter()
        chunks = []
        t_first = None
        prefix_buf = ""
        prefix_done = False
        with requests.post(f"{OLLAMA_URL}/chat", json=payload, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("message", {}).get("content", "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    if not prefix_done:
                        prefix_buf += piece
                        if len(prefix_buf) < 30 and any(p.startswith(prefix_buf.lstrip()) for p in ("คำตอบ:", "คำตอบ :", "มีหลักสูตรอย่างนี้", "มีหลักสูตรดังนี้", "คำตอบคือ")):
                            continue
                        prefix_buf = re.sub(r"^\s*(?:คำตอบ\s*[:\s]|มีหลักสูตรอย่างนี้\s*[:\s]*|มีหลักสูตรดังนี้\s*[:\s]*)", "", prefix_buf)
                        prefix_done = True
                        if prefix_buf:
                            yield {"type": "token", "content": prefix_buf}
                            chunks.append(prefix_buf)
                    else:
                        yield {"type": "token", "content": piece}
                        chunks.append(piece)
        if not prefix_done and prefix_buf:
            cleaned_pref = re.sub(r"^\s*(?:คำตอบ\s*[:\s]|มีหลักสูตรอย่างนี้\s*[:\s]*|มีหลักสูตรดังนี้\s*[:\s]*)", "", prefix_buf)
            if cleaned_pref:
                yield {"type": "token", "content": cleaned_pref}
                chunks.append(cleaned_pref)
        llm_sec = time.perf_counter() - t_llm
        ttft_sec = (t_first - t_llm) if t_first is not None else llm_sec
        answer = "".join(chunks).strip()
        total_sec = time.perf_counter() - began
        yield {"type": "sources", "sources": [{"source": rec["source"], "page": rec.get("page")} for rec in selected]}
        yield {"type": "done", "timings": {"search": round(search_sec, 2), "ttft": round(ttft_sec, 2), "llm": round(llm_sec, 2), "total": round(total_sec, 2)}}
        self.last_topic = topic
        if person:
            self.last_person = person
        if year:
            self.last_year = year
        self.last_question = effective_question
        self.history.append((effective_question, answer))
        self.history = self.history[-4:]

    def ask(self, question):
        for ev in self.ask_stream(question):
            if ev["type"] == "token":
                print(ev["content"], end="", flush=True)
            elif ev["type"] == "sources":
                if ev.get("sources"):
                    print("\n\nเอกสารที่ค้นพบ:")
                    for i, rec in enumerate(ev["sources"], 1):
                        pg = f" หน้า {rec['page']}" if rec.get("page") is not None else ""
                        print(f" [{i}] {rec['source']}{pg}")
            elif ev["type"] == "done":
                t = ev.get("timings", {})
                print(f"[Time] ค้นหา {t.get('search', 0):.2f}s | คำแรก {t.get('ttft', 0):.2f}s | LLM {t.get('llm', 0):.2f}s | รวม {t.get('total', 0):.2f}s\n")

    def status(self):
        cats = Counter(r["meta"][0] for r in self.kb.records)
        print("SQLite:", self.kb.dbpath)
        print("Passages:", len(self.kb.records))
        print("Categories:", dict(sorted(cats.items())))
        print("Professor profiles:", list(self.kb.roster()))


def main():
    try:
        os.system("stty iutf8 2>/dev/null")
    except OSError:
        pass
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    parser = argparse.ArgumentParser(description="AAI RAG v3.1 dynamic conversational chatbot")
    parser.add_argument("mode", nargs="?", default="chat", choices=("chat", "ask", "status"))
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()
    kb = Knowledge()
    chat = Session(kb)
    if args.mode == "status":
        chat.status()
        return
    if args.mode == "ask":
        if not args.question:
            parser.error("ต้องระบุคำถาม")
        chat.ask(" ".join(args.question))
        return

    print("AAI Conversational RAG v3.1 (พิมพ์ 'exit' เพื่อออก, 'reset' เพื่อเริ่มใหม่, 'status' เพื่อดูสถานะ)")
    while True:
        try:
            raw_q = input("\nคุณ: ")
            question = normalize_thai_query(raw_q)
        except UnicodeDecodeError:
            question = safe_input("")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit", "ออก"):
            break
        if question.lower() == "reset":
            chat = Session(kb)
            print("เริ่มบทสนทนาใหม่แล้ว")
            continue
        if question.lower() == "status":
            chat.status()
            continue
        if not question:
            continue
        try:
            chat.ask(question)
        except (RuntimeError, OSError, sqlite3.Error, ValueError) as exc:
            print("[ERROR]", exc)


if __name__ == "__main__":
    main()
