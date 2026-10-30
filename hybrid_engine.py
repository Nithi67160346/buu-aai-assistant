# -*- coding: utf-8 -*-
"""
hybrid_engine.py
Hybrid Fact-Card RAG Engine for BUU AAI on Raspberry Pi 4.
- Tier 1 & 2: Instant Grounded Fact Cards via Precomputed Vector Cosine Similarity (sim >= 0.58)
- Tier 3: Polite In-Domain Decline / General Knowledge Chat (sim < 0.58)
"""

import json
import time
import re
import threading
from collections import OrderedDict
from pathlib import Path
import requests
import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "embeddinggemma:300m-qat-q4_0"
LLM_MODEL = "aai-qwen-1.7b"
SIM_THRESHOLD = 0.58
# Between SIM_THRESHOLD and this, a card is only used if the query contains one of its keywords
# (stops broad cards like domain_difficulty catching general questions such as study tips)
GRAY_ZONE_HIGH = 0.70
LLM_KEEP_ALIVE = "30m"
# While the LLM is generating it uses all 4 cores and embeddings slow down a lot,
# so wait longer and fall back to keywords instead of wrongly declining
EMBED_TIMEOUT = 60
EMBED_CACHE_SIZE = 2000
FALLBACK_STOPWORDS = {"คืออะไร", "อะไร", "หลักสูตร", "คณะ", "สาขา", "aai", "ที่ไหน", "เท่าไหร่", "ไหม", "ยังไง"}
# Ollama answers one LLM request at a time; queue a few, tell the rest to retry
LLM_MAX_WAITING = 2
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
LLM_QUEUE_TIMEOUT = 150  # 2 queued answers x ~50s worst case
BUSY_ANSWER = (
    "ขออภัยครับ ขณะนี้มีผู้ใช้ถามคำถามทั่วไปพร้อมกันหลายคน ระบบตอบได้ทีละคำถาม "
    "กรุณารอสักครู่แล้วถามใหม่อีกครั้งครับ\n"
    "(คำถามเกี่ยวกับหลักสูตร AAI เช่น ค่าเทอม การรับสมัคร TCAS หรืออาจารย์ ยังถามได้ตามปกติครับ)"
)
EMBED_DOWN_ANSWER = (
    "ขออภัยครับ ขณะนี้ระบบค้นหาข้อมูลทำงานช้ากว่าปกติ ทำให้ยังหาคำตอบให้ไม่ได้ "
    "กรุณาลองถามใหม่อีกครั้งในอีกสักครู่ครับ"
)

PROF_NAMES = {
    "ponlawat": ("พลวัต",),
    "nilrat": ("นิลรัตน์", "นิลลัด"),
    "krisana": ("กฤษณะ",),
    "prawit": ("ประวิทย์",),
    "supawadee": ("สุภาวดี",),
    "wyookwan": ("วัชรพงศ์",),
}
# Rooms verified with the faculty on 30 Oct 2026
PROF_ROOM_QUERY = {
    "ponlawat": "ผศ.ดร.พลวัต ช่อผูก ห้องทำงาน IF-913F ชั้น 9",
    "nilrat": "อาจารย์นิลรัตน์ ก้านหยั่นทอง ห้องทำงาน IF-912F ชั้น 9",
    "krisana": "รศ.ดร.กฤษณะ ชินสาร ห้องทำงาน IF-809 ชั้น 8 KST Lab FI-8R06",
    "prawit": "อาจารย์ประวิทย์ บุญมี ห้องทำงาน IF-912C ชั้น 9",
    "supawadee": "ผศ.ดร.สุภาวดี ศรีคำดี ห้องทำงาน IF-911C ชั้น 9",
    "wyookwan": "ดร.วัชรพงศ์ อยู่ขวัญ ห้องทำงาน IF-909E ชั้น 9 KST Lab",
}
PROF_WORK_QUERY = {
    "ponlawat": "ผลงานวิจัยและความเชี่ยวชาญ ผศ.ดร.พลวัต ช่อผูก Lumbar spine AI แขนกล LLM",
    "nilrat": "ผลงานวิจัยและความเชี่ยวชาญ อาจารย์นิลรัตน์ ก้านหยั่นทอง Bone fracture X-ray SEM-EDX",
    "krisana": "ผลงานวิจัยและความเชี่ยวชาญ รศ.ดร.กฤษณะ ชินสาร ThaID 50 papers KST Lab",
    "prawit": "ผลงานวิจัยและความเชี่ยวชาญ อาจารย์ประวิทย์ บุญมี Enterprise Software ตรวจจับขโมย",
    "supawadee": "ผลงานวิจัยและความเชี่ยวชาญ ผศ.ดร.สุภาวดี ศรีคำดี Network Security WSN Smart City",
    "wyookwan": "ผลงานวิจัยและความเชี่ยวชาญ ดร.วัชรพงศ์ อยู่ขวัญ Data Mining หุ่นยนต์อัตโนมัติ",
}


def find_professor(q_low, last_q, last_a):
    """Which professor a follow-up refers to: the current question first, then the previous question,
    then the previous answer only if it names exactly one professor (lists of all six name everyone)."""
    for text in (q_low, last_q or ""):
        hits = [p for p, names in PROF_NAMES.items() if any(n in text for n in names)]
        if hits:
            return hits[0]
    hits = [p for p, names in PROF_NAMES.items() if last_a and any(n in last_a for n in names)]
    return hits[0] if len(hits) == 1 else None


class HybridEngine:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "fact_cards_embedded.json"
        self.data_path = Path(data_path)
        self.cards = []
        self.card_by_id = {}
        self.card_keywords = {}
        self.matrix = None
        self.vector_cards = []
        self.embed_cache = OrderedDict()
        self.embed_lock = threading.Lock()
        self.llm_sem = threading.Semaphore(1)
        self.llm_waiting = 0
        self.llm_lock = threading.Lock()
        self.load_cards()
        self.load_keywords()

    def load_keywords(self):
        # Keywords live only in fact_cards.json (the embedded file drops them)
        kw_path = self.data_path.parent / "fact_cards.json"
        if not kw_path.is_file():
            print(f"[HybridEngine] {kw_path} not found, gray-zone keyword check disabled")
            return
        with open(kw_path, "r", encoding="utf-8") as f:
            for card in json.load(f):
                self.card_keywords[card["id"]] = [k.lower() for k in card.get("keywords", [])]

    def accept_card(self, card, sim, effective_q, rewritten=False):
        if sim >= GRAY_ZONE_HIGH:
            return True
        if sim < SIM_THRESHOLD:
            return False
        # A query rewritten by contextualize_query already names its topic, so trust it as before
        if rewritten or not self.card_keywords:
            return True
        q_low = effective_q.lower()
        return any(k in q_low for k in self.card_keywords.get(card["id"], []))

    def load_cards(self):
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Fact cards not found at {self.data_path}")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.cards = json.load(f)
            
        self.card_by_id = {c["id"]: c for c in self.cards}
        
        vectors = []
        self.vector_cards = []
        for card in self.cards:
            cid = card["id"]
            if "card_embedding" in card and card["card_embedding"]:
                cvec = np.array(card["card_embedding"], dtype=np.float32)
                norm = np.linalg.norm(cvec)
                if norm > 0:
                    vectors.append(cvec / norm)
                    self.vector_cards.append((cid, "card"))
            
            for q_obj in card.get("query_embeddings", []):
                qvec = np.array(q_obj["embedding"], dtype=np.float32)
                norm = np.linalg.norm(qvec)
                if norm > 0:
                    vectors.append(qvec / norm)
                    self.vector_cards.append((cid, q_obj["query"]))
                    # Sample queries were embedded with the same model, so they pre-fill the cache
                    self.embed_cache[q_obj["query"]] = q_obj["embedding"]
                    
        if vectors:
            self.matrix = np.vstack(vectors)
        print(f"[HybridEngine] Loaded {len(self.cards)} cards, vector matrix shape: {self.matrix.shape if self.matrix is not None else None}")

    def get_embedding(self, text):
        with self.embed_lock:
            if text in self.embed_cache:
                self.embed_cache.move_to_end(text)
                return self.embed_cache[text]
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text, "keep_alive": LLM_KEEP_ALIVE}, timeout=EMBED_TIMEOUT)
            if resp.status_code == 200:
                emb = resp.json().get("embedding", [])
                if emb:
                    with self.embed_lock:
                        self.embed_cache[text] = emb
                        while len(self.embed_cache) > EMBED_CACHE_SIZE:
                            self.embed_cache.popitem(last=False)
                return emb
        except Exception as e:
            print(f"[HybridEngine] Embedding error: {e}")
        return []

    def keyword_match(self, effective_q):
        # Fallback when the embedding call fails: the card whose keywords cover most of the query.
        # Generic words ("คืออะไร", "หลักสูตร") and 2-letter codes ("it", "se") match almost anything, so skip them.
        q_low = effective_q.lower()
        best, best_score = None, 0
        for cid, kws in self.card_keywords.items():
            score = sum(len(k) for k in kws
                        if k and k in q_low and k not in FALLBACK_STOPWORDS and not (k.isascii() and len(k) < 3))
            if score > best_score and cid in self.card_by_id:
                best, best_score = self.card_by_id[cid], score
        return best

    def contextualize_query(self, question, session=None):
        q_strip = question.strip()
        q_low = q_strip.lower()

        # Standalone Rule 1: User complaints/frustrations - do not contaminate with previous question
        if any(w in q_low for w in ("ทำไมชอบผิด", "ทำไมตอบผิด", "ตอบผิด", "ตอบมั่ว", "ชอบมั่ว", "ตอบไม่ตรง", "มั่วตลอด", "ยังไม่ได้ถาม")):
            return "ทำไมชอบตอบมั่ว ตอบผิด ไม่ตรงคำถาม"

        # Standalone Rule 2: Bot capability & personality questions
        if any(w in q_low for w in ("เก่งมั้ย", "คุณเก่งมั้ย", "ตัวของคุณเองเก่งมั้ย", "คุณฉลาดมั้ย", "คุณหิวมั้ย", "หิวมั้ย")):
            return "คุณเก่งมั้ย มีความสามารถอะไรบ้าง คุณหิวมั้ย"

        # Standalone Rule 3: TCAS Round 1 ("รอบ 1", "รอบ 1 หละ", "รอบ 1 ล่ะ")
        if any(w in q_low for w in ("รอบ 1", "รอบหนึ่ง", "แล้วรอบ 1", "แล้วรอบหนึ่ง", "รอบ 1 หละ", "รอบ 1 ล่ะ", "พอร์ต", "portfolio")) and not any(k in q_low for k in ("aai", "เกณฑ์", "tcas")) and not any(x in q_low for x in ("วิชา", "จุดเด่น", "ปี 1")):
            return "เกณฑ์ TCAS รอบ 1 Portfolio สาขาวิชา AAI มหาวิทยาลัยบูรพา"

        # Standalone Rule 4: TCAS Round 3 ("รอบ 3", "แล้วรอบ 3", "รอบ 3 หละ")
        if any(w in q_low for w in ("รอบ 3", "รอบสาม", "แล้วรอบ 3", "แล้วรอบสาม", "รอบ 3 หละ", "รอบ 3 ล่ะ", "admission", "แอดมิชชั่น")) and not any(k in q_low for k in ("aai", "เกณฑ์", "tcas")):
            return "เกณฑ์ TCAS รอบ 3 Admission สาขา AAI มหาวิทยาลัยบูรพา"

            # Standalone Rule: Canteen / Food around IF
        if any(w in q_low for w in ("ตึก if มีอะไรกิน", "ของกินตึก if", "โรงอาหารตึก if", "โรงอาหารใต้ตึก if", "ใต้ตึก if มีของกินไหม", "แถวตึก if มีอะไรกิน", "ของกินแถวตึก if", "ร้านอาหารตึก if", "ตึก if มีโรงอาหารไหม", "ตึก if มีของกินอะไรบ้าง", "ใต้ตึก if มีอะไรกิน", "ของกินใกล้ตึก if")):
            return "แนะนำของกิน ร้านอาหาร และโรงอาหารรอบ ม.บูรพา ตึก IF"

        # Standalone Rule: Faculty History & Founding
        if any(w in q_low for w in ("คณะก่อตั้งเมื่อไหร่", "ก่อตั้งเมื่อไหร่", "คณะก่อตั้ง", "ประวัติคณะ", "ปีที่ก่อตั้ง", "วันก่อตั้งคณะ", "ก่อตั้งปีไหน", "ประวัติความเป็นมาคณะ")):
            return "ประวัติความเป็นมาและการก่อตั้งคณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา 9 ธันวาคม 2552"

    # Standalone Rule 5: Cooperative education criteria & preparation hours (42 hours)
        if any(w in q_low for w in ("เกณฑ์ฝึกสหกิจ", "เกณฑ์สหกิจ", "เกณฑ์การฝึกสหกิจ", "เกณฑ์การออกสหกิจ", "เกณฑ์ออกสหกิจ", "เกณฑ์การเก็บชั่วโมงสหกิจ", "เกณฑ์ชั่วโมงสหกิจ", "เก็บชั่วโมงสหกิจ", "ชั่วโมงสหกิจ", "สหกิจ 42", "42 ชม. สหกิจ", "42 ชั่วโมง")) or ("เกณฑ์" in q_low and any(k in q_low for k in ("สหกิจ", "coop", "co-op", "ฝึกงาน"))) or ("ชั่วโมง" in q_low and any(k in q_low for k in ("สหกิจ", "coop", "co-op"))):
            return "การเก็บชั่วโมงเตรียมความพร้อมก่อนออกสหกิจศึกษา 42 ชั่วโมง เกณฑ์ฝึกสหกิจ"

        # Check session history for conversational follow-ups
        if not session or not getattr(session, "history", None):
            return q_strip

        last_q, last_a = session.history[-1]
        last_q_low = last_q.lower()

        # Rule 5: Numbered follow-up e.g. "3 น่าสนใจนะคืออะไรหรอ", "ข้อ 3 คืออะไร", "3 คืออะไร"
        num_m = re.search(r'(?:^|[^\d])([1-9])(?:\s*(?:น่าสนใจ|คืออะไร|คือ|หมายถึงอะไร|แนะนำหน่อย|\?|$))', q_low)
        if num_m and last_a:
            num = num_m.group(1)
            for line in last_a.splitlines():
                line_str = line.strip()
                if line_str.startswith(f"{num}.") or line_str.startswith(f"{num})") or line_str.startswith(f"ข้อ {num}"):
                    topic = re.sub(r'^\d+[\.\)]\s*', '', line_str)
                    topic = re.sub(r'^\*\*(.*?)\*\*.*', r'\1', topic)
                    return f"{topic} คืออะไร รายละเอียด"

        # Rule 6: Coop 42 hours / preparation follow-up: "ต้องเก็บยังไง", "เก็บยังไง", "เก็บกี่ชั่วโมง"
        if any(w in q_low for w in ("ต้องเก็บยังไง", "เก็บยังไง", "เก็บชั่วโมง", "เก็บกี่ชั่วโมง", "ชั่วโมงเตรียมความพร้อม")):
            if "สหกิจ" in last_q or "สหกิจ" in last_a or "coop" in last_q_low or "co-op" in last_q_low:
                return "การเตรียมความพร้อมก่อนไปสหกิจศึกษา AAI ต้องเก็บชั่วโมงอบรมกี่ชั่วโมง 42 ชั่วโมง"

    # Rule 7: "ห้องทำงานอยู่ไหน" or "พักห้องไหน" after professor mention
        if any(w in q_low for w in ("ห้องทำงานอยู่ไหน", "ห้องทำงานอยู่ที่ไหน", "พักห้องไหน", "อยู่ห้องไหน", "ห้องพักอยู่ที่ไหน", "ห้องพักอาจารย์", "ห้องทำงาน")):
            prof = find_professor(q_low, last_q, last_a)
            if prof:
                return PROF_ROOM_QUERY[prof]
            return f"{last_q} ห้องทำงานอยู่ที่ไหน"

        # Rule 7B: Professor Research, Works, and Specialization
        if any(w in q_low for w in ("ผลงาน", "วิจัย", "งานวิจัย", "ความเชี่ยวชาญ", "เก่งด้านไหน", "ทำวิจัยอะไร", "ผลงานเด่น", "มีผลงานอะไรบ้าง", "ผลงานล่ะ")):
            prof = find_professor(q_low, last_q, last_a)
            if prof:
                return PROF_WORK_QUERY[prof]
            return "ผลงานวิจัยและความเชี่ยวชาญเด่นของคณาจารย์ประจำสาขาวิชา AAI"

        # Rule 8: Highlights follow-up
        if re.search(r"^(?:ข้อ|อัน|ลำดับ)?\s*1\s*(?:คืออะไร|คือ|หมายถึงอะไร|\?|$)", q_low):
            if "จุดเด่น" in last_q or "จุดเด่น" in last_a:
                return "จุดเด่นข้อที่ 1 หลักสูตร AAI Project-Based Learning"
            elif "ปี 1" in last_q or "เทอม 1" in last_q:
                return "วิชาข้อที่ 1 ปี 1 เทอม 1 AAI การโปรแกรมคอมพิวเตอร์เบื้องต้น"

        # Rule 9: Subjects follow-up
        if re.search(r"^(?:ข้อ|วิชาที่|อันดับ|ลำดับ)?\s*2\s*(?:เรียนอะไร|คืออะไร|คือ|\?|$)", q_low):
            if "ปี 1" in last_q or "เทอม 1" in last_q or "รายวิชา" in last_q or "วิชา" in last_a:
                return "วิชาข้อที่ 2 ปี 1 เทอม 1 AAI แคลคูลัสสำหรับวิทยาการข้อมูล Calculus for Data Science"

        # Rule 10: "แล้ว ป.เอก ล่ะ"
        if any(w in q_low for w in ("ป.เอก", "ปริญญาเอก")):
            if "data science" in last_q_low or "วิทยาการข้อมูล" in last_q:
                return "หลักสูตรปริญญาเอก Data Science and Advanced Analytics Ph.D. วท.ด."

        # Rule 11: "จบแล้วได้วุฒิอะไร", "ได้วุฒิอะไร"
        if any(w in q_low for w in ("ได้วุฒิอะไร", "วุฒิอะไร", "วุฒิการศึกษา", "จบแล้วได้วุฒิ", "ชื่อปริญญา")):
            return "สำเร็จการศึกษาหลักสูตร AAI ได้รับวุฒิการศึกษาอะไร วท.บ."

        # General short follow-up: only if the question is an incomplete question/reference
        # and NOT a standalone question containing explicit entities/topics:
        standalone_indicators = (
            "cs", "se", "itdi", "aai", "สหกิจ", "coop", "co-op", "ที่ไหน", "ที่นี่",
            "ตึก", "อาคาร", "ก่อตั้ง", "ประวัติ", "ค่าเทอม", "ค่าใช้จ่าย", "หลักสูตร",
            "ปริญญา", "เกณฑ์", "สาขา", "ชั่วโมง", "นั่งชิล", "พักผ่อน", "กินไรดี",
            "ร้านอาหาร", "สวัสดี", "ขอบคุณ", "ใคร", "ชื่อ", "วันอะไร", "เก่ง", "หิว",
            "ผิด", "รอบ", "เข้า", "อยากเข้า", "สมัคร", "tcas", "ผลงาน", "วิจัย", "ประวัติ", "ก่อตั้ง"
        )
        if len(q_strip) <= 15 and not any(x in q_low for x in standalone_indicators):
            return f"{last_q} {q_strip}"

        return q_strip

    def match(self, effective_question):
        raw_emb = self.get_embedding(effective_question)
        if not raw_emb or self.matrix is None:
            # -1 marks "embedding unavailable" so stream_reply can fall back instead of declining
            return None, -1.0, None, 0.0
            
        q_vec = np.array(raw_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None, 0.0, None, 0.0
            
        q_unit = q_vec / q_norm
        sims = np.dot(self.matrix, q_unit)
        
        card_max_sim = {}
        for idx, sim_val in enumerate(sims):
            cid, _ = self.vector_cards[idx]
            if cid not in card_max_sim or sim_val > card_max_sim[cid]:
                card_max_sim[cid] = sim_val
                
        sorted_cards = sorted(card_max_sim.items(), key=lambda x: x[1], reverse=True)
        if not sorted_cards:
            return None, 0.0, None, 0.0
            
        top_cid, top_sim = sorted_cards[0]
        top_card = self.card_by_id.get(top_cid)
        
        sec_card, sec_sim = None, 0.0
        if len(sorted_cards) > 1:
            sec_cid, sec_sim = sorted_cards[1]
            sec_card = self.card_by_id.get(sec_cid)
            
        return top_card, float(top_sim), sec_card, float(sec_sim)

    def notice_reply(self, answer, t_begin, t_search):
        # System notices (busy / search down) are not added to history so follow-ups are unaffected
        yield {"type": "token", "content": answer}
        yield {"type": "sources", "sources": []}
        t_total = time.perf_counter() - t_begin
        yield {"type": "done", "timings": {"search": round(t_search, 3), "ttft": round(t_total, 3), "llm": 0.01, "total": round(t_total, 3)}}

    def stream_reply(self, question, session=None):
        t_begin = time.perf_counter()
        effective_q = self.contextualize_query(question, session)

        top_card, top_sim, sec_card, sec_sim = self.match(effective_q)
        t_search = time.perf_counter() - t_begin
        rewritten = effective_q != question.strip()
        embed_failed = top_sim < 0
        if embed_failed:
            kw_card = self.keyword_match(effective_q)
            print(f"[HybridEngine] Embedding unavailable, keyword fallback -> {kw_card['id'] if kw_card else None}")
            if kw_card:
                top_card, top_sim, sec_card, sec_sim = kw_card, 1.0, None, 0.0

        # Anti-looping check: if consecutively matching exact same card and sec_card is viable, switch.
        # Not when a follow-up rule already resolved the professor: the second card would be another person.
        resolved_prof = effective_q in PROF_ROOM_QUERY.values() or effective_q in PROF_WORK_QUERY.values()
        if session and top_card and not resolved_prof and getattr(session, 'last_card_id', None) == top_card['id']:
            if sec_card and sec_sim >= 0.52 and sec_card['id'] != top_card['id']:
                top_card = sec_card
                top_sim = sec_sim

        # Tier 1 & 2: Instant Grounded Fact Card (sim >= 0.70, or >= 0.58 with a keyword hit)
        if top_card and self.accept_card(top_card, top_sim, effective_q, rewritten):
            answer = top_card["canned_answer"]
            sources = top_card.get("sources", [])
            
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": sources}
            t_total = time.perf_counter() - t_begin
            yield {
                "type": "done",
                "timings": {
                    "search": round(t_search, 3),
                    "ttft": round(t_search, 3),
                    "llm": 0.01,
                    "total": round(t_total, 3)
                }
            }
            
            if session:
                if not hasattr(session, "history"):
                    session.history = []
                session.history.append((question, answer))
                session.history = session.history[-6:]
                session.last_topic = top_card.get("category", "")
                session.last_question = question
            session.last_card_id = top_card['id']
            return

        # Tier 3: Low confidence (< 0.58) or Out-of-Domain
        q_low = effective_q.lower()
        is_domain_inquiry = any(k in q_low for k in (
            "คณะ", "หลักสูตร", "สาขา", "aai", "ม.บูรพา", "อาจารย์", "มหาวิทยาลัย", 
            "เรียนที่นี่", "นิสิต", "ค่าเทอม", "ค่าธรรมเนียม", "tcas", "รอบ 1", "รอบ 2", 
            "รอบ 3", "พอร์ต", "สหกิจ", "coop"
        ))
        # Writing tasks ("ช่วยเขียนแนะนำตัวสัมภาษณ์เข้ามหาวิทยาลัย") mention the university but ask for no facts
        if any(k in q_low for k in ("ช่วยเขียน", "เขียนให้", "แต่งกลอน", "แต่งเพลง", "แต่งเรื่อง", "ช่วยร่าง", "ร่างจดหมาย")):
            is_domain_inquiry = False

        # Without an embedding we cannot tell "no such data" from "search failed", so say so honestly
        if is_domain_inquiry and embed_failed:
            yield from self.notice_reply(EMBED_DOWN_ANSWER, t_begin, t_search)
            return

        if is_domain_inquiry:
            decline_ans = (
                "ขออภัยครับ ข้อมูลดังกล่าวไม่ได้อยู่ในหลักสูตรหรือฐานข้อมูลของสาขาวิชา AAI คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา ครับ\n"
                "หากคุณมีข้อสงสัยเกี่ยวกับหลักสูตรปัญญาประดิษฐ์ประยุกต์ (AAI) ค่าธรรมเนียมการศึกษา แผนการเรียน หรือการรับสมัคร TCAS สามารถสอบถามได้เลยครับ"
            )
            yield {"type": "token", "content": decline_ans}
            yield {"type": "sources", "sources": []}
            t_total = time.perf_counter() - t_begin
            yield {
                "type": "done",
                "timings": {
                    "search": round(t_search, 3),
                    "ttft": round(t_search, 3),
                    "llm": 0.01,
                    "total": round(t_total, 3)
                }
            }
            if session:
                if not hasattr(session, "history"):
                    session.history = []
                session.history.append((question, decline_ans))
                session.history = session.history[-6:]
            session.last_card_id = None
            return

        # General Knowledge / Chit-chat query
        general_sys = ("คุณคือผู้ช่วย AI ตอบคำถามทั่วไปเป็นภาษาไทยเท่านั้น ตอบอย่างสุภาพและถูกต้อง "
                       "ตอบให้จบภายใน 2-3 ประโยคสั้นๆ ไม่ต้องใช้หัวข้อหรือรายการ")
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": general_sys},
                {"role": "user", "content": effective_q}
            ],
            "stream": True,
            "think": False,
            "keep_alive": LLM_KEEP_ALIVE,
            "options": {
                "temperature": 0.4,
                "num_ctx": 1024,
                "num_predict": 160
            }
        }
        
        # One LLM answer at a time; up to LLM_MAX_WAITING wait their turn, the rest get BUSY_ANSWER now
        with self.llm_lock:
            if self.llm_waiting >= LLM_MAX_WAITING + 1:
                queue_full = True
            else:
                queue_full = False
                self.llm_waiting += 1
        if queue_full:
            yield from self.notice_reply(BUSY_ANSWER, t_begin, t_search)
            return
        acquired = False
        try:
            acquired = self.llm_sem.acquire(timeout=LLM_QUEUE_TIMEOUT)
            if not acquired:
                yield from self.notice_reply(BUSY_ANSWER, t_begin, t_search)
                return

            t_llm = time.perf_counter()
            t_first = None
            collected = []
            try:
                with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        data = json.loads(line)
                        msg = data.get("message", {})
                        # qwen3 sometimes drifts into Chinese/Japanese; drop those characters
                        tok = CJK_RE.sub("", msg.get("content", ""))
                        if tok:
                            if t_first is None:
                                t_first = time.perf_counter()
                            collected.append(tok)
                            yield {"type": "token", "content": tok}
                        if data.get("done", False):
                            break
            except Exception as e:
                err_msg = f"ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล: {e}"
                yield {"type": "token", "content": err_msg}
                collected = [err_msg]
        finally:
            # Also runs when the browser disconnects mid-answer (GeneratorExit), so the slot is never lost
            if acquired:
                self.llm_sem.release()
            with self.llm_lock:
                self.llm_waiting -= 1
            
        full_ans = "".join(collected)
        yield {"type": "sources", "sources": []}
        t_end = time.perf_counter()
        ttft = (t_first - t_begin) if t_first else (t_end - t_begin)
        llm_sec = t_end - (t_first or t_llm)
        yield {
            "type": "done",
            "timings": {
                "search": round(t_search, 3),
                "ttft": round(ttft, 3),
                "llm": round(llm_sec, 3),
                "total": round(t_end - t_begin, 3)
            }
        }
        
        if session:
            if not hasattr(session, "history"):
                session.history = []
            session.history.append((question, full_ans))
            session.history = session.history[-6:]
