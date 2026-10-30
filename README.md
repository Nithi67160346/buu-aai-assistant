# BUU AAI Assistant

เว็บผู้ช่วยภาษาไทยสำหรับข้อมูลสาขาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (AAI)
มหาวิทยาลัยบูรพา ใช้ Flask, Ollama และฐานความรู้ในเครื่อง ออกแบบสำหรับ Raspberry Pi 4

นี่คือสำเนาโครงการสำหรับจัดเก็บใน GitHub ณ 30 ตุลาคม 2026
โค้ดการตอบคำถามและหน้าเว็บคงตามต้นฉบับบน Pi ไม่ได้เพิ่มเส้นทาง RAG ใหม่หรือแก้บั๊กการตอบคำถาม
โครงการนี้เป็นต้นแบบเพื่อการศึกษา ไม่ใช่บริการอย่างเป็นทางการของมหาวิทยาลัย

## การทำงานปัจจุบัน

```text
Browser → Flask /api/chat/stream → HybridEngine
                                  ├─ embedding → fact card → คำตอบสำเร็จรูป + sources
                                  ├─ คำถามสาขาที่ไม่มีหลักฐาน → แจ้งข้อมูลไม่เพียงพอ
                                  └─ คำถามทั่วไป → Ollama → stream คำตอบ
```

HybridEngine ใช้ 57 fact cards พร้อม embedding ที่เตรียมไว้
โค้ดเดิมใน `rag_pi4_chat_v3_1.py` มีเส้นทางค้น passages และสร้างคำตอบด้วย LLM
แต่เมื่อ HybridEngine โหลดสำเร็จ จะใช้ Hybrid เป็นเส้นทางหลัก
อ่านข้อจำกัดที่ทราบแล้วใน [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)

## โครงสร้าง

```text
app.py                      Flask API และหน้าเว็บ
hybrid_engine.py            ค้น fact cards, contextualize คำถาม, เรียก Ollama
rag_pi4_chat_v3_1.py         Knowledge, Session และเส้นทาง RAG เดิม
rag_pi4.py                  ตั้งค่าโมเดลและสร้าง SQLite index
rag_pi4_update.py           อัปเดตเอกสารแบบ incremental / OCR
crawl_aai.py                เก็บเอกสารจาก URL ที่กำหนดไว้
precompute_fact_cards.py    สร้าง embedding จาก fact_cards.json
templates/index.html       หน้าเว็บ
data/                      fact cards และ precomputed embeddings
docs_aai/                  เอกสารต้นทางของฐานความรู้
eval/                      ชุดคำถามและเครื่องมือประเมิน
scripts/check_project.py   ตรวจความครบถ้วนโดยไม่เรียก Ollama
docs/                      ข้อจำกัดและข้อมูลการจัดชุดโครงการ
```

## ติดตั้งบน Raspberry Pi / Linux

ต้องมี Python 3.10 ขึ้นไป และ Ollama ติดตั้งและเปิดทำงานไว้แล้ว
คำสั่งต่อไปนี้ให้รันจากรากโครงการ

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull embeddinggemma:300m-qat-q4_0
ollama list
```

โค้ดใช้ชื่อโมเดลสนทนา `aai-qwen-1.7b` ซึ่งเป็นโมเดลที่มีอยู่ในเครื่องเดิม
ไม่ใช่ชื่อที่รับประกันว่าจะดาวน์โหลดได้จาก registry
หากย้ายเครื่อง ต้องนำ weights ต้นฉบับมาเอง แล้วสร้างชื่อโมเดลเดิมด้วย
[Modelfile.example](Modelfile.example):

```bash
# วาง weights ที่ models/aai-qwen-1.7b.gguf ก่อนรัน
ollama create aai-qwen-1.7b -f Modelfile.example
ollama show aai-qwen-1.7b
```

ดูรายละเอียดและ SHA256 ของ weights ที่เครื่องเดิมใช้ใน [models/README.md](models/README.md)
ไม่มี weights รวมอยู่ใน repository และไม่ได้ใช้ Modelfile เก่าของโมเดล 0.5B

### สร้างฐานข้อมูลครั้งแรก

SQLite เป็นไฟล์ที่สร้างใหม่ได้ จึงไม่รวมอยู่ในชุด GitHub แต่ `Knowledge()` ต้องใช้ไฟล์นี้ตอนเปิดแอป
หลังติดตั้ง embedding model แล้ว ให้สร้าง index ก่อน:

```bash
python rag_pi4.py index ./docs_aai
python scripts/check_project.py
python app.py
```

เปิด `http://localhost:8080` หรือ `http://<IP-ของ-Pi>:8080`
เปลี่ยนพอร์ตได้ด้วย `PORT=8081 python app.py`

การ index ต้องเรียก Ollama และใช้เวลาตามจำนวนเอกสาร ไม่จำเป็นต้องทำทุกครั้งที่เปิดแอป
คำสั่ง `index` สร้าง chunks ใหม่แทนของเดิม จำนวน passages ที่สร้างใหม่อาจต่างจาก snapshot เดิม
เอกสารนามสกุล `.txt`, `.md`, `.pdf` จะถูก index; ข้อความที่ไม่มีนามสกุลใช้เป็น fallback ใน Knowledge
PDF แบบสแกนอาจต้อง OCR ก่อนจึงจะค้นจากข้อความได้

โค้ดเดิมเปิด Flask ที่ `0.0.0.0` และไม่มีระบบยืนยันตัวตน จึงควรใช้ในเครื่องหรือเครือข่ายที่ควบคุมได้
หน้าเว็บโหลด Google Fonts และ marked จาก CDN จึงยังต้องใช้อินเทอร์เน็ตสำหรับส่วนนี้

## อัปเดตข้อมูล

แก้ข้อมูลหลักที่ `data/fact_cards.json` แล้วสร้าง embeddings ใหม่:

```bash
python precompute_fact_cards.py
```

ต้องใช้ `embeddinggemma:300m-qat-q4_0` ให้ตรงกับ query embedding และ restart แอปหลังอัปเดต cards
ไฟล์ embedded ที่รวมไว้ช่วยให้ไม่ต้องคำนวณใหม่เมื่อเพิ่ง clone
ไม่รวมสคริปต์ patch หรือ generator รุ่นเก่าที่อาจเขียนทับข้อมูลที่แก้แล้ว

อัปเดตเอกสารลงฐานข้อมูลที่สร้างไว้แล้ว:

```bash
python rag_pi4_update.py ./docs_aai
```

หากต้องใช้ OCR บน Raspberry Pi OS ให้ติดตั้งเครื่องมือเพิ่มเติม:

```bash
sudo apt install ocrmypdf tesseract-ocr-tha tesseract-ocr-eng poppler-utils
```

การอัปเดตเอกสาร SQLite ไม่ได้ปรับ fact cards ให้อัตโนมัติ ต้องดูแลข้อมูลสองส่วนให้สอดคล้องกัน

## ตรวจและประเมิน

```bash
# ตรวจไฟล์, Python syntax, JSON และมิติ embedding โดยไม่เรียกเครือข่าย
python scripts/check_project.py

# ต้องเปิด app.py และ Ollama ก่อน ผลลัพธ์ถูก gitignore
python eval/run_eval.py
```

เครื่องมือ eval เดิมบันทึกคำตอบ เวลา และ sources ไม่ใช่คะแนนความถูกต้องอัตโนมัติ
ชุดคำถามบางส่วนซ้ำกับ sample queries ที่ cache ไว้ จึงต้องแยกการทดสอบคำถามใหม่เมื่อต้องการวัด latency จริง

## ข้อมูลและสิทธิ์การใช้งาน

เอกสารใน `docs_aai/` และข้อเท็จจริงที่อ้างถึงเป็นข้อมูลของแหล่งต้นทาง
รายการ URL ที่มีอยู่บันทึกใน `docs_aai/sources.json` และ `SOURCE_URL` ภายในเอกสารบางไฟล์
metadata อ้างอิงบางรายการยังไม่ตรงกับไฟล์ที่รวมไว้ทั้งหมด
ไม่ได้เพิ่มสัญญาอนุญาตให้เอกสาร โมเดล หรือโค้ดโดยพลการ; repository นี้ยังไม่มี LICENSE ที่เจ้าของเลือก

ข้อมูลการจัดชุดและไฟล์ที่ตัดออกดู [docs/PACKAGING.md](docs/PACKAGING.md)
