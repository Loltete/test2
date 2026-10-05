import glob
import os
import re

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pythainlp.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

TOPIC = "ไกด์ท่องเที่ยวจังหวัดปราจีนบุรี"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # เล็ก รองรับไทย/อังกฤษ
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 500     # ตัวอักษรต่อ chunk
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_SCORE = 0.30     # ต่ำกว่านี้ถือว่าไม่เกี่ยวข้อง

st.set_page_config(page_title=TOPIC, page_icon="🧭")


# ---------- 1) Document loading & chunking ----------
def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str):
    sents = sent_tokenize(text, engine="whitespace+newline")
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > CHUNK_SIZE and cur:
            chunks.append(cur.strip())
            cur = cur[-CHUNK_OVERLAP:] + " " + s
        else:
            cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ---------- 2) Embedding & vector index (โหลดครั้งเดียว) ----------
@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้างดัชนี...")
def build_index():
    model = SentenceTransformer(EMBED_MODEL)
    docs = []
    for path in sorted(glob.glob("data/*.txt")):
        with open(path, encoding="utf-8") as f:
            text = clean(f.read())
        for i, c in enumerate(chunk_text(text)):
            docs.append({"source": os.path.basename(path), "chunk_id": i + 1, "text": c})
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.array(emb, dtype="float32"))
    return model, index, docs


def retrieve(query, model, index, docs):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), TOP_K)
    return [(docs[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


# ---------- 3) Prompt engineering ----------
SYSTEM_PROMPT = f"""คุณคือ "{TOPIC}" ตอบเป็นภาษาเดียวกับผู้ใช้
กฎ:
1. ตอบจาก [บริบท] ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดาเอง
2. ทุกประโยคสำคัญต้องอ้างอิงแหล่งที่มาในรูปแบบ [ชื่อไฟล์#chunk]
3. ถ้าบริบทไม่มีคำตอบ ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร" เท่านั้น
4. ตอบกระชับ ชัดเจน"""


def build_messages(query, hits, history):
    ctx = "\n\n".join(
        f"[{d['source']}#{d['chunk_id']}]\n{d['text']}" for d, _ in hits
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += history[-6:]  # คุยต่อเนื่อง
    msgs.append({"role": "user", "content": f"[บริบท]\n{ctx}\n\n[คำถาม]\n{query}"})
    return msgs


# ---------- 4) LLM ----------
def ask_llm(messages):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.1
    )
    return r.choices[0].message.content


# ---------- 5) Chat UI ----------
st.title(f"🧭 {TOPIC}")
st.caption("ถามเกี่ยวกับสถานที่ท่องเที่ยว ของดี เวลาเปิด-ปิด ตอบจากเอกสารที่รวบรวมไว้เท่านั้น")

model, index, docs = build_index()

with st.sidebar:
    st.subheader("ตัวอย่างคำถาม")
    st.markdown("- (ใส่ตัวอย่างคำถามของคุณ)\n- (ใส่ตัวอย่างคำถามของคุณ)")
    if st.button("ล้างการสนทนา"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("refs"):
            with st.expander("📚 เอกสารอ้างอิง"):
                for r in m["refs"]:
                    st.markdown(f"**{r['source']}#{r['chunk_id']}** (score {r['score']:.2f})")
                    st.write(r["text"])

if query := st.chat_input("พิมพ์คำถามที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    hits = [(d, s) for d, s in retrieve(query, model, index, docs) if s >= MIN_SCORE]
    with st.chat_message("assistant"):
        if not hits:
            answer, refs = "ไม่พบข้อมูลในเอกสาร", []
        else:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                answer = ask_llm(build_messages(query, hits, history))
            except Exception as e:
                answer = f"เกิดข้อผิดพลาดในการเรียก LLM: {e}"
            refs = [{**d, "score": s} for d, s in hits]
        st.markdown(answer)
        if refs:
            with st.expander("📚 เอกสารอ้างอิง"):
                for r in refs:
                    st.markdown(f"**{r['source']}#{r['chunk_id']}** (score {r['score']:.2f})")
                    st.write(r["text"])
    st.session_state.messages.append({"role": "assistant", "content": answer, "refs": refs})
