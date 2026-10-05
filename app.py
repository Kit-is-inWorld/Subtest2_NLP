import os
import glob
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# 1. ตั้งค่าการแสดงผล Streamlit
st.set_page_config(page_title="ผู้ช่วยตอบคำถามระเบียบนักศึกษา", page_icon="🎓", layout="centered")

st.title("🎓 ผู้ช่วยตอบคำถามระเบียบและข้อบังคับนักศึกษา")
st.caption("ระบบตอบคำถามอัตโนมัติจากคลังเอกสารระเบียบการศึกษาด้วยเทคนิค RAG")

# 2. อ่านค่า API Key จาก Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ ไม่พบ GROQ_API_KEY ใน Secrets กรุณาตั้งค่าใน Streamlit Cloud Secrets ก่อนใช้งาน")
    st.stop()

# สร้าง Client
client = Groq(api_key=groq_api_key)

# ฟังก์ชัน Auto-Detect เฉพาะ Chat Model ที่ใช้งานได้จริง
@st.cache_resource
def get_available_model():
    try:
        models_page = client.models.list()
        # ดึงเฉพาะโมเดลที่เป็น Chat Model และไม่ใช่ Guard/Whisper/Embedding
        active_chat_models = [
            m.id for m in models_page.data 
            if not any(x in m.id.lower() for x in ["guard", "whisper", "embed", "vision"])
        ]
        
        # รายชื่อ Chat Models ยอดนิยมเรียงตามลำดับความต้องการ
        preferences = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
            "qwen-2.5-32b"
        ]
        
        # 1. เช็กตาม Priority List ก่อน
        for model in preferences:
            if model in active_chat_models:
                return model
                
        # 2. ถ้าไม่ตรงเลย ให้เอา Chat Model ตัวแรกที่พบ
        if active_chat_models:
            return active_chat_models[0]
            
    except Exception as e:
        pass
    
    # สำรองกรณีดึงรายการไม่สำเร็จ
    return "llama-3.1-8b-instant"

AVAILABLE_MODEL = get_available_model()

# 3. โหลดและสร้าง Vector Database (ใช้ Cache เพื่อความรวดเร็ว)
@st.cache_resource
def load_vector_database():
    file_paths = glob.glob("data/*.txt")
    if not file_paths:
        st.error("❌ ไม่พบไฟล์เอกสารในโฟลเดอร์ data/")
        st.stop()

    documents = []
    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

    # Chunking เอกสาร
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Embedding model ภาษาไทย/อังกฤษ
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # สร้าง FAISS Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

with st.spinner("⏳ กำลังเตรียมคลังข้อมูลระเบียบนักศึกษา..."):
    vector_store = load_vector_database()

# 4. ระบบจัดการ Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"สวัสดีครับ มีข้อสงสัยเกี่ยวกับระเบียบการศึกษา การลงทะเบียน หรือเกณฑ์การวัดผล สอบถามได้เลยครับ! (กำลังใช้งานโมเดล: `{AVAILABLE_MODEL}`)"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. รับคำถามจากผู้ใช้
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่ (เช่น เกรดเท่าไหร่ถูกรีไทร์, ถอนวิชาเรียนทำอย่างไร)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ค้นหา Chunk ที่เกี่ยวข้องที่สุด 3 อัน
    relevant_docs = vector_store.similarity_search(prompt, k=3)
    
    context_text = "\n\n".join([
        f"[เอกสารอ้างอิง: {os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}" 
        for doc in relevant_docs
    ])

    # System Prompt สำหรับบังคับให้ตอบเฉพาะใน Context
    system_prompt = f"""คุณคือผู้ช่วยตอบคำถามอัจฉริยะเกี่ยวกับระเบียบและข้อบังคับนักศึกษา ตอบคำถามโดยใช้ข้อมูลจาก Context ที่กำหนดให้เท่านั้น 

เงื่อนไขการตอบ:
1. ให้ตอบเฉพาะข้อมูลที่มีอยู่ใน Context เท่านั้น
2. หากใน Context ไม่มีข้อมูลที่สามารถตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูลในระบบ" ห้ามคาดเดาหรือคิดคำตอบเองเด็ดขาด
3. ตอบด้วยภาษาไทยที่สุภาพ เรียบร้อย และเข้าใจง่าย

Context:
{context_text}
"""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # เรียกใช้ API ด้วยโมเดลที่ Auto Detect ได้
            completion = client.chat.completions.create(
                model=AVAILABLE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            answer = completion.choices[0].message.content

            # ปรับแต่งคำตอบพร้อมแสดง Reference
            full_response = f"{answer}\n\n---\n**📚 แหล่งข้อมูลอ้างอิงที่ค้นพบ:**\n"
            for i, doc in enumerate(relevant_docs, 1):
                file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                snippet = doc.page_content.replace("\n", " ")[:120]
                full_response += f"\n- **[{i}] `{file_name}`**: *\"{snippet}...\"*\n"

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI API ({AVAILABLE_MODEL}): {str(e)}")