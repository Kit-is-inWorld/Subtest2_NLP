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

# ฟังก์ชันดึง Candidate Models ที่เป็น Chat Model เท่านั้น (ไม่ใช้ Cache เพื่อให้ข้อมูลสดใหม่ตลอด)
def get_candidate_models():
    try:
        models_page = client.models.list()
        active_models = [m.id for m in models_page.data]
        
        # กรองเอาเฉพาะ Chat Model ไม่เอา Guard/Whisper/Vision
        chat_models = [
            m for m in active_models 
            if not any(x in m.lower() for x in ["guard", "whisper", "embed", "vision"])
        ]
        
        # รายชื่อโมเดลคุณภาพสูงเรียงตามลำดับความต้องการ
        preferred = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "qwen-2.5-32b"
        ]
        
        ordered_models = [m for m in preferred if m in chat_models]
        # ใส่โมเดลที่เหลือต่อท้ายไว้กันพลาด
        for m in chat_models:
            if m not in ordered_models:
                ordered_models.append(m)
                
        return ordered_models if ordered_models else ["llama-3.3-70b-versatile"]
    except Exception:
        return ["llama-3.3-70b-versatile", "gemma2-9b-it"]

# 3. โหลดและสร้าง Vector Database (ใช้ Cache เฉพาะส่วน Document Loading & Embedding)
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

    # Chunking เอกสาร (ปรับขนาด 1000 เพื่อความครอบคลุมของเนื้อหา)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
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
        {"role": "assistant", "content": "สวัสดีครับ มีข้อสงสัยเกี่ยวกับระเบียบการศึกษา การลงทะเบียน หรือเกณฑ์การวัดผล สอบถามได้เลยครับ!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. รับคำถามจากผู้ใช้
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่ (เช่น เกรดเท่าไหร่ถูกรีไทร์, ถอนวิชาเรียนทำอย่างไร)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ค้นหา Chunk ที่เกี่ยวข้องที่สุด 4 อัน
    relevant_docs = vector_store.similarity_search(prompt, k=4)
    
    context_text = "\n\n".join([
        f"[เอกสารอ้างอิง: {os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}" 
        for doc in relevant_docs
    ])

    # System Prompt ที่ออกแบบเพื่อการประมวลผล RAG ภาษาไทย
    system_prompt = f"""คุณคือผู้ช่วยตอบคำถามอัจฉริยะเกี่ยวกับระเบียบและข้อบังคับนักศึกษา 
หน้าที่ของคุณคืออ่าน วิเคราะห์ และสรุปคำตอบจากข้อมูล "Context" ที่กำหนดให้ด้านล่างนี้เท่านั้น

คำสั่งในการตอบคำถาม:
1. วิเคราะห์ Context ทั้งหมด แล้วนำข้อมูลที่เกี่ยวข้องมาสังเคราะห์เป็นคำตอบที่ชัดเจน ตรงประเด็น และเข้าใจง่าย
2. หากมีข้อมูลเงื่อนไข ตัวเลข เกรดเฉลี่ย (GPAX) หรือขั้นตอน ให้ระบุให้ครบถ้วนถูกต้องตาม Context
3. ห้ามใช้ความรู้ภายนอกหรือคิดคำตอบขึ้นมาเองเด็ดขาด ให้ใช้เฉพาะข้อมูลที่ปรากฏใน Context เท่านั้น
4. หากใน Context ไม่มีข้อมูลที่เกี่ยวข้องกับคำถามเลย ให้ตอบเพียงว่า "ไม่พบข้อมูลในระบบ" เท่านั้น

Context สำหรับใช้ตอบคำถาม:
{context_text}
"""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # ระบบสลับโมเดลอัตโนมัติหากพบโมเดลพังหรือถูกถอดถอน
        candidate_models = get_candidate_models()
        answer = None
        used_model = None
        last_error = None

        for model_name in candidate_models:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                answer = completion.choices[0].message.content
                used_model = model_name
                break # หากสำเร็จ ให้หลุดออกจาก loop ทันที
            except Exception as e:
                last_error = e
                continue # หากโมเดลนี้ใช้ไม่ได้ ให้ข้ามไปลองโมเดลถัดไป

        if answer:
            # ปรับแต่งคำตอบพร้อมแสดง Reference
            full_response = f"{answer}\n\n---\n*(ประมวลผลด้วยโมเดล: `{used_model}`)*\n\n**📚 แหล่งข้อมูลอ้างอิงที่ค้นพบ:**\n"
            for i, doc in enumerate(relevant_docs, 1):
                file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                snippet = doc.page_content.replace("\n", " ")[:120]
                full_response += f"\n- **[{i}] `{file_name}`**: *\"{snippet}...\"*\n"

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI API ทุกโมเดล: {str(last_error)}")