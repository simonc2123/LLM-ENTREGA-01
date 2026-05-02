"""
app_rag_google.py — Q&A con RAG usando LangChain + FAISS
LLM primario : Google Gemini (vía API)
LLM fallback : Ollama local
Embeddings   : Ollama local (nomic-embed-text-v2-moe) — reutiliza el índice FAISS
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

KB_PATH     = Path("output/knowledge_base_rag.md")
FAISS_PATH  = Path("output/faiss_index")
EMBED_MODEL = "nomic-embed-text-v2-moe:latest"
EMBEDDING_FAMILIES = {"nomic-bert-moe", "bert", "nomic"}

GOOGLE_MODELS = [
    "gemma-4-31b-it",            # 1 500 RPD, TPM ilimitado  ← mejor para uso sostenido
    "gemini-3.1-flash-lite",     # 500 RPD, 250K TPM
    "gemma-3-27b-it",            # 14 400 RPD, 15K TPM
    "gemini-2.5-flash",          # 20 RPD (se agota rápido)
]

OLLAMA_PREFERRED = ["gemma3:12b", "mistral-nemo:latest", "llama3.2:3b", "qwen3:4b"]

# ── Prompts (idénticos a app_rag.py) ─────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente virtual de Smurfit Kappa Colombia \
(Cartón de Colombia / Smurfit Westrock).

Responde la pregunta usando ÚNICAMENTE los fragmentos de contexto proporcionados.

REGLAS:
- Usa solo la información del contexto.
- Combina fragmentos si es necesario para dar una respuesta completa.
- Si el contexto no contiene la respuesta, di: "Esa información no está disponible."
- Nunca uses conocimiento externo sobre la empresa.
- Responde siempre en español, de forma clara y directa."""),
    ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres un analista experto en comunicación corporativa. \
Tu tarea es generar un resumen ejecutivo claro y profesional de la empresa \
basándote ÚNICAMENTE en la información del documento proporcionado.

El resumen debe cubrir:
1. Quién es la empresa (nombre, historia, origen)
2. Qué hace (productos y servicios principales)
3. Dónde opera (plantas y presencia en Colombia)
4. Sus valores y compromisos (sostenibilidad, ética)
5. Su presencia global

Responde en español. Máximo 300 palabras."""),
    ("human", "Documento de la empresa:\n{document}"),
])

FAQ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres un experto en comunicación con clientes. \
Basándote ÚNICAMENTE en el documento proporcionado, genera exactamente 10 \
preguntas frecuentes (FAQs) con sus respuestas, que un cliente nuevo \
haría al conocer esta empresa.

Formato de respuesta:
**P: [pregunta]**
R: [respuesta basada en el documento]

Cubre temas variados: historia, productos, ubicaciones, sostenibilidad, contacto.
Responde en español."""),
    ("human", "Documento de la empresa:\n{document}"),
])

# ── Carga del índice FAISS ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Construyendo índice vectorial...")
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if FAISS_PATH.exists():
        return FAISS.load_local(
            str(FAISS_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    kb_text = KB_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_text(kb_text)
    docs   = [Document(page_content=c) for c in chunks]

    with st.spinner(f"Generando embeddings de {len(docs)} chunks (solo la primera vez)..."):
        vectorstore = FAISS.from_documents(docs, embeddings)

    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_PATH))
    return vectorstore


def get_local_models() -> list[str]:
    try:
        import ollama
        return [
            m.model for m in ollama.list().models
            if not any(f in (m.details.family or "") for f in EMBEDDING_FAMILIES)
            and "embed" not in m.model.lower()
        ]
    except Exception:
        return []


def make_google_llm(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def stream_with_fallback(chain_google, chain_ollama, payload: dict, status_slot):
    """
    Intenta streamear con Google. Si falla, avisa y usa Ollama como fallback.
    Devuelve (texto_completo, proveedor_usado).
    """
    full_text = ""
    try:
        status_slot.info("Generando con Google Gemini...", icon="🌐")
        for token in chain_google.stream(payload):
            full_text += token
            yield token
        status_slot.success("Google Gemini", icon="✅")
    except Exception as exc:
        status_slot.warning(f"Google falló → usando Ollama local  ({exc})", icon="⚠️")
        full_text = ""
        for token in chain_ollama.stream(payload):
            full_text += token
            yield token
        status_slot.info("Ollama local (fallback)", icon="🖥️")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Q&A RAG — Smurfit Kappa (Google)", layout="wide")
st.title("Smurfit Kappa Colombia — Asistente Virtual")
st.caption("RAG con LangChain · FAISS · OllamaEmbeddings · LLM: Google Gemini + fallback Ollama")

api_key = os.getenv("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("Configuración")

    # ── Estado API Key ────────────────────────────────────────────────────────
    if api_key:
        st.success("GOOGLE_API_KEY cargada", icon="🔑")
    else:
        st.error("GOOGLE_API_KEY no encontrada — solo funciona el fallback Ollama")

    # ── Modelo Google ─────────────────────────────────────────────────────────
    st.subheader("Primario — Google Gemini")
    google_model = st.selectbox("Modelo Google", GOOGLE_MODELS, index=0,
                                disabled=not api_key)

    # ── Modelo Ollama fallback ────────────────────────────────────────────────
    st.subheader("Fallback — Ollama local")
    local_models = get_local_models()
    if local_models:
        preferred   = next((m for m in OLLAMA_PREFERRED if m in local_models), None)
        default_idx = local_models.index(preferred) if preferred else 0
        ollama_model = st.selectbox("Modelo Ollama", local_models, index=default_idx)
    else:
        st.warning("Ollama no disponible — fallback desactivado")
        ollama_model = None

    top_k = st.slider("Chunks a recuperar (top-k)", 1, 8, 4)
    st.caption(f"Embeddings: {EMBED_MODEL} (local)")
    st.caption("Vector store: FAISS (local)")

if not KB_PATH.exists():
    st.error(f"No se encontró {KB_PATH}. Ejecuta primero: `make kb`")
    st.stop()

if not api_key and not ollama_model:
    st.error("Sin API Key de Google ni Ollama disponible. No se puede continuar.")
    st.stop()

vectorstore = load_vectorstore()
st.sidebar.success(f"Índice FAISS cargado — {vectorstore.index.ntotal} vectores")

# ── Construir LLMs ────────────────────────────────────────────────────────────

llm_google = make_google_llm(google_model) if api_key else None
llm_ollama = ChatOllama(model=ollama_model, temperature=0) if ollama_model else None

if llm_google is None and llm_ollama is None:
    st.error("No hay ningún LLM disponible.")
    st.stop()

# Si solo hay uno disponible, úsalo para ambos slots
_primary  = llm_google  if llm_google  else llm_ollama
_fallback = llm_ollama  if llm_ollama  else llm_google

tab_qa, tab_summary, tab_faq = st.tabs(["Q&A", "Resumen ejecutivo", "FAQs"])

# ── Tab Q&A ───────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Preguntas y Respuestas")
    question = st.text_area(
        "Escribe tu pregunta:",
        height=100,
        placeholder="Ej: ¿Dónde están las plantas? ¿Qué productos ofrecen?",
        key="qa_input",
    )

    if st.button("Enviar pregunta", type="primary", key="qa_btn"):
        if not question.strip():
            st.warning("Escribe una pregunta primero.")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

            def format_docs(docs):
                return "\n\n---\n\n".join(d.page_content for d in docs)

            chain_google = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | QA_PROMPT | _primary | StrOutputParser()
            )
            chain_ollama = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | QA_PROMPT | _fallback | StrOutputParser()
            )

            with st.spinner("Buscando fragmentos relevantes..."):
                relevant_docs = retriever.invoke(question)

            with st.expander("Fragmentos recuperados", expanded=False):
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Chunk {i}:**\n\n{doc.page_content}")

            status_slot = st.empty()
            answer_box  = st.empty()
            full_answer = ""
            for token in stream_with_fallback(chain_google, chain_ollama,
                                              question, status_slot):
                full_answer += token
                answer_box.markdown(full_answer)

# ── Tab Resumen ───────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Resumen ejecutivo de la empresa")
    st.caption("Generado automáticamente a partir de la base de conocimiento.")

    if st.button("Generar resumen", type="primary", key="summary_btn"):
        document_excerpt = KB_PATH.read_text(encoding="utf-8")[:6000]
        payload = {"document": document_excerpt}

        chain_google = SUMMARY_PROMPT | _primary  | StrOutputParser()
        chain_ollama = SUMMARY_PROMPT | _fallback | StrOutputParser()

        status_slot = st.empty()
        answer_box  = st.empty()
        full_answer = ""
        for token in stream_with_fallback(chain_google, chain_ollama,
                                          payload, status_slot):
            full_answer += token
            answer_box.markdown(full_answer)

# ── Tab FAQ ───────────────────────────────────────────────────────────────────
with tab_faq:
    st.subheader("Preguntas Frecuentes (FAQs)")
    st.caption("10 FAQs generadas a partir de la base de conocimiento.")

    if st.button("Generar FAQs", type="primary", key="faq_btn"):
        document_excerpt = KB_PATH.read_text(encoding="utf-8")[:6000]
        payload = {"document": document_excerpt}

        chain_google = FAQ_PROMPT | _primary  | StrOutputParser()
        chain_ollama = FAQ_PROMPT | _fallback | StrOutputParser()

        status_slot = st.empty()
        answer_box  = st.empty()
        full_answer = ""
        for token in stream_with_fallback(chain_google, chain_ollama,
                                          payload, status_slot):
            full_answer += token
            answer_box.markdown(full_answer)
