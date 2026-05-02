"""
app_rag_comercial.py — Q&A con RAG usando LangChain + FAISS + Kimi K2.6 (API comercial)
======================================================================================

Arquitectura híbrida:
  • Embeddings: Google Gemini text-embedding-004 (API)
  • Vector Store: FAISS (local, persistido en disco)
  • LLM generativo: Kimi K2.6 de Moonshot AI (API)

Ventajas de esta arquitectura:
  - Embeddings de Gemini: alta calidad semántica multilingüe sin necesidad de Ollama.
  - El LLM de generación es de calidad frontera (Kimi K2.6 supera a GPT-5.4 en SWE-Bench).
  - El contexto de 256K tokens permite mandar muchos chunks al modelo sin truncar.

Requisitos:
  1. API key de Google en GOOGLE_API_KEY (embeddings — gratis con Gemini Free Tier)
     → Obtén la key en: https://aistudio.google.com/apikey
  2. API key de Moonshot AI en MOONSHOT_API_KEY (LLM generativo)
     → Obtén la key en: https://platform.moonshot.ai

Uso:
    uv run streamlit run app_rag_comercial.py
"""

import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

from langchain_moonshot import ChatMoonshot
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ── Cargar variables de entorno ───────────────────────────────────────────────

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────

KB_PATH      = Path("output/knowledge_base_rag.md")
FAISS_PATH   = Path("output/faiss_index_gemini")
EMBED_MODEL  = "models/gemini-embedding-2-preview"
LLM_MODEL    = "kimi-k2.6"          # Modelo comercial de Moonshot AI
LLM_FALLBACK = "kimi-k2.5"          # Fallback si K2.6 no está disponible

# ── Prompts (idénticos a app_rag.py para comparación justa) ────────────────────

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

# ── Utilidades ────────────────────────────────────────────────────────────────

def get_moonshot_api_key() -> str | None:
    """Obtiene la API key desde entorno o secrets de Streamlit."""
    key = os.getenv("MOONSHOT_API_KEY")
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("MOONSHOT_API_KEY")
    return key


def get_google_api_key() -> str | None:
    """Obtiene la Google API key desde entorno o secrets de Streamlit."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("GOOGLE_API_KEY")
    return key


def get_embeddings():
    """Embeddings via Google Gemini gemini-embedding-2-preview."""
    api_key = get_google_api_key()
    if not api_key:
        st.error("🔑 No se encontró `GOOGLE_API_KEY`. Agrégala al archivo `.env`.")
        st.info("Obtén tu key gratuita en [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
        st.stop()
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)


@st.cache_resource(show_spinner="Construyendo índice vectorial...")
def load_vectorstore():
    """
    Carga el índice FAISS desde disco si existe.
    Si no, divide el KB en chunks, genera embeddings con Gemini en lotes
    pequeños y construye el índice con FAISS.from_embeddings.
    (gemini-embedding-2-preview no soporta batching masivo de LangChain)
    """
    embeddings = get_embeddings()

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

    BATCH = 20
    SLEEP_BETWEEN_BATCHES = 1  # segundo de cortesía entre lotes
    MAX_RETRIES = 3

    all_vectors = []
    progress = st.progress(0, text=f"Generando embeddings — 0/{len(chunks)} chunks")
    for i in range(0, len(chunks), BATCH):
        lote = chunks[i : i + BATCH]
        for intento in range(MAX_RETRIES):
            try:
                all_vectors.extend(embeddings.embed_documents(lote))
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = SLEEP_BETWEEN_BATCHES * (2 ** intento)
                    avance_txt = min(i + BATCH, len(chunks))
                    progress.progress(
                        i / len(chunks),
                        text=f"Rate limit — esperando {wait}s... ({avance_txt}/{len(chunks)})"
                    )
                    time.sleep(wait)
                else:
                    raise
        avance = min(i + BATCH, len(chunks))
        progress.progress(avance / len(chunks), text=f"Generando embeddings — {avance}/{len(chunks)} chunks")
        if i + BATCH < len(chunks):
            time.sleep(SLEEP_BETWEEN_BATCHES)
    progress.empty()

    text_embedding_pairs = list(zip(chunks, all_vectors))
    vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings)

    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_PATH))
    return vectorstore


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Q&A RAG — Smurfit Kappa Colombia (Kimi K2.6)",
    layout="wide"
)
st.title("🤖 Smurfit Kappa Colombia — Asistente Virtual")
st.caption("RAG con **LangChain · FAISS · OllamaEmbeddings · Kimi K2.6** (comercial)")

# ── Sidebar: Configuración ────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuración")

    # Verificar API key de Moonshot
    api_key = get_moonshot_api_key()
    if not api_key:
        st.error("🔑 No se encontró `MOONSHOT_API_KEY`.")
        st.info(
            "Obtén tu API key en [platform.moonshot.ai](https://platform.moonshot.ai) "
            "y configúrala como variable de entorno o en `.streamlit/secrets.toml`."
        )
        st.code("export MOONSHOT_API_KEY='sk-...'", language="bash")
        st.stop()
    else:
        masked = api_key[:8] + "..." + api_key[-4:]
        st.success(f"🔑 API Key cargada: {masked}")

    # Verificar Google API key (para embeddings)
    google_key = get_google_api_key()
    if not google_key:
        st.error("🔑 No se encontró `GOOGLE_API_KEY`.")
        st.info("Agrégala al archivo `.env` para usar los embeddings de Gemini.")
        st.stop()
    else:
        masked_g = google_key[:8] + "..." + google_key[-4:]
        st.success(f"🔑 Google Key cargada: {masked_g}")

    # Selector de modelo
    modelo_opciones = [LLM_MODEL, LLM_FALLBACK]
    modelo = st.selectbox("Modelo LLM comercial", modelo_opciones, index=0)

    # Toggle thinking mode
    thinking = st.toggle("Modo razonamiento (thinking)", value=False,
                         help="Kimi K2.6 puede razonar antes de responder. Útil para preguntas complejas, pero consume más tokens.")

    top_k = st.slider("Chunks a recuperar (top-k)", 1, 8, 4)

    st.divider()
    st.caption(f"🔧 Embeddings: `{EMBED_MODEL}` (Google Gemini)")
    st.caption(f"🧠 LLM: `{modelo}` (Moonshot AI)")
    st.caption(f"📦 Vector store: FAISS (local)")

# ── Verificar KB ──────────────────────────────────────────────────────────────

if not KB_PATH.exists():
    st.error(f"No se encontró {KB_PATH}. Ejecuta primero: `make kb`")
    st.stop()

# ── Cargar índice FAISS ─────────────────────────────────────────────────────

vectorstore = load_vectorstore()
st.sidebar.success(f"✅ Índice FAISS cargado — {vectorstore.index.ntotal} vectores")

# ── Inicializar LLM comercial ─────────────────────────────────────────────────

llm_kwargs = {
    "model": modelo,
    "temperature": 1,  # kimi-k2.6 solo acepta temperature=0.6
    "max_retries": 2,
}

if thinking:
    llm_kwargs["thinking"] = True

llm = ChatMoonshot(**llm_kwargs)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_qa, tab_summary, tab_faq, tab_info = st.tabs(["❓ Q&A", "📋 Resumen ejecutivo", "🙋 FAQs", "ℹ️ Info"])

# ── Tab Q&A ───────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Preguntas y Respuestas")
    st.caption("El sistema recupera chunks relevantes de la base de conocimiento y usa Kimi K2.6 para generar la respuesta.")

    question = st.text_area(
        "Escribe tu pregunta:",
        height=100,
        placeholder="Ej: ¿Dónde están las plantas? ¿Qué productos ofrecen? ¿Cuándo se fundó la empresa?",
        key="qa_input",
    )

    if st.button("Enviar pregunta", type="primary", key="qa_btn"):
        if not question.strip():
            st.warning("Escribe una pregunta primero.")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

            def format_docs(docs):
                return "\n\n---\n\n".join(d.page_content for d in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | QA_PROMPT
                | llm
                | StrOutputParser()
            )

            with st.spinner("🔍 Buscando chunks relevantes..."):
                relevant_docs = retriever.invoke(question)

            with st.expander("📄 Fragmentos recuperados (contexto enviado a Kimi)", expanded=False):
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Chunk {i}:**\n\n{doc.page_content}")

            answer_box = st.empty()
            full_answer = ""
            with st.spinner("🧠 Kimi K2.6 generando respuesta..."):
                for token in chain.stream(question):
                    full_answer += token
                    answer_box.markdown(full_answer)

# ── Tab Resumen ───────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Resumen ejecutivo de la empresa")
    st.caption("Generado automáticamente con Kimi K2.6 a partir de la base de conocimiento.")

    if st.button("Generar resumen", type="primary", key="summary_btn"):
        kb_text = KB_PATH.read_text(encoding="utf-8")
        document_excerpt = kb_text[:6000]

        chain = SUMMARY_PROMPT | llm | StrOutputParser()

        answer_box = st.empty()
        full_answer = ""
        with st.spinner("🧠 Kimi K2.6 generando resumen..."):
            for token in chain.stream({"document": document_excerpt}):
                full_answer += token
                answer_box.markdown(full_answer)

# ── Tab FAQ ───────────────────────────────────────────────────────────────────
with tab_faq:
    st.subheader("Preguntas Frecuentes (FAQs)")
    st.caption("10 FAQs generadas por Kimi K2.6 a partir de la base de conocimiento.")

    if st.button("Generar FAQs", type="primary", key="faq_btn"):
        kb_text = KB_PATH.read_text(encoding="utf-8")
        document_excerpt = kb_text[:6000]

        chain = FAQ_PROMPT | llm | StrOutputParser()

        answer_box = st.empty()
        full_answer = ""
        with st.spinner("🧠 Kimi K2.6 generando FAQs..."):
            for token in chain.stream({"document": document_excerpt}):
                full_answer += token
                answer_box.markdown(full_answer)

# ── Tab Info ──────────────────────────────────────────────────────────────────
with tab_info:
    st.subheader("ℹ️ Acerca de esta arquitectura")

    st.markdown("""
    ### 🔀 Arquitectura híbrida: Gemini Embeddings + Kimi K2.6

    Este sistema combina dos APIs comerciales de alta calidad:

    | Capa | Tecnología | ¿Por qué? |
    |------|-----------|-----------|
    | **Embeddings** | `GoogleGenerativeAIEmbeddings` — `text-embedding-004` | Alta calidad semántica multilingüe. Gratis con Gemini Free Tier (100 req/min). |
    | **Vector Store** | `FAISS` (persistido en disco) | 100% local, sin servidor, búsqueda por similitud eficiente. |
    | **Retriever** | `FAISS.as_retriever(top_k=4)` | Recupera los chunks más relevantes semánticamente. |
    | **LLM Generativo** | `ChatMoonshot` — **Kimi K2.6** | Modelo frontera de Moonshot AI. 256K contexto. Calidad superior a modelos locales. |
    | **Orquestador** | `LangChain` (LCEL chains) | Pipeline declarativo: retriever → prompt → llm → parser. |
    | **UI** | `Streamlit` | Interfaz web rápida para demo en vivo. |

    ### 🧠 ¿Por qué Kimi K2.6?

    - **Calidad**: Supera a GPT-5.4 en benchmarks de código (SWE-Bench Pro: 58.6)
    - **Contexto**: 256K tokens — puedes enviar muchos chunks sin truncar
    - **Precio**: ~$0.16-$0.95 / MTok input, ~$3.50-$4.00 / MTok output
    - **Español**: Entrenado multilingüe con excelente comprensión del español
    - **Velocidad**: MoE architecture (1T params, 32B activos) — inferencia eficiente

    ### 🔐 Seguridad de la API key

    La API key se lee de (en orden de prioridad):
    1. Variable de entorno: `export MOONSHOT_API_KEY='sk-...'`
    2. Archivo `.env` en la raíz del proyecto
    3. `~/.streamlit/secrets.toml` (para despliegue en Streamlit Cloud)

    
    """)
