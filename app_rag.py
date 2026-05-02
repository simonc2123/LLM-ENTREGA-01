"""
app_rag.py — Q&A con RAG usando LangChain + FAISS + Ollama (100% local)
Tres funcionalidades: Q&A, Resumen y FAQs
"""

from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_PATH = Path("output/knowledge_base_rag.md")
FAISS_PATH = Path("output/faiss_index")
EMBED_MODEL = "nomic-embed-text-v2-moe:latest"
EMBEDDING_FAMILIES = {"nomic-bert-moe", "bert", "nomic"}

# ── Prompts ───────────────────────────────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual de Smurfit Kappa Colombia \
(Cartón de Colombia / Smurfit Westrock).

Responde la pregunta usando ÚNICAMENTE los fragmentos de contexto proporcionados.

REGLAS:
- Usa solo la información del contexto.
- Combina fragmentos si es necesario para dar una respuesta completa.
- Si el contexto no contiene la respuesta, di: "Esa información no está disponible."
- Nunca uses conocimiento externo sobre la empresa.
- Responde siempre en español, de forma clara y directa.""",
        ),
        ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un analista experto en comunicación corporativa. \
Tu tarea es generar un resumen ejecutivo claro y profesional de la empresa \
basándote ÚNICAMENTE en la información del documento proporcionado.

El resumen debe cubrir:
1. Quién es la empresa (nombre, historia, origen)
2. Qué hace (productos y servicios principales)
3. Dónde opera (plantas y presencia en Colombia)
4. Sus valores y compromisos (sostenibilidad, ética)
5. Su presencia global

Responde en español. Máximo 300 palabras.""",
        ),
        ("human", "Documento de la empresa:\n{document}"),
    ]
)

FAQ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un experto en comunicación con clientes. \
Basándote ÚNICAMENTE en el documento proporcionado, genera exactamente 10 \
preguntas frecuentes (FAQs) con sus respuestas, que un cliente nuevo \
haría al conocer esta empresa.

Formato de respuesta:
**P: [pregunta]**
R: [respuesta basada en el documento]

Cubre temas variados: historia, productos, ubicaciones, sostenibilidad, contacto.
Responde en español.""",
        ),
        ("human", "Documento de la empresa:\n{document}"),
    ]
)

# ── Carga del índice FAISS ────────────────────────────────────────────────────


def get_embeddings():
    """Instancia el modelo de embeddings local (nomic-embed-text-v2-moe)."""
    return OllamaEmbeddings(model=EMBED_MODEL)


@st.cache_resource(show_spinner="Construyendo índice vectorial...")
def load_vectorstore():
    """
    Carga el índice FAISS desde disco si existe.
    Si no, divide el KB en chunks con RecursiveCharacterTextSplitter,
    genera embeddings con OllamaEmbeddings y guarda el índice.
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
    docs = [Document(page_content=c) for c in chunks]

    with st.spinner(f"Generando embeddings de {len(docs)} chunks (solo la primera vez)..."):
        vectorstore = FAISS.from_documents(docs, embeddings)

    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_PATH))
    return vectorstore


def get_local_models() -> list[str]:
    """Retorna los modelos LLM disponibles en Ollama excluyendo modelos de embeddings."""
    try:
        import ollama

        return [
            m.model
            for m in ollama.list().models
            if not any(f in (m.details.family or "") for f in EMBEDDING_FAMILIES)
            and "embed" not in m.model.lower()
        ]
    except Exception:
        return []


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Q&A RAG — Smurfit Kappa Colombia", layout="wide")
st.title("Smurfit Kappa Colombia — Asistente Virtual")
st.caption("RAG con LangChain · FAISS · OllamaEmbeddings · 100% local")

with st.sidebar:
    st.header("Configuración")
    models = get_local_models()
    if not models:
        st.error("No se pudo conectar a Ollama o no hay modelos disponibles.")
        st.stop()
    preferred = next(
        (
            m
            for m in ["gemma3:12b", "mistral-nemo:latest", "llama3.2:3b", "qwen3:4b"]
            if m in models
        ),
        None,
    )
    default_idx = models.index(preferred) if preferred else 0
    model_name = st.selectbox("Modelo LLM", models, index=default_idx)
    top_k = st.slider("Chunks a recuperar (top-k)", 1, 8, 4)
    st.caption(f"Embeddings: {EMBED_MODEL}")
    st.caption("Vector store: FAISS (local)")

if not KB_PATH.exists():
    st.error(f"No se encontró {KB_PATH}. Ejecuta primero: `make kb`")
    st.stop()

vectorstore = load_vectorstore()
st.sidebar.success(f"Índice FAISS cargado — {vectorstore.index.ntotal} vectores")

llm = ChatOllama(model=model_name, temperature=0)

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

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | QA_PROMPT
                | llm
                | StrOutputParser()
            )

            with st.spinner("Buscando y generando respuesta..."):
                relevant_docs = retriever.invoke(question)

            with st.expander("Fragmentos recuperados", expanded=False):
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Chunk {i}:**\n\n{doc.page_content}")

            answer_box = st.empty()
            full_answer = ""
            for token in chain.stream(question):
                full_answer += token
                answer_box.markdown(full_answer)

# ── Tab Resumen ───────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Resumen ejecutivo de la empresa")
    st.caption("Generado automáticamente a partir de la base de conocimiento.")

    if st.button("Generar resumen", type="primary", key="summary_btn"):
        kb_text = KB_PATH.read_text(encoding="utf-8")
        # Usar solo los primeros 4000 caracteres del KB para el resumen
        document_excerpt = kb_text[:6000]

        chain = SUMMARY_PROMPT | llm | StrOutputParser()

        answer_box = st.empty()
        full_answer = ""
        with st.spinner("Generando resumen..."):
            for token in chain.stream({"document": document_excerpt}):
                full_answer += token
                answer_box.markdown(full_answer)

# ── Tab FAQ ───────────────────────────────────────────────────────────────────
with tab_faq:
    st.subheader("Preguntas Frecuentes (FAQs)")
    st.caption("10 FAQs generadas a partir de la base de conocimiento.")

    if st.button("Generar FAQs", type="primary", key="faq_btn"):
        kb_text = KB_PATH.read_text(encoding="utf-8")
        document_excerpt = kb_text[:6000]

        chain = FAQ_PROMPT | llm | StrOutputParser()

        answer_box = st.empty()
        full_answer = ""
        with st.spinner("Generando FAQs..."):
            for token in chain.stream({"document": document_excerpt}):
                full_answer += token
                answer_box.markdown(full_answer)
