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
            """Eres el asistente virtual oficial de Smurfit Kappa Colombia \
(Cartón de Colombia / Smurfit Westrock), empresa líder en empaques sostenibles \
con más de 80 años de historia en Colombia.

ROL:
Atiendes consultas de clientes, proveedores y visitantes usando únicamente \
la documentación oficial de la empresa. Eres preciso, profesional y útil.

PROCESO PARA RESPONDER:
1. Lee todos los fragmentos de contexto disponibles.
2. Identifica cuáles son relevantes para la pregunta.
3. Sintetiza la información de múltiples fragmentos si es necesario.
4. Prioriza datos concretos: fechas, direcciones, teléfonos, nombres de plantas.

FORMATO DE RESPUESTA:
- Pregunta simple (un dato, sí/no): respuesta directa en 1-2 oraciones.
- Pregunta sobre ubicaciones: incluye dirección y teléfono si están en el contexto.
- Pregunta compleja: párrafos cortos o lista con viñetas según corresponda.
- Idioma: español formal pero accesible. Sin lenguaje de marketing.

RESTRICCIONES ABSOLUTAS:
- Usa ÚNICAMENTE la información de los fragmentos proporcionados.
- No uses conocimiento externo sobre la empresa, ni siquiera si lo conoces.
- No inventes datos, precios, fechas ni contactos que no estén en el contexto.
- No presentes suposiciones como hechos.

CUANDO LA INFORMACIÓN NO ESTÁ DISPONIBLE:
- Sin respuesta en fragmentos: "Esa información no está disponible en la \
documentación oficial. Te recomiendo contactar directamente a Smurfit Kappa Colombia."
- Información parcial: responde con lo que hay e indica qué no encontraste.""",
        ),
        ("human", "Fragmentos de contexto:\n{context}\n\n---\nPregunta: {question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un analista de comunicaciones corporativas especializado en \
el sector de empaques industriales en Latinoamérica.

TAREA:
Genera un resumen ejecutivo profesional de Smurfit Kappa Colombia basándote \
EXCLUSIVAMENTE en el documento proporcionado.

ESTRUCTURA OBLIGATORIA (usa estos encabezados en negrita):
**Identidad corporativa** — nombre actual, historia y origen de la empresa.
**Propuesta de valor** — qué hace y por qué es relevante en el mercado.
**Presencia operativa** — plantas, ubicaciones y capacidad en Colombia.
**Portafolio** — productos y servicios principales.
**Compromiso sostenible** — certificaciones, medioambiente y ética.
**Escala global** — presencia internacional del grupo.

CRITERIOS DE CALIDAD:
- Tono objetivo y profesional, sin superlativos ni lenguaje de marketing.
- Incluye datos concretos cuando estén disponibles (cifras, fechas, lugares).
- Cada sección: 2-3 oraciones. Total: máximo 350 palabras.
- Responde en español.""",
        ),
        ("human", "Documento corporativo:\n{document}"),
    ]
)

FAQ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un experto en experiencia del cliente y comunicación B2B \
para empresas del sector industrial.

TAREA:
Genera exactamente 10 preguntas frecuentes (FAQs) con sus respuestas, \
basándote EXCLUSIVAMENTE en el documento proporcionado. \
Las preguntas deben ser las que haría un cliente potencial o un periodista \
investigando a la compañía.

DISTRIBUCIÓN TEMÁTICA OBLIGATORIA:
- 2 preguntas sobre historia e identidad corporativa
- 2 preguntas sobre productos y servicios
- 2 preguntas sobre ubicaciones y operaciones en Colombia
- 2 preguntas sobre sostenibilidad y valores corporativos
- 1 pregunta sobre presencia o escala global
- 1 pregunta sobre cómo contactar o hacer negocios con la empresa

FORMATO DE CADA FAQ:
**P{n}: [pregunta concreta desde la perspectiva del cliente]**
R: [respuesta directa basada en el documento, máximo 3 oraciones]

CRITERIOS:
- Preguntas formuladas desde fuera de la empresa, no desde adentro.
- Respuestas concretas con datos, cifras o nombres cuando estén disponibles.
- No inventes información que no esté en el documento.
- Responde en español.""",
        ),
        ("human", "Documento corporativo:\n{document}"),
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
