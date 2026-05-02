"""
app_rag.py — Q&A con RAG usando LangChain + FAISS + Ollama (100% local)
Tres funcionalidades: Q&A, Resumen y FAQs
"""

import re

import streamlit as st
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

KB_PATH     = Path("output/knowledge_base_rag.md")
FAISS_PATH  = Path("output/faiss_index")
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
1.⁠ ⁠Lee todos los fragmentos de contexto disponibles.
2.⁠ ⁠Identifica cuáles son relevantes para la pregunta.
3.⁠ ⁠Sintetiza la información de múltiples fragmentos si es necesario.
4.⁠ ⁠Prioriza datos concretos: fechas, direcciones, teléfonos, nombres de plantas.

FORMATO DE RESPUESTA:
•⁠  ⁠Pregunta simple (un dato, sí/no): respuesta directa en 1-2 oraciones.
•⁠  ⁠Pregunta sobre ubicaciones: incluye dirección y teléfono si están en el contexto.
•⁠  ⁠Pregunta compleja: párrafos cortos o lista con viñetas según corresponda.
•⁠  ⁠Idioma: español formal pero accesible. Sin lenguaje de marketing.

RESTRICCIONES ABSOLUTAS:
•⁠  ⁠Usa ÚNICAMENTE la información de los fragmentos proporcionados.
•⁠  ⁠No uses conocimiento externo sobre la empresa, ni siquiera si lo conoces.
•⁠  ⁠No inventes datos, precios, fechas ni contactos que no estén en el contexto.
•⁠  ⁠No presentes suposiciones como hechos.

CUANDO LA INFORMACIÓN NO ESTÁ DISPONIBLE:
•⁠  ⁠Sin respuesta en fragmentos: "Esa información no está disponible en la \
documentación oficial. Te recomiendo contactar directamente a Smurfit Kappa Colombia."
•⁠  ⁠Información parcial: responde con lo que hay e indica qué no encontraste.""",
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
*Identidad corporativa* — nombre actual, historia y origen de la empresa.
*Propuesta de valor* — qué hace y por qué es relevante en el mercado.
*Presencia operativa* — plantas, ubicaciones y capacidad en Colombia.
*Portafolio* — productos y servicios principales.
*Compromiso sostenible* — certificaciones, medioambiente y ética.
*Escala global* — presencia internacional del grupo.

CRITERIOS DE CALIDAD:
•⁠  ⁠Tono objetivo y profesional, sin superlativos ni lenguaje de marketing.
•⁠  ⁠Incluye datos concretos cuando estén disponibles (cifras, fechas, lugares).
•⁠  ⁠Cada sección: 2-3 oraciones. Total: máximo 350 palabras.
•⁠  ⁠Responde en español.""",
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
•⁠  ⁠2 preguntas sobre historia e identidad corporativa
•⁠  ⁠2 preguntas sobre productos y servicios
•⁠  ⁠2 preguntas sobre ubicaciones y operaciones en Colombia
•⁠  ⁠2 preguntas sobre sostenibilidad y valores corporativos
•⁠  ⁠1 pregunta sobre presencia o escala global
•⁠  ⁠1 pregunta sobre cómo contactar o hacer negocios con la empresa

FORMATO DE CADA FAQ:
*P{{n}}: [pregunta concreta desde la perspectiva del cliente]*
R: [respuesta directa basada en el documento, máximo 3 oraciones]

CRITERIOS:
•⁠  ⁠Preguntas formuladas desde fuera de la empresa, no desde adentro.
•⁠  ⁠Respuestas concretas con datos, cifras o nombres cuando estén disponibles.
•⁠  ⁠No inventes información que no esté en el documento.
•⁠  ⁠Responde en español.""",
        ),
        ("human", "Documento corporativo:\n{document}"),
    ]
)

# ── Carga del índice FAISS ────────────────────────────────────────────────────

def get_embeddings():
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
    preferred   = next(
        (m for m in ["gemma3:12b", "mistral-nemo:latest", "llama3.2:3b", "qwen3:4b"]
         if m in models), None
    )
    default_idx = models.index(preferred) if preferred else 0
    model_name  = st.selectbox("Modelo LLM", models, index=default_idx)
    top_k       = st.slider("Chunks a recuperar (top-k)", 1, 8, 4)

    with st.expander("Parámetros de muestreo del LLM"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1,
                                help="0 = determinista (greedy). >0 introduce aleatoriedad.")
        sampling_top_p = st.slider("top_p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05,
                                   help="Probabilidad acumulada para muestreo. Sin efecto si temperature=0.")
        sampling_top_k = st.slider("top_k (sampling)", 1, 100, 40, 1,
                                   help="Cantidad de tokens candidatos. Sin efecto si temperature=0.")
        repeat_penalty = st.slider("repeat_penalty", 1.0, 2.0, 1.1, 0.05,
                                   help="Penaliza repetición de tokens recientes.")
        num_predict = st.number_input("num_predict (max tokens, -1 = ilimitado)", -1, 8192, -1, 64,
                                      help="-1 deja que el modelo genere libremente hasta su tope natural.")

    st.caption(f"Embeddings: {EMBED_MODEL}")
    st.caption("Vector store: FAISS (local)")

if not KB_PATH.exists():
    st.error(f"No se encontró {KB_PATH}. Ejecuta primero: `make kb`")
    st.stop()

vectorstore = load_vectorstore()
st.sidebar.success(f"Índice FAISS cargado — {vectorstore.index.ntotal} vectores")

llm_kwargs = {
    "model": model_name,
    "temperature": temperature,
    "top_p": sampling_top_p,
    "top_k": int(sampling_top_k),
    "repeat_penalty": repeat_penalty,
}
if num_predict > 0:
    llm_kwargs["num_predict"] = int(num_predict)

llm = ChatOllama(**llm_kwargs)

FAQS_USUARIO = [
    "¿Cuándo fue fundada Cartón de Colombia y en qué ciudad?",
    "¿En qué año se fusionó Smurfit con Kappa Packaging y cómo se llamó la empresa resultante?",
    "¿Cuál es el nombre actual de la empresa tras la fusión de 2024?",
    "¿Cuántos países tiene presencia el grupo Smurfit Westrock y cuántos empleados tiene globalmente?",
    "¿En qué dirección exacta está ubicada la planta corrugadora de Bogotá?",
    "¿Qué tipo de productos fabrica la planta de Guarne en Antioquia?",
    "¿Cuántas plantas corrugadoras tiene Smurfit Kappa en Colombia y en qué ciudades están?",
    "¿Qué certificación tiene la planta de Medellín y qué significa esa certificación?",
    "¿Dónde está ubicada la planta de Sacos de Papel y qué produce?",
    "¿Qué es el sistema Bag-in-Box y para qué industrias está diseñado?",
    "¿Qué es la cartulina Óptima y cuáles son sus principales usos?",
    "¿Qué soluciones de empaque ofrece la empresa para el canal eCommerce?",
    "¿Hasta cuántos colores de impresión pueden tener los empaques corrugados de la empresa?",
    "¿Qué tipos de sacos de papel fabrica la empresa y para qué industrias?",
    "¿Cuántas hectáreas gestiona la División Forestal de Smurfit Kappa en Colombia?",
    "¿Desde qué año tiene la División Forestal la certificación FSC y qué garantiza esa certificación?",
    "¿En cuántos departamentos de Colombia opera la División Forestal?",
    "¿Cuál es el precio por tonelada del cartón corrugado que vende la empresa?",
    "¿Cuántos empleados tiene específicamente la planta de Barranquilla?",
    "¿Cuál es el correo electrónico del gerente general de Smurfit Kappa Colombia?",
]

tab_qa, tab_summary, tab_faq, tab_faqs_user = st.tabs(
    ["Q&A", "Resumen ejecutivo", "FAQs", "FAQs Usuario"]
)

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
            try:
                for token in chain.stream(question):
                    full_answer += token
                    answer_box.markdown(strip_thinking(full_answer) or "_Pensando..._")
                final = strip_thinking(full_answer)
                if not final:
                    answer_box.warning("El modelo no produjo respuesta visible (solo razonamiento interno). Intenta con otro modelo o sube `num_predict`.")
                else:
                    answer_box.markdown(final)
            except Exception as e:
                st.error(f"Error generando respuesta: {e}")

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
        try:
            with st.spinner("Generando resumen..."):
                for token in chain.stream({"document": document_excerpt}):
                    full_answer += token
                    answer_box.markdown(strip_thinking(full_answer) or "_Pensando..._")
            final = strip_thinking(full_answer)
            if not final:
                answer_box.warning("El modelo no produjo respuesta visible. Prueba otro modelo o sube `num_predict`.")
            else:
                answer_box.markdown(final)
        except Exception as e:
            st.error(f"Error generando resumen: {e}")

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
        try:
            with st.spinner("Generando FAQs..."):
                for token in chain.stream({"document": document_excerpt}):
                    full_answer += token
                    answer_box.markdown(strip_thinking(full_answer) or "_Pensando..._")
            final = strip_thinking(full_answer)
            if not final:
                answer_box.warning("El modelo no produjo respuesta visible. Prueba otro modelo o sube `num_predict`.")
            else:
                answer_box.markdown(final)
        except Exception as e:
            st.error(f"Error generando FAQs: {e}")

# ── Tab FAQs Usuario ──────────────────────────────────────────────────────────
with tab_faqs_user:
    st.subheader("Preguntas frecuentes del usuario")
    st.caption("Listado de preguntas predefinidas para evaluar el sistema.")
    for i, q in enumerate(FAQS_USUARIO, 1):
        st.markdown(f"**{i}.** {q}")
