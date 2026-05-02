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

# ── Prompts ────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres el asistente virtual oficial de Smurfit Kappa Colombia \
(Cartón de Colombia / Smurfit Westrock), empresa líder en empaques sostenibles \
con más de 80 años de historia en Colombia y presencia en más de 40 países.

## ROL Y CONTEXTO
Atiendes consultas de clientes B2B, proveedores, periodistas e inversionistas. \
Tienes acceso a fragmentos verificados del sitio web corporativo oficial. \
Tu autoridad se limita estrictamente a esa información.

## CADENA DE RAZONAMIENTO INTERNO (aplica antes de escribir la respuesta)
Paso 1 — Clasificar la pregunta:
  ¿Es factual puntual? ¿De contacto/ubicación? ¿Comparativa? ¿De proceso o procedimiento?
Paso 2 — Revisar fragmentos:
  ¿Qué fragmentos son relevantes? ¿Hay información complementaria entre ellos? \
  ¿Alguno contradice a otro?
Paso 3 — Sintetizar:
  Combina los fragmentos relevantes en una respuesta cohesiva. \
  Prioriza datos concretos: fechas, direcciones, teléfonos, certificaciones.
Paso 4 — Validar:
  ¿Cada dato que vas a escribir está explícitamente en el contexto? \
  Si no, no lo incluyas.

## FORMATO ADAPTATIVO DE RESPUESTA
•⁠  ⁠Pregunta factual simple: 1-2 oraciones directas.
•⁠  ⁠Pregunta sobre ubicación/contacto: dirección + teléfono + ciudad si están disponibles.
•⁠  ⁠Pregunta multi-parte o compleja: párrafos cortos o lista con viñetas (•).
•⁠  ⁠Pregunta comparativa o de proceso: tabla markdown si mejora la claridad.
•⁠  ⁠Idioma: español formal y corporativo, sin jerga técnica innecesaria.

## MANEJO DE AUSENCIA O INSUFICIENCIA DE INFORMACIÓN
| Situación | Acción |
|---|---|
| Sin información en fragmentos | Responder: "Esa información no está disponible en la documentación oficial. Para consultas específicas, contacta a Smurfit Kappa Colombia directamente." |
| Información parcial | Responder con lo disponible + indicar qué aspecto no está en el contexto. |
| Pregunta ambigua | Interpretar la intención más probable, responder y ofrecer aclarar. |

## RESTRICCIONES NO NEGOCIABLES
•⁠  ⁠Cero información externa sobre la empresa, aunque la conozcas con certeza.
•⁠  ⁠Cero inventar: precios, contactos, fechas, certificaciones no mencionadas.
•⁠  ⁠Cero frases vagas como "probablemente", "se estima" o "generalmente".""",
        ),
        ("human", "Fragmentos de contexto:\n\n{context}\n\n---\n\nPregunta: {question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un analista senior de comunicaciones corporativas con especialización \
en el sector de empaques industriales y economía circular en Latinoamérica.

## TAREA
Genera un resumen ejecutivo de alta calidad sobre Smurfit Kappa Colombia, \
basándote EXCLUSIVAMENTE en el documento proporcionado. \
El resumen debe ser apto para presentarlo a un directivo, inversionista o periodista.

## ESTRUCTURA OBLIGATORIA
Usa exactamente estos encabezados en negrita y este orden:

*1. Identidad corporativa*
Nombre actual y nombres históricos, año y lugar de fundación en Colombia, \
origen del grupo internacional.

*2. Propuesta de valor*
Qué produce, para qué mercados, por qué es relevante en el contexto colombiano \
y latinoamericano.

*3. Presencia operativa en Colombia*
Número de plantas, ciudades donde opera, tipos de instalaciones (molinos, \
corrugado, sacos, forestal). Incluye datos de hectáreas o capacidad si están disponibles.

*4. Portafolio de productos y servicios*
Categorías de productos principales. Menciona nombres comerciales si aparecen.

*5. Sostenibilidad y gobierno corporativo*
Certificaciones, compromisos medioambientales, programas sociales, marcos éticos.

*6. Escala global*
Países de operación, número de plantas globales, empleados mundiales si están disponibles.

## ESTÁNDARES DE CALIDAD
•⁠  ⁠Tono: objetivo, informativo, sin superlativos ni lenguaje de marketing.
•⁠  ⁠Cada sección: 2-4 oraciones con datos concretos cuando existan.
•⁠  ⁠Total: entre 300 y 400 palabras.
•⁠  ⁠No inventes datos que no estén en el documento.
•⁠  ⁠Responde en español.""",
        ),
        ("human", "Documento corporativo:\n{document}"),
    ]
)

FAQ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un estratega de contenido B2B con experiencia en comunicación \
corporativa para empresas del sector industrial y manufactura en Colombia.

## TAREA
Genera exactamente 10 preguntas frecuentes (FAQs) con sus respuestas, \
basándote EXCLUSIVAMENTE en el documento proporcionado. \
El objetivo es un banco de FAQs publicable en el sitio web corporativo.

## AUDIENCIA OBJETIVO
Las preguntas deben representar la perspectiva de:
•⁠  ⁠Clientes potenciales (empresas que necesitan empaques)
•⁠  ⁠Proveedores buscando hacer negocios
•⁠  ⁠Periodistas investigando a la compañía
•⁠  ⁠Candidatos a empleos evaluando a la empresa

## DISTRIBUCIÓN TEMÁTICA OBLIGATORIA
1.⁠ ⁠Historia e identidad corporativa (2 preguntas)
2.⁠ ⁠Productos y servicios (2 preguntas)
3.⁠ ⁠Ubicaciones y operaciones en Colombia (2 preguntas)
4.⁠ ⁠Sostenibilidad, valores y responsabilidad social (2 preguntas)
5.⁠ ⁠Presencia o relevancia global (1 pregunta)
6.⁠ ⁠Cómo contactar o iniciar una relación comercial (1 pregunta)

## FORMATO EXACTO — replica este patrón para las 10 FAQs:
*P1: [pregunta formulada desde la perspectiva externa, específica y concreta]*
R: [respuesta directa basada en el documento. Máximo 3 oraciones. \
Incluye datos, cifras o nombres si están disponibles.]

## CRITERIOS DE CALIDAD
•⁠  ⁠Preguntas formuladas desde fuera de la empresa, no en primera persona corporativa.
•⁠  ⁠Respuestas que aporten valor real, no respuestas vagas o genéricas.
•⁠  ⁠Progresión lógica: de lo general (historia) a lo específico (contacto).
•⁠  ⁠No repitas información entre preguntas.
•⁠  ⁠No inventes datos que no estén en el documento.
•⁠  ⁠Responde en español.""",
        ),
        ("human", "Documento corporativo:\n{document}"),
    ]
)

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

    with st.expander("Parámetros de muestreo del LLM"):
        st.caption("⚠️ Kimi K2.6 fija `temperature=1.0` y `top_p=0.95` por diseño del proveedor.")
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, disabled=True,
                                help="Restringido por el proveedor para kimi-k2.6.")
        sampling_top_p = st.slider("top_p (nucleus sampling)", 0.0, 1.0, 0.95, 0.05, disabled=True,
                                   help="Restringido por el proveedor para kimi-k2.6.")
        frequency_penalty = st.slider("frequency_penalty", -2.0, 2.0, 0.0, 0.1,
                                      help="Penaliza tokens según su frecuencia previa.")
        presence_penalty = st.slider("presence_penalty", -2.0, 2.0, 0.0, 0.1,
                                     help="Penaliza tokens que ya aparecieron al menos una vez.")
        unlimited_tokens = st.checkbox("max_tokens ilimitado", value=True,
                                       help="Si está activo, no se envía max_tokens y la API usa el tope natural del modelo (Kimi K2.6: 256K contexto).")
        max_tokens = st.number_input("max_tokens", 64, 32768, 4096, 64,
                                     disabled=unlimited_tokens,
                                     help="Solo se aplica si 'ilimitado' está desactivado. Kimi K2.6 consume parte en razonamiento interno.")

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
    "temperature": 1.0,  # kimi-k2.6 solo acepta temperature=1.0
    "top_p": 0.95,        # kimi-k2.6 solo acepta top_p=0.95
    "frequency_penalty": frequency_penalty,
    "presence_penalty": presence_penalty,
    "max_retries": 2,
}

# Solo se envía max_tokens si el usuario desactivó el modo ilimitado
if not unlimited_tokens:
    llm_kwargs["max_tokens"] = int(max_tokens)

if thinking:
    llm_kwargs["thinking"] = True

llm = ChatMoonshot(**llm_kwargs)

# ── Tabs ──────────────────────────────────────────────────────────────────────

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

tab_qa, tab_summary, tab_faq, tab_faqs_user, tab_info = st.tabs(
    ["❓ Q&A", "📋 Resumen ejecutivo", "🙋 FAQs", "👤 FAQs Usuario", "ℹ️ Info"]
)

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
            try:
                with st.spinner("🧠 Kimi K2.6 generando respuesta..."):
                    for token in chain.stream(question):
                        full_answer += token
                        answer_box.markdown(full_answer or "_Procesando..._")
                if not full_answer.strip():
                    answer_box.warning("La API no devolvió contenido. Revisa los parámetros de muestreo o intenta de nuevo.")
            except Exception as e:
                st.error(f"Error generando respuesta: {e}")

# ── Tab Resumen ───────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Resumen ejecutivo de la empresa")
    st.caption("Generado automáticamente con Kimi K2.6 a partir de la base de conocimiento.")

    if st.button("Generar resumen", type="primary", key="summary_btn"):
        kb_text = KB_PATH.read_text(encoding="utf-8")
        document_excerpt = kb_text[:6000]

        chain = SUMMARY_PROMPT | llm | StrOutputParser()

        answer_box = st.empty()
        try:
            with st.spinner("🧠 Kimi K2.6 generando resumen (puede tardar 20-40s)..."):
                full_answer = chain.invoke({"document": document_excerpt})
            if not full_answer.strip():
                answer_box.warning("La API no devolvió contenido. Sube `max_tokens` o intenta de nuevo.")
            else:
                answer_box.markdown(full_answer)
        except Exception as e:
            st.error(f"Error generando resumen: {e}")

# ── Tab FAQ ───────────────────────────────────────────────────────────────────
with tab_faq:
    st.subheader("Preguntas Frecuentes (FAQs)")
    st.caption("10 FAQs generadas por Kimi K2.6 a partir de la base de conocimiento.")

    if st.button("Generar FAQs", type="primary", key="faq_btn"):
        kb_text = KB_PATH.read_text(encoding="utf-8")
        document_excerpt = kb_text[:6000]

        chain = FAQ_PROMPT | llm | StrOutputParser()

        answer_box = st.empty()
        try:
            with st.spinner("🧠 Kimi K2.6 generando FAQs (puede tardar 30-60s)..."):
                full_answer = chain.invoke({"document": document_excerpt})
            if not full_answer.strip():
                answer_box.warning("La API no devolvió contenido. Sube `max_tokens` o intenta de nuevo.")
            else:
                answer_box.markdown(full_answer)
        except Exception as e:
            st.error(f"Error generando FAQs: {e}")

# ── Tab FAQs Usuario ──────────────────────────────────────────────────────────
with tab_faqs_user:
    st.subheader("Preguntas frecuentes del usuario")
    st.caption("Listado de preguntas predefinidas para evaluar el sistema.")
    for i, q in enumerate(FAQS_USUARIO, 1):
        st.markdown(f"**{i}.** {q}")

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
