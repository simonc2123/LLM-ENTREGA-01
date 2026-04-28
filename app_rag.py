"""
app_rag.py — Q&A con RAG real (embeddings + búsqueda vectorial)
Modelo de embeddings: nomic-embed-text-v2-moe (local, Ollama)
"""

import hashlib
import re
import numpy as np
import streamlit as st
import ollama
from pathlib import Path

KB_PATH         = Path("output/knowledge_base_rag.md")
CACHE_PATH      = Path("output/embeddings_rag.npz")
EMBED_MODEL     = "nomic-embed-text-v2-moe:latest"
CHUNK_SIZE      = 200   # palabras por chunk
CHUNK_OVERLAP   = 40    # palabras de solapamiento entre chunks
TOP_K           = 4     # chunks más relevantes a recuperar
EMBEDDING_FAMILIES = {"nomic-bert-moe", "bert", "nomic"}

SYSTEM_PROMPT = """Eres un asistente virtual de Smurfit Kappa Colombia (Cartón de Colombia / Smurfit Westrock).

Se te proporcionan fragmentos oficiales del sitio web de la empresa. Responde la pregunta usando ÚNICAMENTE esa información.

REGLAS:
- Usa solo la información de los fragmentos proporcionados.
- Combina información de distintos fragmentos si es necesario.
- Si los fragmentos no contienen la respuesta, di: "Esa información no está disponible."
- Nunca uses conocimiento externo sobre la empresa.
- Responde siempre en español, de forma clara y directa."""

USER_PROMPT_TEMPLATE = """Fragmentos relevantes:
\"\"\"
{context}
\"\"\"

Pregunta: {question}"""


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_paragraphs(text: str) -> list[str]:
    """Divide texto en párrafos reales (líneas vacías como separador)."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def chunk_by_words(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide un bloque largo en chunks de palabras con solapamiento."""
    words  = text.split()
    step   = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks


def build_chunks(kb: str) -> list[str]:
    """
    Chunking estándar:
    1. Divide el KB en secciones (## / ###).
    2. Prefija el título de sección a cada chunk para que el embedding
       sepa de qué tema trata aunque el chunk no lo mencione.
    3. Dentro de cada sección: acumula párrafos hasta CHUNK_SIZE palabras.
       Si un párrafo solo ya supera CHUNK_SIZE, lo subdivide con solapamiento.
    """
    section_blocks = re.split(r"\n(?=#{1,3} )", kb)
    all_chunks: list[str] = []

    for block in section_blocks:
        block = block.strip()
        if not block:
            continue

        lines         = block.splitlines()
        first_line    = lines[0].strip()
        section_title = re.sub(r"^#{1,3}\s*", "", first_line) if first_line.startswith("#") else ""
        body          = "\n".join(lines[1:]).strip() if section_title else block
        prefix        = f"[Sección: {section_title}]\n" if section_title else ""

        paragraphs   = split_paragraphs(body)
        buffer_words: list[str] = []

        for para in paragraphs:
            para_words = para.split()

            if len(para_words) > CHUNK_SIZE:
                if buffer_words:
                    all_chunks.append(prefix + " ".join(buffer_words))
                    buffer_words = []
                for sub in chunk_by_words(para):
                    all_chunks.append(prefix + sub)
                continue

            if len(buffer_words) + len(para_words) > CHUNK_SIZE:
                if buffer_words:
                    all_chunks.append(prefix + " ".join(buffer_words))
                buffer_words = buffer_words[-CHUNK_OVERLAP:] + para_words
            else:
                buffer_words.extend(para_words)

        if len(buffer_words) >= 30:
            all_chunks.append(prefix + " ".join(buffer_words))

    return all_chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    """Embedea una lista de textos en batch y devuelve matriz (N, D)."""
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(resp.embeddings, dtype=np.float32)


def kb_hash(kb_text: str) -> str:
    return hashlib.md5(kb_text.encode()).hexdigest()


def load_cache(kb_text: str) -> tuple[list[str], np.ndarray] | None:
    """Devuelve (chunks, embeddings) desde disco si el KB no cambió."""
    if not CACHE_PATH.exists():
        return None
    data = np.load(CACHE_PATH, allow_pickle=True)
    if data["kb_hash"].item() != kb_hash(kb_text):
        return None
    return list(data["chunks"]), data["embeddings"]


def save_cache(kb_text: str, chunks: list[str], embeddings: np.ndarray) -> None:
    np.savez_compressed(
        CACHE_PATH,
        kb_hash=np.array(kb_hash(kb_text)),
        chunks=np.array(chunks, dtype=object),
        embeddings=embeddings,
    )


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Similaridad coseno entre un vector y una matriz de vectores."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    m = matrix / norms
    return m @ q


def bm25_scores(question: str, chunks: list[str]) -> np.ndarray:
    """
    Score BM25 simplificado: frecuencia de términos de la pregunta en cada chunk,
    normalizada por la longitud del chunk. Captura matches exactos que los
    embeddings pierden (nombres propios, direcciones, números).
    """
    stopwords = {"en", "de", "la", "el", "los", "las", "que", "y", "a", "se",
                 "un", "una", "del", "al", "con", "su", "por", "es", "son",
                 "qué", "cuál", "dónde", "cómo", "cuándo", "quién", "cuáles"}
    q_terms = [w.lower().strip("¿?.,;:") for w in question.split()
               if w.lower() not in stopwords and len(w) > 2]

    scores = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        lower  = chunk.lower()
        length = max(len(chunk.split()), 1)
        tf     = sum(lower.count(t) for t in q_terms)
        scores[i] = tf / (length ** 0.5)   # normalizar por raíz de longitud
    return scores


def retrieve(question: str, chunks: list[str], embeddings: np.ndarray, k: int = TOP_K) -> list[str]:
    """
    Búsqueda híbrida: combina score semántico (embeddings) con score léxico (BM25).
    Usa Reciprocal Rank Fusion para fusionar los dos rankings sin necesidad de
    calibrar pesos absolutos.
    """
    # Score semántico
    q_vec    = embed([question])[0]
    sem_scores = cosine_similarity(q_vec, embeddings)

    # Score léxico
    lex_scores = bm25_scores(question, chunks)

    # Reciprocal Rank Fusion (RRF k=60 es el estándar)
    RRF_K = 60
    n = len(chunks)
    sem_ranks = np.argsort(np.argsort(-sem_scores)) + 1   # rank 1 = mejor
    lex_ranks = np.argsort(np.argsort(-lex_scores)) + 1

    rrf = 1 / (RRF_K + sem_ranks) + 1 / (RRF_K + lex_ranks)

    top_k = np.argsort(-rrf)[:k]
    return [chunks[i] for i in top_k]


# ── Ollama ────────────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$",         "", text, flags=re.DOTALL)
    return text.strip()


def get_local_models() -> list[str]:
    try:
        return [
            m.model for m in ollama.list().models
            if not any(f in (m.details.family or "") for f in EMBEDDING_FAMILIES)
            and "embed" not in m.model.lower()
        ]
    except Exception:
        return []


def query_ollama(model: str, context: str, question: str):
    user_msg = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        stream=True,
        options={"temperature": 0.0},
    )
    for chunk in stream:
        yield chunk.message.content


# ── Carga con caché ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando índice...")
def load_index():
    """
    Carga chunks y embeddings. Usa caché en disco si el KB no cambió;
    si cambió, recalcula en batch y guarda el caché nuevo.
    """
    kb     = KB_PATH.read_text(encoding="utf-8")
    cached = load_cache(kb)
    if cached:
        return cached  # instantáneo

    chunks = build_chunks(kb)
    with st.spinner(f"Generando embeddings de {len(chunks)} chunks (solo la primera vez)..."):
        embs = embed(chunks)   # una sola llamada batch a Ollama
    save_cache(kb, chunks, embs)
    return chunks, embs


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Q&A RAG — Smurfit Kappa Colombia", layout="wide")
st.title("Q&A RAG — Cartón de Colombia / Smurfit Kappa")
st.caption("Búsqueda semántica con embeddings locales · nomic-embed-text-v2-moe · Ollama")

with st.sidebar:
    st.header("Configuración")
    models = get_local_models()
    if not models:
        st.error("No se pudo conectar a Ollama o no hay modelos disponibles.")
        st.stop()
    preferred   = next((m for m in ["gemma3:12b", "mistral-nemo:latest", "llama3.2:3b", "qwen3:4b"] if m in models), None)
    default_idx = models.index(preferred) if preferred else 0
    model       = st.selectbox("Modelo LLM", models, index=default_idx)
    top_k       = st.slider("Chunks a recuperar (top-k)", 1, 8, TOP_K)
    st.caption(f"Embedding: {EMBED_MODEL}")

if not KB_PATH.exists():
    st.error(f"No se encontró {KB_PATH}. Ejecuta primero: `make kb`")
    st.stop()

chunks, embeddings = load_index()
st.sidebar.success(f"{len(chunks)} chunks indexados")

question = st.text_area(
    "Escribe tu pregunta:",
    height=120,
    placeholder="Ej: ¿Dónde están las plantas? ¿Qué productos ofrecen? ¿Quién fundó la empresa?",
)

if st.button("Enviar", type="primary"):
    if not question.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        with st.spinner("Buscando fragmentos relevantes..."):
            relevant = retrieve(question.strip(), chunks, embeddings, k=top_k)
            context  = "\n\n---\n\n".join(relevant)

        with st.expander("Fragmentos recuperados", expanded=False):
            for i, chunk in enumerate(relevant, 1):
                st.markdown(f"**Chunk {i}:**\n\n{chunk}")

        answer_box  = st.empty()
        full_answer = ""
        with st.spinner("Generando respuesta..."):
            try:
                for token in query_ollama(model, context, question.strip()):
                    full_answer += token
                    visible = strip_thinking(full_answer)
                    if visible:
                        answer_box.markdown(visible)
            except Exception as e:
                st.error(f"Error: {e}")
