import re
import streamlit as st
import ollama
from pathlib import Path

KB_PATH            = Path("output/knowledge_base.md")
EMBEDDING_FAMILIES = {"nomic-bert-moe", "bert", "nomic"}

SYSTEM_PROMPT = """Eres un asistente virtual de Smurfit Kappa Colombia (Cartón de Colombia / Smurfit Westrock).

Se te proporcionará un fragmento de información oficial sobre la empresa. Tu trabajo es responder la pregunta del usuario ÚNICAMENTE con ese fragmento.

REGLAS:
- USA SOLO la información del fragmento proporcionado.
- Si el fragmento contiene la respuesta, dala de forma clara y completa.
- Si el fragmento NO contiene información suficiente, responde: "Esa información no está disponible."
- NUNCA uses tu conocimiento previo sobre la empresa — puede estar desactualizado o ser incorrecto.
- Responde siempre en español."""

USER_PROMPT_TEMPLATE = """Fragmento oficial:
\"\"\"
{context}
\"\"\"

Pregunta: {question}"""


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def load_knowledge_base() -> str:
    return KB_PATH.read_text(encoding="utf-8") if KB_PATH.exists() else ""


def retrieve_context(kb: str, question: str, max_chars: int = 3000) -> str:
    """
    Recupera los fragmentos más relevantes del KB mediante búsqueda por keywords.
    Devuelve siempre la tabla de Datos Clave + los párrafos más relevantes.
    """
    # Extraer tabla de datos clave (siempre incluir — está al inicio)
    key_table = ""
    table_match = re.search(r"## Datos Clave.*?\n\n(.*?)\n\n", kb, re.DOTALL)
    if table_match:
        key_table = table_match.group(0).strip()

    # Dividir el KB en párrafos (bloques separados por líneas vacías)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", kb) if p.strip()]

    # Palabras clave de la pregunta (quitar stopwords simples)
    stopwords = {"fue", "el", "la", "los", "las", "de", "en", "que", "y", "a",
                 "es", "se", "un", "una", "del", "al", "con", "su", "por",
                 "qué", "quién", "cuándo", "dónde", "cómo", "cuántos", "cuál"}
    q_words = {w.lower().strip("¿?.,") for w in question.split() if w.lower() not in stopwords}

    def score(para: str) -> int:
        lower = para.lower()
        return sum(1 for w in q_words if w in lower)

    # Ordenar párrafos por relevancia
    scored = sorted(paragraphs, key=score, reverse=True)

    # Construir contexto: tabla de datos + párrafos más relevantes
    context_parts = [key_table] if key_table else []
    used_chars = len(key_table)

    for para in scored:
        if para == key_table:
            continue
        if score(para) == 0:
            break
        if used_chars + len(para) > max_chars:
            break
        context_parts.append(para)
        used_chars += len(para)

    return "\n\n".join(context_parts)


def get_local_models() -> list[str]:
    try:
        return [
            m.model for m in ollama.list().models
            if not any(f in (m.details.family or "") for f in EMBEDDING_FAMILIES)
            and "embed" not in m.model.lower()
        ]
    except Exception:
        return []


def query_ollama(model: str, kb: str, question: str):
    context  = retrieve_context(kb, question)
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


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Q&A Smurfit Kappa Colombia", layout="wide")
st.title("Q&A — Cartón de Colombia / Smurfit Kappa")
st.caption("Asistente virtual basado en la información pública del sitio web oficial.")

with st.sidebar:
    st.header("Configuración")
    models = get_local_models()
    if not models:
        st.error("No se pudo conectar a Ollama o no hay modelos disponibles.")
        st.stop()
    preferred   = next((m for m in ["llama3.2:3b", "qwen3-fast:latest", "qwen3:4b", "gemma4:e2b"] if m in models), None)
    default_idx = models.index(preferred) if preferred else 0
    model       = st.selectbox("Modelo", models, index=default_idx)
    st.caption("Ollama local")

kb       = load_knowledge_base()
question = st.text_area(
    "Escribe tu pregunta:",
    height=120,
    placeholder="Ej: ¿Dónde están las plantas? ¿Qué productos ofrecen?",
)

if st.button("Enviar", type="primary"):
    if not question.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        answer_box  = st.empty()
        full_answer = ""
        with st.spinner("Procesando..."):
            try:
                for token in query_ollama(model, kb, question.strip()):
                    full_answer += token
                    visible = strip_thinking(full_answer)
                    if visible:
                        answer_box.markdown(visible)
            except Exception as e:
                st.error(f"Error: {e}")
