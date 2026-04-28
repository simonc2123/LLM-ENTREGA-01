import re
import streamlit as st
import ollama
from pathlib import Path

KB_PATH            = Path("output/knowledge_base.md")
EMBEDDING_FAMILIES = {"nomic-bert-moe", "bert", "nomic"}

USER_PROMPT_TEMPLATE = """Basándote ÚNICAMENTE en el siguiente contexto sobre Smurfit Kappa Colombia, 
responde la pregunta. No uses conocimiento externo ni inventes datos. Si la respuesta no está en el contexto, indícalo claramente.

CONTEXTO:
{knowledge_base}

Pregunta: {question}"""


def strip_thinking(text: str) -> str:
    # Quitar bloques cerrados <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Quitar bloque abierto sin cerrar
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


@st.cache_data
def load_knowledge_base() -> str:
    return KB_PATH.read_text(encoding="utf-8") if KB_PATH.exists() else ""


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
    prompt = USER_PROMPT_TEMPLATE.format(knowledge_base=kb, question=question)
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"temperature": 0.3},
        think=False,
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
        answer_box   = st.empty()
        full_answer  = ""
        with st.spinner("Procesando..."):
            try:
                for token in query_ollama(model, kb, question.strip()):
                    full_answer += token
                    visible = strip_thinking(full_answer)
                    if visible:
                        answer_box.markdown(visible)
            except Exception as e:
                st.error(f"Error: {e}")
