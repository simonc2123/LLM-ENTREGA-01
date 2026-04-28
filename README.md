# Q&A — Smurfit Kappa Colombia

Asistente virtual basado en la información pública del sitio web oficial de Smurfit Kappa Colombia (Cartón de Colombia / Smurfit Westrock). Responde preguntas sobre la empresa usando exclusivamente su contenido scrapeado, sin conocimiento externo.

El proyecto incluye **dos versiones del asistente**:
- `app.py` — búsqueda por keywords (pipeline propio, sin dependencias externas)
- `app_rag.py` — RAG completo con LangChain + FAISS (3 funcionalidades: Q&A, Resumen, FAQs)

---

## Requisitos previos

| Herramienta | Versión mínima | Instalación |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org) |
| [UV](https://docs.astral.sh/uv/) | cualquiera | `pip install uv` |
| [Ollama](https://ollama.com) | cualquiera | [ollama.com/download](https://ollama.com/download) |

### Modelos de Ollama necesarios

```bash
# Modelos de lenguaje (para respuestas) — instalar al menos uno
ollama pull gemma3:12b             # recomendado (mejor calidad)
ollama pull mistral-nemo           # alternativa 7B
ollama pull llama3.2:3b            # alternativa ligera (2GB)

# Modelo de embeddings (solo para la versión RAG)
ollama pull nomic-embed-text-v2-moe
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/simonc2123/LLM-ENTREGA-01.git
cd LLM-ENTREGA-01

# 2. Instalar dependencias con UV (incluye LangChain, FAISS, Streamlit)
uv sync
```

---

## Uso

```bash
make          # muestra todos los comandos disponibles
```

```
Comandos disponibles:
  make scrape   Scrapea el sitio web de Smurfit Kappa Colombia
  make kb       Genera las bases de conocimiento (compacta + RAG)
  make app      Lanza la app Q&A con búsqueda por keywords
  make rag      Lanza la app Q&A con LangChain + FAISS (recomendada)
  make all      Ejecuta scrape + kb + app en orden
```

### Flujo completo (primera vez)

```bash
make scrape   # ~10 min — descarga el sitio completo
make kb       # ~5 seg  — genera knowledge_base.md y knowledge_base_rag.md
make rag      # lanza http://localhost:8501 (primera vez: genera índice FAISS ~2 min)
```

### Solo probar (los archivos ya están en el repo)

```bash
make app      # versión keyword, lista de inmediato
make rag      # versión LangChain+FAISS, genera índice la primera vez
```

---

## Archivos generados

```
output/
├── full_dump.json          # páginas scrapeadas (JSON completo)
├── knowledge_base.md       # KB compacta para versión keyword (~7k palabras)
├── knowledge_base_rag.md   # KB extendida para RAG (~35k palabras)
└── faiss_index/            # índice vectorial FAISS (se regenera automáticamente)
```

---

## Comparación de versiones

| | `app.py` (keyword) | `app_rag.py` (LangChain + FAISS) |
|---|---|---|
| **Framework** | Pipeline propio | LangChain (LCEL chains) |
| **Retrieval** | Conteo de palabras exactas | `OllamaEmbeddings` + FAISS similarity search |
| **Vector store** | Ninguno | FAISS (persistido en disco) |
| **Chunking** | Chunker manual | `RecursiveCharacterTextSplitter` |
| **Prompts** | f-strings | `ChatPromptTemplate` |
| **Funcionalidades** | Solo Q&A | Q&A · Resumen ejecutivo · FAQs |
| **Tamaño del KB** | 7.000 palabras | 35.000 palabras |
| **Tiempo de inicio** | Instantáneo | ~2 min primera vez, luego instantáneo |
| **Entiende sinónimos** | No | Sí — embeddings semánticos |
| **Modelo recomendado** | `llama3.2:3b` | `gemma3:12b` |

---

## Arquitectura (versión RAG)

```
Scraper (scraper.py)
    └─► full_dump.json
            └─► build_kb.py
                    ├─► knowledge_base.md            (compacta, app.py)
                    └─► knowledge_base_rag.md        (extendida, app_rag.py)

app_rag.py — Pipeline LangChain
    [Una vez al iniciar]
    KB → RecursiveCharacterTextSplitter → chunks
       → OllamaEmbeddings → FAISS index (guardado en disco)

    [Tab Q&A — por cada pregunta]
    Pregunta → FAISS retriever (top-k) → ChatPromptTemplate → ChatOllama → Respuesta

    [Tab Resumen]
    KB (primeros 6k chars) → ChatPromptTemplate → ChatOllama → Resumen ejecutivo

    [Tab FAQs]
    KB (primeros 6k chars) → ChatPromptTemplate → ChatOllama → 10 preguntas y respuestas
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| **Scraping** | `aiohttp` + `BeautifulSoup4` (asíncrono) |
| **Framework LLM** | `LangChain` (LCEL, ChatPromptTemplate, chains) |
| **LLM local** | `ChatOllama` — gemma3:12b / mistral-nemo / llama3.2:3b |
| **Embeddings** | `OllamaEmbeddings` — nomic-embed-text-v2-moe |
| **Vector store** | `FAISS` (local, sin servidor) |
| **Chunking** | `RecursiveCharacterTextSplitter` |
| **UI** | `Streamlit` (3 tabs: Q&A, Resumen, FAQs) |
| **Gestión de entorno** | `uv` |
