# Q&A — Smurfit Kappa Colombia

Asistente virtual basado en la información pública del sitio web oficial de Smurfit Kappa Colombia (Cartón de Colombia / Smurfit Westrock). Responde preguntas sobre la empresa usando exclusivamente su contenido scrapeado, sin conocimiento externo.

El proyecto incluye **tres variantes del asistente**:

| App | Motor | Modelo |
|---|---|---|
| `app.py` | Búsqueda por keywords (pipeline propio) | Ollama local |
| `app_rag.py` | RAG semántico — LangChain + FAISS | Ollama local (LLM + embeddings) |
| `app_rag_comercial.py` | RAG semántico — LangChain + FAISS | Google Gemini (embeddings) + Kimi K2.6 (LLM) |

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
ollama pull llama3.2:3b            # alternativa ligera (2 GB)

# Modelo de embeddings (solo para app_rag.py)
ollama pull nomic-embed-text-v2-moe
```

### API keys (solo para `app_rag_comercial.py`)

Crea un archivo `.env` en la raíz del proyecto con:

```bash
GOOGLE_API_KEY=tu_key_aqui       # embeddings — gratis en aistudio.google.com/apikey
MOONSHOT_API_KEY=tu_key_aqui     # LLM Kimi K2.6 — platform.moonshot.ai
```

---

## Instalación

```bash
git clone https://github.com/simonc2123/LLM-ENTREGA-01.git
cd LLM-ENTREGA-01
uv sync
```

---

## Uso

```bash
make          # muestra todos los comandos disponibles
```

```
Comandos disponibles:
  make scrape        Scrapea el sitio web de Smurfit Kappa Colombia
  make kb            Genera las bases de conocimiento (compacta + RAG)
  make app           Lanza la app Q&A con búsqueda por keywords
  make rag           Lanza la app RAG con Ollama 100% local
  make rag-comercial Lanza la app RAG con Gemini Embeddings + Kimi K2.6
  make lint          Verifica estilo del código
  make format        Corrige y formatea automáticamente
  make all           Ejecuta scrape + kb + app en orden
```

### Flujo completo (primera vez)

```bash
make scrape        # ~10 min — descarga el sitio completo
make kb            # ~5 seg  — genera knowledge_base.md y knowledge_base_rag.md
make rag           # lanza http://localhost:8501 (primera vez: genera índice FAISS ~2 min)
```

### Solo probar (los archivos ya están en el repo)

```bash
make app           # versión keyword, lista de inmediato
make rag           # versión RAG local, genera índice FAISS la primera vez
make rag-comercial # versión RAG comercial, requiere API keys en .env
```

---

## Archivos generados

```
output/
├── full_dump.json            # páginas scrapeadas (JSON completo)
├── knowledge_base.md         # KB compacta para app.py (~7k palabras)
├── knowledge_base_rag.md     # KB extendida para RAG (~35k palabras)
├── faiss_index/              # índice vectorial FAISS para app_rag.py (nomic embeddings)
└── faiss_index_gemini/       # índice vectorial FAISS para app_rag_comercial.py (Gemini embeddings)
```

---

## Comparación de versiones

| | `app.py` | `app_rag.py` | `app_rag_comercial.py` |
|---|---|---|---|
| **Retrieval** | Keywords exactas | Embeddings semánticos (nomic local) | Embeddings semánticos (Gemini API) |
| **LLM** | Ollama local | Ollama local | Kimi K2.6 (Moonshot AI) |
| **Entiende sinónimos** | No | Sí | Sí |
| **Contexto del LLM** | ~8K tokens | ~128K tokens | **256K tokens** |
| **Funcionalidades** | Solo Q&A | Q&A · Resumen · FAQs | Q&A · Resumen · FAQs · Modo thinking |
| **API keys requeridas** | Ninguna | Ninguna | Google + Moonshot |
| **Tiempo de inicio** | Instantáneo | ~2 min primera vez | ~5 min primera vez |
| **Costo por consulta** | $0 | $0 | ~$0.001–0.01 |
| **Modelo recomendado** | `llama3.2:3b` | `gemma3:12b` | `kimi-k2.6` |

---

## Arquitectura

```
Scraper (scraper.py)
    └─► output/full_dump.json
            └─► build_kb.py
                    ├─► output/knowledge_base.md         (compacta, app.py)
                    └─► output/knowledge_base_rag.md     (extendida, app_rag.py / app_rag_comercial.py)

app.py — Pipeline propio
    Pregunta → keywords → párrafos del KB puntuados → ChatOllama → Respuesta

app_rag.py — Pipeline LangChain + Ollama
    [Una vez] KB → RecursiveCharacterTextSplitter → OllamaEmbeddings → faiss_index/
    [Q&A]    Pregunta → FAISS top-k → ChatPromptTemplate → ChatOllama → Respuesta

app_rag_comercial.py — Pipeline LangChain + APIs comerciales
    [Una vez] KB → RecursiveCharacterTextSplitter → GoogleGenerativeAIEmbeddings (lotes) → faiss_index_gemini/
    [Q&A]    Pregunta → FAISS top-k → ChatPromptTemplate → ChatMoonshot (Kimi K2.6) → Respuesta
```

---

## Stack tecnológico

| Capa | `app_rag.py` | `app_rag_comercial.py` |
|---|---|---|
| **Scraping** | `aiohttp` + `BeautifulSoup4` (asíncrono) | ← igual |
| **Chunking** | `RecursiveCharacterTextSplitter` | ← igual |
| **Embeddings** | `OllamaEmbeddings` — nomic-embed-text-v2-moe | `GoogleGenerativeAIEmbeddings` — gemini-embedding-2-preview |
| **Vector store** | `FAISS` local | `FAISS` local |
| **LLM** | `ChatOllama` — gemma3:12b / llama3.2:3b | `ChatMoonshot` — kimi-k2.6 |
| **Orquestador** | `LangChain` LCEL | `LangChain` LCEL |
| **UI** | `Streamlit` | `Streamlit` |
| **Entorno** | `uv` | `uv` |
