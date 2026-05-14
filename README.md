# Asistente Virtual — Smurfit Kappa Colombia

Asistente conversacional basado en la información pública del sitio web oficial de Smurfit Kappa Colombia (Cartón de Colombia / Smurfit Westrock). Responde preguntas usando exclusivamente su contenido scrapeado, sin conocimiento externo.

El repositorio está dividido en **dos módulos** correspondientes a las entregas del taller:

| Módulo | Carpeta | Descripción |
|---|---|---|
| **1 — RAG** | raíz del repo (`app*.py`) | Q&A con RAG en tres variantes (keyword, local, comercial). Streamlit. |
| **2 — Agente** | `modulo2-agente/` | Agente conversacional con memoria, router de tools y UI tipo ChatGPT en React. |

---

# 📦 Módulo 1 — Q&A con RAG (Streamlit)

Tres variantes del asistente, todas sobre el mismo `knowledge_base.md`:

| App | Motor | Modelo |
|---|---|---|
| `app.py` | Búsqueda por keywords (pipeline propio) | Ollama local |
| `app_rag.py` | RAG semántico — LangChain + FAISS | Ollama local (LLM + embeddings) |
| `app_rag_comercial.py` | RAG semántico — LangChain + FAISS | Google Gemini (embeddings) + Kimi K2.6 (LLM) |

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

## Instalación

```bash
git clone https://github.com/simonc2123/LLM-ENTREGA-01.git
cd LLM-ENTREGA-01
uv sync
```

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

## Archivos generados

```
output/
├── full_dump.json            # páginas scrapeadas (JSON completo)
├── knowledge_base.md         # KB compacta para app.py (~7k palabras)
├── knowledge_base_rag.md     # KB extendida para RAG (~35k palabras)
├── faiss_index/              # índice vectorial FAISS para app_rag.py (nomic embeddings)
└── faiss_index_gemini/       # índice vectorial FAISS para app_rag_comercial.py (Gemini embeddings)
```

## Comparación de versiones del módulo 1

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

## Arquitectura del módulo 1

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

## Documento académico

El paper IEEE del módulo 1 está en [`Taller1_IEE.pdf`](Taller1_IEE.pdf) — formato IEEEtran, 10 páginas: scraping, build_kb, las 3 variantes de RAG, prompt engineering, análisis comparativo, trade-off soberanía vs. calidad.

---

# 🤖 Módulo 2 — Agente Conversacional ([`modulo2-agente/`](modulo2-agente/))

Evolución del RAG del módulo 1 hacia un **agente con memoria y router de herramientas**. El LLM decide automáticamente entre consultar la base documental o una fuente estructurada según la pregunta.

## Diferencias clave respecto al módulo 1

| Capacidad | Módulo 1 | Módulo 2 |
|---|---|---|
| **Memoria conversacional** | No — preguntas aisladas | Sí — historial persistido en Postgres |
| **Fuentes de información** | Una (RAG) | Dos: RAG + datos estructurados, con **router automático** |
| **Selector de modelo** | Editar código | UI: comercial (OpenAI GPT-5.1) o local (Ollama Qwen3.5:9B) |
| **Usuarios** | Único | 4 usuarios, cada uno con su propio historial |
| **UI** | Streamlit | React + Vite + Tailwind, tipo ChatGPT con streaming |
| **Parámetros de muestreo** | Sliders en sidebar | Modal de ajustes (temperature, top_p, top_k, etc.) |

## Stack del módulo 2

| Capa | Tecnología |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Orquestación** | LangChain — `create_tool_calling_agent` con dos tools |
| **Tool 1: RAG** | Reutiliza el `faiss_index_gemini/` del módulo 1 |
| **Tool 2: Estructurada** | `backend/data/company_info.json` (NIT, sedes, contactos, etc.) |
| **LLM comercial** | OpenAI GPT-5.1 |
| **LLM local** | Ollama Qwen3.5:9B |
| **Persistencia** | PostgreSQL (chats + messages) con connection pool |
| **Streaming** | Server-Sent Events (SSE) |
| **Frontend** | React 18 + Vite + TailwindCSS |

## Setup

### 1. Backend

```bash
cd modulo2-agente/backend
cp .env.example .env
# Edita .env con tus credenciales: OPENAI_API_KEY, GOOGLE_API_KEY, DB_PASSWORD
uv sync
uv run uvicorn main:app --reload --port 8000
```

En el primer arranque, crea las tablas `chats` y `messages` en Postgres automáticamente.

### 2. Frontend

```bash
cd modulo2-agente/frontend
npm install
npm run dev
```

Abre `http://localhost:5173`. El proxy de Vite redirige `/api/*` al backend en `:8000`.

### 3. Modelo local (opcional)

Si quieres usar la variante con Ollama:

```bash
ollama pull qwen3.5:9b
# asegurate de que `ollama serve` esté corriendo
```

## Flujo de la app

1. **Pantalla 1**: selección de modelo (Local / Comercial) — guarda preferencia en localStorage
2. **Pantalla 2**: login (cada usuario tiene su propio historial)
3. **Chat**: 
   - Sidebar con conversaciones previas (filtradas por usuario)
   - Streaming en tiempo real con badges del tool en uso (RAG o datos estructurados)
   - Efecto máquina de escribir con cursor parpadeante
   - Botón de ajustes para tunear temperature, top_p, etc.

## Endpoints principales

| Método | Path | Descripción |
|---|---|---|
| `GET` | `/users` | Lista de usuarios permitidos (público) |
| `GET` | `/chats` | Conversaciones del usuario (filtradas por `X-User-Id`) |
| `POST` | `/chats` | Crear nueva conversación |
| `GET` | `/chats/{id}/messages` | Historial de una conversación |
| `POST` | `/chats/{id}/messages/stream` | Enviar mensaje al agente con streaming SSE |

Headers requeridos en endpoints autenticados:
- `X-User-Id`: uno de `antonio`, `bradley`, `camilo`, `simon`
- `X-Model-Provider`: `commercial` o `local`

## Casos de prueba (cumplimiento del enunciado)

| Caso | Pregunta de ejemplo | Tool esperada |
|---|---|---|
| **RAG** | *¿Qué es la cartulina Óptima y para qué sirve?* | `search_knowledge_base` |
| **Datos estructurados** | *¿Cuál es el NIT de la empresa?* | `get_company_info` |
| **Memoria** | *Háblame de sus productos* → *¿Cuál mencionaste primero?* | RAG, luego resolución por contexto |
| **Multi-tool** | *¿Qué hacen en la planta de Cali y dónde queda?* | Ambas tools |

Ver el [`README.md` del módulo 2`](modulo2-agente/README.md) para detalles de prompt engineering, arquitectura interna y reutilización del módulo 1.

---

## Stack tecnológico global

| Capa | Módulo 1 — Local | Módulo 1 — Comercial | Módulo 2 |
|---|---|---|---|
| **Scraping** | `aiohttp` + `BeautifulSoup4` | ← igual | (reusa output del módulo 1) |
| **Chunking** | `RecursiveCharacterTextSplitter` | ← igual | (reusa FAISS) |
| **Embeddings** | `OllamaEmbeddings` nomic | `GoogleGenerativeAIEmbeddings` Gemini | ← igual que comercial |
| **Vector store** | `FAISS` local | `FAISS` local | ← reusa `faiss_index_gemini/` |
| **LLM** | `ChatOllama` gemma3 | `ChatMoonshot` Kimi K2.6 | `ChatOpenAI` GPT-5.1 / `ChatOllama` Qwen3.5 |
| **Orquestador** | `LangChain` LCEL | `LangChain` LCEL | `LangChain` `create_tool_calling_agent` |
| **Memoria** | — | — | PostgreSQL con `psycopg-pool` |
| **API** | — | — | FastAPI + SSE streaming |
| **UI** | `Streamlit` | `Streamlit` | React + Vite + Tailwind |
| **Entorno** | `uv` | `uv` | `uv` (back) + `npm` (front) |
