# Q&A — Smurfit Kappa Colombia

Asistente virtual basado en la información pública del sitio web oficial de Smurfit Kappa Colombia (Cartón de Colombia / Smurfit Westrock). Responde preguntas sobre la empresa usando exclusivamente su contenido scrapeado, sin conocimiento externo.

El proyecto incluye **dos versiones del asistente**:
- `app.py` — búsqueda por keywords (simple, rápida)
- `app_rag.py` — RAG con embeddings semánticos (más precisa)

---

## Requisitos previos

| Herramienta | Versión mínima | Instalación |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org) |
| [UV](https://docs.astral.sh/uv/) | cualquiera | `pip install uv` |
| [Ollama](https://ollama.com) | cualquiera | [ollama.com/download](https://ollama.com/download) |

### Modelos de Ollama necesarios

```bash
# Modelo de lenguaje (para respuestas)
ollama pull mistral-nemo       # recomendado para RAG (7B)
ollama pull llama3.2:3b        # alternativa ligera

# Modelo de embeddings (solo para la versión RAG)
ollama pull nomic-embed-text-v2-moe
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/simonc2123/LLM-ENTREGA-01.git
cd LLM-ENTREGA-01

# 2. Instalar dependencias con UV
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
  make rag      Lanza la app Q&A con embeddings semánticos (RAG)
  make all      Ejecuta scrape + kb + app en orden
```

### Flujo completo (primera vez)

```bash
make scrape   # ~10 min — descarga el sitio completo
make kb       # ~5 seg  — genera knowledge_base.md y knowledge_base_rag.md
make app      # lanza http://localhost:8501
# o
make rag      # lanza http://localhost:8501 (primera vez: genera embeddings ~2 min)
```

### Solo probar (los archivos ya están en el repo)

```bash
make app      # versión keyword, lista de inmediato
make rag      # versión RAG, genera embeddings la primera vez
```

---

## Archivos generados

```
output/
├── full_dump.json          # páginas scrapeadas (JSON completo)
├── knowledge_base.md       # KB compacta para versión keyword (~7k palabras)
├── knowledge_base_rag.md   # KB extendida para RAG (~35k palabras)
└── embeddings_rag.npz      # caché de embeddings (se regenera automáticamente)
```

---

## Comparación de versiones

| | `app.py` (keyword) | `app_rag.py` (RAG) |
|---|---|---|
| **Cómo encuentra la info** | Cuenta palabras exactas en común con párrafos del KB | Embeddings semánticos + BM25, fusionados con RRF |
| **Modelo de embeddings** | Ninguno | `nomic-embed-text-v2-moe` (local, Ollama) |
| **Tamaño del KB enviado al LLM** | ~400 palabras (los párrafos con más matches) | ~800 palabras (top-K chunks más relevantes) |
| **Tamaño total del KB** | 7.000 palabras | 35.000 palabras |
| **Secciones cubiertas** | About + 1 página de locations + inicio de productos | About completo + todas las plantas + top-25 productos |
| **Tiempo de inicio** | Instantáneo | ~2 min primera vez (embeddings), luego instantáneo |
| **Falla con sinónimos** | Sí — "dónde" no encuentra "ubicada" | No — entiende sinónimos y reformulaciones |
| **Falla con preguntas factuales** (números, direcciones) | Parcialmente | Menos — BM25 complementa donde el embedding falla |
| **Modelo LLM recomendado** | `llama3.2:3b` | `mistral-nemo` o `gemma3:12b` |

### ¿Cuándo usar cada versión?

- **Keyword (`make app`)**: preguntas directas con las mismas palabras que el sitio web, o si se necesita respuesta inmediata sin esperar embeddings.
- **RAG (`make rag`)**: preguntas formuladas con lenguaje natural, sinónimos, o sobre temas específicos de productos y plantas donde el corpus más grande marca la diferencia.

---

## Arquitectura

```
Scraper (scraper.py)
    └─► full_dump.json
            └─► build_kb.py
                    ├─► knowledge_base.md        (compacta, 7k palabras)
                    └─► knowledge_base_rag.md    (extendida, 35k palabras)

Versión keyword (app.py)
    Pregunta → keywords → párrafos con más matches → LLM → Respuesta

Versión RAG (app_rag.py)
    [Una vez] KB → chunks (200 palabras c/u) → embeddings → caché en disco
    [Por pregunta] Pregunta → embedding + BM25 → RRF → top-4 chunks → LLM → Respuesta
```

---

## Stack tecnológico

- **Scraper**: `aiohttp` + `BeautifulSoup4` (scraping asíncrono)
- **LLM inference**: `ollama` Python SDK (100% local, sin API externa)
- **Embeddings**: `nomic-embed-text-v2-moe` vía Ollama
- **Búsqueda vectorial**: `numpy` (coseno) + BM25 + Reciprocal Rank Fusion
- **UI**: `streamlit`
- **Gestión de entorno**: `uv`
