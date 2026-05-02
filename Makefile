.PHONY: help scrape kb app rag rag-google all

# Muestra los comandos disponibles (target por defecto)
help:
	@echo ""
	@echo "Smurfit Kappa Colombia — Q&A System"
	@echo "====================================="
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make scrape   Scrapea el sitio web de Smurfit Kappa Colombia"
	@echo "  make kb       Genera las bases de conocimiento (compacta + RAG)"
	@echo "  make app      Lanza la app Q&A con busqueda por keywords"
	@echo "  make rag        Lanza la app RAG con Ollama 100% local"
	@echo "  make rag-google Lanza la app RAG con Google Gemini (+ fallback Ollama)"
	@echo "  make all        Ejecuta scrape + kb + app en orden"
	@echo ""

# Corre todo en orden
all: scrape kb app

# 1. Scrapear el sitio web de Smurfit Kappa Colombia
scrape:
	uv run scraper.py

# 2. Generar ambas bases de conocimiento (compacta + RAG)
kb:
	uv run build_kb.py

# 3. Lanzar la aplicación Q&A (versión keyword)
app:
	uv run streamlit run app.py

# 4. Lanzar la aplicación Q&A con RAG — Ollama local
rag:
	uv run streamlit run app_rag.py

# 5. Lanzar la aplicación Q&A con RAG — Google Gemini + fallback Ollama
rag-google:
	uv run streamlit run app_rag_google.py
