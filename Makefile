.PHONY: scrape kb app all

# Corre todo en orden
all: scrape kb app

# 1. Scrapear el sitio web de Smurfit Kappa Colombia
scrape:
	uv run scraper.py

# 2. Generar la base de conocimiento en Markdown
kb:
	uv run build_kb.py

# 3. Lanzar la aplicación Q&A
app:
	uv run streamlit run app.py
