"""
build_kb.py
===========
Lee output/full_dump.json y genera output/knowledge_base.md
con las páginas más relevantes de Smurfit Kappa Colombia,
limpias y estructuradas para usarse como system prompt en Ollama.
"""

import json
import re
from pathlib import Path

INPUT  = Path("output/full_dump.json")
OUTPUT = Path("output/knowledge_base.md")

# Solo información corporativa general — sin noticias ni blogs genéricos
PRIORITY_PATTERNS = [
    ("/co/about",                 "## Sobre la Empresa"),
    ("/co/locations",             "## Ubicaciones y Plantas"),
    ("/co/products-and-services", "## Productos y Servicios"),
]

# Páginas concretas que siempre incluir (home + contacto)
ALWAYS_INCLUDE = {
    "https://www.smurfitkappa.com/co",
}

# /about completo + locations + productos disponibles
MAX_WORDS = 15_000

# Mínimo de palabras para que una página valga la pena
MIN_WORDS = 80


def clean_content(text: str) -> str:
    # Quitar artefactos del scraper
    text = re.sub(r"\[\[expand:[^\]]+\]\]", "", text)
    # Quitar líneas que son solo números o separadores de paginación
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Colapsar líneas vacías múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Quitar líneas repetidas de navegación (menos de 4 palabras)
    lines = text.splitlines()
    seen  = set()
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean.append("")
            continue
        words = stripped.split()
        if len(words) < 4 and stripped in seen:
            continue
        seen.add(stripped)
        clean.append(stripped)
    return "\n".join(clean).strip()


def match_section(url: str) -> str | None:
    for pattern, section in PRIORITY_PATTERNS:
        if pattern in url:
            return section
    return None


def build_knowledge_base():
    print(f"Leyendo {INPUT}...")
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    pages = data["pages"]
    print(f"Total páginas disponibles: {len(pages)}")

    # Agrupar páginas por sección (deduplicar por URL canónica)
    sections: dict[str, list[dict]] = {}
    home_page    = None
    pinned_pages = []   # páginas siempre incluidas (ALWAYS_INCLUDE menos home)
    seen_paths: set[str] = set()

    for page in pages:
        url        = page["url"]
        word_count = page.get("word_count", 0)

        if word_count < MIN_WORDS:
            continue

        # Normalizar: forzar https y quitar query strings (UTM, filtros)
        from urllib.parse import urlparse, urlunparse
        parsed     = urlparse(url)
        clean_url  = urlunparse(("https", parsed.netloc, parsed.path, "", "", ""))
        clean_url  = clean_url.rstrip("/")

        if clean_url in seen_paths:
            continue
        seen_paths.add(clean_url)
        page["url"] = clean_url  # usar la URL limpia en el MD

        if clean_url == "https://www.smurfitkappa.com/co":
            home_page = page
            continue

        if clean_url in ALWAYS_INCLUDE:
            pinned_pages.append(page)
            continue

        section = match_section(clean_url)
        if section:
            sections.setdefault(section, []).append(page)

    # Ordenar cada sección por word_count desc
    for key in sections:
        if key == "## Sobre la Empresa":
            limit = 999  # /about completo
        elif key == "## Ubicaciones y Plantas":
            limit = 1    # solo la página principal de locations
        else:
            limit = 999
        sections[key] = sorted(sections[key], key=lambda p: p["word_count"], reverse=True)[:limit]

    # Construir el Markdown
    lines      = []
    total_words = 0

    lines.append("# Base de Conocimiento — Smurfit Kappa Colombia (Smurfit Westrock)\n")
    lines.append(
        "> Documento generado automáticamente a partir del sitio web oficial "
        "https://www.smurfitkappa.com/co. Usado como contexto de conocimiento "
        "para el asistente virtual.\n"
    )

    # Resumen de hechos clave al inicio para evitar "lost in the middle"
    lines.append("## HECHOS CLAVE (referencia rápida)\n")
    lines.append("""**Empresa:** Smurfit Kappa Colombia, también llamada Cartón de Colombia. Desde 2024 opera como Smurfit Westrock tras la fusión global con WestRock.

**Fundación:** Smurfit Kappa (grupo global) fue fundada en 1934 en Irlanda, fabricando cajas de cartón. Fue comprada por Jefferson Smurfit en 1938. En Colombia, Cartón de Colombia nació el 4 de mayo de 1944 en Puerto Isaacs, Yumbo. En los 80s el Grupo Jefferson Smurfit adquirió la compañía colombiana. En 2005 Jefferson Smurfit se fusionó con Kappa Packaging formando Smurfit Kappa Group.

**Plantas y operaciones en Colombia:**
- Smurfit Westrock Corrugado Bogotá
- Smurfit Westrock Corrugado Cali
- Smurfit Westrock Corrugado Medellín
- Smurfit Westrock Corrugado Barranquilla
- Smurfit Westrock Molinos Cali (Yumbo)
- Smurfit Westrock Molino Barranquilla
- Smurfit Westrock Sacos de Papel (Cali)
- Smurfit Westrock División Forestal Colombia
- Nueva planta en Guarne, Antioquia (certificación LEED Plata)
- Sede principal (Headquarters) Colombia

**División Forestal:** 67.000–68.000 hectáreas de plantaciones forestales y bosques naturales en 6 departamentos de la zona andina (eje cafetero al sur del Cauca). Certificación FSC® desde 2003.

**Productos principales:** Empaques corrugados, papel para cartón ondulado, cartulina Óptima, sacos de papel, empaques eCommerce, Bag-in-Box, empaques industriales, empaques retail, displays.

**Presencia global:** Más de 500 plantas en 40 países, +97.000 empleados, 57 molinos de papel.

---
""")
    total_words += 150

    # Home primero
    if home_page:
        content = clean_content(home_page["content"])
        lines.append("## Página Principal\n")
        lines.append(f"**URL:** {home_page['url']}\n")
        lines.append(content + "\n")
        lines.append("---\n")
        total_words += len(content.split())

    # Páginas pinned (siempre incluidas)
    if pinned_pages:
        lines.append("## Información Corporativa Esencial\n")
        for page in pinned_pages:
            content = clean_content(page["content"])
            lines.append(f"### {page['title']}\n")
            lines.append(f"**URL:** {page['url']}\n")
            lines.append(content + "\n")
            lines.append("---\n")
            total_words += len(content.split())
            print(f"  [pinned] {page['url']} ({len(content.split())} palabras)")

    # Secciones ordenadas por PRIORITY_PATTERNS
    for _, section_title in PRIORITY_PATTERNS:
        if section_title not in sections:
            continue
        lines.append(f"{section_title}\n")
        for page in sections[section_title]:
            if total_words >= MAX_WORDS:
                break
            content = clean_content(page["content"])
            words   = len(content.split())
            lines.append(f"### {page['title']}\n")
            lines.append(f"**URL:** {page['url']}\n")
            lines.append(content + "\n")
            lines.append("---\n")
            total_words += words
            print(f"  [+] {page['url']} ({words} palabras)")

        if total_words >= MAX_WORDS:
            print(f"  [LIMITE] {total_words} palabras — deteniendo.")
            break

    md_content = "\n".join(lines)
    OUTPUT.write_text(md_content, encoding="utf-8")

    print(f"\nOK knowledge_base.md generado")
    print(f"  Palabras totales : {total_words:,}")
    print(f"  Tokens estimados : ~{int(total_words * 1.35):,}")
    print(f"  Tamaño archivo   : {OUTPUT.stat().st_size / 1024:.1f} KB")
    print(f"  Ruta             : {OUTPUT.resolve()}")


if __name__ == "__main__":
    build_knowledge_base()
