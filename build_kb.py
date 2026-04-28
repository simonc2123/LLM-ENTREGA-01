"""
build_kb.py
===========
Genera dos archivos de base de conocimiento desde output/full_dump.json:

  knowledge_base.md      — versión compacta (app.py, keyword search)
  knowledge_base_rag.md  — versión completa (app_rag.py, embeddings semánticos)
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

INPUT      = Path("output/full_dump.json")
OUTPUT     = Path("output/knowledge_base.md")
OUTPUT_RAG = Path("output/knowledge_base_rag.md")

PRIORITY_PATTERNS = [
    ("/co/about",                 "## Sobre la Empresa"),
    ("/co/locations",             "## Ubicaciones y Plantas"),
    ("/co/products-and-services", "## Productos y Servicios"),
]

ALWAYS_INCLUDE = {"https://www.smurfitkappa.com/co"}

# Versión compacta (keyword app) — modelos pequeños leen todo el doc
MAX_WORDS_COMPACT = 6_000

# Versión RAG — el modelo solo ve los chunks recuperados, el KB puede ser grande
PRODUCTS_LIMIT_RAG = 25   # top-25 páginas de productos por word_count

MIN_WORDS = 80

NAV_NOISE = {
    "Quiénes somos", "Qué hacemos", "Propósito", "Visión y estrategia",
    "Ética", "Nuestra historia", "Mensaje del CEO", "Servicios Speak Up",
    "Expandir ícono", "Expandir Icono", "Leer más", "Conoce más",
    "Más información", "Ver más", "Conoce cómo", "Lee el comunicado",
    "Filtrar por país", "Country", "selected", "Expandir",
    "Productos y servicios", "Sectores de mercado", "País",
    "Sostenibilidad", "Noticias", "Sala de prensa", "Contacto",
    "Inicio", "Inicio de sesión", "Cerrar sesión", "Buscar",
    "Siguenos", "Síguenos", "Política de privacidad", "Términos y condiciones",
    "Todos los derechos reservados",
}

HEADER = """# Smurfit Kappa Colombia — Base de Conocimiento

## Datos Clave

| Hecho | Valor |
|---|---|
| **Nombre actual** | Smurfit Westrock (desde 2024) |
| **Nombre histórico** | Cartón de Colombia |
| **Fundación en Colombia** | **4 de mayo de 1944**, Puerto Isaacs, Yumbo |
| **Fundación del grupo global** | **1934** en Irlanda (cajas de cartón) |
| **Adquirida por Jefferson Smurfit** | **1938** |
| **Fusión con Kappa Packaging** | **2005** → crea Smurfit Kappa Group |
| **Fusión con WestRock** | **2024** → crea Smurfit Westrock |
| **Productos principales** | Empaques corrugados, cartulina Óptima, sacos de papel, eCommerce, Bag-in-Box |
| **División Forestal** | 67.000 hectáreas en 6 departamentos, certificación FSC desde 2003 |
| **Presencia global** | +500 plantas, 40 países, 97.000 empleados, 57 molinos |

## Direcciones y Ubicaciones de Plantas en Colombia

Estas son las sedes, plantas y direcciones de Smurfit Kappa (Smurfit Westrock / Cartón de Colombia) en Colombia:

- **Corrugado Bogotá** — Avenida de las Américas # 56-41, Bogotá. Tel: +57 (601) 4320690. Empaques corrugados impresos hasta 4 colores, cajas regulares, bandejas, flores.
- **Corrugado Cali** — Calle 15 # 18-109, Puerto Isaacs, Yumbo (Cali). Tel: +57 (602) 4414000. Empaques corrugados impresos hasta 4 colores.
- **Corrugado Medellín (Guarne)** — Vereda Garrido, Lote La María, Guarne, Antioquia. Tel: +57 (604) 5409600. Planta nueva con certificación LEED Plata.
- **Corrugado Barranquilla** — Vía 40 # 62-112, Barranquilla. Empaques corrugados.
- **Molinos Cali** — Calle 15 # 18-109, Puerto Isaacs, Yumbo. Fabricación de pulpa y papeles para corrugar.
- **Molino Barranquilla** — Vía 40 # 85-695 Las Flores, Barranquilla. Tel: +57 (605) 3734500. Papeles testliner y corrugado medio reciclado.
- **Sacos de Papel Palmira** — Cra 33a # 24-59, Palmira (Valle del Cauca). Sacos de papel multicapas para industria.
- **División Forestal** — Sede principal en Yumbo. Gestiona 67.000 hectáreas en 6 departamentos. Certificación FSC desde 2003.

"""


def clean_content(text: str) -> str:
    text = re.sub(r"\[\[expand:[^\]]+\]\]", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.splitlines()
    seen  = set()
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean.append("")
            continue
        if stripped in NAV_NOISE:
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        clean.append(stripped)
    return "\n".join(clean).strip()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(("https", parsed.netloc, parsed.path, "", "", "")).rstrip("/")


def load_pages(data: dict) -> tuple[dict | None, list[dict], dict[str, list[dict]]]:
    """Carga y agrupa páginas deduplicadas por sección."""
    home_page    = None
    pinned_pages = []
    sections: dict[str, list[dict]] = {}
    seen_paths: set[str] = set()

    for page in data["pages"]:
        wc = page.get("word_count", 0)
        if wc < MIN_WORDS:
            continue

        clean_url = normalize_url(page["url"])
        if clean_url in seen_paths:
            continue
        seen_paths.add(clean_url)
        page["url"] = clean_url

        if clean_url == "https://www.smurfitkappa.com/co":
            home_page = page
            continue

        if clean_url in ALWAYS_INCLUDE:
            pinned_pages.append(page)
            continue

        for pattern, section_title in PRIORITY_PATTERNS:
            if pattern in clean_url:
                sections.setdefault(section_title, []).append(page)
                break

    # Ordenar por word_count desc dentro de cada sección
    for key in sections:
        sections[key].sort(key=lambda p: p["word_count"], reverse=True)

    return home_page, pinned_pages, sections


def render_pages(home_page, pinned_pages, sections, limits: dict[str, int],
                 max_words: int, skip_locations_root: bool = False) -> str:
    """Construye el markdown final dado los límites por sección."""
    lines       = [HEADER]
    total_words = 120  # header

    if home_page:
        content = clean_content(home_page["content"])
        lines.append(f"## Página principal\n\n{content}\n")
        total_words += len(content.split())

    for page in pinned_pages:
        content = clean_content(page["content"])
        lines.append(f"## {page['title']}\n\n{content}\n")
        total_words += len(content.split())

    for _, section_title in PRIORITY_PATTERNS:
        if section_title not in sections:
            continue
        limit = limits.get(section_title, 999)
        lines.append(f"\n{section_title}\n")
        for page in sections[section_title][:limit]:
            if 0 < max_words <= total_words:
                break
            # En modo RAG excluir la página raíz de locations (contenido mezclado)
            if skip_locations_root and page["url"] == "https://www.smurfitkappa.com/co/locations":
                print(f"  [skip] {page['url']} (raiz locations excluida en RAG)")
                continue
            content = clean_content(page["content"])
            words   = len(content.split())
            lines.append(f"### {page['title']}\n\n{content}\n")
            total_words += words
            print(f"  [+] {page['url']} ({words} w)")
        if 0 < max_words <= total_words:
            print(f"  [LIMITE] {total_words} palabras")
            break

    return "\n".join(lines), total_words


def write_summary(path: Path, total_words: int) -> None:
    print(f"  Palabras : {total_words:,}  |  Tokens ~{int(total_words*1.35):,}  |  {path.stat().st_size/1024:.1f} KB")
    print(f"  Ruta     : {path.resolve()}")


def build_knowledge_base():
    print(f"Leyendo {INPUT}...")
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    print(f"Total páginas disponibles: {len(data['pages'])}")

    home_page, pinned_pages, sections = load_pages(data)

    # ── Versión compacta ──────────────────────────────────────────────────────
    print("\n[Compacta] knowledge_base.md")
    compact_limits = {
        "## Sobre la Empresa":      999,
        "## Ubicaciones y Plantas": 1,
        "## Productos y Servicios": 999,
    }
    md_compact, words_compact = render_pages(
        home_page, pinned_pages, sections,
        limits=compact_limits, max_words=MAX_WORDS_COMPACT,
    )
    OUTPUT.write_text(md_compact, encoding="utf-8")
    write_summary(OUTPUT, words_compact)

    # ── Versión RAG ───────────────────────────────────────────────────────────
    print("\n[RAG] knowledge_base_rag.md")
    rag_limits = {
        "## Sobre la Empresa":      999,              # todas las páginas /about
        "## Ubicaciones y Plantas": 999,              # todas las plantas
        "## Productos y Servicios": PRODUCTS_LIMIT_RAG,
    }
    md_rag, words_rag = render_pages(
        home_page, pinned_pages, sections,
        limits=rag_limits, max_words=0,  # 0 = sin límite
        skip_locations_root=True,        # usar solo páginas individuales de plantas
    )
    OUTPUT_RAG.write_text(md_rag, encoding="utf-8")
    write_summary(OUTPUT_RAG, words_rag)

    print("\nOK ambos archivos generados.")


if __name__ == "__main__":
    build_knowledge_base()
