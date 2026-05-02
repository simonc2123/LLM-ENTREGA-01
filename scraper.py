"""
Cartón de Colombia (Smurfit Kappa) - Web Scraper Async
=======================================================
Crawl concurrente con aiohttp + guardado incremental cada 200 páginas.

Salida:
  output/pages.jsonl      → Un chunk por línea (para RAG / embeddings)
  output/full_dump.json   → Volcado completo con estadísticas
  output/url_index.txt    → Índice de URLs scrapeadas
  output/scrape_log.txt   → Log de ejecución en tiempo real
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from bs4 import BeautifulSoup

# ─── Configuración ────────────────────────────────────────────────────────────

BASE_URL = "https://www.smurfitkappa.com/co"
OUTPUT_DIR = Path("output")
CONCURRENCY = 10  # Peticiones simultáneas
DELAY_SECONDS = 0.5  # Pausa por worker entre peticiones
MAX_PAGES = 5000
REQUEST_TIMEOUT = 20
SAVE_EVERY = 200  # Guardado incremental cada N páginas
MAX_RETRIES = 3  # Reintentos ante 429 / errores transitorios
RETRY_BACKOFF = 2.0  # Segundos de espera base ante rate-limit (x2 por intento)

USER_AGENT = "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.com/bot)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
}

IGNORE_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
    "input",
    "meta",
    "link",
]

# ─── Logging ──────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "scrape_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── Utilidades ───────────────────────────────────────────────────────────────


def normalize_url(url: str) -> str:
    """Elimina el fragmento (#anchor) y la barra final de una URL."""
    url, _ = urldefrag(url)
    return url.rstrip("/")


def is_internal(url: str, base: str) -> bool:
    """Retorna True si la URL pertenece al mismo host y path base que BASE_URL."""
    base_parsed = urlparse(base)
    url_parsed = urlparse(url)
    same_host = url_parsed.netloc == base_parsed.netloc or url_parsed.netloc == ""
    same_path = url_parsed.path.startswith(base_parsed.path)
    return same_host and same_path


def is_scrapeable(url: str) -> bool:
    """Retorna True si la URL apunta a contenido HTML (excluye binarios, assets, etc.)."""
    skip_ext = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".webp",
        ".mp4",
        ".mp3",
        ".zip",
        ".xlsx",
        ".docx",
        ".pptx",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".css",
        ".js",
    }
    path = urlparse(url).path.lower()
    return not any(path.endswith(ext) for ext in skip_ext)


def url_fingerprint(url: str) -> str:
    """Genera un hash MD5 de la URL para usarlo como ID único de página."""
    return hashlib.md5(url.encode()).hexdigest()


def clean_text(soup: BeautifulSoup) -> str:
    """Extrae el texto visible de la página eliminando scripts, estilos y navegación."""
    for tag in soup.find_all(IGNORE_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_headings(soup: BeautifulSoup) -> list[dict]:
    """Extrae todos los headings (h1–h6) de la página con su nivel y texto."""
    return [
        {"level": int(tag.name[1]), "text": tag.get_text(strip=True)}
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if tag.get_text(strip=True)
    ]


def extract_metadata(soup: BeautifulSoup) -> dict:
    """Extrae meta tags (description, keywords, OpenGraph, canonical, lang) de la página."""
    meta = {}
    desc = soup.find("meta", attrs={"name": "description"})
    if desc:
        meta["description"] = desc.get("content", "").strip()
    kw = soup.find("meta", attrs={"name": "keywords"})
    if kw:
        meta["keywords"] = kw.get("content", "").strip()
    for og in soup.find_all("meta", attrs={"property": re.compile(r"^og:")}):
        key = og.get("property", "").replace("og:", "og_")
        val = og.get("content", "").strip()
        if key and val:
            meta[key] = val
    canonical = soup.find("link", attrs={"rel": "canonical"})
    if canonical:
        meta["canonical"] = canonical.get("href", "")
    html_tag = soup.find("html")
    if html_tag:
        meta["language"] = html_tag.get("lang", "")
    return meta


def extract_images(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Extrae todas las imágenes de la página con su src absoluto, alt y title."""
    images = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if src:
            images.append(
                {
                    "src": urljoin(page_url, src),
                    "alt": img.get("alt", "").strip(),
                    "title": img.get("title", "").strip(),
                }
            )
    return images


def extract_links(soup: BeautifulSoup, page_url: str) -> tuple[list[str], list[dict]]:
    """Separa los enlaces de la página en internos (para crawling) y externos."""
    internal_urls = []
    external_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:")):
            continue
        full_url = normalize_url(urljoin(page_url, href))
        text = a.get_text(strip=True)
        if is_internal(full_url, BASE_URL) and is_scrapeable(full_url):
            internal_urls.append(full_url)
        elif full_url.startswith("http"):
            external_links.append({"href": full_url, "text": text})
    return internal_urls, external_links


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Divide el texto en chunks con overlap para preservar contexto entre fragmentos."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n".join(current))
            overlap_text, overlap_len = [], 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_text.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current, current_len = overlap_text, overlap_len
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n".join(current))
    return chunks


# ─── Robots.txt ───────────────────────────────────────────────────────────────


async def load_robots(session: aiohttp.ClientSession) -> RobotFileParser | None:
    """Descarga y parsea robots.txt; retorna None si no está disponible."""
    robots_url = urljoin(BASE_URL, "/robots.txt")
    try:
        async with session.get(
            robots_url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        ) as resp:
            text = await resp.text()
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.parse(text.splitlines())
        log.info(f"robots.txt cargado desde {robots_url}")
        return rp
    except Exception as e:
        log.warning(f"No se pudo cargar robots.txt ({e}). Se asume acceso permitido.")
        return None


# ─── Parseo de página ─────────────────────────────────────────────────────────


def parse_page(url: str, html: str, status: int) -> dict:
    """Parsea el HTML de una página y retorna un documento estructurado con todo su contenido."""
    soup = BeautifulSoup(html, "lxml")
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    content = clean_text(soup)
    headings = extract_headings(soup)
    metadata = extract_metadata(soup)
    images = extract_images(soup, url)
    internal, ext = extract_links(soup, url)
    chunks = chunk_text(content)

    return {
        "id": url_fingerprint(url),
        "url": url,
        "title": title,
        "content": content,
        "chunks": chunks,
        "headings": headings,
        "metadata": metadata,
        "images": images,
        "external_links": ext,
        "internal_links": internal,
        "http_status": status,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "content_length": len(content),
        "word_count": len(content.split()),
    }


# ─── Guardado incremental ─────────────────────────────────────────────────────


def save_incremental(documents: list[dict], checkpoint: int) -> None:
    """Guarda JSONL y full_dump en disco. Llamado cada SAVE_EVERY páginas."""
    jsonl_path = OUTPUT_DIR / "pages.jsonl"
    chunk_count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in documents:
            for i, chunk in enumerate(doc["chunks"]):
                record = {
                    "id": f"{doc['id']}_chunk_{i}",
                    "source_id": doc["id"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "chunk_index": i,
                    "total_chunks": len(doc["chunks"]),
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "word_count": len(chunk.split()),
                        "scraped_at": doc["scraped_at"],
                        "headings": [h["text"] for h in doc["headings"][:5]],
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

    full_path = OUTPUT_DIR / "full_dump.json"
    summary = {
        "scrape_info": {
            "base_url": BASE_URL,
            "total_pages": len(documents),
            "total_chunks": chunk_count,
            "total_words": sum(d["word_count"] for d in documents),
            "checkpoint": checkpoint,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "pages": documents,
    }
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    index_path = OUTPUT_DIR / "url_index.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# Smurfit Kappa Colombia - URLs scrapeadas\n")
        f.write(f"# Total: {len(documents)} | Guardado: {datetime.now().isoformat()}\n\n")
        for doc in documents:
            f.write(
                f"{doc['url']}\n  Título: {doc['title']}\n  Palabras: {doc['word_count']:,}\n\n"
            )

    log.info(
        f"[GUARDADO] Checkpoint {checkpoint}: {len(documents)} páginas | {chunk_count} chunks → output/"
    )


# ─── Crawler async ────────────────────────────────────────────────────────────


async def fetch_page(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
) -> dict | None:
    """Descarga y parsea una página con reintentos ante rate-limit y errores transitorios."""
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                async with session.get(url, timeout=timeout) as resp:
                    # Rate-limit: esperar y reintentar
                    if resp.status == 429:
                        wait = RETRY_BACKOFF * attempt
                        log.warning(
                            f"[429 RATE LIMIT] {url} → esperando {wait}s (intento {attempt})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return None
                    html = await resp.text()

                # Parseo en thread pool para no bloquear el event loop
                doc = await loop.run_in_executor(None, parse_page, url, html, resp.status)
                log.info(
                    f"[OK] {url} | {doc['content_length']:,} chars | {len(doc['chunks'])} chunks"
                )
                await asyncio.sleep(DELAY_SECONDS)
                return doc

            except aiohttp.ClientResponseError as e:
                log.warning(f"[HTTP {e.status}] {url}")
                return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = RETRY_BACKOFF * attempt
                log.warning(
                    f"[ERROR intento {attempt}/{MAX_RETRIES}] {url}: {e} → reintentando en {wait}s"
                )
                await asyncio.sleep(wait)
            except Exception as e:
                log.error(f"[PARSE ERROR] {url}: {e}")
                return None

        log.warning(f"[ABANDONADO] {url} tras {MAX_RETRIES} intentos")
        return None


async def crawl() -> list[dict]:
    """Ejecuta el crawl completo del sitio de forma asíncrona con guardado incremental."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    visited: set[str] = set()
    queue: list[str] = [normalize_url(BASE_URL)]
    documents: list[dict] = []
    checkpoint = 0

    loop = asyncio.get_event_loop()
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        robots = await load_robots(session)
        log.info(f"Iniciando crawl async: {BASE_URL} | concurrencia={CONCURRENCY}")

        while queue and len(visited) < MAX_PAGES:
            # Tomar un batch del tamaño del semáforo
            batch_size = min(CONCURRENCY * 2, len(queue), MAX_PAGES - len(visited))
            batch = []
            while queue and len(batch) < batch_size:
                url = queue.pop(0)
                if url in visited or not is_scrapeable(url):
                    continue
                if robots and not robots.can_fetch(USER_AGENT, url):
                    log.info(f"[ROBOTS] Bloqueado: {url}")
                    continue
                visited.add(url)
                batch.append(url)

            if not batch:
                break

            # Fetch en paralelo
            tasks = [fetch_page(session, url, semaphore, loop) for url in batch]
            results = await asyncio.gather(*tasks)

            for doc in results:
                if not doc:
                    continue
                documents.append(doc)
                # Encolar nuevos enlaces internos
                for link in doc["internal_links"]:
                    if link not in visited and link not in queue:
                        queue.append(link)

            # Progreso
            log.info(
                f"--- Progreso: {len(documents)} páginas scrapeadas | {len(queue)} en cola ---"
            )

            # Guardado incremental
            if len(documents) >= checkpoint + SAVE_EVERY:
                checkpoint = len(documents)
                save_incremental(documents, checkpoint)

    log.info(f"Crawl completado: {len(documents)} páginas | {len(visited)} URLs visitadas")
    return documents


# ─── Resumen final ────────────────────────────────────────────────────────────


def print_summary(documents: list[dict]) -> None:
    """Imprime un resumen de la extracción con conteos de páginas, chunks y palabras."""
    total_chunks = sum(len(d["chunks"]) for d in documents)
    print("\n" + "=" * 60)
    print("  EXTRACCIÓN COMPLETADA")
    print("=" * 60)
    print(f"  Páginas scrapeadas : {len(documents)}")
    print(f"  Chunks generados   : {total_chunks}")
    print(f"  Total palabras     : {sum(d['word_count'] for d in documents):,}")
    print("\n  Archivos generados:")
    print("    - output/pages.jsonl      (para RAG/embeddings)")
    print("    - output/full_dump.json   (volcado completo)")
    print("    - output/url_index.txt    (índice de URLs)")
    print("    - output/scrape_log.txt   (log de ejecución)")
    print("=" * 60 + "\n")


# ─── Punto de entrada ─────────────────────────────────────────────────────────


def main() -> None:
    """Punto de entrada: ejecuta el crawl, guarda resultados e imprime el resumen."""
    import time

    start = time.time()

    docs = asyncio.run(crawl())

    if docs:
        save_incremental(docs, len(docs))
        print_summary(docs)
    else:
        log.error("No se extrajo ninguna página. Verifica la URL o la conexión.")

    log.info(f"Tiempo total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
