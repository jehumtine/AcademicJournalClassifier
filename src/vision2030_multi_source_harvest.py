#!/usr/bin/env python3
"""
vision2030_multi_source_harvest.py

Harvest metadata from OpenAlex + arXiv (CORE optional) using sector keywords,
auto-label items with the sector used to retrieve them, deduplicate, and export CSV.

Usage examples:
  python vision2030_multi_source_harvest.py --out vision2030_corpus.csv --per-sector 200
  python vision2030_multi_source_harvest.py --out vision2030_corpus.csv --per-sector 100 --core-key YOUR_CORE_KEY

Dependencies:
  pip install requests pandas feedparser
"""

import argparse
import requests
import time
import pandas as pd
import feedparser
import re
from urllib.parse import quote_plus

# ---------- CONFIG: Vision2030 seed mapping (expand as needed) ----------
VISION2030_MAP = {
    "Agriculture": ["agriculture", "farm", "crop", "irrigation", "livestock", "horticulture"],
    "Health": ["health", "medicine", "public health", "clinic", "hospital", "malaria", "immunization", "epidemic"],
    "Education": ["education", "school", "curriculum", "teacher", "university", "learning", "literacy"],
    "Infrastructure": ["infrastructure", "road", "bridge", "rail", "telecom", "water supply", "sewerage", "construction"],
    "Tourism": ["tourism", "tourist", "hotel", "hospitality", "heritage", "ecotourism"],
    "Energy": ["energy", "power", "renewable", "solar", "wind", "hydro", "electricity"],
    "Mining": ["mining", "minerals", "ore", "copper", "extraction", "quarry"],
    "Manufacturing": ["manufacturing", "factory", "industrial", "production", "assembly", "process engineering"],
    "Environment": ["environment", "biodiversity", "climate", "conservation", "pollution", "ecosystem"],
    "ICT/Technology": ["technology", "information technology", "ict", "computer", "software", "hardware", "ai", "sensor"],
    "Governance": ["governance", "policy", "administration", "regulation", "legislation"],
    "Finance/Trade": ["finance", "bank", "trade", "export", "import", "economy", "microfinance"],
    "Transport": ["transport", "logistics", "airport", "port", "shipping", "freight"],
    "Water_and_Sanitation": ["water", "sanitation", "wastewater", "hygiene", "drinking water"]
}
# ---------------------------------------------------------------------

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"  # requires core API key
ARXIV_API = "http://export.arxiv.org/api/query"

HEADERS_TEMPLATE = "vision2030-harvester/1.0 ({email})"

# ---------- helpers ----------
def safe_text(x):
    return (x or "").strip()

def normalize_title(title):
    """Lowercase, remove punctuation and extra whitespace for deduping fallback."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- OpenAlex fetcher ----------
def query_openalex(query, per_page=100, max_items=200, email=None):
    """
    Query OpenAlex works search using 'search' parameter (searches title+abstract) and return list of work dicts.
    We'll page using cursor-based pagination until max_items or end.
    """
    select_fields = ",".join([
        "id", "doi", "display_name", "abstract_inverted_index", "publication_date",
        "best_oa_location", "primary_location", "locations", "topics", "biblio", "open_access"
    ])
    results = []
    params = {
        "search": query,
        "per-page": per_page,
        "select": select_fields,
        "cursor": "*"
    }
    headers = {"User-Agent": HEADERS_TEMPLATE.format(email=email or "no-email")}
    while True:
        r = requests.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=60)
        if r.status_code != 200:
            print("OpenAlex request failed:", r.status_code, r.text[:200])
            break
        data = r.json()
        page_results = data.get("results", [])
        results.extend(page_results)
        if len(results) >= max_items:
            results = results[:max_items]
            break
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        time.sleep(0.5)
    # Normalize extracted fields
    output = []
    for w in results:
        doi = w.get("doi") or ""
        title = w.get("display_name") or ""
        # reconstruct abstract from inverted index if present
        abstract = ""
        ai = w.get("abstract_inverted_index")
        if ai:
            # technique from OpenAlex docs: join tokens in order of positions
            # inverted index: word->list of positions
            try:
                positions = {}
                for token, poslist in ai.items():
                    for p in poslist:
                        positions[p] = token
                abstract = " ".join(positions[i] for i in range(max(positions.keys())+1))
            except Exception:
                abstract = ""
        pub_date = w.get("publication_date") or ""
        # pdf candidate
        pdf = None
        bol = w.get("best_oa_location") or {}
        pdf = bol.get("url") or bol.get("url_for_pdf") or bol.get("pdf_url") or None
        if not pdf:
            pl = w.get("primary_location") or {}
            pdf = pl.get("url") or pl.get("pdf_url") or None
        topics = [t.get("display_name") for t in (w.get("topics") or []) if t.get("display_name")]
        host = (w.get("biblio") or {}).get("journal_title") or (w.get("biblio") or {}).get("journal") or ""
        output.append({
            "source": "openalex",
            "id": w.get("id"),
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": "",  # authors omitted in this select; could fetch if needed
            "published": pub_date,
            "pdf_url": pdf,
            "topics": ";".join(topics),
            "journal": host
        })
    return output

# ---------- arXiv fetcher ----------
def query_arxiv(query, max_results=100):
    """
    Query arXiv via their API (Atom feed). Returns list of dicts with metadata and pdf link.
    query is a simple keyword phrase used as all:<query>
    """
    out = []
    batch = 100 if max_results > 100 else max_results
    start = 0
    while start < max_results:
        n = min(batch, max_results - start)
        q = f"search_query=all:{quote_plus(query)}&start={start}&max_results={n}"
        url = ARXIV_API + "?" + q
        feed = feedparser.parse(url)
        if not feed.entries:
            break
        for e in feed.entries:
            # entry.id like http://arxiv.org/abs/xxxx
            entry_id = e.get("id", "")
            title = safe_text(e.get("title", ""))
            abstract = safe_text(e.get("summary", ""))
            published = e.get("published", "")
            # try to find pdf link among links or construct from id
            pdf_link = None
            for l in e.get("links", []):
                if l.get("type") == "application/pdf":
                    pdf_link = l.get("href")
                    break
            if not pdf_link and entry_id:
                # construct: replace /abs/ with /pdf/ and append .pdf if absent
                pdf_link = entry_id.replace("/abs/", "/pdf/")
                if not pdf_link.endswith(".pdf"):
                    pdf_link = pdf_link + ".pdf"
            authors = ";".join([a.get("name","") for a in (e.get("authors") or [])])
            out.append({
                "source": "arxiv",
                "id": entry_id,
                "doi": "",  # arXiv may have DOI but not always; feedparser can include it in links sometimes
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "pdf_url": pdf_link,
                "topics": "",  # arXiv categories not parsed here
                "journal": "arXiv"
            })
        start += n
        time.sleep(0.5)
    return out

# ---------- CORE fetcher (optional) ----------
def query_core(query, api_key, max_results=100):
    """
    Query CORE (requires an API key). Returns list of dicts. If no key provided, returns [].
    """
    if not api_key:
        return []
    out = []
    per_page = 100 if max_results > 100 else max_results
    url = CORE_SEARCH_URL
    headers = {"Authorization": api_key}
    params = {"q": query, "page": 1, "pageSize": per_page}
    r = requests.get(url, headers=headers, params=params, timeout=60)
    if r.status_code != 200:
        print("CORE request failed:", r.status_code, r.text[:200])
        return []
    data = r.json()
    docs = data.get("results") or []
    for d in docs:
        # CORE fields differ by provider; common: id, title, authors, year, downloadUrl
        doi = d.get("doi") or ""
        title = safe_text(d.get("title") or "")
        abstract = safe_text(d.get("abstract") or "")
        authors = ";".join(d.get("authors") or [])
        published = d.get("year") or ""
        pdf = d.get("downloadUrl") or ""
        out.append({
            "source": "core",
            "id": d.get("id"),
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": published,
            "pdf_url": pdf,
            "topics": ";".join(d.get("tags") or []),
            "journal": d.get("journalTitle") or ""
        })
    return out

# ---------- Orchestration ----------
def harvest_all(out_csv, per_sector=200, core_key=None, email=None):
    seen = {}  # key -> record (dedupe by doi or normalized title)
    rows = []
    total_queries = 0

    for sector, keywords in VISION2030_MAP.items():
        # form a compact query phrase (join keywords with OR for robustness in OpenAlex; but we use simple search)
        # We'll query using the most representative keyword(s) (first 2)
        qs = keywords[:2] if len(keywords) >= 2 else keywords
        # build a search phrase; simple approach: join with space (OpenAlex 'search' matches title/abstract)
        query_phrase = " ".join(qs)
        print(f"\n=== Harvest sector: {sector} using query: '{query_phrase}' ===")

        # 1) OpenAlex
        open_items = query_openalex(query_phrase, per_page=100, max_items=per_sector, email=email)
        print(f"OpenAlex returned {len(open_items)} items for {sector}")
        for item in open_items:
            key = (item.get("doi") or "").lower().strip()
            if not key:
                key = normalize_title(item.get("title") or "")
            if not key:
                continue
            if key in seen:
                # append sector candidate to provenance for later analysis
                seen[key]["assigned_sectors"].add(sector)
                seen[key]["sources"].add("openalex")
            else:
                rec = dict(item)
                rec["assigned_sectors"] = set([sector])
                rec["sources"] = set(["openalex"])
                rec["query_sector"] = sector
                rows.append(rec)
                seen[key] = rec

        time.sleep(0.5)

        # 2) arXiv
        arxiv_items = query_arxiv(query_phrase, max_results=int(per_sector/4))  # arXiv smaller per sector
        print(f"arXiv returned {len(arxiv_items)} items for {sector}")
        for item in arxiv_items:
            key = (item.get("doi") or "").lower().strip()
            if not key:
                key = normalize_title(item.get("title") or "")
            if not key:
                continue
            if key in seen:
                seen[key]["assigned_sectors"].add(sector)
                seen[key]["sources"].add("arxiv")
            else:
                rec = dict(item)
                rec["assigned_sectors"] = set([sector])
                rec["sources"] = set(["arxiv"])
                rec["query_sector"] = sector
                rows.append(rec)
                seen[key] = rec

        time.sleep(0.5)

        # 3) CORE (optional)
        if core_key:
            core_items = query_core(query_phrase, core_key, max_results=int(per_sector/2))
            print(f"CORE returned {len(core_items)} items for {sector}")
            for item in core_items:
                key = (item.get("doi") or "").lower().strip()
                if not key:
                    key = normalize_title(item.get("title") or "")
                if not key:
                    continue
                if key in seen:
                    seen[key]["assigned_sectors"].add(sector)
                    seen[key]["sources"].add("core")
                else:
                    rec = dict(item)
                    rec["assigned_sectors"] = set([sector])
                    rec["sources"] = set(["core"])
                    rec["query_sector"] = sector
                    rows.append(rec)
                    seen[key] = rec

        total_queries += 1

    # Finalize rows: flatten assigned_sectors and sources to strings, prepare dataframe
    final_rows = []
    for rec in rows:
        rr = {
            "source": rec.get("source", ""),
            "id": rec.get("id", ""),
            "doi": rec.get("doi", ""),
            "title": rec.get("title", ""),
            "abstract": rec.get("abstract", ""),
            "authors": rec.get("authors", ""),
            "published": rec.get("published", ""),
            "pdf_url": rec.get("pdf_url", ""),
            "topics": rec.get("topics", ""),
            "journal": rec.get("journal", ""),
            "assigned_sectors": ";".join(sorted(list(rec.get("assigned_sectors") or []))),
            "provenance_sources": ";".join(sorted(list(rec.get("sources") or []))),
            "query_sector": rec.get("query_sector", "")
        }
        final_rows.append(rr)

    df = pd.DataFrame(final_rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} deduplicated records to {out_csv}")
    return df

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Harvest OpenAlex + arXiv + (optional CORE) articles and auto-label to Vision2030 sectors.")
    ap.add_argument("--out", default="vision2030_corpus.csv", help="Output CSV path")
    ap.add_argument("--per-sector", type=int, default=200, help="Target items per sector (OpenAlex).")
    ap.add_argument("--core-key", default=None, help="Optional CORE API key")
    ap.add_argument("--email", default=None, help="Optional contact email for polite user-agent")
    args = ap.parse_args()

    harvest_all(out_csv=args.out, per_sector=args.per_sector, core_key=args.core_key, email=args.email)

if __name__ == "__main__":
    main()
