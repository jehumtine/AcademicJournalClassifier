#!/usr/bin/env python3
"""
download_technology_topics_resolver.py

1) Finds OpenAlex topics matching a search term (default "technology").
2) Lists top-N candidate topics (by works_count).
3) Optionally iterates those topic IDs and downloads OA PDFs for works affiliated with Zambia (country code ZM).
   For each work it attempts:
     - Unpaywall lookup by DOI (preferred)
     - DOI content negotiation (Accept: application/pdf)
     - HEAD check of candidate URL
     - GET landing page + parse for PDF (meta tags, link rel, <a href="*.pdf">, raw .pdf in HTML)
   Writes metadata.csv and saves PDFs to out/<topic_display_name>/.

Usage examples:
  # list candidate topics (no downloads)
  python download_technology_topics_resolver.py --list-only --top-n 8 --email you@domain.com

  # list and then download top 3 topics (Zambia)
  python download_technology_topics_resolver.py --top-n 3 --email you@domain.com

Requirements:
  pip install requests beautifulsoup4
"""

import argparse
import csv
import os
import re
import sys
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

# ---------- Configurable defaults ----------
OPENALEX_TOPICS_URL = "https://api.openalex.org/topics"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2/"   # append DOI, params: email
USER_AGENT_TEMPLATE = "openalex-downloader/1.0 ({email})"

# ---------- Helpers ----------
def safe_filename(s):
    s = re.sub(r'[\\/:"*?<>|]+', '_', s or '')
    s = re.sub(r'\s+', '_', s.strip())
    return s[:180]

def fetch_candidate_topics(search_term="technology", per_page=50, max_topics=10, email=None):
    """
    Query OpenAlex /topics for the search_term and return top max_topics sorted by works_count.
    Returns a list of dicts (id, display_name, works_count, keywords).
    """
    params = {"search": search_term, "per-page": per_page, "select": "id,display_name,description,keywords,works_count"}
    headers = {"User-Agent": USER_AGENT_TEMPLATE.format(email=email or "no-email")}
    try:
        r = requests.get(OPENALEX_TOPICS_URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("Failed to fetch topics from OpenAlex:", e)
        return []

    topics = data.get("results", []) or []
    topics_sorted = sorted(topics, key=lambda t: t.get("works_count", 0), reverse=True)
    return topics_sorted[:max_topics]

# ---------- PDF resolution helpers ----------
def is_doi_url(u):
    if not u:
        return False
    u = u.lower()
    return "doi.org/" in u or re.match(r"^10\.\d+\/", u) is not None

def unpaywall_pdf_for_doi(doi, email, headers):
    """Return an OA pdf URL via Unpaywall (or None)."""
    if not doi or not email:
        return None
    api = UNPAYWALL_API + doi
    try:
        r = requests.get(api, params={"email": email}, headers=headers, timeout=20)
        if r.status_code == 200:
            j = r.json()
            bol = j.get("best_oa_location") or {}
            pdf = bol.get("url_for_pdf") or bol.get("url")
            if pdf:
                return pdf
            for loc in j.get("oa_locations", []) or []:
                if loc.get("url_for_pdf"):
                    return loc.get("url_for_pdf")
                if loc.get("url") and loc.get("url").lower().endswith(".pdf"):
                    return loc.get("url")
    except Exception:
        pass
    return None

def extract_pdf_from_html(url, html_text):
    """Parse HTML for common PDF signals: meta citation_pdf_url, <link rel="alternate" type="application/pdf">, <a href="*.pdf">, regex fallback."""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception:
        return None

    # meta tags
    meta_candidates = ["citation_pdf_url", "pdf_url"]
    for mn in meta_candidates:
        tag = soup.find("meta", attrs={"name": mn}) or soup.find("meta", attrs={"property": mn})
        if tag and tag.get("content"):
            return urljoin(url, tag.get("content"))

    # link rel alternate type application/pdf
    for link in soup.find_all("link", href=True):
        ltype = (link.get("type") or "").lower()
        if "pdf" in ltype:
            return urljoin(url, link.get("href"))

    # anchor tags that end with .pdf
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            return urljoin(url, href)
        if "download" in href.lower() and ("pdf" in href.lower() or "fulltext" in href.lower()):
            return urljoin(url, href)

    # regex fallback
    m = re.search(r"(https?:\/\/[^\s'\"<>]+\.pdf)", html_text, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return None

def try_resolve_pdf_url(candidate_url, doi=None, email=None, headers=None):
    """
    Given a candidate (OA location / DOI / landing page), try to resolve to a direct PDF URL.
    Returns pdf_url string or None.
    """
    headers = headers or {}
    # 1) Unpaywall by DOI (fast & reliable)
    if doi and email:
        pdf = unpaywall_pdf_for_doi(doi, email, headers)
        if pdf:
            return pdf

    # 2) If this looks like a DOI, try content-negotiation at doi.org
    if is_doi_url(candidate_url):
        if candidate_url.startswith("10."):
            doi_url = "https://doi.org/" + candidate_url
        else:
            doi_url = candidate_url if candidate_url.startswith("http") else "https://" + candidate_url
        try:
            r = requests.get(doi_url, headers={**headers, "Accept": "application/pdf"}, stream=True, timeout=30, allow_redirects=True)
            ctype = (r.headers.get("Content-Type") or "").lower()
            final = r.url or doi_url
            if "pdf" in ctype or final.lower().endswith(".pdf"):
                return final
            # check first bytes
            first = r.raw.read(4)
            r.raw.close()
            if first == b'%PDF':
                return final
        except Exception:
            pass

    if not candidate_url:
        return None

    # 3) HEAD quick-check for PDF content-type
    try:
        h = requests.head(candidate_url, headers=headers, allow_redirects=True, timeout=15)
        ctype = (h.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype:
            return h.url
    except Exception:
        pass

    # 4) GET landing page and parse for PDF links/meta
    try:
        r = requests.get(candidate_url, headers=headers, timeout=30, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        final_url = r.url
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            return final_url
        pdf_candidate = extract_pdf_from_html(final_url, r.text)
        if pdf_candidate:
            try:
                h2 = requests.head(pdf_candidate, headers=headers, allow_redirects=True, timeout=15)
                c2 = (h2.headers.get("Content-Type") or "").lower()
                if "pdf" in c2 or h2.url.lower().endswith(".pdf"):
                    return h2.url
            except Exception:
                return pdf_candidate
    except Exception:
        pass

    return None

def download_file_with_validation(url, out_path, headers=None, timeout=40):
    """
    Download streaming; validate PDF by checking first bytes for '%PDF' or URL ending with .pdf.
    Returns (ok_bool, error_msg_or_None).
    """
    headers = headers or {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            it = r.iter_content(chunk_size=4096)
            try:
                first = next(it)
            except StopIteration:
                return False, "empty_response"
            final = r.url or url
            if not (final.lower().endswith(".pdf") or first[:4] == b'%PDF' or b'%PDF' in first):
                return False, "not_pdf"
            with open(out_path, "wb") as f:
                f.write(first)
                for chunk in it:
                    if chunk:
                        f.write(chunk)
        return True, None
    except requests.HTTPError as e:
        return False, f"http_error_{e.response.status_code}"
    except Exception as e:
        return False, str(e)

# ---------- Per-topic downloader ----------
def download_for_topic(topic_id, topic_name, out_base="downloads", per_page=200, sleep=1.0, email=None, max_pages=None):
    safe_topic_name = safe_filename(topic_name or topic_id)
    out_dir = os.path.join(out_base, safe_topic_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metadata.csv")
    writer = None

    select_fields = ",".join([
        "id", "display_name", "doi", "publication_date",
        "best_oa_location", "primary_location", "locations", "topics", "open_access", "biblio"
    ])

    params = {
        "filter": f"topics.id:{topic_id}",
        "per-page": per_page,
        "cursor": "*",
        "select": select_fields
    }
    headers = {"User-Agent": USER_AGENT_TEMPLATE.format(email=email or "no-email")}
    total = 0
    page_count = 0

    print(f"\n=== Topic: {topic_name} ({topic_id}) ===")
    print("Saving into:", out_dir)

    while True:
        page_count += 1
        print(f"[{topic_name}] Querying cursor: {params.get('cursor')} (page {page_count})")
        try:
            r = requests.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                print(f"[{topic_name}] Rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[{topic_name}] Request failed:", e)
            print("Sleeping 10s and retrying...")
            time.sleep(10)
            continue

        if writer is None:
            csvfile = open(csv_path, "w", newline="", encoding="utf-8")
            fieldnames = ["openalex_id", "title", "doi", "publication_date", "pdf_url", "saved_file", "error", "topics", "journal"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        results = data.get("results", [])
        for w in results:
            total += 1
            doi = w.get("doi")
            # candidate from best_oa_location, then primary_location, then first locations entry
            candidate = None
            bol = w.get("best_oa_location") or {}
            candidate = bol.get("url") or bol.get("pdf_url") or bol.get("landing_page_url")
            if not candidate:
                pl = w.get("primary_location") or {}
                candidate = pl.get("url") or pl.get("pdf_url") or pl.get("landing_page_url")
            if not candidate:
                locs = w.get("locations") or []
                if locs:
                    candidate = locs[0].get("url") or locs[0].get("pdf_url") or locs[0].get("landing_page_url")
            pdf_url = try_resolve_pdf_url(candidate, doi=doi, email=email, headers=headers) if (candidate or doi) else None

            filename = ""
            error = ""
            if pdf_url:
                title = w.get("display_name", "no-title")
                doi_safe = doi or ""
                basename = safe_filename(doi_safe or title) + ".pdf"
                filepath = os.path.join(out_dir, basename)
                ok, err = download_file_with_validation(pdf_url, filepath, headers=headers)
                if ok:
                    filename = filepath
                    print(f"[{topic_name}] [{total}] Saved: {filepath}")
                else:
                    error = err or "download_failed"
                    print(f"[{topic_name}] [{total}] Failed to download {pdf_url} => {error}")
            else:
                error = "no_pdf_url_found"
                print(f"[{topic_name}] [{total}] No OA PDF/URL found for: {w.get('display_name')[:80]}")

            topics_list = []
            for t in (w.get("topics") or []):
                try:
                    topics_list.append(t.get("display_name") or t.get("id"))
                except Exception:
                    continue
            b = w.get("biblio") or {}
            journal = b.get("journal_title") or b.get("journal") or b.get("venue") or ""

            writer.writerow({
                "openalex_id": w.get("id", ""),
                "title": w.get("display_name", ""),
                "doi": doi or "",
                "publication_date": w.get("publication_date", ""),
                "pdf_url": pdf_url or "",
                "saved_file": filename,
                "error": error,
                "topics": ";".join(topics_list),
                "journal": journal or ""
            })

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            print(f"[{topic_name}] No next cursor; finished paging.")
            break
        params["cursor"] = next_cursor
        if max_pages and page_count >= max_pages:
            print(f"[{topic_name}] Reached max_pages={max_pages}. Stopping.")
            break
        time.sleep(sleep)

    print(f"[{topic_name}] Done. Total works processed: {total}")
    print(f"[{topic_name}] Metadata CSV: {csv_path}")
    return True

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Find OpenAlex topics for 'technology' and attempt OA PDF downloads for Zambia-affiliated works.")
    ap.add_argument("--search-term", default="technology", help="Topic search term")
    ap.add_argument("--top-n", type=int, default=5, help="Number of candidate topics to pick")
    ap.add_argument("--list-only", action="store_true", help="Only list topics (no downloading)")
    ap.add_argument("--out", default="downloads", help="Output base folder")
    ap.add_argument("--per-page", type=int, default=200, help="Results per OpenAlex works page (max 200)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between pages")
    ap.add_argument("--email", default=None, help="Your contact email for Unpaywall and user-agent")
    ap.add_argument("--max-pages", type=int, default=None, help="Optional: limit pages per topic")
    ap.add_argument("--max-topics", type=int, default=None, help="Optional: stop after this many topics processed")
    args = ap.parse_args()

    candidates = fetch_candidate_topics(search_term=args.search_term, per_page=50, max_topics=args.top_n, email=args.email)
    if not candidates:
        print("No topics found; exiting.")
        sys.exit(1)

    print("\nCandidate topics (top by works_count):")
    for i, t in enumerate(candidates, start=1):
        print(f"{i}. {t.get('display_name')}  (id: {t.get('id')})  works_count: {t.get('works_count')}")
        kw = t.get("keywords") or []
        if kw:
            print("    keywords:", ", ".join(kw[:8]))

    if args.list_only:
        print("\nList-only mode; exiting.")
        return

    processed = 0
    for t in candidates:
        tid = t.get("id") or ""
        if tid.startswith("https://openalex.org/"):
            tid = tid.split("/")[-1]
        tname = t.get("display_name") or tid
        try:
            download_for_topic(topic_id=tid, topic_name=tname, out_base=args.out, per_page=args.per_page, sleep=args.sleep, email=args.email, max_pages=args.max_pages)
        except Exception as e:
            print(f"Error for topic {tname} ({tid}): {e}")
        processed += 1
        if args.max_topics and processed >= args.max_topics:
            print(f"Reached max_topics={args.max_topics}; stopping.")
            break

    print("\nAll done.")

if __name__ == "__main__":
    main()
