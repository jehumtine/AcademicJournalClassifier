#!/usr/bin/env python3
"""
Download PDFs for harvested articles listed in a CSV.

Reads an input CSV (must contain at least one of these columns: 'pdf_url', 'doi', 'title', 'source', 'assigned_sectors').
Attempts in order:
  - Use csv 'pdf_url' (fast path)
  - If source == 'arxiv' and pdf_url missing, construct arXiv PDF link
  - Use Unpaywall (requires --email) to get best OA pdf
  - Try DOI negotiation via doi.org (Accept: application/pdf)
  - HEAD check / GET landing page and parse for meta 'citation_pdf_url' or <a href="*.pdf">
Downloads PDFs and validates they are PDFs (checks '%PDF' in initial bytes).
Outputs an updated CSV with columns: pdf_url_used, saved_path, download_error.

Usage:
  python download_pdfs_from_csv.py --input vision2030_corpus.csv --outdir pdfs --email you@domain.com

Requirements:
  pip install requests beautifulsoup4 pandas tqdm
"""
import argparse
import csv
import os
import re
import time
from urllib.parse import urljoin, urlparse, quote_plus

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# ---------- Config ----------
USER_AGENT_TEMPLATE = "vision2030-pdf-downloader/1.0 ({email})"
SLEEP_BETWEEN = 0.8   # seconds between external requests (adjust as needed)
RETRIES = 2
TIMEOUT = 30
# -------------------------

def safe_filename(s, maxlen=200):
    s = (s or "")[:maxlen]
    s = re.sub(r'[\\/:"*?<>|]+', "_", s)
    s = re.sub(r'\s+', "_", s).strip("_")
    return s or "file"

def is_doi_like(s):
    if not s: return False
    s = s.strip()
    return s.startswith("10.") or "doi.org" in s.lower()

def unpaywall_pdf_for_doi(doi, email, headers):
    """Return a PDF URL from Unpaywall (url_for_pdf) or None."""
    if not doi or not email:
        return None
    api = f"https://api.unpaywall.org/v2/{quote_plus(doi)}"
    try:
        r = requests.get(api, params={"email": email}, headers=headers, timeout=20)
        if r.status_code == 200:
            j = r.json()
            bol = j.get("best_oa_location") or {}
            pdf = bol.get("url_for_pdf") or bol.get("url")
            if pdf:
                return pdf
            for loc in j.get("oa_locations") or []:
                if loc.get("url_for_pdf"):
                    return loc.get("url_for_pdf")
                if loc.get("url") and loc.get("url").lower().endswith(".pdf"):
                    return loc.get("url")
    except Exception:
        pass
    return None

def extract_pdf_from_html(url, html):
    """Parse HTML to find meta citation_pdf_url or <a href=*.pdf> or link rel alternate."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    # 1) meta tags
    for name in ("citation_pdf_url", "pdf_url"):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return urljoin(url, tag.get("content"))

    # 2) <link rel="alternate" type="application/pdf" href="...">
    for link in soup.find_all("link", href=True):
        if "pdf" in (link.get("type") or "").lower():
            return urljoin(url, link.get("href"))

    # 3) <a href="...pdf">
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            return urljoin(url, href)
        if "download" in href.lower() and ("pdf" in href.lower() or "fulltext" in href.lower()):
            return urljoin(url, href)

    # 4) regex fallback
    m = re.search(r"(https?:\/\/[^\s'\"<>]+\.pdf)", html, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return None

def try_doi_content_negotiation(doi, headers):
    """Try doi.org Accept: application/pdf; return final URL if PDF or None."""
    if not doi:
        return None
    if doi.startswith("http"):
        doi_url = doi
    elif doi.startswith("10."):
        doi_url = "https://doi.org/" + doi
    else:
        doi_url = "https://doi.org/" + doi
    try:
        r = requests.get(doi_url, headers={**headers, "Accept":"application/pdf"}, stream=True, timeout=TIMEOUT, allow_redirects=True)
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
    return None

def head_is_pdf(url, headers):
    try:
        h = requests.head(url, headers=headers, allow_redirects=True, timeout=TIMEOUT)
        ctype = (h.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype:
            return h.url
    except Exception:
        pass
    return None

def resolve_pdf_url(row, email, headers):
    """
    Given a row (dict-like with keys 'pdf_url', 'doi', 'source', 'id', 'title'), attempt to resolve a usable pdf URL.
    Returns (pdf_url_or_None, method_string)
    """
    # 1) CSV pdf_url
    pdf = row.get("pdf_url") or row.get("pdf") or None
    if pdf:
        return pdf, "csv_pdf_url"

    # 2) arXiv quick construct (if source is arxiv and id looks like arXiv)
    source = (row.get("source") or "").lower()
    if source == "arxiv":
        # id may be in 'id' column like http://arxiv.org/abs/xxxx or 'id' contains arXiv id
        ident = row.get("id") or row.get("article_id") or ""
        if ident:
            if "arxiv.org/abs/" in ident:
                pdf_link = ident.replace("/abs/", "/pdf/")
                if not pdf_link.endswith(".pdf"):
                    pdf_link = pdf_link + ".pdf"
                return pdf_link, "arxiv_construct"
            # sometimes id is arXiv:xxxx
            if ident.startswith("arXiv:"):
                aid = ident.split(":",1)[1]
                pdf_link = f"https://arxiv.org/pdf/{aid}.pdf"
                return pdf_link, "arxiv_construct"
        # fallback: try forming from title? not reliable
    # 3) Unpaywall by DOI
    doi = row.get("doi") or ""
    if doi and email:
        up = unpaywall_pdf_for_doi(doi, email, headers)
        if up:
            return up, "unpaywall"

    # 4) DOI content-negotiation
    if doi and is_doi_like(doi):
        dn = try_doi_content_negotiation(doi, headers)
        if dn:
            return dn, "doi_negotiation"

    # 5) Try candidate URL from other fields (e.g., 'landing_url' or 'openalex_pdf' if present)
    candidates = []
    for k in ("openalex_pdf", "best_pdf", "landing_page", "landing_url", "url"):
        v = row.get(k)
        if v:
            candidates.append(v)
    # Keep any URL-like candidates from row
    # Try HEAD check first
    for c in candidates:
        hpdf = head_is_pdf(c, headers)
        if hpdf:
            return hpdf, "head_pdf_candidate"
    # 6) landing page GET and parse HTML
    # try candidates again but GET and parse
    for c in candidates:
        try:
            r = requests.get(c, headers=headers, timeout=TIMEOUT)
            ctype = (r.headers.get("Content-Type") or "").lower()
            final_url = r.url
            if "pdf" in ctype or final_url.lower().endswith(".pdf"):
                return final_url, "landing_direct_pdf"
            pdf_from_html = extract_pdf_from_html(final_url, r.text)
            if pdf_from_html:
                # optionally verify via HEAD
                verified = head_is_pdf(pdf_from_html, headers)
                return (verified or pdf_from_html), "landing_parsed"
        except Exception:
            continue

    # 7) nothing found
    return None, "no_pdf_found"

def download_stream_and_validate(url, out_path, headers):
    """Stream download; validate pdf by checking first bytes for '%PDF' or URL ending with .pdf."""
    try:
        with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
            r.raise_for_status()
            it = r.iter_content(chunk_size=4096)
            try:
                first = next(it)
            except StopIteration:
                return False, "empty_response"
            final = r.url or url
            # Validate: either filename ends with .pdf or first bytes contain %PDF
            if not (final.lower().endswith(".pdf") or first[:4] == b'%PDF' or b'%PDF' in first):
                return False, "not_pdf"
            # write out
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

# ---------- Main orchestration ----------
def process_csv(input_csv, out_dir, email=None, cols_map=None, max_rows=None):
    """
    Read CSV into pandas, iterate rows, attempt to download, and write updated CSV with new columns.
    cols_map: optional dict mapping expected column names in CSV to canonical names:
       e.g. {"pdf_url":"pdf_url", "doi":"doi", "title":"title", "source":"source", "assigned_sectors":"assigned_sectors", "id":"id"}
    """
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    if max_rows:
        df = df.iloc[:max_rows]

    # canonical column names
    def getcol(r, names):
        for n in names:
            if n in r:
                return r[n]
        return ""

    # prepare output columns
    out_pdf_url_used = []
    out_saved_path = []
    out_error = []
    headers = {"User-Agent": USER_AGENT_TEMPLATE.format(email=email or "no-email")}

    os.makedirs(out_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        row = row.to_dict()
        # build a small normalized dict to pass to resolver
        norm = {
            "pdf_url": row.get("pdf_url") or row.get("pdf") or "",
            "doi": row.get("doi") or "",
            "source": row.get("source") or "",
            "id": row.get("id") or "",
            "title": row.get("title") or row.get("display_name") or "",
            "openalex_pdf": row.get("openalex_pdf") or row.get("best_pdf") or "",
            "best_pdf": row.get("best_pdf") or "",
            "landing_url": row.get("landing_url") or ""
        }

        pdf_url_used = ""
        saved_path = ""
        error = ""

        # Try resolution with retries
        resolved = None
        method = None
        for attempt in range(RETRIES + 1):
            try:
                resolved, method = resolve_pdf_url(norm, email=email, headers=headers)
                break
            except Exception as e:
                time.sleep(1)
                resolved = None
                method = f"resolve_error_{str(e)}"
        if resolved:
            pdf_url_used = resolved
            # decide directory for saving: prefer assigned sector if present
            assigned_sector = (row.get("assigned_sectors") or "") or (row.get("query_sector") or "")
            if assigned_sector:
                # use first assigned if multiple separated by ;
                folder = assigned_sector.split(";")[0]
            else:
                folder = (row.get("source") or "other").split(";")[0]
            # safe folder and filename
            save_dir = os.path.join(out_dir, safe_filename(folder))
            os.makedirs(save_dir, exist_ok=True)
            # filename from DOI if present else title
            doi = norm["doi"] or ""
            fname_base = None
            if doi:
                fname_base = safe_filename(doi)
            else:
                fname_base = safe_filename(norm["title"][:120])
            out_file = os.path.join(save_dir, fname_base + ".pdf")

            ok, err = download_stream_and_validate(pdf_url_used, out_file, headers=headers)
            if ok:
                saved_path = out_file
            else:
                error = f"{method}|{err}"
                # as a last attempt, if method was doi_negotiation and we got not_pdf, try landing page parsing
                if "doi_negotiation" in method or "unpaywall" in method or "landing" in method or "csv_pdf_url" in method:
                    # attempt GET landing/page and parse for pdf
                    try:
                        r = requests.get(pdf_url_used, headers=headers, timeout=TIMEOUT)
                        cand = extract_pdf_from_html(r.url, r.text)
                        if cand:
                            pdf_url_used = cand
                            ok2, err2 = download_stream_and_validate(pdf_url_used, out_file, headers=headers)
                            if ok2:
                                saved_path = out_file
                                error = ""
                    except Exception:
                        pass
        else:
            error = method or "no_pdf_found"

        # Append outputs
        out_pdf_url_used.append(pdf_url_used or "")
        out_saved_path.append(saved_path or "")
        out_error.append(error or "")

        # polite sleep between iterations
        time.sleep(SLEEP_BETWEEN)

    # write new CSV with added columns
    df["pdf_url_used"] = out_pdf_url_used
    df["saved_path"] = out_saved_path
    df["download_error"] = out_error
    out_csv = os.path.splitext(input_csv)[0] + "_with_pdfs.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nWrote updated CSV with download results: {out_csv}")
    return out_csv

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Download PDFs for harvested items from a CSV.")
    ap.add_argument("--input", "-i", required=True, help="Input CSV file (harvest output)")
    ap.add_argument("--outdir", "-o", default="pdfs", help="Base output directory for PDFs")
    ap.add_argument("--email", "-e", default=None, help="Your email (required for Unpaywall usage; recommended)")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional: limit number of rows to process")
    args = ap.parse_args()

    process_csv(args.input, args.outdir, email=args.email, max_rows=args.max_rows)

if __name__ == "__main__":
    main()
