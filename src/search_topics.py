# search_topics.py
import requests, sys, json

q = "technology"  # change to other search terms if you like
per_page = 50
url = "https://api.openalex.org/topics"
params = {"search": q, "per-page": per_page, "select": "id,display_name,keywords,domain,field,works_count"}

r = requests.get(url, params=params)
r.raise_for_status()
data = r.json()

for t in data.get("results", []):
    print("---")
    print("display_name:", t["display_name"])
    print("id:", t["id"])
    print("domain:", t.get("domain", {}).get("display_name"))
    print("field:", t.get("field", {}).get("display_name"))
    print("works_count:", t.get("works_count"))
    print("keywords:", ", ".join((t.get("keywords") or [])[:10]))
