# msamb_dynamic_ingest.py
"""
Dynamic APMC Scraper + Supabase Upsert
Author: Amarsinh Patil (KisanShaktiAI)
Purpose:
 - Read agri_market_sources rows (active)
 - For each source: load commodity list (local HTML fallback or from main_page)
 - Fetch data from data_endpoint for each commodityCode
 - Parse MSAMB-style table into rows
 - Normalize fields, map commodity -> global_code (commodity_master.aliases)
 - Upsert into public.market_prices using (source_id, commodity_code, price_date, market_location) conflict key
"""

import os
import re
import time
import csv
import json
import datetime
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup
from supabase import create_client

# ---------- Config via env ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
COMMODITY_HTML_DIR = os.environ.get("COMMODITY_HTML_DIR", ".")  # directory for commodity_html_path files
REQUESTS_TIMEOUT = int(os.environ.get("REQUESTS_TIMEOUT", "30"))
THROTTLE_SECONDS = float(os.environ.get("THROTTLE_SECONDS", "1.5"))
USER_AGENT = os.environ.get("USER_AGENT",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables")

# ---------- Setup clients ----------
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/"
})
session.max_redirects = 5

# ---------- Helpers ----------
def clean_number(s: Optional[str]) -> Optional[float]:
    """Strip non-numeric characters and parse to float. Return None if empty or invalid."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s in ("--", "-", "—", "NA", "N/A"):
        return None
    # Remove non-digit except dot and minus
    cleaned = re.sub(r"[^\d\.\-]", "", s)
    try:
        if cleaned == "" or cleaned == "." or cleaned == "-":
            return None
        return float(cleaned)
    except:
        return None

def parse_date_string(date_str: str) -> Optional[str]:
    """Parse date strings like '07/11/2025' or '07-11-2025' and return YYYY-MM-DD or None."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # common formats: DD/MM/YYYY or DD-MM-YYYY
    m = re.search(r"(\d{1,2})\D(\d{1,2})\D(\d{4})", date_str)
    if m:
        d, mo, y = m.groups()
        try:
            dt = datetime.date(int(y), int(mo), int(d))
            return dt.isoformat()
        except:
            return None
    return None

def fetch_text(url: str, params=None, headers=None, timeout=REQUESTS_TIMEOUT) -> Optional[str]:
    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"❌ HTTP GET failed: {url}  {e}")
        return None

def post_text(url: str, data=None, params=None, headers=None, timeout=REQUESTS_TIMEOUT) -> Optional[str]:
    try:
        r = session.post(url, data=data, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"❌ HTTP POST failed: {url}  {e}")
        return None

def load_commodities_from_local(html_path: str, selector: str = "select#drpCommodities option") -> Dict[str, str]:
    """Load commodity code -> name map from a saved HTML file using selector."""
    p = os.path.join(COMMODITY_HTML_DIR, html_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Commodity HTML not found: {p}")
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    opts = soup.select(selector)
    if not opts:
        # try default select
        sel = soup.find("select", {"id": "drpCommodities"}) or soup.find("select")
        opts = sel.find_all("option") if sel else []
    mapping = {}
    for opt in opts:
        code = (opt.get("value") or "").strip()
        name = opt.get_text(strip=True)
        if code and len(code) >= 2:
            mapping[code] = name
    return mapping

def load_commodities_from_remote(main_page_url: str, selector: str = "select#drpCommodities option") -> Dict[str, str]:
    html = fetch_text(main_page_url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    opts = soup.select(selector)
    if not opts:
        sel = soup.find("select", {"id": "drpCommodities"}) or soup.find("select")
        opts = sel.find_all("option") if sel else []
    mapping = {}
    for opt in opts:
        code = (opt.get("value") or "").strip()
        name = opt.get_text(strip=True)
        if code and len(code) >= 2:
            mapping[code] = name
    return mapping

def parse_msamb_table(html: str, commodity_display_name: str) -> List[Dict[str, Any]]:
    """Parse MSAMB-style HTML (date rows + data rows). Returns list of dicts."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    current_date = None
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        # Date row detection: often a single td with colspan
        if len(tds) == 1 or (len(tds) >= 1 and tds[0].has_attr("colspan")):
            date_text = tds[0].get_text(strip=True)
            parsed = parse_date_string(date_text)
            if parsed:
                current_date = parsed
            else:
                # try if date_text contains date at end
                dd = re.search(r"(\d{1,2}(/\-)\d{1,2}(/\-)\d{4})", date_text)
                if dd:
                    parsed = parse_date_string(dd.group(1))
                    if parsed:
                        current_date = parsed
            continue
        # Data row: expect at least 7 columns like MSAMB
        if len(tds) >= 7 and current_date:
            market = tds[0].get_text(strip=True)
            variety = tds[1].get_text(strip=True)
            unit = tds[2].get_text(strip=True)
            arrival = tds[3].get_text(strip=True)
            min_price = tds[4].get_text(strip=True)
            max_price = tds[5].get_text(strip=True)
            modal_price = tds[6].get_text(strip=True)
            rows_out.append({
                "commodity": commodity_display_name,
                "date": current_date,
                "market": market,
                "variety": variety,
                "unit": unit,
                "arrival_raw": arrival,
                "min_price_raw": min_price,
                "max_price_raw": max_price,
                "modal_price_raw": modal_price,
                "raw_html": str(tr)
            })
    return rows_out

# ---------- DB helpers ----------
def load_sources() -> List[Dict[str, Any]]:
    resp = sb.table("agri_market_sources").select("*").eq("active", True).execute()
    return resp.data or []

def load_commodity_alias_map(source_alias: str = "msamb") -> Dict[str, str]:
    """Return mapping: source_code -> global_code"""
    resp = sb.table("commodity_master").select("global_code, aliases").execute()
    rows = resp.data or []
    mapping = {}
    for r in rows:
        aliases = r.get("aliases") or {}
        if isinstance(aliases, dict) and source_alias in aliases:
            mapping[str(aliases[source_alias])] = r["global_code"]
    return mapping

def upsert_market_price(payload: Dict[str, Any]) -> bool:
    """Upsert a single market_prices row. Return True on success."""
    try:
        resp = sb.table("market_prices").upsert(payload,
                                               on_conflict="source_id,commodity_code,price_date,market_location").execute()
        # supabase client returns object with .data
        if hasattr(resp, "data") and resp.data:
            return True
        # fallback: check status code
        if getattr(resp, "status_code", None) in (200, 201):
            return True
        print("⚠️ Upsert response:", resp)
        return False
    except Exception as e:
        print("❌ Upsert failed:", e)
        return False

# ---------- Main scraper flow ----------
def process_source(src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process one agri_market_sources row.
    Returns summary dict.
    """
    summary = {
        "source_id": src.get("id"),
        "board_name": src.get("board_name"),
        "base_url": src.get("base_url"),
        "rows_fetched": 0,
        "rows_upserted": 0,
        "errors": []
    }

    source_id = src.get("id")
    base_url = (src.get("base_url") or "").rstrip("/")
    main_page = src.get("main_page") or ""
    data_endpoint = src.get("data_endpoint')"] if False else src.get("data_endpoint")  # defensive
    if data_endpoint is None:
        data_endpoint = src.get("data_endpoint")  # reassign properly
    commodity_source = src.get("commodity_source") or "dropdown_html"
    commodity_html_path = src.get("commodity_html_path")
    commodity_selector = src.get("commodity_dropdown_selector") or "select#drpCommodities option"
    commodity_value_attr = src.get("commodity_value_attr") or "value"
    page_requires_session = src.get("page_requires_session", True)
    fetch_method = (src.get("fetch_method") or "html_scrape").lower()
    data_request_method = (src.get("data_request_method") or "GET").upper()
    mapping = src.get("mapping") or {"market":0,"variety":1,"unit":2,"arrival":3,"min_price":4,"max_price":5,"modal_price":6}

    # 1) get commodity list
    try:
        if commodity_source == "dropdown_html" and commodity_html_path:
            try:
                commodity_map = load_commodities_from_local(commodity_html_path, selector=commodity_selector)
                print(f"✅ Loaded {len(commodity_map)} commodities from local HTML: {commodity_html_path}")
            except Exception as e:
                print(f"⚠️ Failed to load local commodity HTML ({commodity_html_path}): {e}. Trying remote main_page...")
                commodity_map = load_commodities_from_remote(base_url + main_page, selector=commodity_selector)
                print(f"✅ Loaded {len(commodity_map)} commodities from remote main page")
        else:
            # try remote
            commodity_map = load_commodities_from_remote(base_url + main_page, selector=commodity_selector)
            print(f"✅ Loaded {len(commodity_map)} commodities from remote main page")
    except Exception as e:
        commodity_map = {}
        summary["errors"].append(f"commodity load failed: {e}")
        print("❌ Commodity load failed:", e)

    # fallback: if no commodities found, attempt to parse pre-known mapping (rare)
    if not commodity_map:
        print("⚠️ No commodities found for source. Aborting this source.")
        summary["errors"].append("no_commodities")
        return summary

    # load alias map for this source (use key based on organization or default alias 'msamb')
    source_alias = (src.get("organization") or "").strip().lower() or "msamb"
    alias_map = load_commodity_alias_map(source_alias="msamb")  # for now use 'msamb' alias
    # TODO: derive alias key from source if you maintain different alias names

    # Optionally hit main_page to obtain cookies/session
    if page_requires_session:
        try:
            _ = fetch_text(base_url + (main_page or ""))
            time.sleep(0.5)
        except Exception as e:
            print("⚠️ main_page session fetch failed:", e)

    # Now loop commodities
    for code, display_name in list(commodity_map.items()):
        # small validation for code - some sites use numeric string, others may use full code
        if not code:
            continue
        # Compose data URL - if data_endpoint is a path, join with base_url
        if data_endpoint.startswith("http://") or data_endpoint.startswith("https://"):
            url = data_endpoint
        else:
            url = base_url + data_endpoint

        params = {"commodityCode": code, "apmcCode": "null"}
        try:
            if data_request_method == "GET":
                html = fetch_text(url, params=params)
            else:
                html = post_text(url, data=params)
            if not html:
                print(f"⚠️ Empty/failed response for commodity {code} - {display_name}")
                continue
        except Exception as e:
            print(f"❌ Request failed for {code}: {e}")
            summary["errors"].append(f"request_failed:{code}")
            continue

        # parse HTML table
        parsed_rows = parse_msamb_table(html, display_name)
        if not parsed_rows:
            # maybe the response is JSON or different; skip quietly
            print(f"⚠️ No parsed rows for {display_name} ({code})")
            continue

        summary["rows_fetched"] += len(parsed_rows)

        # Normalize and upsert
        for pr in parsed_rows:
            try:
                arrival = clean_number(pr.get("arrival_raw"))
                min_p = clean_number(pr.get("min_price_raw"))
                max_p = clean_number(pr.get("max_price_raw"))
                modal_p = clean_number(pr.get("modal_price_raw"))

                # compute price_per_unit: prefer modal, else avg of min/max, else None
                price_per_unit = None
                if modal_p is not None:
                    price_per_unit = modal_p
                elif min_p is not None and max_p is not None:
                    price_per_unit = (min_p + max_p) / 2.0
                elif min_p is not None:
                    price_per_unit = min_p
                elif max_p is not None:
                    price_per_unit = max_p

                spread = None
                if min_p is not None and max_p is not None:
                    spread = max_p - min_p

                # map commodity_code -> global
                global_code = alias_map.get(code) or f"{(src.get('organization') or 'SRC').upper()}_{code}"

                # build payload
                payload = {
                    "source_id": source_id,
                    "country_id": src.get("country_id"),
                    "state_id": src.get("state_id"),
                    "commodity_code": code,
                    "global_commodity_code": global_code,
                    "crop_name": pr.get("commodity"),
                    "commodity_name_normalized": None,
                    "commodity_category": None,
                    "variety": pr.get("variety") or None,
                    "unit": pr.get("unit") or None,
                    "arrival": arrival,
                    "min_price": min_p,
                    "max_price": max_p,
                    "modal_price": modal_p,
                    "spread": spread,
                    "price_per_unit": price_per_unit if price_per_unit is not None else 0.0,
                    "market_location": pr.get("market"),
                    "district": None,
                    "state": None,
                    "price_date": pr.get("date"),
                    "price_type": "wholesale",
                    "quality_grade": None,
                    "source": src.get("organization") or None,
                    "metadata": {
                        "raw_row": {
                            "arrival_raw": pr.get("arrival_raw"),
                            "min_price_raw": pr.get("min_price_raw"),
                            "max_price_raw": pr.get("max_price_raw"),
                            "modal_price_raw": pr.get("modal_price_raw")
                        },
                        "ingest_source": src.get("id"),
                    },
                    "raw_html": pr.get("raw_html"),
                    "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "status": "ready"
                }

                # upsert into supabase
                ok = upsert_market_price(payload)
                if ok:
                    summary["rows_upserted"] += 1
                else:
                    summary["errors"].append(f"upsert_failed:{code}:{pr.get('market')}")

            except Exception as e:
                print(f"❌ Error processing row for {code} {pr.get('market')}: {e}")
                summary["errors"].append(f"process_row_error:{code}:{e}")

        # throttle per commodity
        time.sleep(THROTTLE_SECONDS)

    return summary

def main():
    print("🚀 Starting dynamic APMC ingestion")
    sources = load_sources()
    if not sources:
        print("❌ No active agri_market_sources found. Exiting.")
        return

    overall = []
    for s in sources:
        print(f"\n🔎 Processing source: {s.get('organization')} / {s.get('board_name')} (id={s.get('id')})")
        summary = process_source(s)
        overall.append(summary)
        # small pause between sources
        time.sleep(1.0)

    # Print summary
    total_fetched = sum(x.get("rows_fetched", 0) for x in overall)
    total_upserted = sum(x.get("rows_upserted", 0) for x in overall)
    print(f"\n✨ Done. fetched={total_fetched}, upserted={total_upserted}")
    for s in overall:
        print(f"- source {s.get('board_name') or s.get('source_id')}: fetched={s.get('rows_fetched')} upserted={s.get('rows_upserted')} errors={len(s.get('errors') or [])}")
    # Optionally: insert a run-summary into a table `ingest_runs` if you created it
    try:
        sb.table("ingest_runs").insert({
            "run_time": datetime.datetime.utcnow().isoformat() + "Z",
            "fetched_count": total_fetched,
            "upserted_count": total_upserted,
            "detail": json.dumps(overall)
        }).execute()
    except Exception:
        # it's fine if ingest_runs doesn't exist
        pass

if __name__ == "__main__":
    main()
