#!/usr/bin/env python3
"""
msamb_dynamic_ingest.py
Dynamic APMC Scraper + Supabase Upsert
Author: Amarsinh Patil (KisanShaktiAI) - updated
Purpose:
 - Read agri_market_sources rows (active)
 - For each source: load commodity list (local HTML fallback or from main_page)
 - Fetch data from data_endpoint for each commodityCode
 - Parse MSAMB-style table into rows (generic fallback)
 - Normalize fields, map commodity -> global_code (commodity_master.aliases)
 - Batch upsert into public.market_prices using (source_id, commodity_code, price_date, market_location) conflict key
Notes:
 - Configure via environment variables (see README comment below)
"""

import os
import re
import time
import json
import math
import logging
import datetime
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from supabase import create_client

# -------------------------
# Configuration (env)
# -------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

COMMODITY_HTML_DIR = os.environ.get("COMMODITY_HTML_DIR", ".")
REQUESTS_TIMEOUT = int(os.environ.get("REQUESTS_TIMEOUT", "30"))
THROTTLE_SECONDS = float(os.environ.get("THROTTLE_SECONDS", "1.2"))
USER_AGENT = os.environ.get("USER_AGENT",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
BATCH_SIZE = int(os.environ.get("UPSERT_BATCH_SIZE", "100"))

# sanity check
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set as env variables")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("msamb_ingest")

# -------------------------
# Supabase client
# -------------------------
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# HTTP session with retries
# -------------------------
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
})
session.max_redirects = 5

# -------------------------
# Helpers - parsing & cleaning
# -------------------------
def clean_number(s: Optional[str]) -> Optional[float]:
    """Strip non-numeric characters (except dot/minus) and parse to float."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s in ("--", "-", "—", "NA", "N/A"):
        return None
    # convert commonly used non-breaking spaces / unicode commas
    s = s.replace("\u00A0", "").replace(",", "").replace(" ", "")
    cleaned = re.sub(r"[^\d\.\-]", "", s)
    if cleaned in ("", ".", "-"):
        return None
    try:
        return float(cleaned)
    except Exception:
        return None

def parse_date_string(date_str: str) -> Optional[str]:
    """Parse date strings like '07/11/2025' or '07-11-2025' or '07 Nov 2025' into ISO YYYY-MM-DD."""
    if not date_str:
        return None
    s = date_str.strip()
    # common DD/MM/YYYY or DD-MM-YYYY
    m = re.search(r"(\d{1,2})\D(\d{1,2})\D(\d{4})", s)
    if m:
        d, mo, y = m.groups()
        try:
            dt = datetime.date(int(y), int(mo), int(d))
            return dt.isoformat()
        except Exception:
            return None
    # try textual month e.g. "07 Nov 2025"
    m2 = re.search(r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})", s)
    if m2:
        d, mon, y = m2.groups()
        try:
            dt = datetime.datetime.strptime(f"{d} {mon} {y}", "%d %b %Y").date()
            return dt.isoformat()
        except Exception:
            try:
                dt = datetime.datetime.strptime(f"{d} {mon} {y}", "%d %B %Y").date()
                return dt.isoformat()
            except Exception:
                return None
    # fallback: 4-digit year at end
    m3 = re.search(r"(\d{1,2})\D(\d{1,2})\D(\d{2})\b", s)
    if m3:
        d, mo, yy = m3.groups()
        y = int(yy)
        y = 2000 + y if y < 70 else 1900 + y
        try:
            dt = datetime.date(y, int(mo), int(d))
            return dt.isoformat()
        except:
            return None
    return None

def fetch_text(url: str, params=None, timeout=REQUESTS_TIMEOUT) -> Optional[str]:
    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.warning(f"HTTP GET failed: {url} params={params} err={e}")
        return None

def post_text(url: str, data=None, timeout=REQUESTS_TIMEOUT) -> Optional[str]:
    try:
        r = session.post(url, data=data, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.warning(f"HTTP POST failed: {url} data={data} err={e}")
        return None

# -------------------------
# Commodity loading (local or remote)
# -------------------------
def load_commodities_from_local(html_path: str, selector: str = "select#drpCommodities option") -> Dict[str, str]:
    p = os.path.join(COMMODITY_HTML_DIR, html_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Commodity HTML not found: {p}")
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    opts = soup.select(selector)
    if not opts:
        sel = soup.find("select", {"id": "drpCommodities"}) or soup.find("select")
        opts = sel.find_all("option") if sel else []
    mapping = {}
    for opt in opts:
        code = (opt.get("value") or "").strip()
        name = opt.get_text(strip=True)
        if code:
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
        if code:
            mapping[code] = name
    return mapping

# -------------------------
# Parsers
# -------------------------
def parse_msamb_table(html: str, commodity_display_name: str) -> List[Dict[str, Any]]:
    """Parse MSAMB-style HTML (date rows + data rows). Returns list of dicts."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    current_date = None
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        # Date row often single td or colspan
        if len(tds) == 1 or (len(tds) >= 1 and tds[0].has_attr("colspan")):
            date_text = tds[0].get_text(strip=True)
            parsed = parse_date_string(date_text)
            if parsed:
                current_date = parsed
            else:
                # try to find date-like substring
                dd = re.search(r"(\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{2,4})", date_text)
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

# Generic mapping-based parser (if mapping provided in agri_market_sources)
def parse_generic_table_with_mapping(html: str, mapping: dict, date_selector: Optional[str]=None, row_selector: Optional[str]=None) -> List[Dict[str, Any]]:
    """
    mapping example: {"market":0,"variety":1,"unit":2,"arrival":3,"min_price":4,"max_price":5,"modal_price":6}
    row_selector e.g. "table tr"
    date_selector can be CSS that points to a date cell (optional)
    """
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    current_date = None

    # try date selector first
    if date_selector:
        dd = soup.select_one(date_selector)
        if dd:
            d = parse_date_string(dd.get_text(strip=True))
            if d:
                current_date = d

    # choose row_selector
    if row_selector:
        row_elems = soup.select(row_selector)
    else:
        row_elems = soup.find_all("tr")

    for tr in row_elems:
        cols = tr.find_all(["td","th"])
        if not cols:
            continue
        # attempt to detect rows that are date rows (colspan)
        if len(cols) == 1 and (cols[0].has_attr("colspan") or re.search(r"\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{2,4}", cols[0].get_text())):
            maybe = parse_date_string(cols[0].get_text(strip=True))
            if maybe:
                current_date = maybe
            continue
        # if mapping indexes exist
        if mapping and current_date and len(cols) > max(mapping.values()):
            # safe extraction
            def get_col(i):
                return cols[i].get_text(strip=True) if i < len(cols) else ""
            market = get_col(mapping.get("market", 0))
            variety = get_col(mapping.get("variety", 1))
            unit = get_col(mapping.get("unit", 2))
            arrival = get_col(mapping.get("arrival", 3))
            min_price = get_col(mapping.get("min_price", 4))
            max_price = get_col(mapping.get("max_price", 5))
            modal_price = get_col(mapping.get("modal_price", 6))
            rows_out.append({
                "commodity": None,
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

# -------------------------
# DB helpers
# -------------------------
def load_sources() -> List[Dict[str, Any]]:
    resp = sb.table("agri_market_sources").select("*").eq("active", True).execute()
    return resp.data or []

def load_commodity_alias_map(source_alias: str = "msamb") -> Dict[str, str]:
    """Return mapping: source_code -> global_code. Uses commodity_master.aliases JSON.
       It expects in commodity_master.aliases a dict like {"msamb": "08009", "another": ["x","y"]}"""
    resp = sb.table("commodity_master").select("global_code, aliases").execute()
    rows = resp.data or []
    mapping = {}
    for r in rows:
        aliases = r.get("aliases") or {}
        if isinstance(aliases, dict) and source_alias in aliases:
            val = aliases[source_alias]
            if isinstance(val, (list, tuple)):
                for code in val:
                    mapping[str(code)] = r["global_code"]
            else:
                mapping[str(val)] = r["global_code"]
    return mapping

def record_exists(source_id: str, commodity_code: str, price_date: str, market_location: str) -> bool:
    """Quick DB check to avoid redundant upserts (returns True if exists)."""
    try:
        resp = sb.table("market_prices").select("id").eq("source_id", source_id).eq("commodity_code", commodity_code).eq("price_date", price_date).eq("market_location", market_location).limit(1).execute()
        rows = resp.data or []
        return bool(rows)
    except Exception as e:
        log.warning("record_exists check failed: %s", e)
        # fail-open: if check fails, return False so we attempt ingest
        return False

def upsert_batch(payloads: List[Dict[str, Any]], batch_size: int = BATCH_SIZE) -> int:
    """Batch upsert payloads. Returns number of successfully upserted rows (best-effort)."""
    if not payloads:
        return 0
    total = 0
    # chunk payloads
    chunks = math.ceil(len(payloads) / batch_size)
    for i in range(chunks):
        start = i * batch_size
        chunk = payloads[start:start+batch_size]
        try:
            resp = sb.table("market_prices").upsert(chunk, on_conflict="source_id,commodity_code,price_date,market_location").execute()
            # supabase returns .data on success
            if hasattr(resp, "data") and resp.data:
                total += len(resp.data)
            else:
                # fallback counting
                total += len(chunk)
        except Exception as e:
            log.error("Upsert chunk failed: %s", e)
            # continue with next chunk
    return total

# -------------------------
# Source processing
# -------------------------
def process_source(src: Dict[str, Any]) -> Dict[str, Any]:
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
    data_endpoint = src.get("data_endpoint") or ""
    commodity_source = src.get("commodity_source") or "dropdown_html"
    commodity_html_path = src.get("commodity_html_path")
    commodity_selector = src.get("commodity_dropdown_selector") or "select#drpCommodities option"
    commodity_value_attr = src.get("commodity_value_attr") or "value"
    page_requires_session = src.get("page_requires_session", True)
    fetch_method = (src.get("fetch_method") or "html_scrape").lower()
    data_request_method = (src.get("data_request_method") or "GET").upper()
    mapping = src.get("mapping") or None
    row_selector = src.get("row_selector") or None
    date_row_selector = src.get("date_row_selector") or None

    # 1) get commodity list
    try:
        if commodity_source == "dropdown_html" and commodity_html_path:
            try:
                commodity_map = load_commodities_from_local(commodity_html_path, selector=commodity_selector)
                log.info("Loaded %d commodities from local HTML: %s", len(commodity_map), commodity_html_path)
            except Exception as e:
                log.warning("Failed to load local commodity HTML (%s): %s. Trying remote main_page...", commodity_html_path, e)
                commodity_map = load_commodities_from_remote(base_url + main_page, selector=commodity_selector)
                log.info("Loaded %d commodities from remote main page", len(commodity_map))
        else:
            commodity_map = load_commodities_from_remote(base_url + main_page, selector=commodity_selector)
            log.info("Loaded %d commodities from remote main page", len(commodity_map))
    except Exception as e:
        commodity_map = {}
        summary["errors"].append(f"commodity load failed: {e}")
        log.error("Commodity load failed: %s", e)

    if not commodity_map:
        log.warning("No commodities found for source %s (%s). Skipping.", src.get("board_name"), source_id)
        summary["errors"].append("no_commodities")
        return summary

    # alias map for this source (use organization lowercased as alias key)
    source_alias = (src.get("organization") or "").strip().lower() or "msamb"
    alias_map = load_commodity_alias_map(source_alias=source_alias)

    # Optionally hit main_page to obtain cookies/session
    if page_requires_session and main_page:
        try:
            _ = fetch_text(base_url + main_page)
            time.sleep(0.4)
        except Exception as e:
            log.warning("main_page session fetch failed: %s", e)

    # Build data endpoint url
    if data_endpoint.startswith("http://") or data_endpoint.startswith("https://"):
        data_url = data_endpoint
    else:
        data_url = base_url + (data_endpoint or "")

    payloads_to_upsert: List[Dict[str, Any]] = []

    # loop commodities
    for code, display_name in list(commodity_map.items()):
        if not code:
            continue

        params = {"commodityCode": code, "apmcCode": "null"}

        # Request
        html = None
        try:
            if data_request_method == "GET":
                html = fetch_text(data_url, params=params)
            else:
                html = post_text(data_url, data=params)
            if not html:
                log.warning("Empty response for commodity %s (%s)", code, display_name)
                summary["errors"].append(f"empty_response:{code}")
                continue
        except Exception as e:
            log.error("Request failed for %s: %s", code, e)
            summary["errors"].append(f"request_failed:{code}")
            continue

        # Parse using mapping if present else msamb parser
        parsed_rows = []
        try:
            if mapping:
                parsed_rows = parse_generic_table_with_mapping(html, mapping, date_selector=date_row_selector, row_selector=row_selector)
                # set commodity display names where missing
                for r in parsed_rows:
                    if not r.get("commodity"):
                        r["commodity"] = display_name
            else:
                parsed_rows = parse_msamb_table(html, display_name)
        except Exception as e:
            log.exception("Parsing failed for %s: %s", code, e)
            summary["errors"].append(f"parse_failed:{code}")
            continue

        if not parsed_rows:
            log.debug("No parsed rows for %s (%s)", display_name, code)
            continue

        summary["rows_fetched"] += len(parsed_rows)

        # For each parsed row, normalize and prepare payload
        for pr in parsed_rows:
            try:
                price_date = pr.get("date")
                market_location = pr.get("market") or ""
                # quick skip if exists
                exists = record_exists(source_id, code, price_date, market_location)
                if exists:
                    # skip to next row (idempotency)
                    continue

                arrival = clean_number(pr.get("arrival_raw"))
                min_p = clean_number(pr.get("min_price_raw"))
                max_p = clean_number(pr.get("max_price_raw"))
                modal_p = clean_number(pr.get("modal_price_raw"))

                price_per_unit = None
                if modal_p is not None:
                    price_per_unit = modal_p
                elif min_p is not None and max_p is not None:
                    price_per_unit = (min_p + max_p) / 2.0
                elif min_p is not None:
                    price_per_unit = min_p
                elif max_p is not None:
                    price_per_unit = max_p

                spread = (max_p - min_p) if (min_p is not None and max_p is not None) else None

                # map commodity code -> global code (if alias exists)
                global_code = alias_map.get(code) or f"{(src.get('organization') or 'SRC').upper()}_{code}"

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
                    "market_location": market_location,
                    "district": None,
                    "state": None,
                    "price_date": price_date,
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
                        "ingest_source": src.get("id")
                    },
                    "raw_html": pr.get("raw_html"),
                    "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "status": "ready"
                }

                payloads_to_upsert.append(payload)

            except Exception as e:
                log.exception("Error processing row for %s %s: %s", code, market_location, e)
                summary["errors"].append(f"process_row_error:{code}")

        # throttle between commodity requests
        time.sleep(THROTTLE_SECONDS)

    # end commodity loop

    # batch upsert all payloads collected for this source
    if payloads_to_upsert:
        upserted = upsert_batch(payloads_to_upsert, batch_size=BATCH_SIZE)
        summary["rows_upserted"] = upserted
        log.info("Source %s upserted %d rows (fetched=%d)", src.get("board_name"), upserted, summary["rows_fetched"])
    else:
        log.info("No new rows to upsert for source %s", src.get("board_name"))

    return summary

# -------------------------
# Main flow
# -------------------------
def main():
    log.info("Starting dynamic APMC ingestion")
    sources = load_sources()
    if not sources:
        log.error("No active agri_market_sources found. Exiting.")
        return

    overall = []
    for s in sources:
        log.info("Processing source: %s / %s (id=%s)", s.get("organization"), s.get("board_name"), s.get("id"))
        try:
            summary = process_source(s)
            overall.append(summary)
        except Exception as e:
            log.exception("Source processing failed: %s", e)
            overall.append({"source_id": s.get("id"), "errors": [str(e)]})
        time.sleep(1.0)

    total_fetched = sum(x.get("rows_fetched", 0) for x in overall)
    total_upserted = sum(x.get("rows_upserted", 0) for x in overall)
    log.info("Done: fetched=%d upserted=%d", total_fetched, total_upserted)

    # Try to insert run summary if table exists
    try:
        sb.table("ingest_runs").insert({
            "run_time": datetime.datetime.utcnow().isoformat() + "Z",
            "fetched_count": total_fetched,
            "upserted_count": total_upserted,
            "detail": json.dumps(overall)
        }).execute()
    except Exception:
        log.debug("ingest_runs insert failed or table missing - skipping")

if __name__ == "__main__":
    main()
