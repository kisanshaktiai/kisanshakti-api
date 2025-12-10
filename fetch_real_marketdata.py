#!/usr/bin/env python3
# fetch_real_marketdata.py
"""
Dynamic APMC Scraper + Supabase Upsert
Improved: stronger headers, debug logs, retries, JSON-safe checks, and run-summary.
Author: Amarsinh Patil (KisanShaktiAI) - updated
"""

import os
import re
import time
import json
import datetime
import logging
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup
from supabase import create_client

# ---------- Config via env ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
COMMODITY_HTML_DIR = os.environ.get("COMMODITY_HTML_DIR", ".")
REQUESTS_TIMEOUT = int(os.environ.get("REQUESTS_TIMEOUT", "30"))
THROTTLE_SECONDS = float(os.environ.get("THROTTLE_SECONDS", "1.5"))
USER_AGENT = os.environ.get("USER_AGENT",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Setup clients ----------
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})
session.max_redirects = 5

# ---------- Utils ----------
def clean_number(s: Optional[str]) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s in ("", "--", "-", "—", "NA", "N/A"): return None
    cleaned = re.sub(r"[^\d\.\-]", "", s)
    try:
        if cleaned in ("", ".", "-"): return None
        return float(cleaned)
    except:
        return None

def parse_date_string(date_str: str) -> Optional[str]:
    if not date_str: return None
    m = re.search(r"(\d{1,2})\D(\d{1,2})\D(\d{4})", date_str)
    if m:
        d, mo, y = m.groups()
        try:
            dt = datetime.date(int(y), int(mo), int(d))
            return dt.isoformat()
        except:
            return None
    return None

def http_get(url: str, params=None, headers=None, timeout=REQUESTS_TIMEOUT):
    headers = headers or {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            return r
        except Exception as e:
            logging.warning(f"GET error (attempt {attempt}) for {url}: {e}")
            time.sleep(1 + attempt)
    return None

def http_post(url: str, data=None, params=None, headers=None, timeout=REQUESTS_TIMEOUT):
    headers = headers or {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, data=data, params=params, headers=headers, timeout=timeout)
            return r
        except Exception as e:
            logging.warning(f"POST error (attempt {attempt}) for {url}: {e}")
            time.sleep(1 + attempt)
    return None

# ---------- Commodity loaders ----------
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
        if code and len(code) >= 1:
            mapping[code] = name
    return mapping

def load_commodities_from_remote(main_page_url: str, selector: str = "select#drpCommodities option") -> Dict[str, str]:
    r = http_get(main_page_url)
    if not r or not r.ok:
        logging.warning(f"Remote commodity main page fetch failed: {main_page_url} status={(r.status_code if r else 'none')}")
        return {}
    soup = BeautifulSoup(r.text, "lxml")
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

# ---------- MSAMB parser ----------
def parse_msamb_table(html: str, commodity_display_name: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    current_date = None
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        # detect date row
        if len(tds) == 1 or (len(tds) >= 1 and tds[0].has_attr("colspan")):
            date_text = tds[0].get_text(strip=True)
            parsed = parse_date_string(date_text)
            if parsed:
                current_date = parsed
            else:
                dd = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})", date_text)
                if dd:
                    parsed = parse_date_string(dd.group(1))
                    if parsed:
                        current_date = parsed
            continue
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
    resp = sb.table("commodity_master").select("global_code,aliases").execute()
    rows = resp.data or []
    mapping = {}
    for r in rows:
        aliases = r.get("aliases") or {}
        # aliases may be dict or list
        if isinstance(aliases, dict):
            v = aliases.get(source_alias)
            if v:
                mapping[str(v)] = r["global_code"]
    return mapping

def upsert_market_price(payload: Dict[str, Any]) -> bool:
    try:
        resp = sb.table("market_prices").upsert(payload,
                                               on_conflict="source_id,commodity_code,price_date,market_location").execute()
        if hasattr(resp, "data") and resp.data:
            return True
        if getattr(resp, "status_code", None) in (200, 201):
            return True
        logging.warning("Upsert response not OK: %s", resp)
        return False
    except Exception as e:
        logging.error("Upsert failed: %s", e)
        return False

# ---------- Main source processing ----------
def process_source(src: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "source_id": src.get("id"),
        "board_name": src.get("board_name"),
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
    page_requires_session = src.get("page_requires_session", True)
    data_request_method = (src.get("data_request_method") or "GET").upper()

    # 1) load commodity list (local fallback -> remote)
    commodity_map = {}
    if commodity_source == "dropdown_html" and commodity_html_path:
        try:
            commodity_map = load_commodities_from_local(commodity_html_path, selector=commodity_selector)
            logging.info("Loaded %d commodities from local HTML: %s", len(commodity_map), commodity_html_path)
        except Exception as e:
            logging.warning("Failed to load local commodity HTML (%s): %s. Trying remote main page...", commodity_html_path, e)
            commodity_map = load_commodities_from_remote(base_url + main_page, selector=commodity_selector)
            logging.info("Loaded %d commodities from remote main page", len(commodity_map))
    else:
        commodity_map = load_commodities_from_remote(base_url + main_page, selector=commodity_selector)
        logging.info("Loaded %d commodities from remote main page", len(commodity_map))

    if not commodity_map:
        summary["errors"].append("no_commodities")
        logging.warning("No commodities found for source %s", src.get("board_name"))
        return summary

    # load alias map (use 'msamb' as default alias key)
    alias_map = load_commodity_alias_map(source_alias="msamb")

    # optional: fetch main page for cookies
    if page_requires_session:
        try:
            _ = http_get(base_url + (main_page or ""))
            time.sleep(0.5)
        except Exception as e:
            logging.warning("main_page session fetch failed: %s", e)

    # Build effective data URL
    if data_endpoint.startswith("http://") or data_endpoint.startswith("https://"):
        data_url = data_endpoint
    else:
        data_url = base_url + data_endpoint

    # headers for AJAX GET
    ajax_headers = {
        "Referer": base_url + (main_page or ""),
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": USER_AGENT
    }

    for code, display_name in list(commodity_map.items()):
        if not code:
            continue

        params = {"commodityCode": code, "apmcCode": ""}   # note: use empty string rather than "null"
        logging.debug("Requesting %s params=%s", data_url, params)
        r = None
        try:
            if data_request_method == "GET":
                r = http_get(data_url, params=params, headers=ajax_headers)
            else:
                r = http_post(data_url, data=params, headers=ajax_headers)
            if not r:
                logging.warning("No response object for commodity %s (%s)", code, display_name)
                summary["errors"].append(f"no_response:{code}")
                continue

            # Quick debug log for statuses and content shape
            text_snippet = (r.text[:400] + "...") if r.text else "(empty)"
            logging.debug("Response status=%s len=%s snippet=%s", r.status_code, len(r.text or ""), text_snippet)

            # If JSON returned, try to extract html segment or log
            content_type = r.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    payload = r.json()
                    # if payload contains 'html' or 'data', try to use them
                    candidate_html = None
                    if isinstance(payload, dict):
                        # common patterns to inspect
                        for key in ("html", "data", "table", "result"):
                            if key in payload and isinstance(payload[key], str) and "<tr" in payload[key]:
                                candidate_html = payload[key]
                                break
                    if candidate_html:
                        html = candidate_html
                    else:
                        logging.debug("JSON payload received but no html content for %s: keys=%s", code, list(payload.keys()))
                        html = r.text
                except Exception as e:
                    logging.debug("JSON parse failed for %s: %s", code, e)
                    html = r.text
            else:
                html = r.text

            # If HTML contains <tr>, parse; else mark empty
            if html and "<tr" in html:
                parsed_rows = parse_msamb_table(html, display_name)
                if not parsed_rows:
                    logging.info("No parsed rows after html parse for commodity %s (%s)", code, display_name)
                else:
                    summary["rows_fetched"] += len(parsed_rows)
                    for pr in parsed_rows:
                        try:
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

                            ok = upsert_market_price(payload)
                            if ok:
                                summary["rows_upserted"] += 1
                            else:
                                summary["errors"].append(f"upsert_failed:{code}:{pr.get('market')}")
                        except Exception as e:
                            logging.exception("Error processing row for %s %s: %s", code, pr.get("market"), e)
                            summary["errors"].append(f"process_row_error:{code}:{e}")
            else:
                logging.warning("Empty or non-HTML response for commodity %s (%s) status=%s snippet=%s",
                                code, display_name, r.status_code, (r.text[:200] if r.text else "(empty)"))
                summary["errors"].append(f"empty_response:{code}")

        except Exception as e:
            logging.exception("Request/parse failed for %s (%s): %s", code, display_name, e)
            summary["errors"].append(f"request_parse_error:{code}:{e}")

        time.sleep(THROTTLE_SECONDS)

    # update last_checked_at for the source (safe operation)
    try:
        sb.table("agri_market_sources").update({"last_checked_at": datetime.datetime.utcnow().isoformat() + "Z"}).eq("id", source_id).execute()
    except Exception:
        pass

    return summary

def main():
    logging.info("Starting dynamic APMC ingestion")
    sources = load_sources()
    if not sources:
        logging.error("No active agri_market_sources found. Exiting.")
        return

    overall = []
    total_fetched = 0
    total_upserted = 0

    for s in sources:
        logging.info("Processing source: %s / %s (id=%s)", s.get("organization"), s.get("board_name"), s.get("id"))
        summary = process_source(s)
        overall.append(summary)
        total_fetched += summary.get("rows_fetched", 0)
        total_upserted += summary.get("rows_upserted", 0)
        time.sleep(1.0)

    logging.info("Done. fetched=%d upserted=%d", total_fetched, total_upserted)

    # try to insert ingest_run (if table exists)
    try:
        sb.table("ingest_runs").insert({
            "run_time": datetime.datetime.utcnow().isoformat() + "Z",
            "fetched_count": total_fetched,
            "upserted_count": total_upserted,
            "detail": json.dumps(overall)
        }).execute()
    except Exception as e:
        logging.debug("ingest_runs insert failed (maybe table missing): %s", e)

if __name__ == "__main__":
    main()
