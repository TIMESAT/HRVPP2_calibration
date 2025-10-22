#!/usr/bin/env python3
import pandas as pd
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

IN_CSV  = "GPP_callist.csv"           # your input file
OUT_CSV = "GPP_callist_filled.csv"    # final output
TIMEOUT = 25

ISO2_TO_COUNTRY = {
    "AT":"Austria","BE":"Belgium","CH":"Switzerland","CZ":"Czechia","DE":"Germany",
    "DK":"Denmark","ES":"Spain","FI":"Finland","FR":"France","GB":"United Kingdom",
    "IE":"Ireland","IT":"Italy","NL":"Netherlands","NO":"Norway","PL":"Poland",
    "PT":"Portugal","SE":"Sweden","EE":"Estonia","LT":"Lithuania","LV":"Latvia",
    "HU":"Hungary","SK":"Slovakia","SI":"Slovenia","RO":"Romania","BG":"Bulgaria"
}

def station_url(site_code: str) -> str:
    # ICOS Ecosystem station pages usually follow ES_<code>
    # If some codes are Atmosphere (AS_) or Ocean (OS_), adapt here if needed.
    return f"https://meta.icos-cp.eu/resources/stations/ES_{site_code}"

def try_jsonld(url: str) -> Dict[str, Any]:
    """Try to fetch JSON-LD from the station page; returns {} if not available."""
    try:
        r = requests.get(url, headers={"Accept":"application/ld+json"}, timeout=TIMEOUT)
        if r.status_code == 200 and r.headers.get("Content-Type","").lower().find("json") >= 0:
            return r.json()
    except Exception:
        pass
    return {}

def parse_from_jsonld(j: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a few fields from JSON-LD if present."""
    out = {}
    if not j:
        return out
    # JSON-LD may be a list or a dict; search recursively for likely keys
    def walk(obj):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)
    for node in walk(j):
        # country code or country name might appear under address or custom props
        for k in ("countryCode","country_code","country"):
            if k in node and isinstance(node[k], str) and len(node[k]) in (2,3,len(node[k])):
                cc = node[k].upper()
                if len(cc) == 2 and cc.isalpha():
                    out.setdefault("Country code", cc)
                    out.setdefault("Country", ISO2_TO_COUNTRY.get(cc))
                elif k == "country":
                    out.setdefault("Country", node[k])
        # coordinates
        if "latitude" in node:
            try: out.setdefault("Latitude", float(node["latitude"]))
            except: pass
        if "longitude" in node:
            try: out.setdefault("Longitude", float(node["longitude"]))
            except: pass
        # elevation
        for k in ("elevation","elevation_m","altitude"):
            if k in node:
                try:
                    out.setdefault("Elevation_m", float(re.findall(r"[-+]?\d+(\.\d+)?", str(node[k]))[0]))
                except:
                    pass
        # descriptive fields
        for k in ("climateZone","climate_zone","climate"):
            if k in node and isinstance(node[k], str):
                out.setdefault("Climate zone", node[k])
        for k in ("ecosystemType","ecosystem_type","siteType","site_type","landCover","land_cover"):
            if k in node and isinstance(node[k], str):
                out.setdefault("Main ecosystem", node[k])
    return out

def parse_from_html(url: str) -> Dict[str, Any]:
    """Fallback: parse the public landing page text with improved patterns."""
    out = {}
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return out
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text("\n", strip=True)

        # Country code / name
        m = re.search(r"\bCountry code\b[:\s]*([A-Z]{2})\b", text, re.I)
        if m:
            cc = m.group(1).upper()
            out["Country code"] = cc
            out["Country"] = ISO2_TO_COUNTRY.get(cc)
        else:
            m = re.search(r"\bCountry\b[:\s]*([A-Za-z ]+)", text)
            if m: out["Country"] = m.group(1).strip()

        # Climate zone / ecosystem
        m = re.search(r"\bClimate zone\b[:\s]*([^\n]+)", text, re.I)
        if m: out["Climate zone"] = m.group(1).strip()
        m = re.search(r"\b(Main ecosystem|Ecosystem type|Ecosystem)\b[:\s]*([^\n]+)", text, re.I)
        if m: out["Main ecosystem"] = m.group(2).strip()

        # Coordinates & elevation
        m = re.search(r"\bLatitude\b[:\s]*([\-0-9\.]+)", text, re.I)
        if m: out["Latitude"] = float(m.group(1))
        m = re.search(r"\bLongitude\b[:\s]*([\-0-9\.]+)", text, re.I)
        if m: out["Longitude"] = float(m.group(1))
        m = re.search(r"\bElevation\b[:\s]*([0-9\.]+)\s*m\b", text, re.I)
        if m: out["Elevation_m"] = float(m.group(1))
    except Exception:
        pass
    return out

def enrich_site(site_code: str, existing_url: Optional[str]) -> Dict[str, Any]:
    url = existing_url.strip() if (existing_url and str(existing_url).strip()) else station_url(site_code)
    result = {"Website": url}
    # 1) JSON-LD if the server provides it
    j = try_jsonld(url)
    result.update(parse_from_jsonld(j))
    # 2) Fill any gaps from HTML page
    html_info = parse_from_html(url)
    for k, v in html_info.items():
        result.setdefault(k, v)
    return result

def main():
    df = pd.read_csv(IN_CSV)

    # Ensure output columns exist
    cols = ["Sitename","Country code","Country","Latitude","Longitude","Elevation_m",
            "Climate zone","Main ecosystem","Website"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Process every row, never stop on errors
    rows = []
    for i in tqdm(range(len(df)), desc="Enriching ICOS sites"):
        site = str(df.at[i, "Sitename"]).strip()
        website = df.at[i, "Website"] if "Website" in df.columns else None
        try:
            info = enrich_site(site, website)
        except Exception as e:
            info = {"Website": website or station_url(site)}
            print(f"[WARN] {site}: {e}", file=sys.stderr)

        # Write back
        for k, v in info.items():
            df.at[i, k] = v

    # Type coercion
    for c in ("Latitude","Longitude","Elevation_m"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV} with {len(df)} rows.")

if __name__ == "__main__":
    main()
