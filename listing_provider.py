from __future__ import annotations

import re
import FinanceDataReader as fdr
import pandas as pd
import requests
from bs4 import BeautifulSoup


MARKETS = ["KOSPI", "KOSDAQ"]


def load_market_snapshot() -> pd.DataFrame:
    try:
        listing = fdr.StockListing("KRX")
        return normalize_listing(listing)
    except Exception:
        frames = [load_naver_market_page(market) for market in MARKETS]
        return normalize_listing(pd.concat(frames, ignore_index=True))


def load_listing(market: str) -> pd.DataFrame:
    snapshot = load_market_snapshot()
    if market in MARKETS:
        snapshot = snapshot[snapshot["Market"] == market].copy()
    return snapshot[["Code", "Name"]].drop_duplicates()


def normalize_listing(listing: pd.DataFrame) -> pd.DataFrame:
    result = listing.copy()
    rename_map = {"ChagesRatio": "ChagesRatio", "Change": "ChagesRatio"}
    result = result.rename(columns=rename_map)
    if "Market" not in result.columns:
        result["Market"] = ""
    if "Amount" not in result.columns:
        result["Amount"] = pd.to_numeric(result.get("Close"), errors="coerce") * pd.to_numeric(result.get("Volume"), errors="coerce")

    result["Code"] = result["Code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    result["Name"] = result["Name"].astype(str)
    for col in ["Close", "ChagesRatio", "Volume", "Amount", "Marcap"]:
        if col not in result.columns:
            result[col] = 0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result[result["Code"].str.len() == 6].dropna(subset=["Code", "Name", "Market"])


def load_naver_market_page(market: str, max_pages: int = 12) -> pd.DataFrame:
    sosok = "0" if market == "KOSPI" else "1"
    rows = []
    for page in range(1, max_pages + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        response.encoding = "euc-kr"
        soup = BeautifulSoup(response.text, "html.parser")
        page_rows = []
        for tr in soup.select("table.type_2 tr"):
            title = tr.select_one("a.tltle")
            if not title:
                continue
            code_match = re.search(r"code=(\d{6})", title.get("href", ""))
            if not code_match:
                continue
            cells = [cell.get_text(strip=True) for cell in tr.select("td")]
            if len(cells) < 11:
                continue
            close = _to_number(cells[2])
            volume = _to_number(cells[9])
            page_rows.append(
                {
                    "Code": code_match.group(1),
                    "Name": title.get_text(strip=True),
                    "Market": market,
                    "Close": close,
                    "ChagesRatio": _to_number(cells[4].replace("%", "")),
                    "Volume": volume,
                    "Amount": close * volume,
                    "Marcap": _to_number(cells[6]) * 100_000_000,
                }
            )
        if not page_rows:
            break
        rows.extend(page_rows)
    return pd.DataFrame(rows)


def _to_number(value: object) -> float:
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "N/A", "nan"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0
