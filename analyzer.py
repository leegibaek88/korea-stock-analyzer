from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

from listing_provider import load_listing


PERIOD_DAYS = {
    "1년": 365,
    "2년": 365 * 2,
    "3년": 365 * 3,
    "5년": 365 * 5,
}


def load_stock_listing(market: str) -> pd.DataFrame:
    listing = load_listing(market)
    if listing is None or listing.empty:
        raise ValueError("상장 종목 목록을 불러오지 못했습니다.")
    return listing[["Code", "Name"]].drop_duplicates()


def resolve_stock_query(query: str, market: str, listing: pd.DataFrame | None = None) -> dict[str, str]:
    text = str(query).strip()
    if not text:
        raise ValueError("종목코드 또는 종목명을 입력하십시오.")

    if text.isdigit():
        code = normalize_code(text)
        name = ""
        if listing is not None and not listing.empty:
            matched = listing[listing["Code"] == code]
            if not matched.empty:
                name = str(matched.iloc[0]["Name"])
        return {"code": code, "name": name or code}

    listing = listing if listing is not None else load_stock_listing(market)
    exact = listing[listing["Name"].str.lower() == text.lower()]
    if not exact.empty:
        row = exact.iloc[0]
        return {"code": str(row["Code"]), "name": str(row["Name"])}

    contains = listing[listing["Name"].str.contains(text, case=False, na=False, regex=False)]
    if contains.empty:
        raise ValueError("종목명을 찾지 못했습니다. 종목코드 또는 정확한 종목명을 확인하십시오.")
    if len(contains) > 1:
        examples = ", ".join(f"{row.Name}({row.Code})" for row in contains.head(8).itertuples())
        raise ValueError(f"여러 종목이 검색되었습니다. 더 정확히 입력하십시오: {examples}")

    row = contains.iloc[0]
    return {"code": str(row["Code"]), "name": str(row["Name"])}


def load_stock_data(code: str, market: str, period: str = "3년") -> tuple[pd.DataFrame, pd.DataFrame]:
    code = normalize_code(code)
    days = PERIOD_DAYS.get(period, PERIOD_DAYS["3년"])
    start_date = (datetime.today() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
    benchmark_code = "KS11" if market == "KOSPI" else "KQ11"

    stock = fdr.DataReader(code, start_date)
    benchmark = fdr.DataReader(benchmark_code, start_date)

    if stock is None or stock.empty:
        raise ValueError("종목코드 또는 데이터 제공 상태를 확인하십시오.")
    if benchmark is None or benchmark.empty:
        raise ValueError("시장지수 데이터 제공 상태를 확인하십시오.")

    stock = _clean_ohlcv(stock)
    benchmark = _clean_ohlcv(benchmark)
    common_dates = stock.index.intersection(benchmark.index)
    stock = stock.loc[common_dates].copy()
    benchmark = benchmark.loc[common_dates].copy()

    if stock.empty:
        raise ValueError("종목코드 또는 데이터 제공 상태를 확인하십시오.")

    return stock, benchmark


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["MA20"] = result["Close"].rolling(20).mean()
    result["MA50"] = result["Close"].rolling(50).mean()
    result["MA150"] = result["Close"].rolling(150).mean()
    result["MA200"] = result["Close"].rolling(200).mean()
    result["VOL20"] = result["Volume"].rolling(20).mean()
    result["VOL50"] = result["Volume"].rolling(50).mean()
    return result


def calculate_minervini(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    high_52w = float(df["Close"].tail(min(252, len(df))).max())
    low_52w = float(df["Close"].tail(min(252, len(df))).min())
    ma200_20d_ago = _value_days_ago(df["MA200"], 20)

    checks = [
        _check("현재가 > MA150", current, "MA150", latest.get("MA150"), current > latest.get("MA150", np.nan)),
        _check("현재가 > MA200", current, "MA200", latest.get("MA200"), current > latest.get("MA200", np.nan)),
        _check("MA150 > MA200", latest.get("MA150"), "MA200", latest.get("MA200"), latest.get("MA150", np.nan) > latest.get("MA200", np.nan)),
        _check("MA200 상승", latest.get("MA200"), "20거래일 전 MA200", ma200_20d_ago, latest.get("MA200", np.nan) > ma200_20d_ago),
        _check("MA50 > MA150", latest.get("MA50"), "MA150", latest.get("MA150"), latest.get("MA50", np.nan) > latest.get("MA150", np.nan)),
        _check("MA50 > MA200", latest.get("MA50"), "MA200", latest.get("MA200"), latest.get("MA50", np.nan) > latest.get("MA200", np.nan)),
        _check("52주 저점 +30%", current, f"{low_52w * 1.30:,.0f}", low_52w * 1.30, current >= low_52w * 1.30),
        _check("52주 고점 75% 이상", current, f"{high_52w * 0.75:,.0f}", high_52w * 0.75, current >= high_52w * 0.75),
    ]
    valid = [item for item in checks if item["result"] != "데이터 부족"]
    passed = sum(1 for item in valid if item["result"] == "PASS")
    score = passed / len(valid) * 100 if valid else 0
    return {"checks": checks, "score": round(score, 1), "passed": passed, "valid": len(valid), "high_52w": high_52w, "low_52w": low_52w}


def calculate_weinstein(df: pd.DataFrame) -> dict[str, Any]:
    weekly = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    weekly["WMA30"] = weekly["Close"].rolling(30).mean()
    latest = weekly.iloc[-1]
    current = float(latest["Close"])
    wma30 = latest.get("WMA30", np.nan)
    wma30_4w_ago = _value_days_ago(weekly["WMA30"], 4)
    high_20w = float(weekly["Close"].tail(min(20, len(weekly))).max())

    if pd.isna(wma30) or pd.isna(wma30_4w_ago):
        stage = "Stage 1/3 횡보구간"
        score = 50
        reason = "30주 이동평균 데이터가 부족해 보수적으로 횡보구간으로 분류합니다."
    elif current > wma30 and wma30 > wma30_4w_ago and current >= high_20w * 0.90:
        stage = "Stage 2 상승추세"
        score = 100
        reason = "주가가 30주선 위에 있고 30주선이 상승 중입니다."
    elif current < wma30 and wma30 < wma30_4w_ago:
        stage = "Stage 4 하락추세"
        score = 0
        reason = "주가가 30주선 아래에 있고 30주선이 하락 중입니다."
    else:
        stage = "Stage 1/3 횡보구간"
        score = 50
        reason = "명확한 상승 또는 하락 추세로 보기 어렵습니다."

    return {"stage": stage, "score": score, "weekly": weekly, "wma30": _float_or_none(wma30), "reason": reason}


def calculate_oneil(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    high_60d = float(df["High"].tail(min(60, len(df))).max())
    previous_60d_high = df["High"].shift(1).rolling(60).max().iloc[-1]
    distance_to_60d_high = (high_60d - current) / high_60d * 100 if high_60d else np.nan
    near_high = current >= high_60d * 0.95 if high_60d else False
    breakout = bool(pd.notna(previous_60d_high) and current > previous_60d_high)
    volume_ratio = _safe_ratio(latest["Volume"], latest.get("VOL20"))

    checks = {
        "현재가 > MA50": bool(pd.notna(latest.get("MA50")) and current > latest["MA50"]),
        "현재가 > MA200": bool(pd.notna(latest.get("MA200")) and current > latest["MA200"]),
        "MA50 > MA200": bool(pd.notna(latest.get("MA50")) and pd.notna(latest.get("MA200")) and latest["MA50"] > latest["MA200"]),
        "60일 최고가 5% 이내": bool(near_high),
        "60일 신고가 돌파": breakout,
    }

    score = 0
    score += 15 if checks["현재가 > MA50"] else 0
    score += 15 if checks["현재가 > MA200"] else 0
    score += 15 if checks["MA50 > MA200"] else 0
    score += 15 if near_high else 0
    score += 20 if breakout else 0
    score += 20 if volume_ratio >= 1.5 else 10 if volume_ratio >= 1.2 else 0

    if breakout:
        breakout_status = "신고가 돌파"
    elif near_high:
        breakout_status = "돌파 대기권"
    else:
        breakout_status = "돌파권 밖"

    return {
        "score": min(score, 100),
        "checks": checks,
        "high_60d": high_60d,
        "previous_60d_high": _float_or_none(previous_60d_high),
        "distance_to_60d_high": _float_or_none(distance_to_60d_high),
        "near_high": near_high,
        "breakout": breakout,
        "volume_ratio": volume_ratio,
        "volume_status": classify_breakout_volume(volume_ratio),
        "breakout_status": breakout_status,
    }


def calculate_relative_strength(stock: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    periods = {"1개월": 20, "3개월": 60, "6개월": 120, "12개월": 240}
    current_stock = float(stock["Close"].iloc[-1])
    current_market = float(benchmark["Close"].iloc[-1])
    values: dict[str, float | None] = {}
    score = 0

    for label, days in periods.items():
        if len(stock) <= days or len(benchmark) <= days:
            values[label] = None
            continue
        stock_return = current_stock / float(stock["Close"].iloc[-days - 1]) - 1
        market_return = current_market / float(benchmark["Close"].iloc[-days - 1]) - 1
        rs = (stock_return - market_return) * 100
        values[label] = round(rs, 1)
        score += 25 if rs > 0 else 0

    return {"values": values, "score": score}


def calculate_vcp(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 60:
        return {"status": "데이터 부족", "score": 0, "volatilities": {}, "volumes": {}, "volatility_pass": False, "volume_pass": False}

    recent = df.tail(60)
    segments = {
        "A": recent.iloc[0:20],
        "B": recent.iloc[20:40],
        "C": recent.iloc[40:60],
    }
    volatilities = {k: _segment_volatility(v) for k, v in segments.items()}
    volumes = {k: float(v["Volume"].mean()) for k, v in segments.items()}
    volatility_pass = volatilities["A"] > volatilities["B"] > volatilities["C"]
    volume_pass = volumes["C"] < volumes["A"]

    if volatility_pass and volume_pass:
        status = "VCP 유사패턴 감지"
        score = 100
    elif volatility_pass or volume_pass:
        status = "부분적 수축"
        score = 50
    else:
        status = "VCP 조건 미충족"
        score = 0

    return {
        "status": status,
        "score": score,
        "volatilities": {k: round(v, 1) for k, v in volatilities.items()},
        "volumes": {k: round(v, 0) for k, v in volumes.items()},
        "volatility_pass": volatility_pass,
        "volume_pass": volume_pass,
    }


def calculate_volume_analysis(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    ratio = _safe_ratio(latest["Volume"], latest.get("VOL20"))
    if ratio >= 2.0:
        status = "폭발적 거래량"
    elif ratio >= 1.5:
        status = "강한 거래량"
    elif ratio >= 1.2:
        status = "평균 이상"
    elif ratio >= 0.8:
        status = "보통"
    else:
        status = "거래량 감소"
    return {
        "today": int(latest["Volume"]),
        "vol20": _float_or_none(latest.get("VOL20")),
        "vol50": _float_or_none(latest.get("VOL50")),
        "ratio": ratio,
        "status": status,
    }


def calculate_final_score(
    minervini: dict[str, Any],
    oneil: dict[str, Any],
    weinstein: dict[str, Any],
    rs: dict[str, Any],
    vcp: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    ma50 = latest.get("MA50", np.nan)
    ma200 = latest.get("MA200", np.nan)

    total_score = (
        minervini["score"] * 0.35
        + oneil["score"] * 0.25
        + weinstein["score"] * 0.15
        + rs["score"] * 0.15
        + vcp["score"] * 0.10
    )
    total_score = round(total_score, 1)

    forced_exclusion = (
        (pd.notna(ma200) and current < ma200)
        or (pd.notna(ma50) and pd.notna(ma200) and ma50 < ma200)
        or weinstein["stage"] == "Stage 4 하락추세"
    )
    if forced_exclusion:
        rating = "제외 / 하락추세"
        stars = "★"
    elif total_score >= 85:
        rating = "최우선 관심종목"
        stars = "★★★★★"
    elif total_score >= 75:
        rating = "매수후보"
        stars = "★★★★"
    elif total_score >= 65:
        rating = "관심종목"
        stars = "★★★"
    elif total_score >= 50:
        rating = "관망"
        stars = "★★"
    else:
        rating = "제외"
        stars = "★"

    ma20_gap = _gap(current, latest.get("MA20"))
    ma50_gap = _gap(current, ma50)
    entry_status = calculate_entry_status(oneil, total_score, current, ma50, ma20_gap, ma50_gap)
    good_stock = "YES" if not forced_exclusion and total_score >= 75 else "WATCH" if total_score >= 65 else "NO"
    good_price = "BUY ZONE" if entry_status == "돌파 매수구간" else "WAIT" if entry_status in {"돌파 대기", "관망"} else "CAUTION"

    return {
        "total_score": total_score,
        "rating": rating,
        "stars": stars,
        "forced_exclusion": forced_exclusion,
        "entry_status": entry_status,
        "good_stock": good_stock,
        "good_price": good_price,
        "ma20_gap": _float_or_none(ma20_gap),
        "ma50_gap": _float_or_none(ma50_gap),
    }


def calculate_trade_plan(df: pd.DataFrame, oneil: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    ma20 = _float_or_none(latest.get("MA20"))
    ma50 = _float_or_none(latest.get("MA50"))
    low_20d = float(df["Low"].tail(min(20, len(df))).min())
    pivot = oneil.get("previous_60d_high") or oneil.get("high_60d")

    if final["forced_exclusion"]:
        method = "진입 보류"
        buy_low = buy_high = None
        note = "하락추세 또는 핵심 추세조건 미충족으로 신규 진입보다 관망이 우선입니다."
    elif final["entry_status"] == "돌파 매수구간" and pivot:
        method = "돌파 확인 후 분할 진입"
        buy_low = max(float(pivot), current * 0.98)
        buy_high = min(max(current, float(pivot) * 1.03), current * 1.04)
        note = "돌파와 거래량이 확인된 경우 피벗 근처에서 과도한 추격을 피하는 구간입니다."
    elif final["entry_status"] == "돌파 대기" and pivot:
        method = "60일 고점 돌파 확인 대기"
        buy_low = float(pivot)
        buy_high = float(pivot) * 1.03
        note = "신고가권에 있으나 돌파가 확정되지 않아 피벗 돌파와 거래량 확인이 필요합니다."
    elif final["entry_status"] == "과열 / 추격매수 주의":
        method = "눌림목 대기"
        anchor = ma20 or ma50 or current
        buy_low = anchor * 0.98
        buy_high = anchor * 1.03
        note = "현재 이격도가 높아 이동평균선 근처까지 가격이 안정되는지 확인하는 방식입니다."
    elif ma20 and current >= ma20:
        method = "MA20 부근 분할 진입 검토"
        buy_low = ma20 * 0.98
        buy_high = ma20 * 1.03
        note = "추세가 무너지지 않는 범위에서 단기 이동평균선 부근의 눌림을 활용하는 방식입니다."
    elif ma50:
        method = "MA50 회복 확인 후 진입"
        buy_low = ma50
        buy_high = ma50 * 1.03
        note = "현재가가 50일선 아래라면 회복 확인 전 신규 진입은 보수적으로 보는 구간입니다."
    else:
        method = "관망"
        buy_low = buy_high = None
        note = "충분한 이동평균 데이터가 부족해 구체적인 가격대를 제시하기 어렵습니다."

    defense_candidates = [low_20d]
    if ma20:
        defense_candidates.append(ma20)
    if ma50:
        defense_candidates.append(ma50)
    defense_level = min(defense_candidates)
    risk_sell_low = defense_level * 0.97
    risk_sell_high = defense_level

    profit_base = buy_high or current
    profit_sell_low = profit_base * 1.15
    profit_sell_high = profit_base * 1.25

    return {
        "entry_method": method,
        "buy_low": _float_or_none(buy_low),
        "buy_high": _float_or_none(buy_high),
        "risk_sell_low": _float_or_none(risk_sell_low),
        "risk_sell_high": _float_or_none(risk_sell_high),
        "profit_sell_low": _float_or_none(profit_sell_low),
        "profit_sell_high": _float_or_none(profit_sell_high),
        "note": note,
    }


def calculate_entry_status(oneil: dict[str, Any], total_score: float, current: float, ma50: float, ma20_gap: float | None, ma50_gap: float | None) -> str:
    overheated = (ma20_gap is not None and ma20_gap > 15) or (ma50_gap is not None and ma50_gap > 25)
    if oneil["breakout"] and oneil["volume_ratio"] >= 1.5 and total_score >= 75:
        return "돌파 매수구간"
    if overheated:
        return "과열 / 추격매수 주의"
    if oneil["near_high"] and not oneil["breakout"] and total_score >= 70:
        return "돌파 대기"
    if pd.notna(ma50) and current < ma50:
        return "진입 보류"
    return "관망"


def generate_analysis_text(analysis: dict[str, Any]) -> dict[str, Any]:
    latest = analysis["df"].iloc[-1]
    current = float(latest["Close"])
    oneil = analysis["oneil"]
    rs = analysis["rs"]
    vcp = analysis["vcp"]
    volume = analysis["volume"]
    final = analysis["final"]
    minervini = analysis["minervini"]

    positives: list[str] = []
    risks: list[str] = []
    paragraphs: list[str] = []

    if _ordered_ma(latest):
        positives.append("50 > 150 > 200일선 정배열")
        paragraphs.append("현재 주가는 주요 이동평균선 위에 있고 50일선 > 150일선 > 200일선 구조가 형성되어 장기 상승추세가 양호합니다.")
    elif pd.notna(latest.get("MA200")) and current < latest["MA200"]:
        risks.append("현재가가 200일선 아래")
        paragraphs.append("현재 주가는 200일 이동평균선 아래에 있어 추세 추종 관점에서는 보수적인 접근이 필요합니다.")
    else:
        paragraphs.append("주요 이동평균선 구조는 일부만 충족되어 추세 확인이 더 필요합니다.")

    ma200_rising = next((item for item in minervini["checks"] if item["condition"] == "MA200 상승"), None)
    if ma200_rising and ma200_rising["result"] == "PASS":
        positives.append("200일선 상승")
    elif ma200_rising and ma200_rising["result"] == "FAIL":
        risks.append("200일선 상승 조건 미충족")

    high_gap = _gap_from_high(current, minervini["high_52w"])
    if high_gap <= 5:
        positives.append(f"52주 신고가 {high_gap:.1f}% 이내")
        paragraphs.append(f"현재가는 52주 최고가에서 {high_gap:.1f}% 떨어진 위치로 신고가권에 있습니다.")
    else:
        risks.append(f"52주 최고가 대비 {high_gap:.1f}% 아래")

    rs_3m = rs["values"].get("3개월")
    if rs_3m is not None:
        if rs_3m > 0:
            positives.append(f"시장 대비 3개월 {rs_3m:+.1f}%p")
        else:
            risks.append(f"시장 대비 3개월 {rs_3m:+.1f}%p")
        paragraphs.append(f"3개월 시장대비 상대강도는 {rs_3m:+.1f}%p로 {'시장보다 강합니다' if rs_3m > 0 else '시장보다 약합니다'}.")

    if volume["ratio"] >= 1.2:
        positives.append(f"최근 거래량 20일 평균 대비 {volume['ratio']:.2f}배")
        paragraphs.append(f"최근 거래량은 20일 평균의 {volume['ratio']:.2f}배로 {volume['status']} 상태입니다.")
    else:
        risks.append(f"거래량이 20일 평균 대비 {volume['ratio']:.2f}배")

    if vcp["status"] == "VCP 유사패턴 감지":
        positives.append("최근 60일 변동성 및 거래량 수축")
        paragraphs.append("최근 60일 변동성이 단계적으로 축소되고 거래량도 줄어 VCP와 유사한 가격 압축이 관찰됩니다.")
    elif vcp["status"] == "부분적 수축":
        risks.append("VCP 조건 일부만 충족")
    else:
        risks.append("VCP 조건 미충족")

    if final["ma20_gap"] is not None and final["ma20_gap"] > 15:
        risks.append(f"MA20 이격도 +{final['ma20_gap']:.1f}%")
        paragraphs.append(f"다만 현재가는 20일 이동평균선보다 {final['ma20_gap']:.1f}% 높아 단기 추격매수 위험이 있습니다.")
    if final["ma50_gap"] is not None and final["ma50_gap"] > 25:
        risks.append(f"MA50 이격도 +{final['ma50_gap']:.1f}%")

    if oneil["breakout"]:
        positives.append("직전 60거래일 고점 돌파")
    else:
        risks.append("아직 60일 신고가 완전 돌파 전")

    trade_plan = analysis["trade_plan"]
    paragraphs.append(
        f"추천 진입방식은 '{trade_plan['entry_method']}'입니다. 매수가격대와 매도가격대는 자동 주문 기준이 아니라 추세와 방어선을 바탕으로 한 참고 범위입니다."
    )

    trend_summary = "강한 상승추세" if analysis["weinstein"]["score"] == 100 and minervini["score"] >= 75 else "하락추세 주의" if final["forced_exclusion"] else "추세 확인 필요"
    strength_summary = "시장 대비 매우 강함" if rs["score"] >= 75 else "시장 대비 보통" if rs["score"] >= 50 else "시장 대비 약함"
    if final["entry_status"] == "과열 / 추격매수 주의":
        timing_summary = "좋은 종목이어도 현재는 단기 과열"
    elif final["entry_status"] == "돌파 대기":
        timing_summary = "신고가권 근처에서 돌파 확인 대기"
    elif final["entry_status"] == "돌파 매수구간":
        timing_summary = "돌파와 거래량이 함께 확인된 구간"
    else:
        timing_summary = final["entry_status"]

    return {
        "paragraphs": paragraphs,
        "positive_factors": positives,
        "risk_factors": risks,
        "diagnosis": {"추세": trend_summary, "종목의 강도": strength_summary, "현재 매수 위치": timing_summary},
    }


def analyze_stock(code: str, market: str = "KOSPI", period: str = "3년", name: str | None = None) -> dict[str, Any]:
    stock, benchmark = load_stock_data(code, market, period)
    df = calculate_indicators(stock)
    benchmark = calculate_indicators(benchmark)
    minervini = calculate_minervini(df)
    weinstein = calculate_weinstein(df)
    oneil = calculate_oneil(df)
    rs = calculate_relative_strength(df, benchmark)
    vcp = calculate_vcp(df)
    volume = calculate_volume_analysis(df)
    final = calculate_final_score(minervini, oneil, weinstein, rs, vcp, df)
    trade_plan = calculate_trade_plan(df, oneil, final)
    analysis = {
        "code": normalize_code(code),
        "name": name or normalize_code(code),
        "market": market,
        "period": period,
        "df": df,
        "benchmark": benchmark,
        "minervini": minervini,
        "weinstein": weinstein,
        "oneil": oneil,
        "rs": rs,
        "vcp": vcp,
        "volume": volume,
        "final": final,
        "trade_plan": trade_plan,
    }
    analysis["text"] = generate_analysis_text(analysis)
    return analysis


def normalize_code(code: str) -> str:
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    return digits.zfill(6)[:6]


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    for col in required:
        if col not in result.columns:
            raise ValueError("종목코드 또는 데이터 제공 상태를 확인하십시오.")
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result[required].dropna(subset=["Close"]).sort_index()


def _check(condition: str, current_value: Any, benchmark_label: str, benchmark_value: Any, passed: bool) -> dict[str, Any]:
    if pd.isna(current_value) or pd.isna(benchmark_value):
        result = "데이터 부족"
    else:
        result = "PASS" if passed else "FAIL"
    return {
        "condition": condition,
        "current": _format_number(current_value),
        "criteria": benchmark_label if isinstance(benchmark_label, str) and not benchmark_label.replace(",", "").replace(".", "").isdigit() else str(benchmark_label),
        "result": result,
    }


def _value_days_ago(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return float(series.iloc[-days - 1])


def _safe_ratio(value: Any, base: Any) -> float:
    if pd.isna(value) or pd.isna(base) or float(base) == 0:
        return 0.0
    return round(float(value) / float(base), 2)


def _gap(current: float, base: Any) -> float | None:
    if pd.isna(base) or float(base) == 0:
        return None
    return (current / float(base) - 1) * 100


def _gap_from_high(current: float, high: float) -> float:
    if not high:
        return 0.0
    return max((high - current) / high * 100, 0.0)


def _segment_volatility(segment: pd.DataFrame) -> float:
    high = float(segment["High"].max())
    low = float(segment["Low"].min())
    return (high - low) / high * 100 if high else 0.0


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return "데이터 부족"
    return f"{float(value):,.0f}"


def _ordered_ma(latest: pd.Series) -> bool:
    values = [latest.get("MA50"), latest.get("MA150"), latest.get("MA200")]
    return all(pd.notna(v) for v in values) and values[0] > values[1] > values[2] and latest["Close"] > values[0]


def classify_breakout_volume(volume_ratio: float) -> str:
    if volume_ratio >= 1.5:
        return "강한 거래량"
    if volume_ratio >= 1.2:
        return "양호"
    return "거래량 부족"
