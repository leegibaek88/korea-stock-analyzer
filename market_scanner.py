from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import FinanceDataReader as fdr
import pandas as pd

from analyzer import (
    calculate_final_score,
    calculate_indicators,
    calculate_minervini,
    calculate_oneil,
    calculate_relative_strength,
    calculate_trade_plan,
    calculate_vcp,
    calculate_volume_analysis,
    calculate_weinstein,
)
from listing_provider import load_market_snapshot


RECOMMEND_LOOKBACKS = {
    "일간": 1,
    "주간": 5,
    "월간": 20,
}

MOVER_LOOKBACKS = {
    "일간": 1,
    "월간": 20,
    "연간": 240,
}


def analysis_link(code: str, market: str) -> str:
    return f"?analyze_code={str(code).zfill(6)}&market={market}"


def market_snapshot() -> pd.DataFrame:
    return load_market_snapshot()


def recommendation_top10_by_market(mode: str = "일간", per_market_universe: int = 30) -> dict[str, pd.DataFrame]:
    listing = market_snapshot()
    start_date = (datetime.today() - timedelta(days=365 * 3 + 45)).strftime("%Y-%m-%d")
    benchmarks = {
        "KOSPI": calculate_indicators(_clean_ohlcv(fdr.DataReader("KS11", start_date))),
        "KOSDAQ": calculate_indicators(_clean_ohlcv(fdr.DataReader("KQ11", start_date))),
    }

    output = {}
    for market in ["KOSPI", "KOSDAQ"]:
        universe = _recommendation_universe(listing, market, per_market_universe)
        rows = _scan_recommendations(universe, benchmarks[market], start_date, mode)
        frame = pd.DataFrame(rows)
        if frame.empty:
            output[market] = frame
            continue
        frame = frame.sort_values(
            ["추천점수", "종합점수", "시장대비3개월", "거래량배수"],
            ascending=[False, False, False, False],
        ).head(10)
        output[market] = _format_recommendation_frame(frame)
    return output


def movers_top10_by_market(period: str = "일간", direction: str = "상승", per_market_universe: int = 60) -> dict[str, pd.DataFrame]:
    bundle = movers_top10_bundle_by_market(period, per_market_universe)
    return {market: bundle[market][direction] for market in ["KOSPI", "KOSDAQ"]}


def movers_top10_bundle_by_market(period: str = "일간", per_market_universe: int = 60) -> dict[str, dict[str, pd.DataFrame]]:
    listing = market_snapshot()
    output = {}
    for market in ["KOSPI", "KOSDAQ"]:
        market_listing = listing[listing["Market"] == market].copy()
        if period == "일간":
            frame = market_listing.dropna(subset=["ChagesRatio"]).copy()
            frame["등락률"] = frame["ChagesRatio"]
        else:
            universe = _mover_universe(market_listing, per_market_universe)
            frame = pd.DataFrame(_scan_movers(universe, MOVER_LOOKBACKS[period]))

        if frame.empty:
            output[market] = {"상승": frame, "하락": frame}
            continue

        gainers = frame.sort_values("등락률", ascending=False).head(10)
        losers = frame.sort_values("등락률", ascending=True).head(10)
        output[market] = {
            "상승": _format_mover_frame(gainers),
            "하락": _format_mover_frame(losers),
        }
    return output


def unusual_risk_top10_by_market(per_market_universe: int = 80) -> dict[str, pd.DataFrame]:
    listing = market_snapshot()
    output = {}
    start_date = (datetime.today() - timedelta(days=210)).strftime("%Y-%m-%d")
    for market in ["KOSPI", "KOSDAQ"]:
        universe = _unusual_universe(listing, market, per_market_universe)
        rows = _scan_unusual(universe, start_date)
        frame = pd.DataFrame(rows)
        if frame.empty:
            output[market] = frame
            continue
        frame = frame.sort_values(["주의점수", "거래량배수", "월간등락률"], ascending=False).head(10)
        output[market] = _format_unusual_frame(frame)
    return output


def historical_success_cases(per_market_universe: int = 35, forward_days: int = 60, limit: int = 10) -> list[dict[str, Any]]:
    listing = market_snapshot()
    start_date = (datetime.today() - timedelta(days=365 * 3 + 90)).strftime("%Y-%m-%d")
    benchmarks = {
        "KOSPI": calculate_indicators(_clean_ohlcv(fdr.DataReader("KS11", start_date))),
        "KOSDAQ": calculate_indicators(_clean_ohlcv(fdr.DataReader("KQ11", start_date))),
    }
    universes = [
        _recommendation_universe(listing, "KOSPI", per_market_universe),
        _recommendation_universe(listing, "KOSDAQ", per_market_universe),
    ]
    universe = pd.concat(universes, ignore_index=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_historical_case_candidate, row.Code, row.Name, row.Market, benchmarks[row.Market], start_date, forward_days): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)

    return sorted(rows, key=lambda item: (item["이후수익률"], item["당시점수"]), reverse=True)[:limit]


def _scan_recommendations(universe: pd.DataFrame, benchmark: pd.DataFrame, start_date: str, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_analyze_candidate, row.Code, row.Name, row.Market, benchmark, start_date, mode): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    return rows


def _analyze_candidate(code: str, name: str, market: str, benchmark: pd.DataFrame, start_date: str, mode: str) -> dict[str, Any] | None:
    try:
        stock = calculate_indicators(_clean_ohlcv(fdr.DataReader(code, start_date)))
        common_dates = stock.index.intersection(benchmark.index)
        stock = stock.loc[common_dates].copy()
        aligned_benchmark = benchmark.loc[common_dates].copy()
        if len(stock) < 80:
            return None

        minervini = calculate_minervini(stock)
        weinstein = calculate_weinstein(stock)
        oneil = calculate_oneil(stock)
        rs = calculate_relative_strength(stock, aligned_benchmark)
        vcp = calculate_vcp(stock)
        volume = calculate_volume_analysis(stock)
        final = calculate_final_score(minervini, oneil, weinstein, rs, vcp, stock)
        if final["forced_exclusion"]:
            return None

        trade_plan = calculate_trade_plan(stock, oneil, final)
        latest = stock.iloc[-1]
        lookback = RECOMMEND_LOOKBACKS.get(mode, 1)
        recent_return = _return_over_days(stock, lookback)
        momentum_bonus = _momentum_bonus(recent_return)
        recommendation_score = round(final["total_score"] * 0.9 + momentum_bonus, 1)

        return {
            "종목": f"{name} ({code})",
            "분석 링크": analysis_link(code, market),
            "시장": market,
            "현재가": float(latest["Close"]),
            "종합점수": final["total_score"],
            "추천점수": recommendation_score,
            "기간수익률": recent_return,
            "시장대비3개월": rs["values"].get("3개월"),
            "거래량배수": volume["ratio"],
            "등급": f"{final['stars']} {final['rating']}",
            "진입상태": final["entry_status"],
            "추천 진입방식": trade_plan["entry_method"],
            "추천 매수가격대": _price_range(trade_plan["buy_low"], trade_plan["buy_high"]),
            "방어적 매도가격대": _price_range(trade_plan["risk_sell_low"], trade_plan["risk_sell_high"]),
            "분할 매도가격대": _price_range(trade_plan["profit_sell_low"], trade_plan["profit_sell_high"]),
        }
    except Exception:
        return None


def _scan_movers(universe: pd.DataFrame, lookback: int) -> list[dict[str, Any]]:
    start_date = (datetime.today() - timedelta(days=max(lookback * 2, 365) + 15)).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_mover_candidate, row.Code, row.Name, row.Market, start_date, lookback): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    return rows


def _mover_candidate(code: str, name: str, market: str, start_date: str, lookback: int) -> dict[str, Any] | None:
    try:
        df = _clean_ohlcv(fdr.DataReader(code, start_date))
        if len(df) <= lookback:
            return None
        current = float(df["Close"].iloc[-1])
        past = float(df["Close"].iloc[-lookback - 1])
        if past <= 0:
            return None
        return {
            "종목": f"{name} ({code})",
            "분석 링크": analysis_link(code, market),
            "시장": market,
            "현재가": current,
            "등락률": (current / past - 1) * 100,
            "거래대금": None,
        }
    except Exception:
        return None


def _scan_unusual(universe: pd.DataFrame, start_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_unusual_candidate, row.Code, row.Name, row.Market, row.Close, row.ChagesRatio, row.Volume, row.Amount, row.Marcap, start_date): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    return rows


def _unusual_candidate(
    code: str,
    name: str,
    market: str,
    close: float,
    daily_change: float,
    volume: float,
    amount: float,
    marcap: float,
    start_date: str,
) -> dict[str, Any] | None:
    try:
        df = calculate_indicators(_clean_ohlcv(fdr.DataReader(code, start_date)))
        if len(df) < 40:
            return None
        latest = df.iloc[-1]
        monthly_return = _return_over_days(df, 20)
        weekly_return = _return_over_days(df, 5)
        volume_ratio = float(volume) / float(latest["VOL20"]) if pd.notna(latest.get("VOL20")) and latest["VOL20"] else 0
        ma20_gap = (float(close) / float(latest["MA20"]) - 1) * 100 if pd.notna(latest.get("MA20")) and latest["MA20"] else 0
        intraday_range = (float(latest["High"]) - float(latest["Low"])) / float(latest["Close"]) * 100 if latest["Close"] else 0

        score = 0.0
        reasons = []
        if abs(float(daily_change)) >= 20:
            score += 25
            reasons.append(f"일간 등락률 {float(daily_change):+.1f}%")
        elif abs(float(daily_change)) >= 10:
            score += 15
            reasons.append(f"일간 등락률 {float(daily_change):+.1f}%")
        if monthly_return >= 80:
            score += 25
            reasons.append(f"월간 급등 {monthly_return:+.1f}%")
        elif monthly_return >= 40:
            score += 15
            reasons.append(f"월간 상승 {monthly_return:+.1f}%")
        if volume_ratio >= 8:
            score += 25
            reasons.append(f"거래량 {volume_ratio:.1f}배")
        elif volume_ratio >= 4:
            score += 15
            reasons.append(f"거래량 {volume_ratio:.1f}배")
        if ma20_gap >= 35:
            score += 15
            reasons.append(f"MA20 이격 {ma20_gap:+.1f}%")
        elif ma20_gap >= 20:
            score += 8
            reasons.append(f"MA20 이격 {ma20_gap:+.1f}%")
        if intraday_range >= 20:
            score += 10
            reasons.append(f"장중 변동폭 {intraday_range:.1f}%")
        if float(marcap) < 150_000_000_000:
            score += 8
            reasons.append("소형 시가총액")

        if score < 25:
            return None

        return {
            "종목": f"{name} ({code})",
            "분석 링크": analysis_link(code, market),
            "시장": market,
            "현재가": float(close),
            "주의점수": round(min(score, 100), 1),
            "일간등락률": float(daily_change),
            "주간등락률": weekly_return,
            "월간등락률": monthly_return,
            "거래량배수": volume_ratio,
            "거래대금": float(amount),
            "시가총액": float(marcap),
            "주의 사유": ", ".join(reasons[:4]),
        }
    except Exception:
        return None


def _historical_case_candidate(code: str, name: str, market: str, benchmark: pd.DataFrame, start_date: str, forward_days: int) -> dict[str, Any] | None:
    try:
        stock = calculate_indicators(_clean_ohlcv(fdr.DataReader(code, start_date)))
        common_dates = stock.index.intersection(benchmark.index)
        stock = stock.loc[common_dates].copy()
        aligned_benchmark = benchmark.loc[common_dates].copy()
        if len(stock) < 280:
            return None

        best_case = None
        end_limit = len(stock) - forward_days
        start_idx = max(220, len(stock) - 360)
        for idx in range(start_idx, end_limit, 5):
            sample = stock.iloc[:idx].copy()
            sample_benchmark = aligned_benchmark.iloc[:idx].copy()
            if len(sample) < 220:
                continue

            minervini = calculate_minervini(sample)
            weinstein = calculate_weinstein(sample)
            oneil = calculate_oneil(sample)
            rs = calculate_relative_strength(sample, sample_benchmark)
            vcp = calculate_vcp(sample)
            volume = calculate_volume_analysis(sample)
            final = calculate_final_score(minervini, oneil, weinstein, rs, vcp, sample)
            if final["forced_exclusion"] or final["total_score"] < 75:
                continue

            entry_price = float(stock["Close"].iloc[idx - 1])
            future_window = stock.iloc[idx : idx + forward_days]
            if future_window.empty:
                continue
            high_after = float(future_window["High"].max())
            low_after = float(future_window["Low"].min())
            final_after = float(future_window["Close"].iloc[-1])
            max_return = (high_after / entry_price - 1) * 100
            close_return = (final_after / entry_price - 1) * 100
            drawdown = (low_after / entry_price - 1) * 100
            if max_return < 15 or close_return < 5:
                continue

            trade_plan = calculate_trade_plan(sample, oneil, final)
            score = max_return + final["total_score"] * 0.2 - abs(min(drawdown, 0)) * 0.5
            case = {
                "종목": f"{name} ({code})",
                "시장": market,
                "분석일": sample.index[-1].strftime("%Y-%m-%d"),
                "당시가격": entry_price,
                "당시점수": final["total_score"],
                "당시판정": final["rating"],
                "진입상태": final["entry_status"],
                "추천진입": trade_plan["entry_method"],
                "추천매수가": _price_range(trade_plan["buy_low"], trade_plan["buy_high"]),
                "이후수익률": round(max_return, 1),
                "종가수익률": round(close_return, 1),
                "중간하락폭": round(drawdown, 1),
                "분석 링크": analysis_link(code, market),
                "사례점수": score,
                "핵심근거": _case_reasons(minervini, oneil, rs, volume, vcp),
            }
            if best_case is None or case["사례점수"] > best_case["사례점수"]:
                best_case = case

        return best_case
    except Exception:
        return None


def _case_reasons(minervini: dict[str, Any], oneil: dict[str, Any], rs: dict[str, Any], volume: dict[str, Any], vcp: dict[str, Any]) -> str:
    reasons = []
    if minervini["score"] >= 75:
        reasons.append(f"Minervini {minervini['score']:.1f}점")
    if oneil["breakout"]:
        reasons.append("60일 고점 돌파")
    elif oneil["near_high"]:
        reasons.append("60일 고점 5% 이내")
    rs_3m = rs["values"].get("3개월")
    if rs_3m is not None and rs_3m > 0:
        reasons.append(f"3개월 RS {rs_3m:+.1f}%p")
    if volume["ratio"] >= 1.2:
        reasons.append(f"거래량 {volume['ratio']:.1f}배")
    if vcp["score"] >= 50:
        reasons.append(vcp["status"])
    return ", ".join(reasons[:5])


def _recommendation_universe(listing: pd.DataFrame, market: str, limit: int) -> pd.DataFrame:
    market_listing = listing[
        (listing["Market"] == market)
        & (listing["Marcap"] >= 100_000_000_000)
        & (listing["Amount"] >= 3_000_000_000)
    ].copy()
    return market_listing.sort_values(["Amount", "Marcap"], ascending=False).head(limit)


def _mover_universe(listing: pd.DataFrame, limit: int) -> pd.DataFrame:
    return listing[
        (listing["Marcap"] >= 30_000_000_000)
        & (listing["Amount"] >= 500_000_000)
    ].sort_values(["Amount", "Marcap"], ascending=False).head(limit)


def _unusual_universe(listing: pd.DataFrame, market: str, limit: int) -> pd.DataFrame:
    market_listing = listing[
        (listing["Market"] == market)
        & (listing["Marcap"] >= 20_000_000_000)
        & (listing["Amount"] >= 300_000_000)
    ].copy()
    market_listing["AbsChange"] = market_listing["ChagesRatio"].abs()
    return market_listing.sort_values(["AbsChange", "Amount"], ascending=False).head(limit)


def _format_recommendation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in ["현재가"]:
        result[col] = result[col].map(lambda v: f"{v:,.0f}원")
    for col in ["종합점수", "추천점수", "기간수익률", "시장대비3개월", "거래량배수"]:
        result[col] = result[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):,.1f}")
    result.insert(0, "순위", range(1, len(result) + 1))
    return result[
        [
            "순위",
            "종목",
            "분석 링크",
            "추천점수",
            "종합점수",
            "기간수익률",
            "시장대비3개월",
            "거래량배수",
            "등급",
            "진입상태",
            "추천 진입방식",
            "추천 매수가격대",
            "방어적 매도가격대",
            "분할 매도가격대",
        ]
    ]


def _format_mover_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "종목" not in result.columns and {"Name", "Code"}.issubset(result.columns):
        result["종목"] = result["Name"].astype(str) + " (" + result["Code"].astype(str).str.zfill(6) + ")"
    if "분석 링크" not in result.columns and {"Code", "Market"}.issubset(result.columns):
        result["분석 링크"] = [analysis_link(code, market) for code, market in zip(result["Code"], result["Market"])]
    if "현재가" not in result.columns and "Close" in result.columns:
        result["현재가"] = pd.to_numeric(result["Close"], errors="coerce")
    result["현재가"] = result["현재가"].map(lambda v: f"{v:,.0f}원")
    result["등락률"] = result["등락률"].map(lambda v: f"{float(v):+.1f}%")
    result.insert(0, "순위", range(1, len(result) + 1))
    return result[["순위", "종목", "분석 링크", "현재가", "등락률"]]


def _format_unusual_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["현재가"] = result["현재가"].map(lambda v: f"{v:,.0f}원")
    result["주의점수"] = result["주의점수"].map(lambda v: f"{float(v):.1f}")
    for col in ["일간등락률", "주간등락률", "월간등락률"]:
        result[col] = result[col].map(lambda v: f"{float(v):+.1f}%")
    result["거래량배수"] = result["거래량배수"].map(lambda v: f"{float(v):.1f}배")
    result["거래대금"] = result["거래대금"].map(lambda v: f"{float(v) / 100_000_000:,.0f}억")
    result["시가총액"] = result["시가총액"].map(lambda v: f"{float(v) / 100_000_000:,.0f}억")
    result.insert(0, "순위", range(1, len(result) + 1))
    return result[
        [
            "순위",
            "종목",
            "분석 링크",
            "주의점수",
            "현재가",
            "일간등락률",
            "주간등락률",
            "월간등락률",
            "거래량배수",
            "거래대금",
            "시가총액",
            "주의 사유",
        ]
    ]


def _return_over_days(df: pd.DataFrame, days: int) -> float:
    if len(df) <= days:
        return 0.0
    return (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-days - 1]) - 1) * 100


def _momentum_bonus(value: float) -> float:
    if value >= 20:
        return 10.0
    if value >= 10:
        return 7.5
    if value >= 5:
        return 5.0
    if value > 0:
        return 2.5
    return 0.0


def _price_range(low: float | None, high: float | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "제시 보류"
    return f"{low:,.0f}~{high:,.0f}원"


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    result.index = pd.to_datetime(result.index)
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.dropna(subset=["Close"]).sort_index()
