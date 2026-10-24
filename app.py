from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots

from analyzer import analyze_stock, load_stock_listing, normalize_code, resolve_stock_query
from market_scanner import historical_success_cases, movers_top10_bundle_by_market, recommendation_top10_by_market, unusual_risk_top10_by_market


st.set_page_config(page_title="추세·돌파형 한국주식 분석기", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze_stock(code: str, market: str, period: str) -> dict:
    return analyze_stock(code, market, period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_listing(market: str) -> pd.DataFrame:
    return load_stock_listing(market)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze_resolved(code: str, name: str, market: str, period: str) -> dict:
    return analyze_stock(code, market, period, name)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_company_overview(code: str) -> dict[str, str]:
    url = f"https://finance.naver.com/item/main.naver?code={normalize_code(code)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    summary = soup.select_one("#summary_info")
    if not summary:
        return {"text": "회사정보를 불러오지 못했습니다.", "source": url}

    lines = [line.strip() for line in summary.get_text("\n", strip=True).splitlines()]
    lines = [line for line in lines if line and line != "기업개요"]
    text = "\n".join(lines[:5]) if lines else "회사정보를 불러오지 못했습니다."
    return {"text": text, "source": url}


@st.cache_data(ttl=86400, show_spinner=False)
def cached_recommendations(mode: str, universe_size: int, cache_key: str) -> dict[str, pd.DataFrame]:
    return recommendation_top10_by_market(mode, per_market_universe=universe_size)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_movers(period: str, universe_size: int, cache_key: str) -> dict[str, dict[str, pd.DataFrame]]:
    return movers_top10_bundle_by_market(period, per_market_universe=universe_size)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_unusual_risk(universe_size: int, cache_key: str) -> dict[str, pd.DataFrame]:
    return unusual_risk_top10_by_market(per_market_universe=universe_size)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_success_cases(universe_size: int, forward_days: int, cache_key: str) -> list[dict]:
    return historical_success_cases(per_market_universe=universe_size, forward_days=forward_days, limit=10)


def main() -> None:
    init_session()
    render_style()

    render_sidebar_membership()
    render_product_header()

    linked = render_query_analysis()
    if linked:
        st.divider()

    tab_single, tab_compare, tab_recommend, tab_movers, tab_unusual, tab_simulator, tab_cases, tab_board = st.tabs(
        ["개별 종목 분석", "다중 종목 비교", "추천 TOP10", "상승/하락 TOP10", "이상 급등/과열 주의", "투자 시뮬레이션", "과거 적중 사례", "투자자 게시판"]
    )

    with tab_single:
        render_single_analysis()

    with tab_compare:
        render_multi_compare()

    with tab_recommend:
        render_market_recommendations()

    with tab_movers:
        render_market_movers()

    with tab_unusual:
        render_unusual_risk()

    with tab_simulator:
        render_investment_simulator()

    with tab_cases:
        render_success_cases()

    with tab_board:
        render_investor_board()


def render_product_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">Korea Breakout Intelligence</div>
                <h1>추세·돌파형 한국주식 분석기</h1>
                <p>Minervini + O'Neil + Weinstein 원칙으로 좋은 종목과 좋은 가격을 분리해 보는 투자 분석 대시보드</p>
            </div>
            <div class="hero-panel">
                <div class="panel-label">오늘의 분석 흐름</div>
                <div class="panel-main">종목 검색 → 추세 점검 → 진입 가격대 → 시장 TOP10</div>
                <div class="panel-sub">자동매매 없이, 판단에 필요한 숫자와 이유만 선명하게 정리합니다.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_membership() -> None:
    with st.sidebar:
        st.markdown("### Investor Club")
        st.caption("관심 등록을 남기면 현재 세션에서만 저장됩니다.")
        with st.expander("회원가입 관심 등록", expanded=False):
            email = st.text_input("이메일", placeholder="you@example.com", key="signup_email")
            style = st.selectbox("관심 스타일", ["추세 돌파", "눌림목", "저변동성", "고위험 단타 제외"], key="signup_style")
            if st.button("관심 등록", use_container_width=True):
                if email and "@" in email:
                    st.session_state["members"].insert(0, {"이메일": email, "관심 스타일": style, "등록시각": datetime.now().strftime("%Y-%m-%d %H:%M")})
                    st.success("등록되었습니다.")
                else:
                    st.warning("이메일 형식을 확인하십시오.")


def render_query_analysis() -> bool:
    params = st.query_params
    code = params.get("analyze_code")
    market = params.get("market", "KOSPI")
    if not code:
        return False
    code = normalize_code(code)
    market = market if market in {"KOSPI", "KOSDAQ"} else "KOSPI"
    try:
        listing = cached_stock_listing(market)
        resolved = resolve_stock_query(code, market, listing)
        with st.spinner(f"{resolved['name']} ({code}) 개별 분석을 불러오는 중입니다."):
            analysis = cached_analyze_resolved(resolved["code"], resolved["name"], market, "3년")
        st.subheader("링크로 연 개별 종목 분석")
        render_analysis(analysis)
        if st.button("링크 분석 닫기"):
            st.query_params.clear()
        return True
    except Exception as exc:
        st.error(str(exc) if str(exc) else "링크 종목을 분석하지 못했습니다.")
        return True


def render_single_analysis() -> None:
    with st.sidebar:
        st.header("분석 조건")
        market = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"], index=0)
        query = st.text_input("종목코드 또는 종목명", value="005930", placeholder="005930 또는 삼성전자", key="single_query")
        render_search_suggestions("single_query", market)
        period = st.selectbox("분석기간", ["1년", "2년", "3년", "5년"], index=2)
        run = st.button("분석하기", type="primary", use_container_width=True)

    if not run and not st.session_state.get("last_analysis"):
        st.info("종목코드 6자리 또는 종목명을 입력하고 분석하기를 누르십시오.")
        render_history()
        return

    if run:
        try:
            with st.spinner("데이터를 불러오고 추세·돌파 조건을 계산하는 중입니다."):
                listing = cached_stock_listing(market)
                resolved = resolve_stock_query(query, market, listing)
                analysis = cached_analyze_resolved(resolved["code"], resolved["name"], market, period)
            st.session_state["last_analysis"] = analysis
            add_history(analysis)
        except Exception as exc:
            st.error(str(exc) if str(exc) else "종목코드 또는 데이터 제공 상태를 확인하십시오.")
            return

    analysis = st.session_state["last_analysis"]
    render_analysis(analysis)
    render_history()


def render_multi_compare() -> None:
    st.subheader("다중 종목 비교")
    market = st.selectbox("비교 시장 선택", ["KOSPI", "KOSDAQ"], index=0, key="compare_market")
    codes_text = st.text_input("비교 종목", value="005930,000660,NAVER", help="종목코드 또는 종목명을 쉼표로 구분하여 최대 10개까지 입력합니다.", key="compare_query")
    render_search_suggestions("compare_query", market, multi=True)
    period = st.selectbox("비교 분석기간", ["1년", "2년", "3년", "5년"], index=2, key="compare_period")

    if st.button("비교 분석하기", type="primary"):
        queries = [item.strip() for item in codes_text.split(",") if item.strip()]
        queries = list(dict.fromkeys(queries))[:10]
        if not queries:
            st.error("비교할 종목코드 또는 종목명을 입력하십시오.")
            return

        rows = []
        errors = []
        progress = st.progress(0)
        listing = cached_stock_listing(market)
        for idx, query in enumerate(queries, start=1):
            try:
                resolved = resolve_stock_query(query, market, listing)
                analysis = cached_analyze_resolved(resolved["code"], resolved["name"], market, period)
                rows.append(comparison_row(analysis))
            except Exception as exc:
                errors.append(f"{query}: {exc}")
            progress.progress(idx / len(queries))

        if rows:
            result = pd.DataFrame(rows).sort_values("종합점수", ascending=False)
            st.dataframe(result, use_container_width=True, hide_index=True)
        if errors:
            st.warning("일부 종목은 분석하지 못했습니다.\n\n" + "\n".join(errors))


def render_market_recommendations() -> None:
    st.subheader("현시점 기준 진입 추천 TOP10")
    st.caption("기존 분석 로직으로 하락추세 강제 제외 조건을 적용한 뒤, KOSPI와 KOSDAQ을 각각 스캔합니다.")
    mode = st.radio("추천 기준", ["일간", "주간", "월간"], horizontal=True)
    scan_mode = st.segmented_control("스캔 범위", ["빠름", "정밀"], default="빠름", key="recommend_scan_mode")
    universe_size = 30 if scan_mode == "빠름" else 80
    cache_key = cache_bucket(mode)

    run = st.button("추천 TOP10 계산", type="primary", key="refresh_recommendations")
    if run:
        cached_recommendations.clear()
        try:
            with st.spinner(f"{mode} 추천 TOP10을 계산하는 중입니다. 빠름은 시장별 {universe_size}개 후보를 스캔합니다."):
                st.session_state["recommendation_result"] = {
                    "mode": mode,
                    "scan_mode": scan_mode,
                    "data": cached_recommendations(mode, universe_size, cache_key),
                }
        except Exception as exc:
            st.error(f"추천 TOP10 데이터를 불러오지 못했습니다: {exc}")
            return

    saved = st.session_state.get("recommendation_result")
    if not saved:
        st.info("계산 버튼을 누르면 추천 TOP10을 조회합니다. 시간이 오래 걸리면 '빠름' 범위를 사용하십시오.")
        return

    data = saved["data"]
    st.caption(f"표시 중인 결과: {saved['mode']} · {saved['scan_mode']}")

    kospi, kosdaq = st.columns(2)
    with kospi:
        st.markdown("**KOSPI 추천 TOP10**")
        render_rank_table(data.get("KOSPI"))
    with kosdaq:
        st.markdown("**KOSDAQ 추천 TOP10**")
        render_rank_table(data.get("KOSDAQ"))

    st.info("추천 가격대는 자동 주문이나 확정 수익 신호가 아니라 조건식 기반 참고 범위입니다.")


def render_market_movers() -> None:
    st.subheader("상승률 / 하락률 TOP10")
    st.caption("일간은 KRX 현재 등락률, 월간·연간은 최근 종가와 과거 종가를 비교해 계산합니다.")
    period = st.radio("등락률 기간", ["일간", "월간", "연간"], horizontal=True)
    scan_mode = st.segmented_control("월간·연간 스캔 범위", ["빠름", "정밀"], default="빠름", key="mover_scan_mode")
    universe_size = 60 if scan_mode == "빠름" else 160
    cache_key = cache_bucket("일간")

    run = st.button("상승/하락 TOP10 계산", type="primary", key="refresh_movers")
    if run:
        cached_movers.clear()
        try:
            with st.spinner(f"{period} 상승률·하락률 TOP10을 계산하는 중입니다."):
                st.session_state["mover_result"] = {
                    "period": period,
                    "scan_mode": scan_mode,
                    "data": cached_movers(period, universe_size, cache_key),
                }
        except Exception as exc:
            st.error(f"상승률·하락률 TOP10 데이터를 불러오지 못했습니다: {exc}")
            return

    saved = st.session_state.get("mover_result")
    if not saved:
        st.info("계산 버튼을 누르면 상승률/하락률 TOP10을 조회합니다. 일간은 빠르고, 월간·연간은 후보군 데이터 조회 시간이 걸립니다.")
        return

    movers = saved["data"]
    st.caption(f"표시 중인 결과: {saved['period']} · {saved['scan_mode']}")

    for market in ["KOSPI", "KOSDAQ"]:
        st.markdown(f"### {market}")
        up_col, down_col = st.columns(2)
        with up_col:
            st.markdown("**가장 상승률이 높은 주식 10개**")
            render_rank_table(movers.get(market, {}).get("상승"))
        with down_col:
            st.markdown("**가장 하락폭이 큰 주식 10개**")
            render_rank_table(movers.get(market, {}).get("하락"))


def render_unusual_risk() -> None:
    st.subheader("이상 급등/과열 주의 후보")
    st.caption("거래량 급증, 단기 급등락, 이동평균 이격, 장중 변동성, 소형 시가총액을 조합한 기술적 주의 목록입니다.")
    scan_mode = st.segmented_control("스캔 범위", ["빠름", "정밀"], default="빠름", key="unusual_scan_mode")
    universe_size = 60 if scan_mode == "빠름" else 140
    cache_key = cache_bucket("일간")

    run = st.button("주의 후보 TOP10 계산", type="primary", key="refresh_unusual")
    if run:
        cached_unusual_risk.clear()
        try:
            with st.spinner(f"이상 급등/과열 주의 후보를 계산하는 중입니다. 시장별 {universe_size}개 후보를 확인합니다."):
                st.session_state["unusual_result"] = {
                    "scan_mode": scan_mode,
                    "data": cached_unusual_risk(universe_size, cache_key),
                }
        except Exception as exc:
            st.error(f"이상 급등/과열 주의 후보를 불러오지 못했습니다: {exc}")
            return

    saved = st.session_state.get("unusual_result")
    if not saved:
        st.info("계산 버튼을 누르면 KOSPI/KOSDAQ 주의 후보를 각각 10개씩 조회합니다.")
        return

    st.caption(f"표시 중인 결과: {saved['scan_mode']} · 매일 갱신")
    data = saved["data"]
    kospi, kosdaq = st.columns(2)
    with kospi:
        st.markdown("**KOSPI 주의 후보 TOP10**")
        render_rank_table(data.get("KOSPI"))
    with kosdaq:
        st.markdown("**KOSDAQ 주의 후보 TOP10**")
        render_rank_table(data.get("KOSDAQ"))

    st.warning("이 목록은 불공정거래나 작전주를 단정하지 않습니다. 기술적 이상 징후가 큰 종목을 주의 후보로 분류한 참고 자료입니다.")


def render_investment_simulator() -> None:
    st.subheader("참여형 투자 시뮬레이션")
    st.caption("기간, 성향, 자금, 리스크 허용폭을 넣으면 기존 분석 로직으로 진입·익절·방어 시나리오를 구성합니다.")

    source = st.radio("시뮬레이션 대상", ["원하는 종목 직접 입력", "추천 TOP10 중 선택"], horizontal=True)
    selected_query = "005930"
    selected_market = "KOSPI"

    if source == "추천 TOP10 중 선택":
        sim_mode = st.selectbox("추천 기준", ["일간", "주간", "월간"], index=1, key="sim_reco_mode")
        if st.button("추천 후보 불러오기", key="sim_load_recommendations"):
            with st.spinner("추천 후보를 불러오는 중입니다."):
                data = cached_recommendations(sim_mode, 30, cache_bucket(sim_mode))
                choices = []
                for market, frame in data.items():
                    if frame is not None and not frame.empty:
                        for row in frame.itertuples(index=False):
                            choices.append({"label": f"{market} · {row.종목}", "market": market, "query": row.종목.split("(")[-1].replace(")", "")})
                st.session_state["simulation_choices"] = choices
        choices = st.session_state.get("simulation_choices", [])
        if choices:
            choice_label = st.selectbox("종목 선택", [item["label"] for item in choices])
            selected = next(item for item in choices if item["label"] == choice_label)
            selected_query = selected["query"]
            selected_market = selected["market"]
        else:
            st.info("추천 후보를 먼저 불러오거나 직접 입력 방식을 사용하십시오.")
    else:
        col_a, col_b = st.columns([0.7, 0.3])
        with col_b:
            selected_market = st.selectbox("시장", ["KOSPI", "KOSDAQ"], index=0, key="sim_market")
        with col_a:
            selected_query = st.text_input("종목코드 또는 종목명", value="005930", key="sim_query")
            render_search_suggestions("sim_query", selected_market)

    with st.form("simulation_form"):
        c1, c2, c3, c4 = st.columns(4)
        horizon = c1.selectbox("투자 기간", ["1개월", "3개월", "6개월", "1년"], index=1)
        temperament = c2.selectbox("투자 성향", ["보수형", "균형형", "공격형"], index=1)
        capital = c3.number_input("투자 예정 금액", min_value=100000, max_value=1000000000, value=5000000, step=100000)
        max_risk = c4.slider("허용 손실폭", min_value=2.0, max_value=20.0, value=7.0, step=0.5)
        submitted = st.form_submit_button("시뮬레이션 실행", type="primary")

    if not submitted:
        render_simulator_guide()
        return

    try:
        listing = cached_stock_listing(selected_market)
        resolved = resolve_stock_query(selected_query, selected_market, listing)
        with st.spinner(f"{resolved['name']} 시뮬레이션을 계산하는 중입니다."):
            analysis = cached_analyze_resolved(resolved["code"], resolved["name"], selected_market, "3년")
        plan = build_simulation_plan(analysis, horizon, temperament, capital, max_risk)
        render_simulation_result(analysis, plan)
    except Exception as exc:
        st.error(str(exc) if str(exc) else "시뮬레이션을 실행하지 못했습니다.")


def render_success_cases() -> None:
    st.subheader("과거 적중 사례")
    st.caption("과거 특정 시점에 이 분석기의 조건을 적용했을 때 좋은 점수가 나왔고, 이후 상승이 확인된 사례를 찾습니다.")
    horizon = st.radio("성과 확인 기간", ["1개월", "3개월", "6개월"], index=1, horizontal=True)
    scan_mode = st.segmented_control("스캔 범위", ["빠름", "정밀"], default="빠름", key="case_scan_mode")
    forward_days = {"1개월": 20, "3개월": 60, "6개월": 120}[horizon]
    universe_size = 25 if scan_mode == "빠름" else 60
    cache_key = cache_bucket("일간")

    run = st.button("과거 사례 찾기", type="primary", key="find_success_cases")
    if run:
        cached_success_cases.clear()
        try:
            with st.spinner(f"과거 {horizon} 성과 사례를 찾는 중입니다. 최초 실행은 시간이 걸릴 수 있습니다."):
                st.session_state["success_cases"] = {
                    "horizon": horizon,
                    "scan_mode": scan_mode,
                    "cases": cached_success_cases(universe_size, forward_days, cache_key),
                }
        except Exception as exc:
            st.error(f"과거 적중 사례를 불러오지 못했습니다: {exc}")
            return

    saved = st.session_state.get("success_cases")
    if not saved:
        st.info("버튼을 누르면 과거 데이터에서 조건 부합 후 상승한 사례 10개를 블로그 글 형태로 보여줍니다.")
        return

    st.caption(f"표시 중인 결과: {saved['horizon']} · {saved['scan_mode']} · 매일 갱신")
    cases = saved["cases"]
    if not cases:
        st.warning("조건에 맞는 과거 사례를 찾지 못했습니다. 정밀 스캔 또는 다른 기간을 선택해보십시오.")
        return

    for idx, case in enumerate(cases, start=1):
        render_case_post(idx, case, saved["horizon"])

    st.info("과거에 잘 맞았던 사례가 미래 성과를 보장하지 않습니다. 시장 환경과 종목별 뉴스, 재무 이벤트를 함께 확인해야 합니다.")


def render_case_post(rank: int, case: dict, horizon: str) -> None:
    st.markdown(
        f"""
        <div class="case-card">
            <div class="case-kicker">CASE #{rank} · {case['시장']} · 분석일 {case['분석일']}</div>
            <div class="case-title">{case['종목']} - 당시 분석기가 잡아낸 추세 전환 후보</div>
            <div class="case-lead">
                당시 종합점수는 <b>{case['당시점수']:.1f}점</b>, 판정은 <b>{case['당시판정']}</b>이었고
                {horizon} 안에 고점 기준 <b>{case['이후수익률']:+.1f}%</b>까지 상승했습니다.
            </div>
            <div class="case-grid">
                <div><span>당시 가격</span><b>{case['당시가격']:,.0f}원</b></div>
                <div><span>진입 상태</span><b>{case['진입상태']}</b></div>
                <div><span>추천 진입</span><b>{case['추천진입']}</b></div>
                <div><span>추천 매수가</span><b>{case['추천매수가']}</b></div>
                <div><span>종가 기준 성과</span><b>{case['종가수익률']:+.1f}%</b></div>
                <div><span>중간 하락폭</span><b>{case['중간하락폭']:+.1f}%</b></div>
            </div>
            <div class="case-body">
                <b>왜 의미 있었나:</b> {case['핵심근거'] or '주요 추세 조건과 상대강도 조건이 동시에 개선되었습니다.'}
            </div>
            <a class="case-link" href="{case['분석 링크']}">현재 기준 개별 분석 열기</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simulator_guide() -> None:
    a, b, c = st.columns(3)
    a.markdown('<div class="sim-card"><div class="sim-title">1. 종목 선택</div><div class="sim-body">직접 입력하거나 추천 TOP10에서 후보를 고릅니다.</div></div>', unsafe_allow_html=True)
    b.markdown('<div class="sim-card"><div class="sim-title">2. 성향 설정</div><div class="sim-body">보수형·균형형·공격형에 따라 분할 비중과 익절 폭이 달라집니다.</div></div>', unsafe_allow_html=True)
    c.markdown('<div class="sim-card"><div class="sim-title">3. 계획 확인</div><div class="sim-body">진입, 방어, 익절 시나리오와 예상 주식 수를 확인합니다.</div></div>', unsafe_allow_html=True)


def build_simulation_plan(analysis: dict, horizon: str, temperament: str, capital: int, max_risk: float) -> dict:
    latest = analysis["df"].iloc[-1]
    current = float(latest["Close"])
    trade = analysis["trade_plan"]
    final = analysis["final"]
    oneil = analysis["oneil"]

    months = {"1개월": 1, "3개월": 3, "6개월": 6, "1년": 12}[horizon]
    settings = {
        "보수형": {"risk": 0.6, "first": 0.35, "second": 0.35, "third": 0.30, "target": (0.08, 0.16), "trail": 0.07},
        "균형형": {"risk": 1.0, "first": 0.45, "second": 0.35, "third": 0.20, "target": (0.12, 0.24), "trail": 0.10},
        "공격형": {"risk": 1.35, "first": 0.60, "second": 0.25, "third": 0.15, "target": (0.18, 0.35), "trail": 0.14},
    }[temperament]

    buy_low = trade["buy_low"] or current * 0.98
    buy_high = trade["buy_high"] or current * 1.02
    entry_mid = (buy_low + buy_high) / 2
    stop_reference = trade["risk_sell_high"] or current * (1 - max_risk / 100)
    stop_price = max(min(stop_reference, entry_mid * (1 - max_risk / 100)), entry_mid * 0.75)
    per_share_risk = max(entry_mid - stop_price, entry_mid * 0.01)
    risk_budget = capital * (max_risk / 100) * settings["risk"]
    max_position_value = min(capital, risk_budget / per_share_risk * entry_mid)
    shares = int(max_position_value // entry_mid)
    used_cash = shares * entry_mid

    target_low = max(trade["profit_sell_low"] or entry_mid * 1.12, entry_mid * (1 + settings["target"][0]))
    target_high = max(trade["profit_sell_high"] or entry_mid * 1.22, entry_mid * (1 + settings["target"][1]))
    time_stop_days = {1: 10, 3: 20, 6: 35, 12: 55}[months]

    if final["forced_exclusion"]:
        scenario = "진입 보류 시나리오"
        summary = "핵심 추세 조건이 약해 신규 진입보다 관찰 목록 유지가 우선입니다."
    elif final["entry_status"] == "돌파 매수구간":
        scenario = "돌파 추종 시나리오"
        summary = "돌파와 거래량이 확인된 상태이므로 피벗 근처 분할 진입 후 빠른 방어선 관리가 핵심입니다."
    elif final["entry_status"] == "돌파 대기":
        scenario = "피벗 돌파 대기 시나리오"
        summary = "아직 완전 돌파 전이므로 가격보다 확인 신호를 먼저 기다리는 구성이 적합합니다."
    elif final["entry_status"] == "과열 / 추격매수 주의":
        scenario = "눌림목 대기 시나리오"
        summary = "이격도가 커서 바로 추격하기보다 MA20 또는 이전 돌파 가격 부근에서 반응을 확인합니다."
    else:
        scenario = "관망 후 확인 시나리오"
        summary = "추세가 완전히 선명하지 않아 소액 분할 또는 조건 충족 후 진입이 더 적합합니다."

    entry_steps = [
        {"단계": "1차", "조건": trade["entry_method"], "비중": f"{settings['first'] * 100:.0f}%", "가격대": price_range(buy_low, buy_high)},
        {"단계": "2차", "조건": "진입 후 피벗/MA20 위 지지 확인", "비중": f"{settings['second'] * 100:.0f}%", "가격대": price_range(entry_mid * 0.98, entry_mid * 1.04)},
        {"단계": "3차", "조건": "거래량 유지 또는 신고가 재돌파 확인", "비중": f"{settings['third'] * 100:.0f}%", "가격대": price_range(entry_mid * 1.02, entry_mid * 1.08)},
    ]
    exit_steps = [
        {"구분": "방어", "조건": "진입 근거 훼손 또는 기준선 이탈", "가격대": price_range(stop_price * 0.99, stop_price), "행동": "비중 축소 또는 전량 방어 검토"},
        {"구분": "1차 익절", "조건": "목표 구간 하단 도달", "가격대": won(target_low), "행동": "30~50% 분할 매도 검토"},
        {"구분": "2차 익절", "조건": "목표 구간 상단 또는 과열 재발", "가격대": won(target_high), "행동": "잔여 물량 분할 매도 검토"},
        {"구분": "추세 추적", "조건": f"고점 대비 {settings['trail'] * 100:.0f}% 이상 되돌림", "가격대": "시장가 기준 추적", "행동": "잔여 물량 방어 검토"},
    ]

    return {
        "scenario": scenario,
        "summary": summary,
        "horizon": horizon,
        "temperament": temperament,
        "capital": capital,
        "shares": shares,
        "used_cash": used_cash,
        "entry_mid": entry_mid,
        "stop_price": stop_price,
        "risk_budget": risk_budget,
        "target_low": target_low,
        "target_high": target_high,
        "time_stop_days": time_stop_days,
        "entry_steps": entry_steps,
        "exit_steps": exit_steps,
        "signals": [
            f"종합점수 {final['total_score']:.1f}점 / {final['rating']}",
            f"현재 진입상태: {final['entry_status']}",
            f"60일 돌파 여부: {'돌파' if oneil['breakout'] else '대기'}",
            f"거래량: 20일 평균 대비 {analysis['volume']['ratio']:.2f}배",
        ],
    }


def render_simulation_result(analysis: dict, plan: dict) -> None:
    st.markdown(
        f"""
        <div class="stock-header">
            <div>
                <div class="stock-meta">{analysis['market']} · {analysis['code']} · {plan['horizon']} · {plan['temperament']}</div>
                <div class="stock-name">{analysis['name']} 시뮬레이션</div>
                <div class="stock-sub">{plan['summary']}</div>
            </div>
            <div class="badge-watch">{plan['scenario']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("예정 투자금", won(plan["capital"]))
    c2.metric("시뮬레이션 수량", f"{plan['shares']:,}주")
    c3.metric("평균 진입 기준", won(plan["entry_mid"]))
    c4.metric("방어 기준", won(plan["stop_price"]))

    c5, c6, c7 = st.columns(3)
    c5.metric("투입 예상금액", won(plan["used_cash"]))
    c6.metric("리스크 예산", won(plan["risk_budget"]))
    c7.metric("목표 익절 구간", price_range(plan["target_low"], plan["target_high"]))

    st.markdown("**진입 계획**")
    st.dataframe(pd.DataFrame(plan["entry_steps"]), use_container_width=True, hide_index=True)
    st.markdown("**익절 / 방어 계획**")
    st.dataframe(pd.DataFrame(plan["exit_steps"]), use_container_width=True, hide_index=True)
    st.markdown("**핵심 신호**")
    for signal in plan["signals"]:
        st.markdown(f"- {signal}")
    st.info(f"{plan['time_stop_days']}거래일 안에 진입 근거가 강화되지 않으면 기간 리스크를 줄이는 관점에서 재검토합니다. 이 시뮬레이션은 투자 판단 보조용이며 수익을 보장하지 않습니다.")


def render_rank_table(frame: pd.DataFrame | None) -> None:
    if frame is None or frame.empty:
        st.warning("표시할 종목이 없습니다.")
        return
    config = {}
    if "분석 링크" in frame.columns:
        config["분석 링크"] = st.column_config.LinkColumn("개별 분석", display_text="열기")
    st.dataframe(frame, use_container_width=True, hide_index=True, column_config=config)


def render_analysis(analysis: dict) -> None:
    df = analysis["df"]
    latest = df.iloc[-1]
    final = analysis["final"]
    rs_3m = analysis["rs"]["values"].get("3개월")

    render_stock_header(analysis, latest, final)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(f"""
        <div class="question-card">
            <div class="question-title">좋은 종목인가?</div>
            <div class="question-answer">{final["good_stock"]} - {final["total_score"]:.1f}점</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown(f"""
        <div class="question-card">
            <div class="question-title">지금 좋은 가격인가?</div>
            <div class="question-answer">{final["good_price"]} - {final["entry_status"]}</div>
        </div>
        """, unsafe_allow_html=True)

    cols = st.columns(5)
    cols[0].metric("현재가", f"{latest['Close']:,.0f}원")
    cols[1].metric("종합점수", f"{final['total_score']:.1f} / 100")
    cols[2].metric("최종등급", final["stars"], final["rating"])
    cols[3].metric("현재 진입상태", final["entry_status"])
    cols[4].metric("시장대비 3개월 RS", format_rs(rs_3m))

    st.divider()
    left, right = st.columns([1.1, 0.9])
    with left:
        render_scoreboard(analysis)
        render_diagnosis(analysis)
    with right:
        render_positive_risk(analysis)

    render_trade_plan(analysis)

    render_company_overview(analysis)
    render_chart(analysis)

    c1, c2 = st.columns(2)
    with c1:
        render_minervini_table(analysis)
        render_price_table(analysis)
    with c2:
        render_volume_vcp(analysis)
        render_support_levels(analysis)

    render_text_analysis(analysis)


def render_stock_header(analysis: dict, latest: pd.Series, final: dict) -> None:
    badge_class = "badge-good" if final["good_stock"] == "YES" else "badge-watch" if final["good_stock"] == "WATCH" else "badge-risk"
    st.markdown(
        f"""
        <div class="stock-header">
            <div>
                <div class="stock-meta">{analysis['market']} · {analysis['code']}</div>
                <div class="stock-name">{analysis['name']}</div>
                <div class="stock-sub">현재가 {float(latest['Close']):,.0f}원 · 종합점수 {final['total_score']:.1f} · {final['stars']} {final['rating']}</div>
            </div>
            <div class="{badge_class}">{final['good_stock']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scoreboard(analysis: dict) -> None:
    st.subheader("세부 점수판")
    rows = [
        {"항목": "Minervini Trend", "점수": analysis["minervini"]["score"]},
        {"항목": "O'Neil Breakout", "점수": analysis["oneil"]["score"]},
        {"항목": "Weinstein Stage", "점수": analysis["weinstein"]["score"]},
        {"항목": "Relative Strength", "점수": analysis["rs"]["score"]},
        {"항목": "VCP", "점수": analysis["vcp"]["score"]},
        {"항목": "종합", "점수": analysis["final"]["total_score"]},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_diagnosis(analysis: dict) -> None:
    st.subheader("핵심 진단")
    diagnosis = analysis["text"]["diagnosis"]
    st.write(f"**추세:** {diagnosis['추세']}")
    st.write(f"**종목의 강도:** {diagnosis['종목의 강도']}")
    st.write(f"**현재 매수 위치:** {diagnosis['현재 매수 위치']}")


def render_positive_risk(analysis: dict) -> None:
    st.subheader("이유 표시")
    pos_col, risk_col = st.columns(2)
    with pos_col:
        st.markdown("**긍정 요소**")
        for item in analysis["text"]["positive_factors"] or ["뚜렷한 긍정 요소가 제한적입니다."]:
            st.markdown(f"- {item}")
    with risk_col:
        st.markdown("**위험 요소**")
        for item in analysis["text"]["risk_factors"] or ["뚜렷한 위험 요소가 제한적입니다."]:
            st.markdown(f"- {item}")


def render_trade_plan(analysis: dict) -> None:
    st.subheader("추천 진입방식 / 가격대")
    plan = analysis["trade_plan"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("추천 진입방식", plan["entry_method"])
    c2.metric("추천 매수가격대", price_range(plan["buy_low"], plan["buy_high"]))
    c3.metric("방어적 매도가격대", price_range(plan["risk_sell_low"], plan["risk_sell_high"]))
    c4.metric("분할 매도가격대", price_range(plan["profit_sell_low"], plan["profit_sell_high"]))
    st.caption(plan["note"])
    st.caption("가격대는 분석 참고값이며 자동 주문, 확정 수익, 무조건 매수·매도 기준이 아닙니다.")


def render_company_overview(analysis: dict) -> None:
    st.subheader("회사정보")
    try:
        overview = cached_company_overview(analysis["code"])
        st.markdown(overview["text"].replace("\n", "\n\n"))
        st.caption(f"출처: 네이버 금융 / 에프앤가이드 · {overview['source']}")
    except Exception:
        st.caption("회사정보를 불러오지 못했습니다.")


def render_chart(analysis: dict) -> None:
    st.subheader("차트")
    chart_period = st.radio("차트 기간", ["3개월", "6개월", "1년", "2년"], index=2, horizontal=True)
    days = {"3개월": 63, "6개월": 126, "1년": 252, "2년": 504}[chart_period]
    df = analysis["df"].tail(min(days, len(analysis["df"])))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="가격"), row=1, col=1)
    for ma, color in [("MA20", "#2563eb"), ("MA50", "#16a34a"), ("MA150", "#f59e0b"), ("MA200", "#dc2626")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1.6, color=color)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color="#64748b"), row=2, col=1)
    fig.update_layout(
        title=f"{analysis['name']} ({analysis['code']}) 가격 / 이동평균 / 거래량",
        height=660,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=30),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def render_minervini_table(analysis: dict) -> None:
    st.subheader("Minervini 체크리스트")
    rows = [
        {"조건": item["condition"], "현재값": item["current"], "기준": item["criteria"], "결과": item["result"]}
        for item in analysis["minervini"]["checks"]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_price_table(analysis: dict) -> None:
    st.subheader("주요 가격 위치")
    df = analysis["df"]
    latest = df.iloc[-1]
    minervini = analysis["minervini"]
    oneil = analysis["oneil"]
    final = analysis["final"]
    low_20d = float(df["Low"].tail(min(20, len(df))).min())
    high_gap = (minervini["high_52w"] - latest["Close"]) / minervini["high_52w"] * 100 if minervini["high_52w"] else 0
    rows = [
        ("현재가", won(latest["Close"])),
        ("MA20", won(latest.get("MA20"))),
        ("MA50", won(latest.get("MA50"))),
        ("MA150", won(latest.get("MA150"))),
        ("MA200", won(latest.get("MA200"))),
        ("52주 최고가", won(minervini["high_52w"])),
        ("52주 최저가", won(minervini["low_52w"])),
        ("60일 최고가", won(oneil["high_60d"])),
        ("20일 최저가", won(low_20d)),
        ("현재가와 52주 최고가 거리", f"{high_gap:.1f}%"),
        ("현재가와 MA20 이격도", pct(final["ma20_gap"])),
        ("현재가와 MA50 이격도", pct(final["ma50_gap"])),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["항목", "값"]), use_container_width=True, hide_index=True)


def render_volume_vcp(analysis: dict) -> None:
    st.subheader("거래량 / VCP")
    volume = analysis["volume"]
    vcp = analysis["vcp"]
    st.write(f"**오늘 거래량:** {volume['today']:,.0f}주")
    st.write(f"**20일 평균:** {num(volume['vol20'])}주")
    st.write(f"**50일 평균:** {num(volume['vol50'])}주")
    st.write(f"**거래량 배수:** {volume['ratio']:.2f}배 - {volume['status']}")
    st.write(f"**VCP:** {vcp['status']}")
    if vcp["volatilities"]:
        st.dataframe(pd.DataFrame([
            {"구간": k, "변동성": f"{vcp['volatilities'][k]:.1f}%", "평균 거래량": f"{vcp['volumes'][k]:,.0f}주"}
            for k in ["A", "B", "C"]
        ]), use_container_width=True, hide_index=True)


def render_support_levels(analysis: dict) -> None:
    st.subheader("기술적 방어선 참고")
    df = analysis["df"]
    latest = df.iloc[-1]
    low_20d = float(df["Low"].tail(min(20, len(df))).min())
    st.caption("자동 매매나 확정 손절가가 아닌 분석용 참고값입니다.")
    rows = [
        {"항목": "MA20", "가격": won(latest.get("MA20"))},
        {"항목": "MA50", "가격": won(latest.get("MA50"))},
        {"항목": "최근 20거래일 최저가", "가격": won(low_20d)},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_text_analysis(analysis: dict) -> None:
    st.subheader("분석 설명")
    for paragraph in analysis["text"]["paragraphs"]:
        st.write(paragraph)
    st.info("이 프로그램은 자동 매매 프로그램이 아니며, 주문·증권사 로그인·매수/매도 API 기능을 제공하지 않습니다.")


def render_investor_board() -> None:
    st.subheader("투자자 게시판")
    st.caption("현재 세션에서만 저장되는 커뮤니티 데모입니다. 투자 권유, 허위 정보, 확정 수익 표현은 피하십시오.")

    left, right = st.columns([0.72, 0.28])
    with right:
        st.markdown("**새 글 작성**")
        nickname = st.text_input("닉네임", value="익명 투자자", key="board_nickname")
        category = st.selectbox("분류", ["관심종목", "돌파 관찰", "리스크 공유", "질문"], key="board_category")
        title = st.text_input("제목", placeholder="예: 005930 거래량 어떻게 보시나요?", key="board_title")
        body = st.text_area("내용", height=140, placeholder="근거와 관찰 포인트를 함께 적어주세요.", key="board_body")
        if st.button("게시하기", type="primary", use_container_width=True):
            if title.strip() and body.strip():
                st.session_state["board_posts"].insert(
                    0,
                    {
                        "작성시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "닉네임": nickname.strip() or "익명 투자자",
                        "분류": category,
                        "제목": title.strip(),
                        "내용": body.strip(),
                    },
                )
                st.success("게시되었습니다.")
            else:
                st.warning("제목과 내용을 입력하십시오.")

    with left:
        st.markdown("**실시간 토론**")
        if not st.session_state["board_posts"]:
            st.info("아직 게시글이 없습니다. 첫 관찰 메모를 남겨보십시오.")
            return
        for post in st.session_state["board_posts"][:20]:
            st.markdown(
                f"""
                <div class="post-card">
                    <div class="post-top"><span>{post['분류']}</span><span>{post['작성시각']} · {post['닉네임']}</span></div>
                    <div class="post-title">{post['제목']}</div>
                    <div class="post-body">{post['내용']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history() -> None:
    st.subheader("최근 분석목록")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("현재 세션에서 분석한 종목이 여기에 표시됩니다.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def comparison_row(analysis: dict) -> dict:
    latest = analysis["df"].iloc[-1]
    return {
        "종목": f"{analysis['name']} ({analysis['code']})",
        "현재가": round(float(latest["Close"]), 0),
        "종합점수": analysis["final"]["total_score"],
        "Minervini": analysis["minervini"]["score"],
        "O'Neil": analysis["oneil"]["score"],
        "Weinstein": analysis["weinstein"]["score"],
        "RS": analysis["rs"]["score"],
        "VCP": analysis["vcp"]["score"],
        "등급": f"{analysis['final']['stars']} {analysis['final']['rating']}",
        "진입상태": analysis["final"]["entry_status"],
        "추천 진입방식": analysis["trade_plan"]["entry_method"],
        "추천 매수가격대": price_range(analysis["trade_plan"]["buy_low"], analysis["trade_plan"]["buy_high"]),
    }


def add_history(analysis: dict) -> None:
    latest = analysis["df"].iloc[-1]
    row = {
        "종목명": analysis["name"],
        "종목코드": analysis["code"],
        "분석시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "현재가": f"{latest['Close']:,.0f}원",
        "종합점수": analysis["final"]["total_score"],
        "등급": f"{analysis['final']['stars']} {analysis['final']['rating']}",
        "진입상태": analysis["final"]["entry_status"],
    }
    st.session_state["history"] = [row] + [item for item in st.session_state["history"] if item["종목코드"] != analysis["code"]]
    st.session_state["history"] = st.session_state["history"][:20]


def render_search_suggestions(input_key: str, market: str, multi: bool = False) -> None:
    current = str(st.session_state.get(input_key, "")).strip()
    search_text = current.split(",")[-1].strip() if multi else current
    if len(search_text) < 1 or search_text.isdigit():
        return

    try:
        listing = cached_stock_listing(market)
        matches = listing[listing["Name"].str.contains(search_text, case=False, na=False, regex=False)].head(6)
    except Exception:
        return

    if matches.empty:
        st.caption("추천 검색어가 없습니다.")
        return

    st.markdown('<div class="suggest-label">추천 검색어</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, len(matches)))
    for idx, row in enumerate(matches.itertuples(index=False)):
        label = f"{row.Name} ({row.Code})"
        value = _suggestion_value(input_key, row.Name, multi)
        cols[idx % len(cols)].button(
            label,
            key=f"suggest_{input_key}_{row.Code}_{idx}",
            on_click=set_search_value,
            args=(input_key, value),
            use_container_width=True,
        )


def set_search_value(input_key: str, value: str) -> None:
    st.session_state[input_key] = value


def _suggestion_value(input_key: str, name: str, multi: bool) -> str:
    if not multi:
        return name
    current = str(st.session_state.get(input_key, ""))
    parts = current.split(",")
    prefix = ",".join(parts[:-1]).strip()
    return f"{prefix},{name}" if prefix else name


def init_session() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_analysis", None)
    st.session_state.setdefault("recommendation_result", None)
    st.session_state.setdefault("mover_result", None)
    st.session_state.setdefault("unusual_result", None)
    st.session_state.setdefault("simulation_choices", [])
    st.session_state.setdefault("success_cases", None)
    st.session_state.setdefault("members", [])
    st.session_state.setdefault(
        "board_posts",
        [
            {
                "작성시각": "오늘 09:20",
                "닉네임": "BreakoutLab",
                "분류": "돌파 관찰",
                "제목": "거래량 동반 돌파는 다음날 지지가 더 중요해 보입니다",
                "내용": "당일 급등보다 피벗 위에서 거래량이 줄며 버티는지 확인하는 쪽이 리스크 관리에 유리해 보입니다.",
            },
            {
                "작성시각": "오늘 08:45",
                "닉네임": "RS체크",
                "분류": "관심종목",
                "제목": "시장대비 3개월 RS가 계속 양수인 종목 위주로 보고 있습니다",
                "내용": "지수 조정에도 50일선 위를 유지하는 종목은 별도 관심목록으로 분리해두면 좋겠습니다.",
            },
        ],
    )


def valid_code(code: str) -> bool:
    return len(str(code).strip()) == 6 and str(code).strip().isdigit()


def won(value) -> str:
    if value is None or pd.isna(value):
        return "데이터 부족"
    return f"{float(value):,.0f}원"


def price_range(low, high) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "제시 보류"
    return f"{float(low):,.0f}~{float(high):,.0f}원"


def num(value) -> str:
    if value is None or pd.isna(value):
        return "데이터 부족"
    return f"{float(value):,.0f}"


def pct(value) -> str:
    if value is None or pd.isna(value):
        return "데이터 부족"
    return f"{float(value):+.1f}%"


def format_rs(value) -> str:
    if value is None:
        return "데이터 부족"
    return f"{value:+.1f}%p"


def cache_bucket(mode: str) -> str:
    now = datetime.now()
    if mode == "주간":
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"
    if mode == "월간":
        return now.strftime("%Y-%m")
    return now.strftime("%Y-%m-%d")


def render_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748b;
            --line: #d8dee9;
            --surface: #ffffff;
            --soft: #f6f8fb;
            --accent: #0f766e;
            --accent-2: #1d4ed8;
            --danger: #b91c1c;
        }
        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, #eef3f8 46%, #f8fafc 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.4rem;
            max-width: 1500px;
        }
        section[data-testid="stSidebar"] {
            background: #0f172a;
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #e5e7eb;
        }
        .hero {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
            gap: 18px;
            align-items: stretch;
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 26px 28px;
            background:
                linear-gradient(135deg, rgba(255,255,255,.96), rgba(240,247,244,.94)),
                repeating-linear-gradient(90deg, rgba(15,118,110,.08) 0, rgba(15,118,110,.08) 1px, transparent 1px, transparent 42px);
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }
        .eyebrow {
            color: var(--accent);
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .hero h1 {
            font-size: 2.4rem;
            line-height: 1.15;
            margin: 0 0 10px 0;
            letter-spacing: 0;
            color: #0b1220;
        }
        .hero p {
            margin: 0;
            max-width: 860px;
            color: #475569;
            font-size: 1.04rem;
        }
        .hero-panel {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 18px;
            background: rgba(255,255,255,.82);
        }
        .panel-label {
            color: #64748b;
            font-size: .86rem;
            font-weight: 700;
            margin-bottom: 7px;
        }
        .panel-main {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .panel-sub {
            color: #64748b;
            font-size: .92rem;
            line-height: 1.45;
        }
        .stock-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 22px 24px;
            background: #ffffff;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin: 12px 0 16px 0;
        }
        .stock-meta {
            color: var(--accent-2);
            font-size: .9rem;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .stock-name {
            color: #0f172a;
            font-size: 2.2rem;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: 0;
        }
        .stock-sub {
            color: #64748b;
            margin-top: 7px;
            font-size: 1rem;
        }
        .badge-good,
        .badge-watch,
        .badge-risk {
            min-width: 96px;
            text-align: center;
            border-radius: 8px;
            padding: 12px 14px;
            font-weight: 900;
            border: 1px solid;
        }
        .badge-good { color: #065f46; background: #ecfdf5; border-color: #a7f3d0; }
        .badge-watch { color: #92400e; background: #fffbeb; border-color: #fde68a; }
        .badge-risk { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
        .question-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            background: var(--surface);
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .question-title {
            color: #475569;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .question-answer {
            color: #0f172a;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.25;
            word-break: keep-all;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stTabs"] button {
            font-weight: 750;
        }
        .post-card {
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 16px 18px;
            background: #ffffff;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .post-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: #64748b;
            font-size: .86rem;
            margin-bottom: 8px;
        }
        .post-top span:first-child {
            color: var(--accent);
            font-weight: 800;
        }
        .post-title {
            font-size: 1.05rem;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 7px;
        }
        .post-body {
            color: #475569;
            line-height: 1.55;
        }
        .suggest-label {
            color: #64748b;
            font-size: .82rem;
            font-weight: 800;
            margin: 4px 0 6px 0;
        }
        .sim-card {
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 18px;
            background: #ffffff;
            min-height: 118px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .sim-title {
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .sim-body {
            color: #64748b;
            line-height: 1.5;
        }
        .case-card {
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 22px 24px;
            background: #ffffff;
            margin: 0 0 18px 0;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        .case-kicker {
            color: #0f766e;
            font-size: .86rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .case-title {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 900;
            line-height: 1.28;
            margin-bottom: 10px;
        }
        .case-lead {
            color: #475569;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 16px;
        }
        .case-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }
        .case-grid div {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            background: #f8fafc;
        }
        .case-grid span {
            display: block;
            color: #64748b;
            font-size: .78rem;
            font-weight: 750;
            margin-bottom: 4px;
        }
        .case-grid b {
            color: #0f172a;
            font-size: .96rem;
        }
        .case-body {
            color: #475569;
            line-height: 1.55;
            margin-bottom: 12px;
        }
        .case-link {
            color: #1d4ed8;
            font-weight: 850;
            text-decoration: none;
        }
        @media (max-width: 900px) {
            .hero {
                grid-template-columns: 1fr;
                padding: 22px;
            }
            .hero h1 {
                font-size: 1.9rem;
            }
            .stock-header {
                align-items: flex-start;
                flex-direction: column;
            }
            .stock-name {
                font-size: 1.8rem;
            }
            .case-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
