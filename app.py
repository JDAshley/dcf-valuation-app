"""DCF Valuation App — Streamlit.

Data: SEC EDGAR Company Facts (historical fundamentals), Financial Modeling Prep
(profile, price, ratios, estimates) with a yfinance fallback.

For education only. Not financial advice.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="DCF Valuation App", page_icon="📈", layout="wide")

CACHE_TTL = 24 * 60 * 60  # 24 hours
FOOTER = "For education only. Not financial advice."
DEFAULT_SEC_UA = "DCF Valuation App (educational project) admin@example.com"
FMP_BASE = "https://financialmodelingprep.com/stable"
HTTP_TIMEOUT = 20

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A", "20-F", "20-F/A", "40-F", "40-F/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}


class DataError(Exception):
    """A data source failed in a way we want to report to the user."""


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def get_secret(name: str) -> str | None:
    """Read a value from st.secrets, falling back to an environment variable."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets.toml present
        value = None
    return value or os.environ.get(name) or None


def num(x) -> float | None:
    """Coerce to a finite float, or None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def pos(x) -> float | None:
    """Return x as float if strictly positive, else None."""
    f = num(x)
    return f if f is not None and f > 0 else None


def safe_div(a, b) -> float | None:
    a, b = num(a), num(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def fmt_money(x, decimals: int = 2) -> str:
    x = num(x)
    if x is None:
        return "N/A"
    sign = "-" if x < 0 else ""
    a = abs(x)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= divisor:
            return f"{sign}${a / divisor:,.{decimals}f}{suffix}"
    return f"{sign}${a:,.{decimals}f}"


def fmt_price(x) -> str:
    x = num(x)
    if x is None:
        return "N/A"
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def md_escape(text: str) -> str:
    """Escape "$" for Markdown, where a pair of dollar signs would render as LaTeX."""
    return text.replace("$", r"\$")


def fmt_mult(x) -> str:
    """Multiples are N/A when negative, zero or unavailable."""
    x = pos(x)
    return "N/A" if x is None else f"{x:,.1f}x"


def fmt_pct(x, signed: bool = False) -> str:
    x = num(x)
    if x is None:
        return "N/A"
    return f"{x:+.1%}" if signed else f"{x:.2%}"


# --------------------------------------------------------------------------- #
# SEC EDGAR
# --------------------------------------------------------------------------- #
def sec_headers() -> dict:
    return {
        "User-Agent": get_secret("SEC_USER_AGENT") or DEFAULT_SEC_UA,
        "Accept-Encoding": "gzip, deflate",
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_sec_tickers() -> pd.DataFrame:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=sec_headers(),
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    df = pd.DataFrame(list(r.json().values()))
    df["ticker"] = df["ticker"].str.upper()
    return df[["ticker", "title", "cik_str"]]


def _units(facts: dict, concept: str, unit: str, taxonomy: str = "us-gaap") -> list:
    try:
        return facts["facts"][taxonomy][concept]["units"][unit]
    except (KeyError, TypeError):
        return []


def _better(cand: tuple, cur: tuple | None) -> bool:
    """Candidate = (concept priority, filed date, value). Lower priority wins,
    then the most recently filed value (picks up restatements)."""
    if cur is None:
        return True
    return cand[0] < cur[0] or (cand[0] == cur[0] and cand[1] > cur[1])


def annual_flow(facts: dict, concepts: list[str], unit: str = "USD") -> pd.Series:
    """Fiscal-year duration values (income / cash-flow items), deduplicated.

    The same fiscal year is repeated in several 10-Ks (as a comparative
    period) and sometimes restated. We key each fact by its period end date,
    keep only ~12-month durations from annual forms, prefer facts that carry
    an SEC `frame` and otherwise keep the latest filing for that period.
    """
    best: dict[pd.Timestamp, tuple] = {}
    for pri, concept in enumerate(concepts):
        for f in _units(facts, concept, unit):
            if f.get("form") not in ANNUAL_FORMS or "start" not in f:
                continue
            try:
                start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
            except (KeyError, ValueError):
                continue
            if not 350 <= (end - start).days <= 380:
                continue
            framed = 0 if f.get("frame") else 1
            cand = (pri, (-framed, f.get("filed", "")), num(f.get("val")))
            if cand[2] is not None and _better(cand, best.get(end)):
                best[end] = cand
    return _one_per_fiscal_year({k: v[2] for k, v in best.items()})


def _one_per_fiscal_year(values: dict) -> pd.Series:
    s = pd.Series(values, dtype="float64").sort_index()
    if s.empty:
        return s
    # Guard against two periods landing in the same year (fiscal year-end change).
    s = s.groupby(s.index.year).tail(1)
    return s


def instant_values(facts: dict, concepts: list[str], unit: str = "USD") -> pd.Series:
    """Balance-sheet (point-in-time) values from 10-K/10-Q, one per date."""
    best: dict[pd.Timestamp, tuple] = {}
    for pri, concept in enumerate(concepts):
        for f in _units(facts, concept, unit):
            if "start" in f or f.get("form") not in ANNUAL_FORMS | QUARTERLY_FORMS:
                continue
            try:
                end = pd.Timestamp(f["end"])
            except (KeyError, ValueError):
                continue
            cand = (pri, f.get("filed", ""), num(f.get("val")))
            if cand[2] is not None and _better(cand, best.get(end)):
                best[end] = cand
    return pd.Series({k: v[2] for k, v in best.items()}, dtype="float64").sort_index()


def total_debt_series(facts: dict) -> pd.Series:
    total = instant_values(facts, ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"])
    noncurrent = instant_values(facts, ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"])
    current = instant_values(facts, ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"])
    short = instant_values(facts, ["ShortTermBorrowings", "CommercialPaper"])
    debt_current = instant_values(facts, ["DebtCurrent"])

    dates = sorted(set(total.index) | set(noncurrent.index) | set(current.index))
    out = {}
    for d in dates:
        if d in total.index:
            base = total[d]
        else:
            base = noncurrent.get(d, 0.0) + current.get(d, 0.0)
        if d in short.index:
            base += short[d]
        elif d not in current.index and d in debt_current.index:
            base += debt_current[d]
        out[d] = base
    return pd.Series(out, dtype="float64").sort_index()


def latest_shares(facts: dict) -> tuple[float | None, str | None]:
    """Shares outstanding from the cover page (summed across share classes)."""
    rows = [f for f in _units(facts, "EntityCommonStockSharesOutstanding", "shares", "dei") if num(f.get("val"))]
    if rows:
        last_end = max(f["end"] for f in rows)
        on_date = [f for f in rows if f["end"] == last_end]
        last_accn = max(on_date, key=lambda f: f.get("filed", ""))["accn"]
        total = sum(num(f["val"]) for f in on_date if f["accn"] == last_accn)
        return total, last_end
    s = instant_values(facts, ["CommonStockSharesOutstanding"], unit="shares")
    if not s.empty:
        return float(s.iloc[-1]), s.index[-1].date().isoformat()
    s = annual_flow(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding"], unit="shares")
    if not s.empty:
        return float(s.iloc[-1]), s.index[-1].date().isoformat()
    return None, None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def edgar_fundamentals(cik: int) -> dict:
    """Fetch EDGAR company facts and reduce them to a small, cacheable dict."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    r = requests.get(url, headers=sec_headers(), timeout=HTTP_TIMEOUT)
    if r.status_code == 404:
        return {"name": None, "history": pd.DataFrame(), "note": "No XBRL company facts on EDGAR."}
    r.raise_for_status()
    facts = r.json()

    revenue = annual_flow(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ])
    net_income = annual_flow(facts, ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"])
    ocf = annual_flow(facts, [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ])
    capex = annual_flow(facts, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
    ])
    op_income = annual_flow(facts, ["OperatingIncomeLoss"])
    sbc = annual_flow(facts, ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"])
    eps_diluted = annual_flow(facts, ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
                              unit="USD/shares")
    dna = annual_flow(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ])
    cash = instant_values(facts, [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ])
    debt = total_debt_series(facts)

    # Fiscal years = period ends of the flow statements; keep the last 5.
    by_year = {d.year: d for d in sorted(set(revenue.index) | set(net_income.index) | set(ocf.index))}
    ends = sorted(by_year.values())[-5:]

    def at(series: pd.Series, d) -> float | None:
        return float(series[d]) if d in series.index else None

    rows = []
    for d in ends:
        o, c = at(ocf, d), at(capex, d)
        oi, da = at(op_income, d), at(dna, d)
        rows.append({
            "Fiscal Year": f"FY{d.year}",
            "Period End": d.date().isoformat(),
            "Revenue": at(revenue, d),
            "Net Income": at(net_income, d),
            "Operating Cash Flow": o,
            "Capex": c,
            "Free Cash Flow": (o - c) if o is not None and c is not None else None,
            "EBITDA": (oi + da) if oi is not None and da is not None else None,
            "EPS (diluted)": at(eps_diluted, d),
            "Stock-Based Compensation": at(sbc, d),
            "Cash": at(cash, d),
            "Total Debt": at(debt, d),
        })
    history = pd.DataFrame(rows).set_index("Fiscal Year") if rows else pd.DataFrame()

    shares, shares_date = latest_shares(facts)
    return {
        "name": facts.get("entityName"),
        "history": history,
        "cash": float(cash.iloc[-1]) if not cash.empty else None,
        "cash_date": cash.index[-1].date().isoformat() if not cash.empty else None,
        "debt": float(debt.iloc[-1]) if not debt.empty else None,
        "debt_date": debt.index[-1].date().isoformat() if not debt.empty else None,
        "shares": shares,
        "shares_date": shares_date,
        "note": None if rows else "No US-GAAP annual data found (foreign/IFRS filers are not supported).",
    }


# --------------------------------------------------------------------------- #
# Market data: Financial Modeling Prep, with yfinance fallback
# --------------------------------------------------------------------------- #
MARKET_KEYS = [
    "name", "sector", "industry", "currency", "price", "market_cap", "shares",
    "eps_ttm", "eps_fwd", "pe_ttm", "pe_fwd", "peg", "ps", "pb", "ev",
    "ev_sales", "ev_ebitda", "ebitda", "revenue_ttm", "fcf_ttm", "div_yield",
    "total_debt", "total_cash", "estimates", "annual", "ttm",
]


def _pick(d: dict | None, *keys):
    if not d:
        return None
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "None"):
            return v
    return None


def _fmp_get(endpoint: str, symbol: str, api_key: str, **params):
    try:
        r = requests.get(
            f"{FMP_BASE}/{endpoint}",
            params={"symbol": symbol, "apikey": api_key, **params},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if isinstance(data, dict) and ("Error Message" in data or "error" in data):
        return None
    return data


def _first(data):
    if isinstance(data, list):
        return data[0] if data else None
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fmp_market_data(symbol: str, api_key: str) -> dict:
    profile = _first(_fmp_get("profile", symbol, api_key))
    if not profile:
        return {}
    ratios = _first(_fmp_get("ratios-ttm", symbol, api_key)) or {}
    metrics = _first(_fmp_get("key-metrics-ttm", symbol, api_key)) or {}
    estimates = _fmp_get("analyst-estimates", symbol, api_key, period="annual", page=0, limit=10) or []
    income = _fmp_get("income-statement", symbol, api_key, period="annual", limit=5) or []
    cashflow = _fmp_get("cash-flow-statement", symbol, api_key, period="annual", limit=5) or []
    sbc_by_date = {str(c.get("date", "")): num(c.get("stockBasedCompensation"))
                   for c in (cashflow if isinstance(cashflow, list) else []) if isinstance(c, dict)}
    q_income = _fmp_get("income-statement", symbol, api_key, period="quarter", limit=4) or []
    q_cash = {str(c.get("date", "")): c for c in (_fmp_get("cash-flow-statement", symbol, api_key,
                                                            period="quarter", limit=4) or [])
              if isinstance(c, dict)}
    quarters = [
        {"date": str(i.get("date", "")), "ebitda": num(i.get("ebitda")), "ebitda_norm": None,
         "sbc": num(q_cash.get(str(i.get("date", "")), {}).get("stockBasedCompensation")),
         "fcf": num(q_cash.get(str(i.get("date", "")), {}).get("freeCashFlow"))}
        for i in (q_income if isinstance(q_income, list) else []) if isinstance(i, dict)
    ]

    price = num(_pick(profile, "price"))
    mcap = num(_pick(profile, "marketCap", "mktCap"))
    pe = num(_pick(ratios, "priceToEarningsRatioTTM", "peRatioTTM"))
    ps = num(_pick(ratios, "priceToSalesRatioTTM"))
    ev = num(_pick(metrics, "enterpriseValueTTM"))
    ev_ebitda = num(_pick(metrics, "evToEBITDATTM", "enterpriseValueOverEBITDATTM"))

    # Nearest fiscal year still in the future = forward estimate.
    eps_fwd = None
    today = dt.date.today().isoformat()
    if isinstance(estimates, list):
        future = sorted((e for e in estimates if str(e.get("date", "")) > today), key=lambda e: e["date"])
        if future:
            eps_fwd = num(_pick(future[0], "epsAvg", "estimatedEpsAvg"))

    eps_ttm = num(_pick(ratios, "netIncomePerShareTTM")) or safe_div(price, pe)
    consensus = [
        {
            "date": str(e.get("date", "")),
            "revenue": num(_pick(e, "revenueAvg", "estimatedRevenueAvg")),
            "ebitda": num(_pick(e, "ebitdaAvg", "estimatedEbitdaAvg")),
            "analysts": num(_pick(e, "numAnalystsRevenue", "numberAnalystEstimatedRevenue")),
            "eps": num(_pick(e, "epsAvg", "estimatedEpsAvg")),
            "eps_analysts": num(_pick(e, "numAnalystsEps", "numberAnalystsEstimatedEps")),
        }
        for e in (estimates if isinstance(estimates, list) else [])
        if isinstance(e, dict)
    ]
    fcf_yield = num(_pick(metrics, "freeCashFlowYieldTTM"))
    return {
        "name": _pick(profile, "companyName"),
        "sector": _pick(profile, "sector"),
        "industry": _pick(profile, "industry"),
        "currency": _pick(profile, "currency"),
        "price": price,
        "market_cap": mcap,
        "eps_ttm": eps_ttm,
        "eps_fwd": eps_fwd,
        "pe_ttm": pe,
        "peg": num(_pick(ratios, "priceToEarningsGrowthRatioTTM", "pegRatioTTM")),
        "ps": ps,
        "pb": num(_pick(ratios, "priceToBookRatioTTM")),
        "ev": ev,
        "ev_sales": num(_pick(metrics, "evToSalesTTM")),
        "ev_ebitda": ev_ebitda,
        "ebitda": safe_div(ev, ev_ebitda),
        "revenue_ttm": safe_div(mcap, ps),
        "fcf_ttm": fcf_yield * mcap if fcf_yield is not None and mcap else None,
        "div_yield": num(_pick(ratios, "dividendYieldTTM", "dividendYielTTM")),
        "estimates": consensus,
        "annual": [
            {"date": str(i.get("date", "")), "source": "Financial Modeling Prep", "ebitda": num(i.get("ebitda")),
             "ebitda_norm": None, "eps_diluted": num(_pick(i, "epsDiluted", "epsdiluted")),
             "sbc": sbc_by_date.get(str(i.get("date", "")))}
            for i in (income if isinstance(income, list) else []) if isinstance(i, dict)
        ],
        "ttm": ttm_adjusted(quarters, "Financial Modeling Prep"),
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def yf_market_data(symbol: str) -> dict:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    price = num(_pick(info, "currentPrice", "regularMarketPrice", "previousClose"))
    if price is None:
        return {}
    div_rate = num(info.get("dividendRate"))
    div_yield = safe_div(div_rate, price) if div_rate else num(info.get("trailingAnnualDividendYield"))
    return {
        "name": _pick(info, "longName", "shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency"),
        "price": price,
        "market_cap": num(info.get("marketCap")),
        "shares": num(_pick(info, "impliedSharesOutstanding", "sharesOutstanding")),
        "eps_ttm": num(info.get("trailingEps")),
        "eps_fwd": num(info.get("forwardEps")),
        "pe_ttm": num(info.get("trailingPE")),
        "pe_fwd": num(info.get("forwardPE")),
        "peg": num(_pick(info, "trailingPegRatio", "pegRatio")),
        "ps": num(info.get("priceToSalesTrailing12Months")),
        "pb": num(info.get("priceToBook")),
        "ev": num(info.get("enterpriseValue")),
        "ev_sales": num(info.get("enterpriseToRevenue")),
        "ev_ebitda": num(info.get("enterpriseToEbitda")),
        "ebitda": num(info.get("ebitda")),
        "revenue_ttm": num(info.get("totalRevenue")),
        "fcf_ttm": num(info.get("freeCashflow")),
        "div_yield": div_yield or 0.0,
        "total_debt": num(info.get("totalDebt")),
        "total_cash": num(info.get("totalCash")),
        "annual": _yf_periods(ticker, quarterly=False),
        "ttm": ttm_adjusted(_yf_periods(ticker, quarterly=True), "yfinance"),
    }


def _yf_periods(ticker, quarterly: bool) -> list[dict]:
    """EBITDA (reported and normalized), stock-based compensation, FCF and diluted EPS per period."""
    try:
        inc = ticker.quarterly_income_stmt if quarterly else ticker.income_stmt
        cf = ticker.quarterly_cashflow if quarterly else ticker.cashflow
    except Exception:
        return []
    if inc is None or inc.empty:
        return []

    def get(df, row, col):
        if df is None or df.empty or row not in df.index or col not in df.columns:
            return None
        return num(df.at[row, col])

    return [
        {"date": pd.Timestamp(col).date().isoformat(), "source": "yfinance",
         "ebitda": get(inc, "EBITDA", col), "ebitda_norm": get(inc, "Normalized EBITDA", col),
         "eps_diluted": get(inc, "Diluted EPS", col), "sbc": get(cf, "Stock Based Compensation", col),
         "fcf": get(cf, "Free Cash Flow", col)}
        for col in inc.columns
    ]


def ttm_adjusted(quarters: list[dict], source: str) -> dict | None:
    """TTM FCF and TTM Adjusted EBITDA from the latest four quarters of ONE provider.

    Adjusted EBITDA = EBITDA (normalized, i.e. excluding unusual items, when the provider has it
    for all four quarters; otherwise reported) + stock-based compensation."""
    latest = sorted(quarters, key=lambda q: q["date"], reverse=True)[:4]
    if len(latest) < 4 or any(q["fcf"] is None or q["sbc"] is None for q in latest):
        return None
    kind = "ebitda_norm" if all(q["ebitda_norm"] is not None for q in latest) else "ebitda"
    if any(q[kind] is None for q in latest):
        return None
    core = sum(q[kind] for q in latest)
    sbc = sum(q["sbc"] for q in latest)
    return {
        "source": source,
        "period": f"4 quarters to {latest[0]['date']}",
        "fcf": sum(q["fcf"] for q in latest),
        "core": core,
        "core_kind": "normalized EBITDA" if kind == "ebitda_norm" else "reported EBITDA",
        "sbc": sbc,
        "adj_ebitda": core + sbc,
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def yf_estimates(symbol: str) -> dict:
    """Consensus for the current (0y) and next (+1y) fiscal years: revenue growth and EPS."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    out: dict = {"revenue": [], "eps": []}
    df = ticker.revenue_estimate
    prev = None
    for period in ("0y", "+1y"):
        if df is None or df.empty or period not in df.index:
            break
        row = df.loc[period]
        avg = pos(row.get("avg"))
        base = prev if prev is not None else pos(row.get("yearAgoRevenue"))
        if not avg or not base:
            break
        out["revenue"].append({"growth": avg / base - 1, "analysts": num(row.get("numberOfAnalysts"))})
        prev = avg
    df = ticker.earnings_estimate
    for period in ("0y", "+1y"):
        if df is None or df.empty or period not in df.index or num(df.loc[period].get("avg")) is None:
            break
        out["eps"].append({"eps": num(df.loc[period].get("avg")),
                           "analysts": num(df.loc[period].get("numberOfAnalysts"))})
    return out


def get_market_data(symbol: str) -> tuple[dict, str, list[str]]:
    """Return (normalized market data, source label, warnings)."""
    warnings: list[str] = []
    data: dict = {}
    source = "yfinance"
    api_key = get_secret("FMP_API_KEY")

    if api_key:
        try:
            data = fmp_market_data(symbol, api_key)
        except Exception:
            data = {}
        if data.get("price"):
            source = "Financial Modeling Prep"
        else:
            warnings.append("Financial Modeling Prep returned no data (invalid key, rate limit or "
                            "unsupported symbol) — falling back to yfinance.")

    # yfinance is the fallback, and also fills any gaps left by FMP's free tier.
    if not data.get("price") or any(not data.get(k) for k in ("eps_fwd", "ebitda", "revenue_ttm", "annual", "ttm")):
        try:
            yf_data = yf_market_data(symbol)
        except Exception:
            yf_data = {}
            if not data.get("price"):
                warnings.append("yfinance request failed.")
        if not data.get("price"):
            data = yf_data
        else:
            for k, v in yf_data.items():
                if data.get(k) in (None, []) and v not in (None, []):
                    data[k] = v
            if yf_data:
                source += " (gaps filled from yfinance)"

    return {k: data.get(k) for k in MARKET_KEYS}, source, warnings


# --------------------------------------------------------------------------- #
# Projection model
# --------------------------------------------------------------------------- #
YEARS = 5
LONG_RUN_GROWTH = 3.0  # % that defaults fade toward after consensus runs out
ACTUAL_COLORS = ["#374151", "#6b7280", "#9ca3af"]     # Revenue, EBITDA, FCF — muted greys
PROJECTED_COLORS = ["#1d4ed8", "#3b82f6", "#93c5fd"]  # same order — accent blues
EPS_COLORS = ["#b8a07a", "#d97706"]                   # EPS line: muted tan (actual), amber (projected)

ASSUMPTION_CSS = """
<style>
.st-key-assumptions {
    border: 2px solid #1d4ed8 !important;
    border-radius: 0.6rem;
    background: rgba(29, 78, 216, 0.05);
    padding: 1rem;
}
.st-key-assumptions [data-testid="stNumberInputContainer"],
.st-key-assumptions [data-testid="stNumberInput"] input {
    background-color: #e8f0fe !important;
    color: #0b57d0 !important;
    font-weight: 600;
}
</style>
"""


def fade(start: float, n: int) -> list[float]:
    """n growth rates moving linearly from `start` toward LONG_RUN_GROWTH."""
    return [start + (LONG_RUN_GROWTH - start) * k / (n + 1) for k in range(1, n + 1)]


def build_defaults(base: dict, years: list[int], fmp_estimates: list | None, yf_est: dict | None) -> dict:
    """Default assumptions (all in %): consensus where available, otherwise latest actuals."""
    growth: list[float] = []
    margin: list[float] = []
    source, analysts = None, None

    # FMP: consecutive fiscal-year estimates after the base period.
    if fmp_estimates and pos(base["revenue"]):
        prev, last_date = base["revenue"], base["end"]
        for e in sorted(fmp_estimates, key=lambda e: e["date"]):
            try:
                d = pd.Timestamp(e["date"])
            except ValueError:
                continue
            if d <= base["end"] + pd.Timedelta(days=200) or not pos(e["revenue"]):
                continue
            if (d - last_date).days > 400 or len(growth) == YEARS:  # gap in coverage
                break
            growth.append((e["revenue"] / prev - 1) * 100)
            if e["ebitda"] is not None:
                margin.append(e["ebitda"] / e["revenue"] * 100)
            prev, last_date = e["revenue"], d
            analysts = analysts or e["analysts"]
        if growth:
            source = "Financial Modeling Prep"
        margin = margin[:len(growth)] if len(margin) == len(growth) else []

    yf_est = yf_est or {}
    if not growth and yf_est.get("revenue"):
        growth = [e["growth"] * 100 for e in yf_est["revenue"]][:YEARS]
        analysts = yf_est["revenue"][0]["analysts"]
        source = "yfinance"

    # Consensus EPS per projection year (None = not covered): FMP, else yfinance.
    eps: list[float | None] = [None] * YEARS
    eps_source, eps_analysts = None, None
    for e in sorted(fmp_estimates or [], key=lambda e: e["date"]):
        try:
            offset = round((pd.Timestamp(e["date"]) - base["end"]).days / 365.25)
        except ValueError:
            continue
        if 1 <= offset <= YEARS and e.get("eps") is not None:
            eps[offset - 1] = e["eps"]
            eps_source = "Financial Modeling Prep"
            eps_analysts = eps_analysts or e.get("eps_analysts")
    if eps_source is None and yf_est.get("eps"):
        for i, e in enumerate(yf_est["eps"][:YEARS]):
            eps[i] = e["eps"]
        eps_source, eps_analysts = "yfinance", yf_est["eps"][0]["analysts"]

    n_growth, n_margin = len(growth), len(margin)
    if growth:
        growth += fade(growth[-1], YEARS - n_growth)
    else:
        actual = safe_div(base["revenue"], base.get("prior_revenue"))
        start = min(max((actual - 1) * 100, -20.0), 30.0) if actual else LONG_RUN_GROWTH
        growth = [start] + fade(start, YEARS - 1)

    base_margin = (safe_div(base["ebitda"], base["revenue"]) or 0.0) * 100
    margin += [margin[-1] if margin else base_margin] * (YEARS - n_margin)

    conversion, conv_note = base["conversion"], base["conv_note"]

    span = lambda n: f"{years[0]}" if n == 1 else f"{years[0]}–{years[n - 1]}"
    if source:
        who = source + (f", {int(analysts)} analysts" if analysts else "")
        parts = [f"**Defaults: analyst consensus average ({who})** for revenue growth in {span(n_growth)}"]
        if n_margin:
            parts.append(f" and EBITDA margin in {span(n_margin)}.")
        else:
            parts.append(". No EBITDA consensus from this source — Adj. EBITDA margins default to "
                         + ("the latest actual Adjusted EBITDA margin." if base["ebitda"] is not None
                            else "0% (no base-year EBITDA)."))
        if n_growth < YEARS:
            parts.append(f" Later years fade revenue growth toward {LONG_RUN_GROWTH:.0f}%"
                         + (" and carry the last consensus margin forward." if n_margin else "."))
        parts.append(f" FCF conversion: {conv_note}.")
        caption = "".join(parts)
    else:
        caption = ("**No analyst consensus available — defaults based on latest actual results** "
                   f"(latest revenue growth fading toward {LONG_RUN_GROWTH:.0f}%, latest EBITDA margin, "
                   f"FCF conversion {conv_note}).")

    return {
        "growth": [round(g, 1) for g in growth],
        "margin": [round(m, 1) for m in margin],
        "conversion": round(conversion, 1),
        "caption": caption,
        "is_consensus": source is not None,
        "eps": eps,
        "eps_source": eps_source,
        "eps_analysts": eps_analysts,
    }


# --------------------------------------------------------------------------- #
# Saved inputs — kept in the viewer's browser (localStorage), never on the server
# --------------------------------------------------------------------------- #
STORE_KEY = "browser_store"
LAST_TICKER_ITEM = "dcf_last_ticker"
_browser_store = components.declare_component(
    "browser_store", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_store")
)


def stored_items() -> dict | None:
    """Saved items from browser storage, or None until the browser has reported them."""
    store = st.session_state.get(STORE_KEY)
    return (store.get("items") or {}) if isinstance(store, dict) else None


def storage_available() -> bool:
    store = st.session_state.get(STORE_KEY)
    return isinstance(store, dict) and bool(store.get("available"))


def saved_inputs(ticker: str) -> dict | None:
    raw = (stored_items() or {}).get(f"dcf_inputs_{ticker}")
    try:
        saved = json.loads(raw) if raw else None
    except ValueError:
        return None
    return saved if isinstance(saved, dict) and isinstance(saved.get("values"), dict) else None


def input_key(ticker: str, name: str) -> str:
    return f"in_{ticker}_{name}"


def seed_input(ticker: str, name: str, default, lo=None, hi=None, options=None) -> str:
    """Register an input's default and, before its widget is created, seed the
    widget's session state with the saved value (if any) or the default.
    Returns the widget key."""
    st.session_state.setdefault(f"_defaults_{ticker}", {})[name] = default
    key = input_key(ticker, name)
    if key not in st.session_state:
        value = default
        saved = saved_inputs(ticker)
        if saved and name in saved["values"]:
            v = saved["values"][name]
            if options is not None:
                value = v if v in options else default
            elif num(v) is not None:
                value = min(max(float(v), lo if lo is not None else -math.inf), hi if hi is not None else math.inf)
            st.session_state[f"_restored_{ticker}"] = saved.get("saved_at")
        st.session_state[key] = value
    return key


def reset_inputs(ticker: str) -> None:
    """Restore every input for this ticker to its default and drop the saved copy."""
    for name, default in st.session_state.get(f"_defaults_{ticker}", {}).items():
        st.session_state[input_key(ticker, name)] = default
    st.session_state.pop(f"_restored_{ticker}", None)
    st.session_state[f"_reset_{ticker}"] = True


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def persist_inputs(ticker: str | None, slot) -> None:
    """Write this ticker's inputs to browser storage when they change.

    Inputs equal to the defaults are not stored (and any saved copy is deleted),
    so "Reset to consensus" makes the defaults come back and stay back.
    """
    items = stored_items()
    writes: dict[str, str | None] = {}
    if items is not None and ticker:
        defaults = st.session_state.get(f"_defaults_{ticker}", {})
        current = {n: st.session_state[input_key(ticker, n)] for n in defaults
                   if input_key(ticker, n) in st.session_state}
        if current:
            if items.get(LAST_TICKER_ITEM) != ticker:
                writes[LAST_TICKER_ITEM] = ticker
            saved = saved_inputs(ticker)
            reset = st.session_state.pop(f"_reset_{ticker}", False)
            # Keep saved values for inputs not on screen this run (e.g. the P/S multiple while on P/E).
            values = current if reset or not saved else {**saved["values"], **current}
            item = f"dcf_inputs_{ticker}"
            if all(n in defaults and _same(v, defaults[n]) for n, v in values.items()):
                if item in items:
                    writes[item] = None
            elif not saved or values.keys() != saved["values"].keys() or not all(
                    _same(v, saved["values"][n]) for n, v in values.items()):
                saved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                writes[item] = json.dumps({"saved_at": saved_at, "values": values})
    with slot:
        _browser_store(writes=writes, key=STORE_KEY, default=None)


def fmt_saved_at(iso: str | None) -> str:
    try:
        return dt.datetime.fromisoformat(iso).strftime("%b %d, %Y at %H:%M UTC")
    except (TypeError, ValueError):
        return "in an earlier session"


def project(base_revenue: float, growth: list[float], margin: list[float], conversion: float) -> list[dict]:
    rows, revenue = [], base_revenue
    for g, m in zip(growth, margin):
        revenue = revenue * (1 + g / 100)
        ebitda = revenue * m / 100
        rows.append({"revenue": revenue, "ebitda": ebitda, "fcf": ebitda * conversion / 100})
    return rows


def project_eps(base_eps, base_ebitda, proj: list[dict], consensus: list) -> list[float | None]:
    """Consensus EPS where covered; beyond that, prior EPS grown at the projected EBITDA growth rate.
    A non-positive prior EPS or EBITDA is never compounded — those years are N/A."""
    out: list[float | None] = []
    prev_eps, prev_ebitda = base_eps, base_ebitda
    for p, cons in zip(proj, consensus):
        if cons is not None:
            eps = cons
        elif pos(prev_eps) and pos(prev_ebitda) and p["ebitda"] > 0:
            eps = prev_eps * p["ebitda"] / prev_ebitda
        else:
            eps = None
        out.append(eps)
        prev_eps, prev_ebitda = eps, p["ebitda"]
    return out


def _zero_aligned_ranges(primary: list, secondary: list) -> tuple[list, list]:
    """Axis ranges that put y=0 at the same height on both axes, so negative bars on
    either axis sit below one shared zero line."""
    def bounds(vals):
        vals = [v for v in vals if v is not None] or [0.0]
        lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
        return (lo - 0.08 * (hi - lo) if lo < 0 else lo), hi  # headroom so low markers aren't cut off

    (lo1, hi1), (lo2, hi2) = bounds(primary), bounds(secondary)
    frac = lambda lo, hi: -lo / (hi - lo) if hi > lo else 0.0
    f = min(max(frac(lo1, hi1), frac(lo2, hi2)), 0.6)  # share of each axis below zero (capped for legibility)

    def rng(lo, hi):
        span = max(hi / (1 - f) if hi > 0 else 0.0, -lo / f if f > 0 else 0.0) or 1.0
        span *= 1.1
        return [-f * span, (1 - f) * span]

    return rng(lo1, hi1), rng(lo2, hi2)


def model_chart(labels: list[str], base: dict, proj: list[dict], eps: list):
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (key, name) in enumerate((("revenue", "Revenue"), ("ebitda", "Adj. EBITDA"), ("fcf", "Free Cash Flow"))):
        fig.add_bar(x=[labels[0]], y=[(base[key] or 0) / 1e9], name=name, offsetgroup=key,
                    marker_color=ACTUAL_COLORS[i], legendgroup="actual", legendgrouptitle_text="Actual",
                    hovertemplate=f"{name} (actual): $%{{y:,.2f}}B<extra></extra>")
    for i, (key, name) in enumerate((("revenue", "Revenue"), ("ebitda", "Adj. EBITDA"), ("fcf", "Free Cash Flow"))):
        fig.add_bar(x=labels[1:], y=[p[key] / 1e9 for p in proj], name=name, offsetgroup=key,
                    marker_color=PROJECTED_COLORS[i], legendgroup="projected", legendgrouptitle_text="Projected",
                    hovertemplate=f"{name} (projected): $%{{y:,.2f}}B<extra></extra>")
    # EPS as a line on the secondary axis: a solid marker for the actual year, then a dashed
    # line (starting from the actual point) with open markers through the projection years.
    fig.add_scatter(x=labels, y=[base["eps"]] + eps, cliponaxis=False, name="EPS (diluted), dashed", yaxis="y2",
                    mode="lines+markers", line=dict(color=EPS_COLORS[1], width=2.5, dash="dash"),
                    marker=dict(size=[0] + [10] * len(eps), symbol="circle-open", line=dict(width=2.5)),
                    text=["actual"] + ["projected"] * len(eps), legendgroup="projected",
                    legendgrouptitle_text="Projected",
                    hovertemplate="EPS (%{text}): $%{y:,.2f}<extra></extra>")
    fig.add_scatter(x=[labels[0]], y=[base["eps"]], cliponaxis=False, name="EPS (diluted)", yaxis="y2", mode="markers",
                    marker=dict(size=12, color=EPS_COLORS[0], line=dict(color="#78633f", width=1.5)),
                    legendgroup="actual", legendgrouptitle_text="Actual",
                    hovertemplate="EPS (actual): $%{y:,.2f}<extra></extra>")
    money = [(base[k] or 0) / 1e9 for k in ("revenue", "ebitda", "fcf")] + \
            [p[k] / 1e9 for p in proj for k in ("revenue", "ebitda", "fcf")]
    y1, y2 = _zero_aligned_ranges(money, [base["eps"]] + eps)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#9ca3af")
    fig.add_annotation(x=0.5, y=1.06, yref="paper", text="Projections begin →", showarrow=False,
                       xanchor="left", font=dict(size=12))
    fig.update_layout(
        barmode="group", height=440, margin=dict(t=40, b=10, l=10, r=10),
        yaxis=dict(title="USD billions", range=y1, zeroline=True, zerolinecolor="#6b7280"),
        yaxis2=dict(title="EPS ($)", overlaying="y", side="right", range=y2, showgrid=False,
                    zeroline=True, zerolinecolor="#6b7280", tickformat="$,.2f"),
        legend=dict(groupclick="toggleitem", x=1.08),
    )
    return fig


# --------------------------------------------------------------------------- #
# Valuation math
# --------------------------------------------------------------------------- #
def run_dcf(flows: list[float], terminal, wacc, net_debt, shares) -> dict | None:
    """Discount explicit FCFs, add a Gordon Growth terminal value on the final year."""
    if not flows or not pos(flows[-1]) or not pos(shares) or wacc <= terminal:
        return None
    factors = [(1 + wacc) ** t for t in range(1, len(flows) + 1)]
    pv_flows = [f / d for f, d in zip(flows, factors)]
    tv = flows[-1] * (1 + terminal) / (wacc - terminal)
    pv_tv = tv / factors[-1]
    ev = sum(pv_flows) + pv_tv
    equity = ev - (net_debt or 0.0)
    return {
        "flows": flows, "factors": factors, "pv_flows": pv_flows,
        "tv": tv, "pv_tv": pv_tv, "ev": ev, "equity": equity,
        "price": equity / shares,
    }


def is_financial(sector: str | None, industry: str | None) -> bool:
    s, i = (sector or "").lower(), (industry or "").lower()
    return "financial" in s or any(k in i for k in ("bank", "insurance", "insurer", "reinsurance"))


def upside(target, price) -> float | None:
    if num(target) is None or not pos(price):
        return None
    return target / price - 1


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def resolve_ticker(query: str) -> str | None:
    """Accept a ticker or a company name; returns a ticker or None."""
    q = query.strip().upper().replace(".", "-")
    if not q:
        return None
    try:
        tickers = load_sec_tickers()
    except Exception:
        st.warning("Could not load the SEC ticker list — using your input as the ticker.")
        return q
    if q in set(tickers["ticker"]):
        return q
    # Match on letters/digits only ("coca-cola" finds "COCA COLA CO"), shortest names first.
    needle = re.sub(r"[^A-Z0-9]", "", query.upper())
    if not needle:
        return None
    hay = tickers["title"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    matches = tickers[hay.str.contains(needle, regex=False)]
    matches = matches.loc[matches["title"].str.len().sort_values(kind="stable").index].head(15)
    if matches.empty:
        # Not on EDGAR (e.g. foreign listing) — still try market data sources.
        return q
    labels = [f"{r.ticker} — {r.title}" for r in matches.itertuples()]
    choice = st.selectbox("Matching companies", labels)
    return choice.split(" — ")[0]


def lookup_cik(ticker: str) -> int | None:
    try:
        tickers = load_sec_tickers()
    except Exception:
        return None
    hit = tickers[tickers["ticker"] == ticker]
    return int(hit.iloc[0]["cik_str"]) if not hit.empty else None


def render_snapshot(m: dict, fcf_for_yield: float | None) -> None:
    st.header("Valuation Snapshot")
    eps_neg = m["eps_ttm"] is not None and m["eps_ttm"] <= 0
    items = [
        ("Trailing P/E", "N/A" if eps_neg else fmt_mult(m["pe_ttm"])),
        ("Forward P/E", "N/A" if not pos(m["eps_fwd"]) else fmt_mult(m["pe_fwd"])),
        ("PEG", "N/A" if eps_neg else fmt_mult(m["peg"])),
        ("Price / Sales", fmt_mult(m["ps"])),
        ("Price / Book", fmt_mult(m["pb"])),
        ("EV / Sales", fmt_mult(m["ev_sales"])),
        ("EV / Adj. EBITDA", "N/A" if not pos(m["ebitda"]) else fmt_mult(m["ev_ebitda"])),
        ("FCF Yield", fmt_pct(safe_div(fcf_for_yield, m["market_cap"]))),
    ]
    if pos(m["div_yield"]):
        items.append(("Dividend Yield", fmt_pct(m["div_yield"])))
    else:
        items.append(("Dividend Yield", "None"))
    for start in range(0, len(items), 5):
        cols = st.columns(5)
        for col, (label, value) in zip(cols, items[start:start + 5]):
            col.metric(label, value)
    st.caption("Multiples show N/A when the underlying earnings, EBITDA or book value is negative or unavailable.")


def render_model(ticker: str, base: dict, defaults: dict, years: list[int]) -> list[dict] | None:
    """Assumption boxes, calculated table and bar chart. Returns the projected years."""
    st.header("Financial Model")
    if not pos(base["revenue"]):
        st.warning("Base-year revenue is unavailable, so the projection model and DCF cannot be built.")
        return None
    if base["ebitda"] is None:
        st.warning("Base-year Adjusted EBITDA is unavailable (no provider EBITDA, and operating income or D&A not "
                   "reported on EDGAR). Adj. EBITDA margin defaults start at 0% — enter your own margins.")
    elif base["ebitda"] <= 0:
        st.warning("Base-year Adjusted EBITDA is negative. The Adjusted EBITDA → FCF bridge (FCF conversion %) is unreliable "
                   "for this company; treat the projected FCF and DCF with caution.")
    for note in base["notes"]:
        st.caption(note)

    # Seed every box (saved value first, else default) before any widget is created.
    for i in range(YEARS):
        seed_input(ticker, f"g{i}", defaults["growth"][i])
        seed_input(ticker, f"m{i}", defaults["margin"][i])
    seed_input(ticker, "conv", defaults["conversion"])

    st.markdown(ASSUMPTION_CSS, unsafe_allow_html=True)
    with st.container(border=True, key="assumptions"):
        st.subheader("Projection Assumptions — edit any box to run your own case")
        restored = st.session_state.get(f"_restored_{ticker}")
        if restored is not None:
            st.caption(f"💾 Restored your saved assumptions for this ticker (saved {fmt_saved_at(restored)}).")
        widths = [1.6] + [1] * YEARS
        for label, kind, help_text in (
            ("Revenue growth %", "g", "Projected revenue = prior-year revenue × (1 + growth %)."),
            ("Adj. EBITDA margin %", "m",
             "Projected Adjusted EBITDA = projected revenue × Adjusted EBITDA margin %."),
        ):
            cols = st.columns(widths, vertical_alignment="bottom")
            cols[0].markdown(f"**{label}**", help=help_text)
            for i, col in enumerate(cols[1:]):
                col.number_input(f"{years[i]}E", key=input_key(ticker, f"{kind}{i}"), step=0.5, format="%.1f")
        cols = st.columns([1.6, 1, 4], vertical_alignment="bottom")
        cols[0].markdown("**FCF conversion %**",
                         help="FCF ÷ Adjusted EBITDA. Projected FCF = projected Adjusted EBITDA × conversion %.")
        cols[1].number_input("All years", key=input_key(ticker, "conv"), step=1.0, format="%.1f")
        cols[2].caption(md_escape(base["conv_calc"]))
        conv_now = st.session_state[input_key(ticker, "conv")]
        if base["conv_unusual"]:
            st.warning(f"⚠️ The company's actual conversion is {base['conversion']:.1f}%, which is unusually "
                       f"{'high' if base['conversion'] > 100 else 'low'} and may not be sustainable. "
                       "Consider entering a long-run rate instead.")
        if conv_now > 100 and not _same(conv_now, defaults["conversion"]):
            st.warning("FCF conversion above 100% means free cash flow exceeds Adjusted EBITDA every year. That can "
                       "happen briefly (e.g. working-capital releases) but is not sustainable long-term.")
        cols = st.columns([3, 1], vertical_alignment="center")
        save_note = (" Your changes save automatically in this browser." if storage_available()
                     else " Browser storage is unavailable, so changes will not be saved.")
        cols[0].caption(defaults["caption"] + save_note)
        cols[1].button("Reset to consensus" if defaults["is_consensus"] else "Reset to defaults",
                       on_click=reset_inputs, args=(ticker,), width="stretch",
                       help="Restores every input for this ticker to its default and deletes your saved inputs.")

    growth = [st.session_state[input_key(ticker, f"g{i}")] for i in range(YEARS)]
    margin = [st.session_state[input_key(ticker, f"m{i}")] for i in range(YEARS)]
    conversion = st.session_state[input_key(ticker, "conv")]
    proj = project(base["revenue"], growth, margin, conversion)
    eps = project_eps(base["eps"], base["ebitda"], proj, defaults["eps"])

    labels = [f"{base['label']} Actual"] + [f"{y}E" for y in years]
    base_margin = safe_div(base["ebitda"], base["revenue"])
    table = pd.DataFrame(
        {
            "Revenue": [fmt_money(base["revenue"])] + [fmt_money(p["revenue"]) for p in proj],
            "Adjusted EBITDA": [fmt_money(base["ebitda"])] + [fmt_money(p["ebitda"]) for p in proj],
            "Adj. EBITDA margin %": [fmt_pct(base_margin)] + [fmt_pct(p["ebitda"] / p["revenue"]) for p in proj],
            "Free Cash Flow": [fmt_money(base["fcf"])] + [fmt_money(p["fcf"]) for p in proj],
            "EPS (diluted)": [fmt_price(base["eps"])] + [fmt_price(e) for e in eps],
        },
        index=labels,
    ).T
    st.dataframe(table, width="stretch")
    st.caption(f"Actual column: {base['source']}, period ending {base['end'].date().isoformat()}. "
               f"Actual FCF = operating cash flow − capex. Actual Adjusted EBITDA: {base['ebitda_source']}. "
               f"Actual EPS: {base['eps_source']}.")
    try:
        st.plotly_chart(model_chart(labels, base, proj, eps), width="stretch")
    except ImportError:
        st.warning("Install plotly (`pip install plotly`) to see the chart.")
    st.caption(eps_caption(defaults, years, eps, base))
    return proj


def eps_caption(defaults: dict, years: list[int], eps: list, base: dict) -> str:
    """Where each EPS figure in the chart and table comes from."""
    cons_years = [y for y, c in zip(years, defaults["eps"]) if c is not None]
    parts = [f"EPS (diluted, $/share, right axis): {base['label']} actual = {base['eps_source']}."]
    if cons_years:
        who = defaults["eps_source"] + (f", {int(defaults['eps_analysts'])} analysts"
                                        if defaults["eps_analysts"] else "")
        span = f"{cons_years[0]}" if len(cons_years) == 1 else f"{cons_years[0]}–{cons_years[-1]}"
        parts.append(f" {span}: analyst consensus average EPS ({who}); consensus is usually on the analysts' "
                     "adjusted basis, so it may not be comparable with reported (GAAP) actual EPS.")
    later = [y for y, c in zip(years, defaults["eps"]) if c is None]
    if later:
        parts.append(f" {', '.join(map(str, later))}: prior-year EPS × (1 + that year's projected EBITDA growth), "
                     "so it follows your assumptions.")
    if any(e is None for e in eps):
        parts.append(" N/A where the prior-year EPS or EBITDA is not positive (a loss is not compounded).")
    return "".join(parts)


def render_dcf(ticker: str, proj: list[dict] | None, years: list[int], net_debt, shares, price) -> float | None:
    """Render the DCF section. Returns the implied price."""
    st.header("Discounted Cash Flow")
    c1, c2 = st.columns(2)
    terminal = c1.slider("Terminal growth (%)", 0.0, 5.0, step=0.25,
                         key=seed_input(ticker, "terminal", 2.5, 0.0, 5.0)) / 100
    wacc = c2.slider("WACC / discount rate (%)", 4.0, 20.0, step=0.25,
                     key=seed_input(ticker, "wacc", 9.0, 4.0, 20.0)) / 100

    if not proj:
        st.warning("No projection model — DCF cannot be computed.")
        return None
    flows = [p["fcf"] for p in proj]
    if not pos(flows[-1]):
        st.warning(f"Projected {years[-1]} free cash flow is not positive, so a Gordon Growth terminal value is "
                   "not meaningful. Adjust the margin / conversion assumptions, or use the P/S and EV/Adj. EBITDA "
                   "methods below.")
        return None
    if not pos(shares):
        st.warning("Shares outstanding unavailable — cannot compute a per-share value.")
        return None
    if wacc <= terminal:
        st.error("WACC must be greater than terminal growth for the Gordon Growth Model.")
        return None
    if net_debt is None:
        st.warning("Cash/debt unavailable — net debt assumed to be zero.")

    r = run_dcf(flows, terminal, wacc, net_debt, shares)
    st.dataframe(pd.DataFrame(
        {
            "Projected FCF": [fmt_money(v) for v in r["flows"]],
            "Discount factor": [f"{1 / d:.4f}" for d in r["factors"]],
            "Present value": [fmt_money(v) for v in r["pv_flows"]],
        },
        index=[f"{y}E" for y in years],
    ).T, width="stretch")

    cols = st.columns(4)
    cols[0].metric("Sum of PV (FCF)", fmt_money(sum(r["pv_flows"])))
    cols[1].metric("Terminal value (PV)", fmt_money(r["pv_tv"]), help=f"Undiscounted: {fmt_money(r['tv'])}")
    cols[2].metric("Enterprise value", fmt_money(r["ev"]))
    cols[3].metric("Net debt", fmt_money(net_debt or 0.0))
    cols = st.columns(4)
    cols[0].metric("Equity value", fmt_money(r["equity"]))
    cols[1].metric("Shares outstanding", f"{shares / 1e6:,.1f}M")
    implied = r["price"]
    up = upside(implied, price)
    cols[2].metric("Implied price", fmt_price(implied),
                   delta=fmt_pct(up, signed=True) + " vs current" if up is not None else None)
    cols[3].metric("Current price", fmt_price(price))
    tv_share = r["pv_tv"] / r["ev"] if r["ev"] > 0 else None
    if tv_share is not None and tv_share > 0.75:
        st.info(f"Terminal value is {tv_share:.0%} of enterprise value — the result is very sensitive "
                "to the terminal growth and WACC assumptions.")
    if implied <= 0:
        st.warning("Net debt exceeds the DCF enterprise value, so the implied equity value is negative.")

    return implied if implied > 0 else None


def render_price_targets(ticker, m, shares, net_debt, y1: dict | None, year1: int, base_fcf, eps_ttm) -> list[tuple]:
    """Render the Price Target Setter. Returns [(method, target price)]."""
    st.header("Price Target Setter")
    price = m["price"]
    profitable = eps_ttm is not None and eps_ttm > 0
    auto = "P/E" if profitable else "P/S"
    st.caption(f"TTM EPS: {fmt_price(eps_ttm)} → auto-selected method: **{auto}** "
               f"({'profitable' if profitable else 'not profitable'}). Revenue, EBITDA and FCF methods use "
               f"your {year1}E projections from the model above.")
    options = ["Auto", "P/E", "P/S"]
    choice = st.radio("Method", options, horizontal=True,
                      format_func=lambda o: f"Auto ({auto})" if o == "Auto" else o,
                      key=seed_input(ticker, "method", "Auto", options=options))
    method = auto if choice == "Auto" else choice
    results: list[tuple] = []

    def show(label: str, target: float | None, formula: str):
        col1, col2 = st.columns([1, 2])
        up = upside(target, price)
        col1.metric(label, fmt_price(target) if target is not None and target > 0 else "N/A",
                    delta=fmt_pct(up, signed=True) + " vs current" if up is not None and target > 0 else None)
        col2.markdown(f"<br/>{formula}", unsafe_allow_html=True)
        if target is not None and target > 0:
            results.append((label.replace(" target", ""), target))
        else:
            st.warning(f"{label}: inputs produce a non-positive price, excluded from the summary.")

    if not pos(shares):
        st.warning("Shares outstanding unavailable — per-share targets cannot be computed.")
        return results

    if method == "P/E":
        default_eps = num(m["eps_fwd"]) if m["eps_fwd"] is not None else (eps_ttm or 0.0)
        default_pe = pos(m["pe_fwd"]) or pos(m["pe_ttm"]) or 15.0
        c1, c2 = st.columns(2)
        eps = c1.number_input("Forecast EPS ($)", step=0.05,
                              key=seed_input(ticker, "pe_eps", round(float(default_eps), 2)),
                              help="Default: consensus forward EPS (falls back to TTM EPS).")
        target_pe = c2.number_input("Target P/E", min_value=0.0, step=0.5,
                                    key=seed_input(ticker, "pe_mult", round(float(default_pe), 1), lo=0.0), help="Default: current forward P/E.")
        show("P/E target", eps * target_pe, f"`{eps:,.2f} EPS × {target_pe:,.1f} P/E`")
    else:
        if not y1 or not pos(y1["revenue"]):
            st.warning("Projected revenue unavailable — P/S method cannot be computed.")
        else:
            target_ps = st.number_input("Target P/S", min_value=0.0, step=0.1,
                                        key=seed_input(ticker, "ps_mult", round(float(pos(m["ps"]) or 2.0), 2),
                                                       lo=0.0), help="Default: current P/S.")
            show("P/S target", y1["revenue"] / shares * target_ps,
                 f"`({fmt_money(y1['revenue'])} {year1}E revenue ÷ {shares / 1e6:,.1f}M shares) "
                 f"× {target_ps:,.2f} P/S`")

        if y1 and pos(y1["ebitda"]):
            st.markdown(f"**EV/Adj. EBITDA method** ({year1}E Adjusted EBITDA is positive)")
            target_mult = st.number_input("Target EV/Adj. EBITDA", min_value=0.0, step=0.5,
                                          key=seed_input(ticker, "ev_mult",
                                                         round(float(pos(m["ev_ebitda"]) or 12.0), 1), lo=0.0), help="Default: current EV / TTM Adjusted EBITDA.")
            target = (y1["ebitda"] * target_mult - (net_debt or 0.0)) / shares
            show("EV/Adj. EBITDA target", target,
                 f"`({fmt_money(y1['ebitda'])} {year1}E Adj. EBITDA × {target_mult:,.1f} − "
                 f"{fmt_money(net_debt or 0)} net debt) ÷ {shares / 1e6:,.1f}M shares`")

    if y1 and pos(y1["fcf"]):
        st.markdown(f"**Price / FCF method** ({year1}E free cash flow is positive)")
        current_pfcf = safe_div(m["market_cap"], base_fcf) if pos(base_fcf) else None
        default_pfcf = current_pfcf if current_pfcf and current_pfcf <= 100 else 20.0
        target_pfcf = st.number_input("Target Price / FCF", min_value=0.0, step=0.5,
                                      key=seed_input(ticker, "pfcf_mult", round(float(default_pfcf), 1), lo=0.0), help="Default: current Price/FCF.")
        show("FCF target", y1["fcf"] / shares * target_pfcf,
             f"`({fmt_money(y1['fcf'])} {year1}E FCF ÷ {shares / 1e6:,.1f}M shares) × {target_pfcf:,.1f} P/FCF`")
    return results


def render_summary(rows: list[tuple], price) -> None:
    st.header("Summary")
    if not rows:
        st.warning("No valuation method produced a positive price target for this company.")
        return
    blended = sum(p for _, p in rows) / len(rows)
    table = pd.DataFrame(
        [{"Method": name, "Target price": fmt_price(p), "Upside / downside": fmt_pct(upside(p, price), signed=True)}
         for name, p in rows]
        + [{"Method": "Blended target (simple average)", "Target price": fmt_price(blended),
            "Upside / downside": fmt_pct(upside(blended, price), signed=True)}]
    ).set_index("Method")
    st.dataframe(table, width="stretch")
    up = upside(blended, price)
    st.metric("Blended target", fmt_price(blended),
              delta=fmt_pct(up, signed=True) + " vs current" if up is not None else None)


def render_how_it_works() -> None:
    with st.expander("How this works"):
        st.markdown("**Data.** The base (Actual) year comes from SEC EDGAR Company Facts (US-GAAP XBRL): the most "
                    "recent fiscal year, keeping only the latest filing for that period so restated or repeated "
                    "figures are not double counted. Price, profile, ratios and analyst consensus come from "
                    "Financial Modeling Prep (or yfinance). Everything is cached for 24 hours.")
        st.markdown("**Base year (Actual)**")
        st.latex(r"FCF_0 = \text{Operating Cash Flow} - \text{Capex}")
        st.latex(r"\text{Adjusted EBITDA} = \text{EBITDA (excl. unusual items)} + "
                 r"\text{Stock-Based Compensation}")
        st.markdown("Adjusted EBITDA is used everywhere EBITDA appears (model, margins, EV/Adj. EBITDA, FCF "
                    "conversion). The EBITDA part is the data provider's normalized EBITDA (FMP or yfinance), "
                    "or its reported EBITDA when no normalized figure exists. For the base year the stock-based "
                    "compensation add-back is SEC EDGAR's ShareBasedCompensation, disclosed in the model "
                    "footnote. EDGAR operating income + D&A is used only when the provider has no EBITDA. "
                    "EPS₀ is reported diluted EPS (EDGAR, else the provider).")
        st.markdown("**Projection model** (t = 1…5; g_t, m_t and c are the editable assumption boxes)")
        st.latex(r"Revenue_t = Revenue_{t-1} \times (1 + g_t)")
        st.latex(r"\text{Adj. EBITDA}_t = Revenue_t \times m_t")
        st.latex(r"FCF_t = \text{Adj. EBITDA}_t \times c")
        st.latex(r"c_{\text{default}} = \frac{\text{TTM FCF}}{\text{TTM Adjusted EBITDA}} "
                 r"\quad \text{(same provider, same four quarters)}")
        st.markdown("**Default assumptions.** Revenue growth: consensus revenue estimate ÷ prior-year revenue. "
                    "EBITDA margin: consensus EBITDA ÷ consensus revenue for that year. Beyond consensus coverage, "
                    f"growth fades linearly toward {LONG_RUN_GROWTH:.0f}% and the last margin carries forward. "
                    "FCF conversion: the company's actual TTM ratio above, uncapped; a warning appears when it "
                    "is above 100% or negative. If the provider has no quarterly data, the base fiscal year's "
                    "FCF ÷ Adjusted EBITDA is used and labeled as such. Without consensus, defaults use the "
                    "latest actual growth and margin.")
        st.markdown("**EPS tracker** (diluted, $ per share)")
        st.latex(r"EPS_t = \text{consensus average EPS}_t \quad \text{(years covered by analysts)}")
        st.latex(r"EPS_t = EPS_{t-1} \times \frac{EBITDA_t}{EBITDA_{t-1}} \quad \text{(later years)}")
        st.markdown("Consensus EPS comes from FMP, else yfinance. EPS is shown as N/A when the prior year's EPS "
                    "or EBITDA is not positive, so a loss is never compounded.")
        st.markdown("**DCF** (r = WACC, g_T = terminal growth)")
        st.latex(r"PV(FCF_t) = \frac{FCF_t}{(1+r)^t}")
        st.latex(r"TV = \frac{FCF_5 (1+g_T)}{r - g_T} \qquad PV(TV) = \frac{TV}{(1+r)^5}")
        st.latex(r"EV = \sum_{t=1}^{5} PV(FCF_t) + PV(TV)")
        st.latex(r"\text{Equity} = EV - (\text{Total Debt} - \text{Cash}) \qquad "
                 r"\text{Implied Price} = \frac{\text{Equity}}{\text{Shares Outstanding}}")
        st.latex(r"\text{Upside} = \frac{\text{Target}}{\text{Current Price}} - 1")
        st.markdown("**Price target methods** (year-1 figures come from the projection model)")
        st.latex(r"\text{P/E target} = \text{Forecast EPS} \times \text{Target P/E}")
        st.latex(r"\text{P/S target} = \frac{Revenue_1}{\text{Shares}} \times \text{Target P/S}")
        st.latex(r"\text{EV/EBITDA target} = \frac{EBITDA_1 \times \text{Target EV/EBITDA} "
                 r"- \text{Net Debt}}{\text{Shares}}")
        st.latex(r"\text{P/FCF target} = \frac{FCF_1}{\text{Shares}} \times \text{Target P/FCF}")
        st.latex(r"\text{Blended} = \frac{1}{n}\sum_{i=1}^{n} \text{Target}_i")
        st.markdown("**Snapshot multiples**")
        st.latex(r"P/E = \frac{Price}{EPS} \quad PEG = \frac{P/E}{\text{EPS growth (\%)}} \quad "
                 r"P/S = \frac{\text{Mkt Cap}}{\text{Revenue}} \quad P/B = \frac{Price}{\text{Book/Share}}")
        st.latex(r"EV = \text{Mkt Cap} + \text{Debt} - \text{Cash} \quad EV/Sales \quad EV/EBITDA \quad "
                 r"\text{FCF Yield} = \frac{FCF}{\text{Mkt Cap}}")
        st.markdown("The auto method is **P/E** when TTM EPS > 0, otherwise **P/S** (plus **EV/EBITDA** when "
                    "year-1 EBITDA > 0). **P/FCF** is added whenever year-1 FCF > 0. Total debt is approximate: "
                    "long-term debt (including current portion) plus short-term borrowings/commercial paper; "
                    "cash excludes marketable securities.")


def render_footer() -> None:
    st.divider()
    render_how_it_works()
    st.caption(f"**{FOOTER}**")


def build_base(history: pd.DataFrame, m: dict, source: str) -> dict:
    """The single Actual year: latest EDGAR fiscal year, with market-data fallbacks."""
    notes: list[str] = []
    edgar_ebitda = sbc_edgar = None
    if not history.empty:
        row = history.iloc[-1]
        edgar_ebitda = num(row.get("EBITDA"))
        sbc_edgar = num(row.get("Stock-Based Compensation"))
        base = {
            "label": history.index[-1],
            "end": pd.Timestamp(row["Period End"]),
            "source": "SEC EDGAR",
            "revenue": num(row.get("Revenue")),
            "fcf": num(row.get("Free Cash Flow")),
            "eps": num(row.get("EPS (diluted)")),
            "eps_source": "reported diluted EPS (SEC EDGAR)",
            "prior_revenue": num(history.iloc[-2]["Revenue"]) if len(history) > 1 else None,
        }
        if base["fcf"] is None and num(m["fcf_ttm"]) is not None:
            base["fcf"] = num(m["fcf_ttm"])
            notes.append("Base FCF not derivable from EDGAR — using trailing-twelve-month FCF from market data.")
    else:
        today = pd.Timestamp(dt.date.today())
        base = {
            "label": "TTM", "end": today, "source": "market data (trailing twelve months)",
            "revenue": num(m["revenue_ttm"]), "fcf": num(m["fcf_ttm"]),
            "eps": num(m["eps_ttm"]), "eps_source": f"trailing-twelve-month EPS ({source})",
            "prior_revenue": None,
        }

    # Provider figures for the same fiscal year (period end within ~3 weeks).
    provider = None
    for row in m.get("annual") or []:
        try:
            if abs((pd.Timestamp(row["date"]) - base["end"]).days) <= 21:
                provider = row
                break
        except ValueError:
            continue

    fy = base["label"]
    ttm = m.get("ttm")
    # Adjusted EBITDA = EBITDA (normalized when available) + stock-based compensation.
    core, core_label = None, None
    if provider and provider.get("ebitda_norm") is not None:
        core, core_label = provider["ebitda_norm"], f"{provider['source']} normalized EBITDA"
    elif provider and provider.get("ebitda") is not None:
        core, core_label = provider["ebitda"], f"{provider['source']} reported EBITDA"
    elif edgar_ebitda is not None:
        core, core_label = edgar_ebitda, "SEC EDGAR operating income + D&A (no provider EBITDA)"
    if sbc_edgar is not None:
        sbc, sbc_label = sbc_edgar, "SEC EDGAR ShareBasedCompensation"
    elif provider and provider.get("sbc") is not None:
        sbc, sbc_label = provider["sbc"], f"{provider['source']} stock-based compensation"
    else:
        sbc, sbc_label = None, None

    if core is not None:
        base["ebitda"] = core + (sbc or 0.0)
        base["ebitda_source"] = f"{core_label} + stock-based compensation ¹"
        if sbc is not None:
            notes.append(f"¹ Adjusted EBITDA {fy} = {core_label} {fmt_money(core)} + stock-based compensation "
                         f"{fmt_money(sbc)} ({sbc_label}) = {fmt_money(base['ebitda'])}.")
        else:
            notes.append(f"¹ Adjusted EBITDA {fy} = {core_label} {fmt_money(core)}; stock-based compensation is "
                         "not reported, so nothing is added back.")
    elif history.empty and ttm:
        base["ebitda"], base["ebitda_source"] = ttm["adj_ebitda"], f"TTM Adjusted EBITDA ({ttm['source']})"
    else:
        base["ebitda"], base["ebitda_source"] = None, "not available"

    # Default FCF conversion: the company's actual TTM ratio, same provider and period, uncapped.
    if ttm and ttm["adj_ebitda"]:
        ratio = ttm["fcf"] / ttm["adj_ebitda"] * 100
        base["conv_calc"] = (f"TTM FCF {fmt_money(ttm['fcf'])} ÷ TTM Adj. EBITDA {fmt_money(ttm['adj_ebitda'])} "
                             f"= {ratio:.1f}% ({ttm['source']}, {ttm['period']}; Adj. EBITDA = "
                             f"{ttm['core_kind']} {fmt_money(ttm['core'])} + SBC {fmt_money(ttm['sbc'])})")
        base["conv_note"] = f"actual TTM FCF ÷ TTM Adjusted EBITDA ({ttm['source']})"
    elif base["fcf"] is not None and base["ebitda"]:
        ratio = base["fcf"] / base["ebitda"] * 100
        base["conv_calc"] = (f"{fy} FCF {fmt_money(base['fcf'])} ÷ {fy} Adj. EBITDA {fmt_money(base['ebitda'])} "
                             f"= {ratio:.1f}% (no quarterly TTM data from the provider; base fiscal year used)")
        base["conv_note"] = f"actual {fy} FCF ÷ Adjusted EBITDA (no TTM data)"
    else:
        ratio = 0.0
        base["conv_calc"] = "Not computable: no TTM or base-year Adjusted EBITDA. Enter your own conversion."
        base["conv_note"] = "not computable — enter your own"
    base["conversion"] = round(ratio, 1)
    base["conv_unusual"] = ratio > 100 or ratio < 0

    if base["eps"] is None and provider and provider["eps_diluted"] is not None:
        base["eps"], base["eps_source"] = provider["eps_diluted"], f"reported diluted EPS ({provider['source']})"
    if base["eps"] is None:
        base["eps_source"] = "not available"
    base["notes"] = notes
    return base


def main() -> str | None:
    """Render the app. Returns the ticker whose inputs should be saved, if any."""
    st.title("📈 DCF Valuation App")

    # Saved inputs live in the browser, which reports them a moment after the
    # first run. Wait for them so widgets start from saved values, not defaults.
    items = stored_items()
    if items is None and not st.session_state.get("_store_skip"):
        st.info("Loading your saved inputs…")
        st.button("Continue without saved inputs", on_click=st.session_state.update, kwargs={"_store_skip": True})
        return None
    if "ticker_query" not in st.session_state:
        st.session_state["ticker_query"] = (items or {}).get(LAST_TICKER_ITEM) or "AAPL"

    query = st.text_input("Search by ticker or company name", key="ticker_query",
                          placeholder="e.g. MSFT, NVDA, Coca-Cola")
    ticker = resolve_ticker(query)
    if not ticker:
        st.info("Enter a ticker symbol or company name to begin.")
        return None

    with st.spinner(f"Loading {ticker}…"):
        m, source, warnings = get_market_data(ticker)
        cik = lookup_cik(ticker)
        edgar: dict = {"history": pd.DataFrame(), "note": "Ticker not found in the SEC company list."}
        if cik is not None:
            try:
                edgar = edgar_fundamentals(cik)
            except Exception as exc:
                edgar = {"history": pd.DataFrame(), "note": f"SEC EDGAR request failed ({type(exc).__name__})."}

    if not m["price"] and edgar["history"].empty:
        st.error(f"Could not find data for **{ticker}**. Check the ticker symbol and try again.")
        return None
    for w in warnings:
        st.warning(w)
    if not m["price"]:
        st.warning("Current price unavailable from market data sources — upside/downside cannot be computed.")
    if edgar["history"].empty:
        st.warning(f"Fundamentals unavailable from SEC EDGAR — the base year uses market data. {edgar.get('note') or ''}")

    history: pd.DataFrame = edgar["history"]
    latest = history.iloc[-1] if not history.empty else pd.Series(dtype="object")

    # ---- Reconcile inputs (EDGAR first for fundamentals, market data as fallback)
    shares = pos(edgar.get("shares")) or pos(m["shares"]) or safe_div(m["market_cap"], m["price"])
    cash = num(edgar.get("cash")) if edgar.get("cash") is not None else num(m["total_cash"])
    debt = num(edgar.get("debt")) if edgar.get("debt") is not None else num(m["total_debt"])
    net_debt = (debt or 0.0) - (cash or 0.0) if cash is not None or debt is not None else None
    base = build_base(history, m, source)
    # EV/EBITDA uses Adjusted EBITDA too: TTM when the provider has quarters, else the base year.
    m["ebitda"] = m["ttm"]["adj_ebitda"] if m.get("ttm") else base["ebitda"]
    m["ev_ebitda"] = None  # recomputed below on Adjusted EBITDA
    revenue_ttm = pos(m["revenue_ttm"]) or pos(base["revenue"])
    eps_ttm = num(m["eps_ttm"])
    if eps_ttm is None:
        eps_ttm = safe_div(latest.get("Net Income"), shares)

    # Fill multiples that the provider did not supply.
    price = m["price"]
    if not m["market_cap"] and price and shares:
        m["market_cap"] = price * shares
    if m["pe_fwd"] is None and pos(m["eps_fwd"]) and price:
        m["pe_fwd"] = price / m["eps_fwd"]
    if m["pe_ttm"] is None and pos(eps_ttm) and price:
        m["pe_ttm"] = price / eps_ttm
    if m["ev"] is None and m["market_cap"]:
        m["ev"] = m["market_cap"] + (net_debt or 0.0)
    if m["ps"] is None:
        m["ps"] = safe_div(m["market_cap"], revenue_ttm)
    if m["ev_sales"] is None:
        m["ev_sales"] = safe_div(m["ev"], revenue_ttm)
    if m["ev_ebitda"] is None and pos(m["ebitda"]):
        m["ev_ebitda"] = safe_div(m["ev"], m["ebitda"])
    if m["peg"] is None and pos(eps_ttm) and pos(m["eps_fwd"]) and pos(m["pe_ttm"]):
        eps_growth = (m["eps_fwd"] / eps_ttm - 1) * 100
        m["peg"] = m["pe_ttm"] / eps_growth if eps_growth > 0 else None
    m["eps_ttm"] = eps_ttm

    # ---- 1. Header
    name = m["name"] or edgar.get("name") or ticker
    st.subheader(f"{name} ({ticker})")
    cols = st.columns(4)
    cols[0].metric("Sector", m["sector"] or "N/A")
    cols[1].metric("Industry", m["industry"] or "N/A")
    cols[2].metric("Current price", fmt_price(price))
    cols[3].metric("Market cap", fmt_money(m["market_cap"]))
    caption = f"Market data: {source}. Fundamentals: SEC EDGAR."
    if m["currency"] and m["currency"] != "USD":
        caption += f" ⚠️ Quote currency is {m['currency']}; EDGAR figures are USD."
    st.caption(caption)

    # ---- 7. Financials warning
    financial = is_financial(m["sector"], m["industry"])
    if financial:
        st.error(
            "### ⚠️ Firm-level DCF does not apply to banks and insurers\n"
            "For banks and insurers, debt and deposits are *raw material* for the business, not financing. "
            "That breaks the WACC and net-debt steps of an enterprise DCF, so the result is not meaningful. "
            "Use **equity models** instead: dividend discount model, free cash flow to equity (FCFE), "
            "residual income, or justified P/B. The multiples sections below still apply. "
            "(For non-bank financials such as payment networks, a DCF may still be reasonable.)"
        )

    # ---- 2. Snapshot
    fcf_for_yield = num(m["fcf_ttm"]) if m["fcf_ttm"] is not None else base["fcf"]
    render_snapshot(m, fcf_for_yield)

    # ---- 3. Projection model (1 actual year + 5 projected years)
    years = [base["end"].year + i for i in range(1, YEARS + 1)]
    yf_est = None
    fmp_est = m.get("estimates") or []
    if not fmp_est or not any(e.get("eps") is not None for e in fmp_est):
        try:
            yf_est = yf_estimates(ticker)
        except Exception:
            yf_est = None
    defaults = build_defaults(base, years, m.get("estimates"), yf_est)
    proj = render_model(ticker, base, defaults, years)

    # ---- 4. DCF
    dcf_price = None
    show_dcf = True
    if financial:
        show_dcf = st.checkbox("Show DCF anyway (not recommended for banks/insurers)", value=False,
                               key=f"force_dcf_{ticker}")
    if show_dcf:
        dcf_price = render_dcf(ticker, proj, years, net_debt, shares, price)
    else:
        st.header("Discounted Cash Flow")
        st.info("DCF hidden for this financial company — see the warning above.")

    # ---- 5. Price targets
    rows: list[tuple] = []
    if dcf_price:
        rows.append(("DCF", dcf_price))
    rows += render_price_targets(ticker, m, shares, net_debt, proj[0] if proj else None, years[0],
                                 base["fcf"], eps_ttm)

    # ---- 6. Summary
    render_summary(rows, price)
    return ticker


storage_slot = st.container()  # fixed position for the (invisible) browser-storage component
shown_ticker = None
try:
    shown_ticker = main()
except Exception as exc:  # last-resort guard: never show a traceback
    st.error(f"Something went wrong while building the valuation: {type(exc).__name__}: {exc}")
finally:
    try:
        persist_inputs(shown_ticker, storage_slot)
    except Exception:
        pass  # saving is best-effort; never break the page
    render_footer()
