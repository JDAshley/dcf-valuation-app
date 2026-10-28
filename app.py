"""DCF Valuation App — Streamlit.

Data: SEC EDGAR Company Facts (historical fundamentals), Financial Modeling Prep
(profile, price, ratios, estimates) with a yfinance fallback.

For education only. Not financial advice.
"""

from __future__ import annotations

import base64
import datetime as dt
import difflib
import functools
import html
import io
import json
import math
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

APP_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "jd-dcf-model-icon.png")


def _page_icon():
    """The JD-DCF icon, scaled down for use as a favicon; falls back to an emoji if the file is missing."""
    try:
        with Image.open(APP_ICON_PATH) as img:
            img.load()
            return img.convert("RGB").resize((192, 192), Image.LANCZOS)
    except OSError:
        return "📈"


st.set_page_config(page_title="JD-DCF Model", page_icon=_page_icon(), layout="wide")


@functools.lru_cache(maxsize=1)
def _logo_data_uri() -> str | None:
    """The logo as a 192px PNG data URI (sharp at 64px on high-DPI screens), or None if unavailable."""
    try:
        with Image.open(APP_ICON_PATH) as img:
            buf = io.BytesIO()
            img.convert("RGB").resize((192, 192), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


HEADER_CSS = """
<style>
.jd-header {
    position: relative; box-sizing: border-box; min-height: 64px;
    display: flex; flex-direction: column; justify-content: center;
    padding: 0.35rem 96px 0.35rem 0;
}
.jd-header .jd-brand {
    color: #b38b22; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; line-height: 1.2;
}
.jd-header h1.jd-title {
    margin: 0; padding: 0; font-size: 2.5rem; font-weight: 700; line-height: 1.15; color: inherit;
}
.jd-header .jd-logo {
    position: absolute; right: 0; top: 50%; transform: translateY(-50%);
    width: 64px; height: 64px; border-radius: 12px; opacity: 1;
}
@media (max-width: 767px) {
    .jd-header { padding: 0.45rem 64px 0.45rem 0; }
    .jd-header h1.jd-title { font-size: 2rem; }
    .jd-header .jd-logo { width: 46px; height: 46px; border-radius: 9px; }
}
</style>
"""


def loading_html(ticker: str) -> str:
    return ('<style>.jd-loading{display:flex;align-items:center;gap:.65rem;padding:1.25rem 0;font-size:1rem}'
            '.jd-loading .spin{width:20px;height:20px;box-sizing:border-box;border-radius:50%;'
            'border:3px solid rgba(128,128,128,.3);border-top-color:#3b82f6;animation:jdspin .8s linear infinite}'
            '@keyframes jdspin{to{transform:rotate(360deg)}}</style>'
            f'<div class="jd-loading" role="status"><span class="spin"></span>'
            f'<span>Loading {html.escape(ticker)}…</span></div>')


def render_header(title: str = "DCF Valuation") -> None:
    """Compact page header: gold brand line and title on the left, JD logo on the right."""
    logo = _logo_data_uri()
    logo_html = f'<img class="jd-logo" src="{logo}" alt="JD-DCF Model logo">' if logo else ""
    st.html(f'{HEADER_CSS}<div class="jd-header"><div class="jd-brand">JD-DCF MODEL</div>'
            f'<h1 class="jd-title">{html.escape(title)}</h1>{logo_html}</div>')

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
    diluted_shares = annual_flow(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding"], unit="shares")
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
    income_tax = annual_flow(facts, ["IncomeTaxExpenseBenefit"])
    pretax = annual_flow(facts, [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ])
    assets_cur = instant_values(facts, ["AssetsCurrent"])
    liab_cur = instant_values(facts, ["LiabilitiesCurrent"])
    debt_cur_parts = [instant_values(facts, ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]),
                      instant_values(facts, ["ShortTermBorrowings", "CommercialPaper"])]
    debt_cur_total = instant_values(facts, ["DebtCurrent"])

    # Fiscal years = period ends of the flow statements; keep the last 5.
    by_year = {d.year: d for d in sorted(set(revenue.index) | set(net_income.index) | set(ocf.index))}
    ends = sorted(by_year.values())[-5:]

    def at(series: pd.Series, d) -> float | None:
        return float(series[d]) if d in series.index else None

    def operating_nwc(d) -> float | None:
        """(current assets − cash) − (current liabilities − current debt) at a balance-sheet date."""
        ca, cl, c = at(assets_cur, d), at(liab_cur, d), at(cash, d)
        if ca is None or cl is None or c is None:
            return None
        if any(d in part.index for part in debt_cur_parts):
            cur_debt = sum(at(part, d) or 0.0 for part in debt_cur_parts)
        else:
            cur_debt = at(debt_cur_total, d) or 0.0
        return (ca - c) - (cl - cur_debt)

    rows = []
    for d in ends:
        o, c = at(ocf, d), at(capex, d)
        oi, da = at(op_income, d), at(dna, d)
        pre, tax_exp, ni = at(pretax, d), at(income_tax, d), at(net_income, d)
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
            "Diluted shares": at(diluted_shares, d),
            "D&A": da,
            "Income tax": tax_exp,
            # Pre-tax income: the reported line, else net income + income tax expense.
            "Pretax income": pre if pre is not None else (ni + tax_exp if ni is not None and tax_exp is not None else None),
            "Pretax derived": pre is None and ni is not None and tax_exp is not None,
            "Operating NWC": operating_nwc(d),
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
            "revenue_high": num(_pick(e, "revenueHigh", "estimatedRevenueHigh")),
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
             "diluted_shares": num(_pick(i, "weightedAverageShsOutDil")),
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
         "eps_diluted": get(inc, "Diluted EPS", col), "diluted_shares": get(inc, "Diluted Average Shares", col),
         "sbc": get(cf, "Stock Based Compensation", col),
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
        out["revenue"].append({"growth": avg / base - 1, "analysts": num(row.get("numberOfAnalysts")),
                               "avg": avg, "high": pos(row.get("high")), "ref": base})
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
    background: linear-gradient(rgba(29, 78, 216, 0.05), rgba(29, 78, 216, 0.05)), var(--jd-base, transparent);
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


# Phones (< 768px) get compact two-column tiles; desktop keeps the Streamlit metric rows.
# Colours are translucent greys over the inherited text colour, so tiles suit light and dark themes.
MOBILE_CSS = """
<style>
/* Hide the inactive variant together with its layout wrapper, so it leaves no gap. */
.st-key-mob_header, .st-key-mob_snapshot, .st-key-mob_dcf, .st-key-mob_chart,
[data-testid="stLayoutWrapper"]:has(> .st-key-mob_header),
[data-testid="stLayoutWrapper"]:has(> .st-key-mob_snapshot),
[data-testid="stLayoutWrapper"]:has(> .st-key-mob_dcf),
[data-testid="stLayoutWrapper"]:has(> .st-key-mob_chart) { display: none !important; }
@media (max-width: 767px) {
    .st-key-desk_header, .st-key-desk_snapshot, .st-key-desk_dcf, .st-key-desk_chart,
    [data-testid="stLayoutWrapper"]:has(> .st-key-desk_header),
    [data-testid="stLayoutWrapper"]:has(> .st-key-desk_snapshot),
    [data-testid="stLayoutWrapper"]:has(> .st-key-desk_dcf),
    [data-testid="stLayoutWrapper"]:has(> .st-key-desk_chart) { display: none !important; }
    .st-key-mob_header, .st-key-mob_snapshot, .st-key-mob_dcf, .st-key-mob_chart { display: flex !important; }
    [data-testid="stLayoutWrapper"]:has(> .st-key-mob_header),
    [data-testid="stLayoutWrapper"]:has(> .st-key-mob_snapshot),
    [data-testid="stLayoutWrapper"]:has(> .st-key-mob_dcf),
    [data-testid="stLayoutWrapper"]:has(> .st-key-mob_chart) { display: flex !important; }
}
/* Desktop chart legend: one centred row; if it must wrap, two balanced rows of four. */
.chart-legend {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem 0.75rem;
    font-size: 0.72rem; line-height: 1.2; margin-top: -0.5rem;
}
.chart-legend .half { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem 0.75rem; }
.chart-legend .item { display: inline-flex; align-items: center; gap: 0.3rem; white-space: nowrap; }
/* Swatches are CSS-drawn: st.html's sanitizer strips inline SVG. */
.chart-legend .sw { flex: none; width: 14px; height: 14px; border-radius: 2px; }
.chart-legend .dot { flex: none; width: 12px; height: 12px; border-radius: 50%; border: 1.5px solid #78633f; }
.chart-legend .eps-line { flex: none; position: relative; width: 36px; height: 14px; }
.chart-legend .eps-line .dash { position: absolute; left: 0; right: 0; top: 6px; border-top: 2.5px dashed; }
.chart-legend .eps-line .ring {
    position: absolute; left: 12px; top: 1px; width: 12px; height: 12px; box-sizing: border-box;
    border-radius: 50%; border: 2.5px solid; background: transparent;
}
.tile-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.5rem;
    width: 100%;
}
.tile {
    min-width: 0;
    min-height: 4.4rem;
    box-sizing: border-box;
    padding: 0.6rem 0.75rem;
    border: 1px solid rgba(128, 128, 128, 0.30);
    border-radius: 0.6rem;
    background: linear-gradient(rgba(128, 128, 128, 0.08), rgba(128, 128, 128, 0.08)), var(--jd-base, transparent);
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.15rem;
}
.tile.wide { grid-column: 1 / -1; }
.tile.emph { border-color: rgba(59, 130, 246, 0.60); background: rgba(59, 130, 246, 0.12); }
.tile .lbl {
    font-size: 0.78rem; line-height: 1.2; opacity: 0.75;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.tile .val {
    font-size: 1.3rem; font-weight: 600; line-height: 1.25; font-variant-numeric: tabular-nums;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.tile.emph .val { font-size: 1.75rem; }
.tile .delta { font-size: 0.82rem; font-weight: 600; }
.tile .delta.up { color: #16a34a; }
.tile .delta.down { color: #dc2626; }
</style>
"""


def desktop_legend_html() -> str:
    """HTML legend for the desktop chart, matching the trace colours and EPS line styles."""
    def square(color):
        return f'<span class="sw" style="background-color: {color}"></span>'

    eps_actual = f'<span class="dot" style="background-color: {EPS_COLORS[0]}"></span>'
    eps_projected = (f'<span class="eps-line" style="color: {EPS_COLORS[1]}">'
                     '<span class="dash"></span><span class="ring"></span></span>')
    entries = [
        ("Actual Revenue", square(ACTUAL_COLORS[0])), ("Projected Revenue", square(PROJECTED_COLORS[0])),
        ("Actual Adj. EBITDA", square(ACTUAL_COLORS[1])), ("Projected Adj. EBITDA", square(PROJECTED_COLORS[1])),
        ("Actual Free Cash Flow", square(ACTUAL_COLORS[2])), ("Projected Free Cash Flow", square(PROJECTED_COLORS[2])),
        ("Actual EPS (diluted)", eps_actual), ("Projected EPS (diluted, dashed)", eps_projected),
    ]
    items = [f'<span class="item">{icon}<span>{html.escape(label)}</span></span>' for label, icon in entries]
    return (f'<div class="chart-legend"><div class="half">{"".join(items[:4])}</div>'
            f'<div class="half">{"".join(items[4:])}</div></div>')


def tile_grid(tiles: list[tuple], grid_cls: str = "") -> str:
    """HTML for a two-column tile grid. Each tile: (label, value[, css class[, delta %]])."""
    cells = []
    for label, value, *rest in tiles:
        cls = rest[0] if rest else ""
        delta = rest[1] if len(rest) > 1 else None
        delta_html = ""
        if delta is not None:
            delta_html = (f'<span class="delta {"up" if delta >= 0 else "down"}">'
                          f'{html.escape(fmt_pct(delta, signed=True))} vs current</span>')
        cells.append(f'<div class="tile {cls}"><span class="lbl">{html.escape(label)}</span>'
                     f'<span class="val">{html.escape(value)}</span>{delta_html}</div>')
    return f'<div class="tile-grid {grid_cls}">{"".join(cells)}</div>'


def fade(start: float, n: int) -> list[float]:
    """n growth rates moving linearly from `start` toward LONG_RUN_GROWTH."""
    return [start + (LONG_RUN_GROWTH - start) * k / (n + 1) for k in range(1, n + 1)]


def build_defaults(base: dict, years: list[int], fmp_estimates: list | None, yf_est: dict | None) -> dict:
    """Default assumptions (all in %): consensus where available, otherwise latest actuals."""
    growth: list[float] = []
    margin: list[float] = []
    levels: list[dict] = []   # estimate levels behind each consensus growth year (for the upper-range mode)
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
            levels.append({"avg": e["revenue"], "high": pos(e.get("revenue_high")), "ref": prev})
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
        levels = [{"avg": e.get("avg"), "high": e.get("high"), "ref": e.get("ref")} for e in yf_est["revenue"]][:YEARS]
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
        "levels": levels[:n_growth],
        "source": source,
    }


UPPER_MARGIN_CAP = 1.5  # max margin expansion per year in the upper-range mode (percentage points)


def margin_trend(history: pd.DataFrame, m: dict) -> float | None:
    """Average annual change in the Adjusted EBITDA margin over up to the last 3 fiscal years
    (provider EBITDA — normalized when available — plus SBC, over EDGAR revenue)."""
    margins = []
    for _, row in history.iterrows():
        rev, end = pos(row.get("Revenue")), pd.Timestamp(row["Period End"])
        prov = None
        for a in m.get("annual") or []:
            try:
                if abs((pd.Timestamp(a["date"]) - end).days) <= 21:
                    prov = a
                    break
            except (KeyError, ValueError):
                continue
        core = prov.get("ebitda_norm") if prov and prov.get("ebitda_norm") is not None else (prov or {}).get("ebitda")
        sbc = num(row.get("Stock-Based Compensation"))
        if sbc is None and prov:
            sbc = prov.get("sbc")
        margins.append(((core + (sbc or 0.0)) / rev * 100) if rev and core is not None else None)
    changes = [b - a for a, b in zip(margins, margins[1:]) if a is not None and b is not None][-3:]
    return sum(changes) / len(changes) if changes else None


def upper_range_defaults(base: dict, years: list[int], cons: dict, trend: float | None) -> dict:
    """Upper-range mode: the two analyst-estimate years use avg + 0.5 × (high − avg) (≈75th percentile),
    out-years fade from Year 2; margins step up from the base-year margin by the company's own recent
    trend (capped at +1.5 pts/yr), or stay flat if the trend is not positive."""
    upper = [lv for lv in cons["levels"][:2] if lv.get("avg") and lv.get("high") and lv.get("ref")]
    growth, shown = list(cons["growth"]), []
    if upper and len(upper) == len(cons["levels"][:2]):
        prev = upper[0]["ref"]
        g = []
        for lv in upper:
            level = lv["avg"] + 0.5 * (lv["high"] - lv["avg"])
            g.append((level / prev - 1) * 100)
            shown.append((level, lv["avg"], lv["high"]))
            prev = level
        growth = g + fade(g[-1], YEARS - len(g))
    base_margin = (safe_div(base["ebitda"], base["revenue"]) or 0.0) * 100
    step = min(trend, UPPER_MARGIN_CAP) if trend is not None and trend > 0 else 0.0
    margin = [base_margin + step * (k + 1) for k in range(YEARS)]
    if shown:
        est = "; ".join(f"{years[i]} revenue ${lv / 1e6:,.1f}M (avg ${a / 1e6:,.1f}M, high ${h / 1e6:,.1f}M)"
                        for i, (lv, a, h) in enumerate(shown))
        note = (f"**Upper-range default (≈75th percentile of analyst estimates)** for revenue growth: {est}; later years "
                f"fade toward {LONG_RUN_GROWTH:.0f}%.")
    else:
        note = "**Upper-range mode:** no analyst estimate range is available, so revenue growth matches the consensus defaults."
    note += (f" Adj. EBITDA margins expand {step:.2f} pts a year from the {base_margin:.1f}% base (the company's average "
             f"annual margin change of {trend:+.2f} pts over up to 3 years, capped at +{UPPER_MARGIN_CAP} pts)."
             if step > 0 else
             f" Adj. EBITDA margins stay flat at the {base_margin:.1f}% base-year margin "
             + ("(no positive margin trend)." if trend is not None else "(margin history unavailable)."))
    return {"growth": [round(x, 1) for x in growth], "margin": [round(x, 1) for x in margin], "note": note}


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


def bound(key: str) -> dict:
    """Widget kwargs: the key plus an explicit value equal to its session state.

    A widget created without ``value=`` declares its built-in default (a slider's minimum, 0.0
    for a number box, "" for a text box). The app computes with the session-state value, but a
    freshly mounted widget displays that declared default, so the screen could show 4.00% WACC
    while the DCF used 9.00%. Declaring the real value keeps the two identical. Keyed sliders,
    number and text inputs exclude ``value`` from their identity, so this never resets them."""
    return {"key": key, "value": st.session_state[key]}


# Inputs that are saved per ticker; older saved names (e.g. a removed method radio) are dropped.
SAVED_INPUTS = {*(f"g{i}" for i in range(5)), *(f"m{i}" for i in range(5)), *(f"s{i}" for i in range(5)),
                *(f"d{i}" for i in range(5)), *(f"c{i}" for i in range(5)), *(f"w{i}" for i in range(5)),
                "tax", "dmode", "build", "sbc_method", "conv", "wacc", "terminal",
                "pe_mult", "ps_mult", "ev_mult", "pfcf_mult"}


MODE_CHOICES = ("dmode", "build")  # the user's mode choices survive "Reset to defaults"


def reset_inputs(ticker: str) -> None:
    """Restore every input for this ticker to the active mode's defaults and drop the saved copy."""
    for name, default in st.session_state.get(f"_defaults_{ticker}", {}).items():
        if name not in MODE_CHOICES:
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
            kept = {n: v for n, v in (saved or {}).get("values", {}).items() if n in SAVED_INPUTS}
            values = current if reset or not saved else {**kept, **current}
            item = f"dcf_inputs_{ticker}"
            if all(n in defaults and _same(v, defaults[n]) for n, v in values.items()):
                if item in items:
                    writes[item] = None
            elif not saved or values.keys() != saved["values"].keys() or not all(
                    _same(v, saved["values"][n]) for n, v in values.items()):
                saved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                writes[item] = json.dumps({"saved_at": saved_at, "values": values})
    if items is not None:
        writes.update(global_writes(items))
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


def project_detailed(base_revenue: float, growth, margin, da_pct, capex_pct, nwc_pct, tax_rate) -> list[dict]:
    """Detailed build, per year: Revenue → Adj. EBITDA → EBIT (− D&A) → NOPAT (tax on EBIT, no floor:
    a negative EBIT gives a tax benefit) → Unlevered FCF = NOPAT + D&A − Capex − ΔNWC."""
    rows, revenue, t = [], base_revenue, tax_rate / 100
    for i in range(len(growth)):
        revenue = revenue * (1 + growth[i] / 100)
        ebitda = revenue * margin[i] / 100
        da = revenue * da_pct[i] / 100
        ebit = ebitda - da
        tax = ebit * t
        nopat = ebit - tax
        capex = revenue * capex_pct[i] / 100
        dnwc = revenue * nwc_pct[i] / 100
        rows.append({"revenue": revenue, "ebitda": ebitda, "da": da, "ebit": ebit, "tax": tax, "nopat": nopat,
                     "capex": capex, "dnwc": dnwc, "fcf": nopat + da - capex - dnwc})
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
    fig.add_annotation(x=0.5, y=1.0, yref="paper", yanchor="bottom", text="Projections begin →",
                       showarrow=False, xanchor="left", font=dict(size=11))
    axis_font = dict(size=10)
    fig.update_layout(
        barmode="group", height=440, margin=dict(t=28, b=8, l=6, r=6),
        dragmode=False,
        xaxis=dict(fixedrange=True, tickangle=0, tickfont=axis_font, automargin=True),
        yaxis=dict(title=dict(text="USD billions", font=dict(size=11), standoff=4), range=y1, zeroline=True,
                   zerolinecolor="#6b7280", fixedrange=True, tickfont=axis_font, automargin=True),
        yaxis2=dict(title=dict(text="EPS ($)", font=dict(size=11), standoff=4), overlaying="y", side="right",
                    range=y2, showgrid=False, zeroline=True, zerolinecolor="#6b7280", tickformat="$,.2f",
                    fixedrange=True, tickfont=axis_font, automargin=True),
        # Legend below the plot, one row per group (Actual / Projected).
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.17, yanchor="top", font=dict(size=11),
                    groupclick="toggleitem", tracegroupgap=4, entrywidth=0, itemsizing="constant"),
    )
    return fig


# No modebar, no zoom/pan: a touch-drag over the chart scrolls the page. Hover still works.
CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "responsive": True}


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


def render_snapshot(m: dict, fcf_for_yield: float | None, price) -> None:
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
    with st.container(key="desk_snapshot"):
        for start in range(0, len(items), 5):
            cols = st.columns(5)
            for col, (label, value) in zip(cols, items[start:start + 5]):
                col.metric(label, value)
    with st.container(key="mob_snapshot"):
        tiles = [("Current price", fmt_price(price)), ("Market cap", fmt_money(m["market_cap"]))]
        tiles += [("PEG ratio" if label == "PEG" else label, value) for label, value in items[:-1]]
        tiles.append((items[-1][0], items[-1][1], "wide"))  # Dividend Yield spans both columns
        st.html(tile_grid(tiles))
    st.caption("Multiples show N/A when the underlying earnings, EBITDA or book value is negative or unavailable.")


DEFAULT_MODES = ["Consensus", "Upper-range"]
BUILD_MODES = ["Detailed", "Simple"]


def switch_defaults_mode(ticker: str, by_mode: dict) -> None:
    """Mode switch: re-default only boxes still at the previous mode's default; edited boxes are kept."""
    new = st.session_state[input_key(ticker, "dmode")]
    old = next(m for m in DEFAULT_MODES if m != new)
    for kind in ("growth", "margin"):
        for i in range(YEARS):
            key = input_key(ticker, f"{kind[0]}{i}")
            if key in st.session_state and _same(st.session_state[key], by_mode[old][kind][i]):
                st.session_state[key] = by_mode[new][kind][i]


def render_model(ticker: str, base: dict, defaults: dict, years: list[int], sbc_rows: list[dict] | None,
                 market_cap, upper: dict | None = None, share_info: dict | None = None
                 ) -> tuple[list[dict], list, list | None] | None:
    """Assumption boxes, SBC projection, calculated table and bar chart.
    Returns (projected years, projected EPS, projected SBC $ per year or None)."""
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
        st.caption(md_escape(note))

    comp = base.get("components") or {"available": False}
    by_mode = {"Consensus": {"growth": defaults["growth"], "margin": defaults["margin"]},
               "Upper-range": upper or {"growth": defaults["growth"], "margin": defaults["margin"]}}
    # Seed every box (saved value first, else the active mode's default) before any widget is created.
    dmode_key = seed_input(ticker, "dmode", "Consensus", options=DEFAULT_MODES)
    build_key = seed_input(ticker, "build", "Detailed" if comp["available"] else "Simple", options=BUILD_MODES)
    active = by_mode[st.session_state[dmode_key]]
    for i in range(YEARS):
        seed_input(ticker, f"g{i}", active["growth"][i])
        seed_input(ticker, f"m{i}", active["margin"][i])
        if comp["available"]:
            seed_input(ticker, f"d{i}", round(comp["da_pct"], 2))
            seed_input(ticker, f"c{i}", round(comp["capex_pct"], 2))
            seed_input(ticker, f"w{i}", round(comp["nwc_pct"], 2))
    if comp["available"]:
        seed_input(ticker, "tax", round(comp["tax_rate"], 1))
    seed_input(ticker, "conv", defaults["conversion"])
    detailed = st.session_state[build_key] == "Detailed" and comp["available"]
    # Streamlit drops the state of widgets that are not drawn, so carry the hidden build's boxes across reruns.
    hidden = ["conv"] if detailed else [f"{p}{i}" for p in "dcw" for i in range(YEARS)] + ["tax"] if comp["available"] else []
    for k in hidden:
        st.session_state[input_key(ticker, k)] = st.session_state[input_key(ticker, k)]

    st.markdown(ASSUMPTION_CSS, unsafe_allow_html=True)
    with st.container(border=True, key="assumptions"):
        top = st.columns([3, 1], vertical_alignment="center")
        top[0].subheader("Projection Assumptions — edit any box to run your own case")
        top[1].button("Reset to defaults", on_click=reset_inputs, args=(ticker,), width="stretch",
                      help="Restores every input for this ticker to the active mode's defaults and deletes your saved inputs.")
        modes = st.columns(2, vertical_alignment="bottom")
        modes[0].segmented_control("Defaults", DEFAULT_MODES, required=True, key=dmode_key,
                                   default=st.session_state[dmode_key], on_change=switch_defaults_mode,
                                   args=(ticker, by_mode),
                                   help="Consensus: analyst averages. Upper-range: ≈75th percentile of analyst revenue "
                                        "estimates and the company's own margin-improvement trend.")
        modes[1].segmented_control("Build", BUILD_MODES, required=True, key=build_key,
                                   default=st.session_state[build_key],
                                   help="Detailed: EBITDA → EBIT → NOPAT → unlevered FCF from D&A, capex, ΔNWC and tax. "
                                        "Simple: FCF = Adj. EBITDA × FCF conversion %.")
        if not comp["available"]:
            st.caption("Detailed build unavailable for this ticker (SEC EDGAR lacks base-year D&A or capex), so the "
                       "Simple (FCF conversion) build is used.")
        restored = st.session_state.get(f"_restored_{ticker}")
        if restored is not None:
            st.caption(f"💾 Restored your saved assumptions for this ticker (saved {fmt_saved_at(restored)}).")
        # Year boxes sit in a grid: 5 across on desktop; CSS wraps them 3 + 2 on phones.
        for label, kind, help_text in (
            ("Revenue growth %", "g", "Projected revenue = prior-year revenue × (1 + growth %)."),
            ("Adj. EBITDA margin %", "m",
             "Projected Adjusted EBITDA = projected revenue × Adjusted EBITDA margin %."),
        ):
            with st.container(key=f"row_{kind}"):  # label left of the grid on desktop, above it on phones
                st.markdown(f"**{label}**", help=help_text)
                with st.container(key=f"grid_{kind}"):
                    for i, col in enumerate(st.columns(YEARS)):
                        col.number_input(f"{years[i]}E", step=0.5, format="%.1f",
                                         **bound(input_key(ticker, f"{kind}{i}")))
        if detailed:
            for label, kind, help_text in (
                ("D&A % of revenue", "d", "D&A = revenue × D&A %. Deducted to get EBIT, then added back to NOPAT."),
                ("Capex % of revenue", "c", "Capex = revenue × capex %, deducted from FCF."),
                ("ΔNWC % of revenue", "w", "Change in net working capital = revenue × ΔNWC %, deducted from FCF. "
                                          "Enter a negative % for working capital releasing cash."),
            ):
                with st.container(key=f"row_{kind}"):
                    st.markdown(f"**{label}**", help=help_text)
                    with st.container(key=f"grid_{kind}"):
                        for i, col in enumerate(st.columns(YEARS)):
                            col.number_input(f"{years[i]}E", step=0.1, format="%.2f",
                                             **bound(input_key(ticker, f"{kind}{i}")))
            with st.container(key="conv_row"):  # same compact single-box row style as the conversion box
                cols = st.columns([1.6, 1, 3], vertical_alignment="center")
                cols[0].markdown("**Tax rate %**", help="Applied once to EBIT, with no floor: a negative EBIT gives a tax benefit.")
                cols[1].number_input("Tax rate % (all years)", step=0.5, format="%.1f", label_visibility="collapsed",
                                     **bound(input_key(ticker, "tax")))
        else:
            with st.container(key="conv_row"):
                cols = st.columns([1.6, 1, 3], vertical_alignment="center")
                cols[0].markdown("**FCF conversion %**",
                                 help="FCF ÷ Adjusted EBITDA. Projected FCF = projected Adjusted EBITDA × conversion %.")
                cols[1].number_input("FCF conversion % (all years)", step=1.0, format="%.1f", label_visibility="collapsed",
                                     **bound(input_key(ticker, "conv")))
            st.caption(md_escape(base["conv_calc"]))
            conv_now = st.session_state[input_key(ticker, "conv")]
            if base["conv_unusual"]:
                st.warning(f"⚠️ The company's actual conversion is {base['conversion']:.1f}%, which is unusually "
                           f"{'high' if base['conversion'] > 100 else 'low'} and may not be sustainable. "
                           "Consider entering a long-run rate instead.")
            if conv_now > 100 and not _same(conv_now, defaults["conversion"]):
                st.warning("FCF conversion above 100% means free cash flow exceeds Adjusted EBITDA every year. That can "
                           "happen briefly (e.g. working-capital releases) but is not sustainable long-term.")
        save_note = (" Your changes save automatically in this browser." if storage_available()
                     else " Browser storage is unavailable, so changes will not be saved.")
        caption = defaults["caption"]
        if st.session_state[dmode_key] == "Upper-range" and upper:
            caption = upper["note"] + " Consensus reference: " + caption.replace("**", "").replace("Defaults: ", "", 1)
        if detailed:
            caption += (f" Detailed build defaults (flat across years): D&A {comp['da_pct']:.2f}% and capex "
                        f"{comp['capex_pct']:.2f}% of revenue ({base['label']}, SEC EDGAR); ΔNWC {comp['nwc_pct']:.2f}% "
                        f"({comp['nwc_note']}); tax {comp['tax_rate']:.1f}% ({comp['tax_note']}). These do not change "
                        "with the defaults mode.")
        st.caption(md_escape(caption + save_note))

    growth = [st.session_state[input_key(ticker, f"g{i}")] for i in range(YEARS)]
    margin = [st.session_state[input_key(ticker, f"m{i}")] for i in range(YEARS)]
    conversion = st.session_state[input_key(ticker, "conv")]
    if detailed:
        read = lambda kind: [st.session_state[input_key(ticker, f"{kind}{i}")] for i in range(YEARS)]
        proj = project_detailed(base["revenue"], growth, margin, read("d"), read("c"), read("w"),
                                st.session_state[input_key(ticker, "tax")])
        diluted = (share_info or {}).get("count")
        eps = [p["nopat"] / diluted if pos(diluted) else None for p in proj]  # EPS = NOPAT ÷ diluted shares
    else:
        proj = project(base["revenue"], growth, margin, conversion)
        eps = project_eps(base["eps"], base["ebitda"], proj, defaults["eps"])
    if sbc_rows is None:  # gated: not a software company under $20B
        st.caption(SBC_GATE_NOTE)
        sbc = None
    else:
        sbc = render_sbc(ticker, base, years, proj, sbc_rows, market_cap)

    labels = [f"{base['label']} Actual"] + [f"{y}E" for y in years]
    chart_labels = [f"{base['label']}<br>Actual"] + labels[1:]  # two-line label keeps ticks horizontal
    base_margin = safe_div(base["ebitda"], base["revenue"])
    if detailed:
        actual = base_detailed_actuals(base, comp)
        chart_base = {**base, "fcf": actual["fcf"] if actual["fcf"] is not None else base["fcf"]}
        rows = [("Revenue", "revenue"), ("Adjusted EBITDA", "ebitda"), ("Less: D&A", "da"), ("EBIT", "ebit"),
                ("Less: Tax", "tax"), ("NOPAT", "nopat"), ("Plus: D&A", "da"), ("Less: Capex", "capex"),
                ("Less: Change in NWC", "dnwc"), ("Unlevered FCF", "fcf")]
        table = pd.DataFrame({name: [fmt_money(actual[k])] + [fmt_money(p[k]) for p in proj] for name, k in rows},
                             index=labels).T
    else:
        chart_base = base
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
    if sbc:
        base_sbc = sbc_rows[-1]["sbc"] if sbc_rows and sbc_rows[-1]["end"] == base["end"] else None
        table.loc["Stock-Based Compensation"] = [fmt_money(base_sbc)] + [fmt_money(v) for v in sbc]
        actual_fcf = chart_base["fcf"]
        base_after = actual_fcf - base_sbc if actual_fcf is not None and base_sbc is not None else None
        table.loc["FCF after SBC"] = [fmt_money(base_after)] + [fmt_money(p["fcf"] - v) for p, v in zip(proj, sbc)]
    if detailed:
        table.loc["Adj. EBITDA margin %"] = [fmt_pct(base_margin)] + [fmt_pct(p["ebitda"] / p["revenue"]) for p in proj]
        table.loc["EPS (NOPAT ÷ diluted shares)"] = [fmt_price(base["eps"])] + [fmt_price(e) for e in eps]
    st.dataframe(table, width="stretch", height=(len(table) + 1) * 35 + 3,  # every row visible, no inner scroll
                 column_config={"_index": st.column_config.Column(width=215)})  # longest row label fits
    if detailed:
        st.caption(f"Actual column: {base['source']}, period ending {base['end'].date().isoformat()}. Revenue, D&A, capex "
                   f"and the change in operating working capital are reported; EBIT, tax and NOPAT are derived with the "
                   f"{comp['tax_rate']:.1f}% default tax rate. Actual Adjusted EBITDA: {base['ebitda_source']}. "
                   f"Actual EPS: {base['eps_source']}.")
    else:
        st.caption(f"Actual column: {base['source']}, period ending {base['end'].date().isoformat()}. "
                   f"Actual FCF = operating cash flow − capex. Actual Adjusted EBITDA: {base['ebitda_source']}. "
                   f"Actual EPS: {base['eps_source']}.")
    try:
        import plotly.graph_objects as go

        fig = model_chart(chart_labels, chart_base, proj, eps)
        # Desktop: same figure without Plotly's grouped legend, plus a horizontal HTML legend.
        with st.container(key="desk_chart"):
            st.plotly_chart(go.Figure(fig).update_layout(showlegend=False), width="stretch",
                            config=CHART_CONFIG, key="chart_desktop")
            st.html(desktop_legend_html())
        # Phones: the figure as is, with its grouped legend below the plot.
        with st.container(key="mob_chart"):
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key="chart_mobile")
    except ImportError:
        st.warning("Install plotly (`pip install plotly`) to see the chart.")
    if detailed:
        cnt = (share_info or {}).get("count")
        st.caption(f"EPS (diluted, $/share, right axis): {base['label']} actual = {base['eps_source']}. Projected EPS = "
                   f"NOPAT ÷ diluted shares" + (f" ({cnt / 1e6:,.1f}M, {share_info['source']})." if pos(cnt) else
                                               " — unavailable (no share count)."))
    else:
        st.caption(eps_caption(defaults, years, eps, base))
    return proj, eps, sbc


def base_detailed_actuals(base: dict, comp: dict) -> dict:
    """Base-year column for the detailed build: reported revenue, EBITDA, D&A, capex and ΔNWC; EBIT, tax and
    NOPAT derived with the default tax rate; unlevered FCF from those."""
    out = {"revenue": base["revenue"], "ebitda": base["ebitda"], "da": comp["da"], "capex": comp["capex"],
           "dnwc": comp.get("dnwc")}
    if base["ebitda"] is not None:
        out["ebit"] = base["ebitda"] - comp["da"]
        out["tax"] = out["ebit"] * comp["tax_rate"] / 100
        out["nopat"] = out["ebit"] - out["tax"]
        out["fcf"] = out["nopat"] + comp["da"] - comp["capex"] - (comp.get("dnwc") or 0.0)
    else:
        out.update(ebit=None, tax=None, nopat=None, fcf=None)
    return out


def ntm_figures(base_end: pd.Timestamp, years: list[int], proj: list[dict], eps: list,
                today: dt.date | None = None) -> dict | None:
    """Next-twelve-months figures from the model's own projections.

    The window runs from today to today + 12 months. Projected fiscal year i covers
    (base end + i-1 years, base end + i years]; each year's projected value is weighted by the
    share of the window it overlaps (time-weighted by days)."""
    start = pd.Timestamp(today or dt.date.today())
    end = start + pd.DateOffset(months=12)
    span = (end - start).days
    parts = []
    for i in range(len(proj)):
        fy_start, fy_end = base_end + pd.DateOffset(years=i), base_end + pd.DateOffset(years=i + 1)
        days = (min(end, fy_end) - max(start, fy_start)).days
        if days > 0:
            parts.append((i, days / span))
    if not parts or abs(sum(w for _, w in parts) - 1) > 0.01:
        return None  # window not covered by the projection years
    out = {"end": end, "parts": [(years[i], w) for i, w in parts]}
    for k in ("revenue", "ebitda", "fcf"):
        out[k] = sum(w * proj[i][k] for i, w in parts)
    out["eps"] = sum(w * eps[i] for i, w in parts) if all(eps[i] is not None for i, _ in parts) else None
    return out


def ntm_basis_text(ntm: dict) -> str:
    """e.g. "12-month forward basis (to Oct 2027): 17% FY2026E + 83% FY2027E" (weights sum to 100%)."""
    pcts = [round(w * 100) for _, w in ntm["parts"]]
    pcts[-1] = 100 - sum(pcts[:-1])
    mix = " + ".join(f"{pct}% FY{year}E" for (year, _), pct in zip(ntm["parts"], pcts))
    return f"12-month forward basis (to {ntm['end'].strftime('%b %Y')}): {mix}"

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


def dcf_share_count(history: pd.DataFrame, base: dict, m: dict, shares_outstanding) -> dict:
    """Shares the DCF divides by: FY weighted-average DILUTED shares from SEC EDGAR, else the provider's
    figure for the same fiscal year, else current shares outstanding (labelled as such). Reported figures
    are used as-is — in loss years diluted equals basic, and no higher count is invented."""
    fy = base["label"]
    if not history.empty and pos(history.iloc[-1].get("Diluted shares")):
        return {"count": float(history.iloc[-1]["Diluted shares"]), "diluted": True,
                "source": f"{fy} weighted-average diluted shares, SEC EDGAR"}
    for row in m.get("annual") or []:
        try:
            same_year = abs((pd.Timestamp(row["date"]) - base["end"]).days) <= 21
        except (KeyError, ValueError):
            continue
        if same_year and pos(row.get("diluted_shares")):
            return {"count": float(row["diluted_shares"]), "diluted": True,
                    "source": f"{fy} weighted-average diluted shares, {row['source']}"}
    return {"count": pos(shares_outstanding), "diluted": False, "source": "current shares outstanding"}


def shares_used_text(info: dict) -> str:
    if info["diluted"]:
        return f"Shares used (diluted): {info['count'] / 1e6:,.1f}M"
    return f"Shares used: {info['count'] / 1e6:,.1f}M (diluted unavailable — shares outstanding used)"


def render_dcf(ticker: str, proj: list[dict] | None, years: list[int], net_debt, share_info: dict, price,
               sbc: list | None = None, shares_outstanding=None) -> float | None:
    """Render the DCF section. Returns the implied price (equity value ÷ diluted shares)."""
    st.header("Discounted Cash Flow")
    c1, c2 = st.columns(2)
    terminal = c1.slider("Terminal growth (%)", 0.0, 5.0, step=0.25,
                         **bound(seed_input(ticker, "terminal", 2.5, 0.0, 5.0))) / 100
    wacc = c2.slider("WACC / discount rate (%)", 4.0, 20.0, step=0.25,
                     **bound(seed_input(ticker, "wacc", 9.0, 4.0, 20.0))) / 100
    shares = share_info["count"]
    if pos(shares):
        st.markdown(f"**{shares_used_text(share_info)}**")
        st.caption(share_info["source"].capitalize() if not share_info["diluted"] else share_info["source"])

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

    implied = r["price"]
    up = upside(implied, price)
    with st.container(key="desk_dcf"):
        cols = st.columns(4)
        cols[0].metric("Sum of PV (FCF)", fmt_money(sum(r["pv_flows"])))
        cols[1].metric("Terminal value (PV)", fmt_money(r["pv_tv"]), help=f"Undiscounted: {fmt_money(r['tv'])}")
        cols[2].metric("Enterprise value", fmt_money(r["ev"]))
        cols[3].metric("Net debt", fmt_money(net_debt or 0.0))
        cols = st.columns(4)
        cols[0].metric("Equity value", fmt_money(r["equity"]))
        cols[1].metric("Shares used (diluted)" if share_info["diluted"] else "Shares used", f"{shares / 1e6:,.1f}M",
                       help=share_info["source"])
        cols[2].metric("Implied price", fmt_price(implied),
                       delta=fmt_pct(up, signed=True) + " vs current" if up is not None else None)
        cols[3].metric("Current price", fmt_price(price))
    with st.container(key="mob_dcf"):
        st.html(tile_grid([
            ("Sum of PV of projected FCF", fmt_money(sum(r["pv_flows"]))),
            ("PV of terminal value", fmt_money(r["pv_tv"])),
            ("Enterprise value", fmt_money(r["ev"])),
            ("Equity value", fmt_money(r["equity"])),
            ("Net debt", fmt_money(net_debt or 0.0)),
            ("Shares used (diluted)" if share_info["diluted"] else "Shares used", f"{shares / 1e6:,.1f}M"),
            (f"Implied price (current {fmt_price(price)})", fmt_price(implied), "wide emph", up),
        ]))
    tv_share = r["pv_tv"] / r["ev"] if r["ev"] > 0 else None
    if tv_share is not None and tv_share > 0.75:
        st.info(f"Terminal value is {tv_share:.0%} of enterprise value — the result is very sensitive "
                "to the terminal growth and WACC assumptions.")
    if implied <= 0:
        st.warning("Net debt exceeds the DCF enterprise value, so the implied equity value is negative.")
    if sbc:
        # The after-SBC cash flows already deduct SBC, so that block divides by shares outstanding.
        render_sbc_dcf(proj, sbc, terminal, wacc, net_debt, pos(shares_outstanding) or shares, implied, price)

    return implied if implied > 0 else None


def render_price_targets(ticker, m, shares, net_debt, ntm: dict | None, base_fcf, base_label) -> list[tuple]:
    """Render the Price Target Setter on a 12-month-forward basis. Returns [(method, target price)]."""
    st.header("Price Target Setter")
    price = m["price"]
    results: list[tuple] = []
    if not ntm:
        st.warning("The projection model is unavailable, so 12-month-forward targets cannot be computed.")
        return results
    if not pos(shares):
        st.warning("Shares outstanding unavailable — per-share targets cannot be computed.")
        return results

    st.markdown(f"**{ntm_basis_text(ntm)}**")
    cols = st.columns(4)
    cols[0].metric("NTM revenue", fmt_money(ntm["revenue"]))
    cols[1].metric("NTM Adj. EBITDA", fmt_money(ntm["ebitda"]))
    cols[2].metric("NTM free cash flow", fmt_money(ntm["fcf"]))
    cols[3].metric("NTM EPS (diluted)", fmt_price(ntm["eps"]))
    st.caption("NTM = next twelve months from today, calendarized from your projections in the model above. "
               "Every method below applies a target multiple to these NTM figures, so each result is a "
               "12-month price target. Target multiples default to the company's multiple today.")

    def target_input(label: str, name: str, today_value, fallback: float, basis: str, step: float) -> float:
        today_value = pos(today_value)
        default = round(float(today_value or fallback), 2 if step < 0.5 else 1)
        note = (f"Default = today's {basis} ({today_value:,.1f}x)" if today_value
                else f"Default = {fallback:,.1f}x fallback (today's {basis} is unavailable or not meaningful)")
        value = st.number_input(label, min_value=0.0, step=step, help=note,
                                **bound(seed_input(ticker, name, default, lo=0.0)))
        st.caption(note)
        return value

    def show(label: str, target: float, formula: str):
        col1, col2 = st.columns([1, 2])
        up = upside(target, price)
        col1.metric(f"{label} 12-month target", fmt_price(target) if target > 0 else "N/A",
                    delta=fmt_pct(up, signed=True) + " vs current" if up is not None and target > 0 else None)
        col2.markdown(f"<br/>{formula}", unsafe_allow_html=True)
        if target > 0:
            results.append((label, target))
        else:
            st.warning(f"{label}: inputs produce a non-positive price, excluded from the summary.")

    per_share = f"{shares / 1e6:,.1f}M shares"
    if pos(ntm["eps"]):
        st.subheader("P/E method")
        mult = target_input("Target P/E", "pe_mult", pos(m["pe_fwd"]) or pos(m["pe_ttm"]), 15.0,
                            "forward P/E" if pos(m["pe_fwd"]) else "trailing P/E", 0.5)
        show("P/E", ntm["eps"] * mult, f"`{fmt_price(ntm['eps'])} NTM EPS × {mult:,.1f} P/E`")
    else:
        st.subheader("P/S method")
        st.caption("Used because NTM EPS is not positive, so a P/E target is not meaningful.")
        mult = target_input("Target P/S", "ps_mult", m["ps"], 2.0, "P/S (TTM revenue)", 0.1)
        show("P/S", ntm["revenue"] / shares * mult,
             f"`({fmt_money(ntm['revenue'])} NTM revenue ÷ {per_share}) × {mult:,.2f} P/S`")

    if pos(ntm["ebitda"]):
        st.subheader("EV / Adj. EBITDA method")
        mult = target_input("Target EV/Adj. EBITDA", "ev_mult", m["ev_ebitda"], 12.0,
                            "EV / TTM Adjusted EBITDA", 0.5)
        nd = net_debt or 0.0
        # Subtracting net debt; with net cash (negative net debt) that is an addition.
        bridge = f"− net debt of {fmt_money(nd)}" if nd >= 0 else f"+ net cash of {fmt_money(-nd)}"
        show("EV/Adj. EBITDA", (ntm["ebitda"] * mult - nd) / shares,
             f"`({fmt_money(ntm['ebitda'])} NTM Adj. EBITDA × {mult:,.1f} {bridge}) ÷ {per_share}`")

    if pos(ntm["fcf"]):
        st.subheader("Price / FCF method")
        today_pfcf = safe_div(m["market_cap"], base_fcf) if pos(base_fcf) else None
        mult = target_input("Target Price / FCF", "pfcf_mult", today_pfcf if today_pfcf and today_pfcf <= 100 else None,
                            20.0, f"Price / FCF (market cap ÷ {base_label} FCF)", 0.5)
        show("P/FCF", ntm["fcf"] / shares * mult,
             f"`({fmt_money(ntm['fcf'])} NTM FCF ÷ {per_share}) × {mult:,.1f} P/FCF`")
    return results


def render_summary(dcf_price: float | None, multiples: list[tuple], price) -> None:
    """Three blocks: intrinsic value today, the 12-month multiples target, and a mixed-horizon blend."""
    st.header("Summary")
    left, right = st.columns(2)
    with left.container(border=True, key="sum_dcf"):
        st.markdown("**Intrinsic Value (DCF) — value today**")
        if dcf_price:
            up = upside(dcf_price, price)
            st.metric("DCF implied price", fmt_price(dcf_price),
                      delta=fmt_pct(up, signed=True) + " vs current" if up is not None else None)
        else:
            st.metric("DCF implied price", "N/A")
            st.caption("No DCF value: it is hidden for banks/insurers or not meaningful for this company's cash flows.")
    with right.container(border=True, key="sum_ntm"):
        if len(multiples) >= 2:
            names = ", ".join(n for n, _ in multiples)
            target = sum(p for _, p in multiples) / len(multiples)
            st.markdown("**12-Month Price Target — multiples average**")
            label, note = "12-month target (average)", f"Simple average of {len(multiples)} NTM methods: {names}."
        elif len(multiples) == 1:
            name, target = multiples[0]
            st.markdown(f"**12-Month Price Target — {name}**")
            label, note = f"12-month target ({name})", f"{name} is the only multiples method that applies, so this is its value."
        else:
            target = None
            st.markdown("**12-Month Price Target**")
            label, note = "12-month target", "No multiples method produced a positive 12-month target."
        up = upside(target, price) if target else None
        st.metric(label, fmt_price(target) if target else "N/A",
                  delta=fmt_pct(up, signed=True) + " vs current" if up is not None else None)
        st.caption(note)

    rows = ([("DCF", "Intrinsic value today", dcf_price)] if dcf_price else []) + \
           [(n, "12-month target (NTM)", p) for n, p in multiples]
    if rows:
        st.dataframe(pd.DataFrame(
            [{"Method": n, "Horizon": h, "Target price": fmt_price(p),
              "Upside / downside": fmt_pct(upside(p, price), signed=True)} for n, h, p in rows]
        ).set_index("Method"), width="stretch")
    if len(rows) >= 2:
        blended = sum(p for _, _, p in rows) / len(rows)
        up = upside(blended, price)
        st.caption(md_escape(
            f"Blended (all methods — mixed horizon): {fmt_price(blended)}"
            + (f" ({fmt_pct(up, signed=True)} vs current)" if up is not None else "")
            + f" — simple average of {len(rows)} methods; it mixes the DCF value today with 12-month targets."))


def render_how_it_works() -> None:
    has_fmp = bool(get_secret("FMP_API_KEY"))
    source = st.session_state.get("_data_source") or ("Financial Modeling Prep" if has_fmp else "yfinance")
    provider = "the data provider (Financial Modeling Prep, else yfinance)" if has_fmp else "yfinance"
    with st.expander("How this works"):
        st.markdown("**Data.** The base (Actual) year comes from SEC EDGAR Company Facts (US-GAAP XBRL): the most "
                    "recent fiscal year, keeping only the latest filing for that period so restated or repeated "
                    f"figures are not double counted. Price, profile, ratios and analyst consensus come from "
                    f"{source}" + ("" if has_fmp else " (no Financial Modeling Prep API key is configured)")
                    + ". Everything is cached for 24 hours.")
        st.markdown("**Base year (Actual)**")
        st.latex(r"FCF_0 = \text{Operating Cash Flow} - \text{Capex}")
        st.latex(r"\text{Adjusted EBITDA} = \text{EBITDA (excl. unusual items)} + "
                 r"\text{Stock-Based Compensation}")
        st.markdown("Adjusted EBITDA is used everywhere EBITDA appears (model, margins, EV/Adj. EBITDA, FCF "
                    "conversion). " f"The EBITDA part is the normalized EBITDA from {provider}, "
                    "or its reported EBITDA when no normalized figure exists. For the base year the stock-based "
                    "compensation add-back is SEC EDGAR's ShareBasedCompensation, disclosed in the model "
                    "footnote. EDGAR operating income + D&A is used only when the provider has no EBITDA. "
                    "EPS₀ is reported diluted EPS (EDGAR, else the provider).")
        st.markdown("**Projection model** (t = 1…5; every percentage is an editable assumption box). The default "
                    "**Detailed** build constructs unlevered free cash flow explicitly:")
        st.latex(r"Revenue_t = Revenue_{t-1} \times (1 + g_t) \qquad \text{Adj. EBITDA}_t = Revenue_t \times m_t")
        st.latex(r"EBIT_t = \text{Adj. EBITDA}_t - D\&A_t \qquad NOPAT_t = EBIT_t \times (1 - \tau)")
        st.latex(r"\text{Unlevered FCF}_t = NOPAT_t + D\&A_t - Capex_t - \Delta NWC_t")
        st.markdown("D&A, capex and ΔNWC are revenue × their %; tax is applied once to EBIT with no floor (a negative "
                    "EBIT gives a tax benefit). Defaults come from the latest fiscal year on SEC EDGAR: D&A ÷ revenue, "
                    "capex ÷ revenue, the change in operating working capital ((current assets − cash) − (current "
                    "liabilities − current debt)) ÷ revenue averaged over up to the three latest fiscal years, positive "
                    "meaning cash outflow (0.0 if the tags are missing), "
                    "and tax expense ÷ pre-tax income clamped to 0–40% (21% if pre-tax income is not positive). Projected "
                    "EPS = NOPAT ÷ diluted shares. If EDGAR lacks D&A or capex, the **Simple** build is used instead, "
                    "and it can be chosen manually:")
        st.latex(r"FCF_t = \text{Adj. EBITDA}_t \times c \quad \text{(Simple build)}")
        st.markdown("**Defaults mode.** Consensus uses analyst averages. Upper-range uses avg + 0.5 × (high − avg) "
                    "(≈75th percentile) for the two analyst-estimate years, fades out-years the same way, and expands "
                    "Adj. EBITDA margins by the company's own average annual margin change over up to 3 years (capped at "
                    "+1.5 pts a year; flat if the trend is not positive). D&A, capex, ΔNWC and tax defaults do not change "
                    "with the mode. Switching modes re-defaults only boxes you have not edited.")
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
        st.markdown(f"Consensus EPS comes from {provider}. EPS is shown as N/A when the prior year's EPS "
                    "or EBITDA is not positive, so a loss is never compounded.")
        st.markdown("**DCF** (r = WACC, g_T = terminal growth). The implied price divides equity value by "
                    "**diluted** shares — the latest fiscal year's weighted-average diluted count from SEC EDGAR "
                    "(else the provider's figure for that year, else current shares outstanding, labelled) — so "
                    "options, RSUs and convertibles that will become shares are counted in the per-share value. "
                    "In loss years companies report diluted equal to basic, and that reported figure is used as-is. "
                    "The DCF (after SBC) block instead divides by shares outstanding: its cash flows already deduct "
                    "projected SBC, so also counting the awards in a diluted share count would charge for them twice. "
                    "The snapshot market cap and the multiples price targets keep using shares outstanding.")
        st.latex(r"PV(FCF_t) = \frac{FCF_t}{(1+r)^t}")
        st.latex(r"TV = \frac{FCF_5 (1+g_T)}{r - g_T} \qquad PV(TV) = \frac{TV}{(1+r)^5}")
        st.latex(r"EV = \sum_{t=1}^{5} PV(FCF_t) + PV(TV)")
        st.latex(r"\text{Equity} = EV - (\text{Total Debt} - \text{Cash}) \qquad "
                 r"\text{Implied Price} = \frac{\text{Equity}}{\text{Shares Outstanding}}")
        st.latex(r"\text{Upside} = \frac{\text{Target}}{\text{Current Price}} - 1")
        st.markdown("**12-month-forward (NTM) figures.** The window runs from today to today + 12 months. "
                    "Each projected fiscal year is weighted by the share of the window it overlaps, so a window "
                    "with 2 months left in FY1 and 10 months in FY2 is 17% FY1 + 83% FY2.")
        st.latex(r"X_{NTM} = \sum_{t} w_t \, X_t \qquad w_t = \frac{\text{days of FY}_t \text{ inside the window}}"
                 r"{\text{days in the window}} \qquad X \in \{Revenue,\ Adj.\ EBITDA,\ FCF,\ EPS\}")
        st.markdown("**12-month price targets** (target multiples default to the company's multiple today)")
        st.latex(r"\text{P/E} = EPS_{NTM} \times \text{Target P/E} \quad (EPS_{NTM} > 0)")
        st.latex(r"\text{P/S} = \frac{Revenue_{NTM}}{\text{Shares}} \times \text{Target P/S} "
                 r"\quad (\text{used when } EPS_{NTM} \le 0)")
        st.latex(r"\text{EV/Adj. EBITDA} = \frac{\text{Adj. EBITDA}_{NTM} \times \text{Target multiple} "
                 r"- \text{Net Debt}}{\text{Shares}} \quad (\text{Adj. EBITDA}_{NTM} > 0)")
        st.latex(r"\text{P/FCF} = \frac{FCF_{NTM}}{\text{Shares}} \times \text{Target P/FCF} \quad (FCF_{NTM} > 0)")
        st.markdown("**Horizons.** The DCF gives intrinsic value *today*. The multiples methods give a *12-month* "
                    "price target; when two or more apply, the 12-month target is their simple average, and when "
                    "only one applies it is that method's value. The blended line averages all methods and mixes "
                    "both horizons, so it is shown only as a secondary reference.")
        st.markdown("**Snapshot multiples**")
        st.latex(r"P/E = \frac{Price}{EPS} \quad PEG = \frac{P/E}{\text{EPS growth (\%)}} \quad "
                 r"P/S = \frac{\text{Mkt Cap}}{\text{Revenue}} \quad P/B = \frac{Price}{\text{Book/Share}}")
        st.latex(r"EV = \text{Mkt Cap} + \text{Debt} - \text{Cash} \quad EV/Sales \quad EV/EBITDA \quad "
                 r"\text{FCF Yield} = \frac{FCF}{\text{Mkt Cap}}")
        st.markdown("**Stock-based compensation** is modelled only for software companies under $20B market "
                    "cap, where dilution is typically material. SBC history comes from SEC EDGAR (ShareBasedCompensation), "
                    "else the provider's cash-flow statement. Projected SBC = projected revenue × SBC % (Trend, Flat "
                    "or Dollar-growth defaults, all editable). FCF after SBC = FCF − SBC; the DCF (after SBC) "
                    "discounts it with the same WACC, terminal growth, net debt and share count, and the share "
                    "count is not grown for SBC so the cost is not counted twice. SBC drag = standard DCF − "
                    "DCF (after SBC), per share.")
        st.markdown("**Compounding calculator** (i = annual return ÷ 12, n = months, contributions at month end)")
        st.latex(r"B_{m} = B_{m-1}(1+i) + PMT \qquad \text{Goal: } PMT = \frac{(\text{Target} - "
                 r"\text{Start}(1+i)^n)\, i}{(1+i)^n - 1}")
        st.markdown("Total debt is approximate: long-term debt (including current portion) plus short-term "
                    "borrowings/commercial paper; cash excludes marketable securities.")


def render_footer() -> None:
    st.divider()
    render_how_it_works()
    st.caption(f"**{FOOTER}**")


def detailed_components(history: pd.DataFrame, base: dict) -> dict:
    """Default D&A %, capex % and tax rate for the detailed build from the latest EDGAR fiscal year; ΔNWC % from up to three."""
    comp = {"available": False}
    if history.empty or not pos(base.get("revenue")):
        return comp
    row = history.iloc[-1]
    prev = history.iloc[-2] if len(history) > 1 else None
    rev, da, capex, fy = base["revenue"], num(row.get("D&A")), num(row.get("Capex")), base["label"]
    if da is None or capex is None:
        return comp
    comp.update(available=True, da=da, capex=capex, da_pct=da / rev * 100, capex_pct=capex / rev * 100)
    nwc_now, nwc_prev = num(row.get("Operating NWC")), num(prev.get("Operating NWC")) if prev is not None else None
    comp["dnwc"] = nwc_now - nwc_prev if nwc_now is not None and nwc_prev is not None else None  # base-year actual
    # Default: average ΔNWC % of revenue over up to the three latest fiscal years (base year and the two before it).
    per_year = []
    for k in range(len(history) - 1, max(len(history) - 4, 0), -1):
        cur, before = history.iloc[k], history.iloc[k - 1]
        now, then = num(cur.get("Operating NWC")), num(before.get("Operating NWC"))
        year_rev = rev if k == len(history) - 1 else num(cur.get("Revenue"))
        if now is not None and then is not None and pos(year_rev):
            per_year.append((history.index[k], (now - then) / year_rev * 100))
    if per_year:
        used = ", ".join(f"{label} {v:.2f}%" for label, v in reversed(per_year))
        comp.update(nwc_pct=sum(v for _, v in per_year) / len(per_year), nwc_years=per_year,
                    nwc_note=f"{len(per_year)}-year average of the change in operating working capital ÷ revenue "
                             f"({used}), SEC EDGAR; positive = cash outflow")
    else:
        comp.update(nwc_pct=0.0, nwc_years=[],
                    nwc_note="defaulted to 0.0 — working-capital tags were not available on SEC EDGAR")
    tax, pre = num(row.get("Income tax")), num(row.get("Pretax income"))
    if tax is not None and pos(pre):
        comp.update(tax_rate=min(max(tax / pre * 100, 0.0), 40.0),
                    tax_note=f"{fy} effective rate, SEC EDGAR, clamped to 0–40%"
                             + ("; pre-tax income = net income + tax" if bool(row.get("Pretax derived")) else ""))
    else:
        comp.update(tax_rate=21.0, tax_note="21% default — pre-tax income not positive or tax tags missing")
    return comp


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

    base["components"] = detailed_components(history, base)

    if base["eps"] is None and provider and provider["eps_diluted"] is not None:
        base["eps"], base["eps_source"] = provider["eps_diluted"], f"reported diluted EPS ({provider['source']})"
    if base["eps"] is None:
        base["eps_source"] = "not available"
    base["notes"] = notes
    return base


def main() -> str | None:
    """Route to the active view. Returns the ticker whose inputs should be saved, if any."""
    view = st.session_state.get("view", "valuation")
    st.html(app_css())  # texture, tab bar, compact grids; style-only HTML takes no layout space
    render_header(VIEW_TITLES[view])
    render_topnav(view)

    # Saved inputs live in the browser, which reports them a moment after the
    # first run. Wait for them so widgets start from saved values, not defaults.
    items = stored_items()
    if items is None and not st.session_state.get("_store_skip"):
        st.info("Loading your saved inputs…")
        st.button("Continue without saved inputs", on_click=st.session_state.update, kwargs={"_store_skip": True})
        return None
    if view == "compounding":
        render_compounding()
        return None
    if view == "feed":
        render_feed()
        return None
    if view == "portfolio":
        render_portfolio()
        return None
    return valuation_view(items)


def valuation_view(items: dict | None) -> str | None:
    """The DCF valuation view (the original app)."""
    if "ticker_query" not in st.session_state:
        st.session_state["ticker_query"] = (items or {}).get(LAST_TICKER_ITEM) or "AAPL"

    query = st.text_input("Search by ticker or company name", placeholder="e.g. MSFT, NVDA, Coca-Cola",
                          **bound("ticker_query"))
    ticker = resolve_ticker(query)
    if not ticker:
        st.info("Enter a ticker symbol or company name to begin.")
        return None

    # Everything for this ticker renders into one block at this position. On a ticker change an
    # st.empty() slot replaces the previous ticker's sections with a loading state at once, and the
    # new heading appears only together with its own data, after every fetch has finished.
    # Same-ticker reruns (input edits) use a plain container so the page updates in place;
    # st.empty() would blank the block on every rerun, collapsing the page and its scroll position.
    new_ticker = st.session_state.get("_shown_ticker") != ticker
    slot = st.empty() if new_ticker else None
    if slot is not None:
        slot.html(loading_html(ticker))
    m, source, warnings = get_market_data(ticker)
    cik = lookup_cik(ticker)
    edgar: dict = {"history": pd.DataFrame(), "note": "Ticker not found in the SEC company list."}
    if cik is not None:
        try:
            edgar = edgar_fundamentals(cik)
        except Exception as exc:
            edgar = {"history": pd.DataFrame(), "note": f"SEC EDGAR request failed ({type(exc).__name__})."}
    yf_est = None
    fmp_est = m.get("estimates") or []
    if (m["price"] or not edgar["history"].empty) and not any(e.get("eps") is not None for e in fmp_est):
        try:
            yf_est = yf_estimates(ticker)
        except Exception:
            yf_est = None
    st.session_state["_data_source"] = source
    st.session_state["_shown_ticker"] = ticker
    with slot.container() if slot is not None else st.container():
        return render_ticker(ticker, m, source, warnings, edgar, yf_est)


def render_ticker(ticker: str, m: dict, source: str, warnings: list, edgar: dict, yf_est) -> str | None:
    """All sections for one ticker, rendered after its data has loaded."""
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
    st.html(MOBILE_CSS)  # style-only HTML takes no space in the layout
    st.subheader(f"{name} ({ticker})")
    with st.container(key="desk_header"):
        cols = st.columns(4)
        cols[0].metric("Sector", m["sector"] or "N/A")
        cols[1].metric("Industry", m["industry"] or "N/A")
        cols[2].metric("Current price", fmt_price(price))
        cols[3].metric("Market cap", fmt_money(m["market_cap"]))
    with st.container(key="mob_header"):
        st.markdown(f"**{m['sector'] or 'N/A'}** · {m['industry'] or 'N/A'}")
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
    render_snapshot(m, fcf_for_yield, price)

    # ---- 3. Projection model (1 actual year + 5 projected years)
    years = [base["end"].year + i for i in range(1, YEARS + 1)]
    defaults = build_defaults(base, years, m.get("estimates"), yf_est)
    # SBC is modelled only where dilution is typically material: software companies under $20B.
    # Otherwise its history is not built at all (sbc_rows = None) and saved SBC inputs are simply ignored.
    sbc_rows = sbc_history(history, m) if sbc_gate(m) else None
    share_info = dcf_share_count(history, base, m, shares)
    upper = upper_range_defaults(base, years, defaults, margin_trend(history, m) if not history.empty else None)
    model = render_model(ticker, base, defaults, years, sbc_rows, m["market_cap"], upper=upper, share_info=share_info)
    proj, eps, sbc = model if model else (None, None, None)

    # ---- 4. DCF
    dcf_price = None
    show_dcf = True
    if financial:
        show_dcf = st.checkbox("Show DCF anyway (not recommended for banks/insurers)", value=False,
                               key=f"force_dcf_{ticker}")
    if show_dcf:
        dcf_price = render_dcf(ticker, proj, years, net_debt, share_info, price, sbc, shares_outstanding=shares)
    else:
        st.header("Discounted Cash Flow")
        st.info("DCF hidden for this financial company — see the warning above.")

    # ---- 5. Price targets (12-month forward)
    ntm = ntm_figures(base["end"], years, proj, eps) if proj else None
    multiples = render_price_targets(ticker, m, shares, net_debt, ntm, base["fcf"], base["label"])

    # ---- 6. Summary
    render_summary(dcf_price, multiples, price)
    return ticker


# --------------------------------------------------------------------------- #
# App shell: views, bottom tab bar, gold-textured background
# --------------------------------------------------------------------------- #
VIEWS = (("valuation", "📊 Valuation"), ("compounding", "🧮 Compounding"),
         ("feed", "📰 My Feed"), ("portfolio", "💼 Portfolio"))
VIEW_TITLES = {"valuation": "DCF Valuation", "compounding": "Compounding Calculator",
               "feed": "My Daily Feed", "portfolio": "My Portfolio"}
NAVY = "#10204a"   # the logo's background
GOLD = "#b38b22"
_RUN: dict = {}    # per-run flags (the script re-executes on every rerun, so this starts empty)

APP_CSS = f"""
/* Toolbar: mostly opaque so scrolled content never sits under it, while the gold glow still shows below. */
[data-testid="stHeader"] {{ background: color-mix(in srgb, var(--jd-base) 97%, transparent) !important;
    backdrop-filter: blur(6px); }}
@media (min-width: 768px) {{
    [class*="st-key-row_"] {{ flex-direction: row !important; align-items: flex-end !important; flex-wrap: nowrap !important; }}
    [class*="st-key-row_"] > div:first-child {{ flex: 0 0 14% !important; width: 14% !important; padding-bottom: 0.55rem; }}
    [class*="st-key-row_"] > div:last-child {{ flex: 1 1 auto !important; min-width: 0 !important; width: auto !important; }}
}}
/* Content surfaces stay solid over the texture. */
[data-testid="stExpander"] details, [data-testid="stPlotlyChart"], .st-key-sbc_proj, .st-key-sbc_dcf,
.st-key-sum_dcf, .st-key-sum_ntm, .st-key-pf_rows, .feed-card {{ background-color: var(--jd-base); }}

/* Top navigation: a segmented control on the page background, sticky just below Streamlit's toolbar
   (3.75rem tall), so it never covers the toolbar's menu. Desktop and phone variants differ only in labels. */
.st-key-topnav_mob, [data-testid="stLayoutWrapper"]:has(> .st-key-topnav_mob) {{ display: none !important; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-topnav_desk), [data-testid="stLayoutWrapper"]:has(> .st-key-topnav_mob) {{
    position: sticky; top: 3.75rem; z-index: 50; }}
.st-key-topnav_desk, .st-key-topnav_mob {{
    background: var(--jd-base); border: 1px solid rgba(179, 139, 34, 0.45); border-radius: 12px;
    padding: 4px !important; gap: 0 !important; box-sizing: border-box; }}
.st-key-topnav_desk [data-testid="stHorizontalBlock"], .st-key-topnav_mob [data-testid="stHorizontalBlock"] {{
    flex-wrap: nowrap !important; flex-direction: row !important; gap: 4px !important; }}
.st-key-topnav_desk [data-testid="stColumn"], .st-key-topnav_mob [data-testid="stColumn"] {{
    min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }}
.st-key-topnav_desk button, .st-key-topnav_mob button {{
    background: transparent !important; border: 0 !important; box-shadow: none !important; color: inherit !important;
    min-height: 38px; padding: 4px 6px !important; border-radius: 9px !important; }}
.st-key-topnav_desk button p, .st-key-topnav_mob button p {{
    font-size: 0.9rem; font-weight: 600; line-height: 1.1; white-space: nowrap; opacity: 0.72; }}
.st-key-topnav_desk button[data-testid="stBaseButton-primary"], .st-key-topnav_mob button[data-testid="stBaseButton-primary"] {{
    background: {GOLD} !important; }}
.st-key-topnav_desk button[data-testid="stBaseButton-primary"] p, .st-key-topnav_mob button[data-testid="stBaseButton-primary"] p {{
    color: #1c1508 !important; opacity: 1; }}

/* Compact year grids (Projection Assumptions, SBC %) */
@media (max-width: 767px) {{
    .st-key-topnav_desk, [data-testid="stLayoutWrapper"]:has(> .st-key-topnav_desk) {{ display: none !important; }}
    .st-key-topnav_mob {{ display: flex !important; }}
    [data-testid="stLayoutWrapper"]:has(> .st-key-topnav_mob) {{ display: flex !important; }}
    .st-key-topnav_mob button {{ padding: 4px 2px !important; }}
    .st-key-topnav_mob button p {{ font-size: 0.8rem; }}
    [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important; flex-direction: row !important; gap: 0.5rem !important; }}
    [class*="st-key-grid_"] [data-testid="stColumn"] {{
        flex: 0 0 calc((100% - 1rem) / 3) !important; width: calc((100% - 1rem) / 3) !important; min-width: 0 !important; }}
    [class*="st-key-grid_"] [data-testid="stNumberInputStepDown"], [class*="st-key-grid_"] [data-testid="stNumberInputStepUp"],
    .st-key-conv_row [data-testid="stNumberInputStepDown"], .st-key-conv_row [data-testid="stNumberInputStepUp"],
    .st-key-pf_rows [data-testid="stNumberInputStepDown"], .st-key-pf_rows [data-testid="stNumberInputStepUp"] {{
        display: none !important; }}
    .st-key-conv_row [data-testid="stHorizontalBlock"], .st-key-pf_rows [data-testid="stHorizontalBlock"] {{
        flex-wrap: nowrap !important; flex-direction: row !important; gap: 0.5rem !important; }}
    .st-key-conv_row [data-testid="stColumn"], .st-key-pf_rows [data-testid="stColumn"] {{
        min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }}
    .st-key-conv_row [data-testid="stColumn"]:last-child {{ display: none !important; }}
}}

/* Four-across tiles on desktop (two on phones) */
.tile-grid.cols4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
@media (max-width: 767px) {{ .tile-grid.cols4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.big-figure {{ font-size: 2.6rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.big-label {{ font-size: 0.85rem; opacity: 0.75; }}

/* Feed */
.feed-list {{ display: flex; flex-direction: column; gap: 0.6rem; }}
.feed-card {{ border: 1px solid rgba(128, 128, 128, 0.28); border-radius: 0.6rem; padding: 0.7rem 0.85rem; }}
.feed-meta {{ display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; font-size: 0.78rem; }}
.feed-meta .muted {{ opacity: 0.72; }}
.feed-badge {{ font-weight: 700; font-size: 0.68rem; letter-spacing: 0.04em; padding: 0.1rem 0.45rem; border-radius: 999px; color: #fff; }}
.feed-badge.news {{ background: #1d4ed8; }}
.feed-badge.posts {{ background: #7c3aed; }}
.feed-badge.filings {{ background: {GOLD}; }}
.feed-title {{ margin-top: 0.3rem; font-size: 0.95rem; line-height: 1.35; overflow-wrap: anywhere; }}
.feed-title a {{ color: inherit; text-decoration: none; }}
.feed-title a:hover {{ text-decoration: underline; }}
.feed-sent.bull {{ color: #16a34a; font-weight: 600; }}
.feed-sent.bear {{ color: #dc2626; font-weight: 600; }}
"""


def _texture_rules(dark: bool) -> str:
    """Warm base colour, a faint 26px gold dot grid, and a soft gold glow behind the header."""
    base = "#16120d" if dark else "#fcfaf5"
    dot = "rgba(179, 139, 34, 0.10)" if dark else "rgba(179, 139, 34, 0.16)"
    glow = "rgba(179, 139, 34, 0.12)" if dark else "rgba(214, 178, 94, 0.18)"
    return (f":root {{ --jd-base: {base}; }}"
            f".stApp {{ background-color: {base} !important; }}"
            f'[data-testid="stMain"] {{ background-color: transparent;'
            f" background-image: radial-gradient(ellipse 75% 400px at 50% 0, {glow}, transparent 100%),"
            f" radial-gradient(circle at 13px 13px, {dot} 0.8px, transparent 1.3px);"
            f" background-size: 100% 400px, 26px 26px; background-repeat: no-repeat, repeat;"
            f" background-attachment: local, local; }}")


def app_css() -> str:
    """Global CSS. Uses Streamlit's active theme when known, else the system colour scheme."""
    try:
        kind = st.context.theme.type
    except Exception:
        kind = None
    if kind in ("light", "dark"):
        texture = _texture_rules(kind == "dark")
    else:
        texture = _texture_rules(False) + "@media (prefers-color-scheme: dark) {" + _texture_rules(True) + "}"
    return MOBILE_CSS + "<style>" + texture + APP_CSS + "</style>"


def set_view(view: str) -> None:
    st.session_state["view"] = view


NAV_SHORT = {"valuation": "📊 Value", "compounding": "🧮 Growth", "feed": "📰 Feed", "portfolio": "💼 Portfolio"}


def render_topnav(active: str) -> None:
    """Segmented view switcher under the header: full labels on desktop, short labels on phones."""
    for key, prefix, labels in (("topnav_desk", "tab", dict(VIEWS)), ("topnav_mob", "tabm", NAV_SHORT)):
        with st.container(key=key):
            for col, (view, _) in zip(st.columns(len(VIEWS)), VIEWS):
                col.button(labels[view], key=f"{prefix}_{view}", type="primary" if view == active else "secondary",
                           on_click=set_view, args=(view,), width="stretch")


# --------------------------------------------------------------------------- #
# Browser-wide saved items: calculator inputs and holdings
# --------------------------------------------------------------------------- #
CALC_ITEM = "dcf_calc"
HOLDINGS_ITEM = "dcf_holdings"
UI_ITEM = "dcf_ui"                   # small UI preferences, e.g. whether the holdings editor is open
PF_EDITOR_KEY = "pf_editor_open"
CALC_DEFAULTS = {"calc_mode": "Calculator", "calc_start": 1000.0, "calc_monthly": 200.0, "calc_return": 8.0,
                 "calc_years": 20, "calc_increase": 0.0, "goal_target": 100000.0, "goal_years": 10,
                 "goal_start": 1000.0, "goal_return": 8.0}
CALC_MODES = ["Calculator", "Goal solver"]
DEFAULT_HOLDINGS = [{"ticker": "ZETA", "shares": None, "cost": None}]
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,9}$")
PRIVACY_NOTE = ("🔒 Your holdings are stored only in this browser (local storage). The app uses them to build "
                "this page but never saves them on a server.")


def stored_json(item: str, default):
    raw = (stored_items() or {}).get(item)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


def seed_calc(name: str) -> str:
    """Seed a calculator input from the browser's saved values (or its default) before its widget exists."""
    if name not in st.session_state:
        default = CALC_DEFAULTS[name]
        saved = stored_json(CALC_ITEM, {})
        value = saved.get(name, default) if isinstance(saved, dict) else default
        if isinstance(default, str):
            value = value if value in CALC_MODES else default
        elif isinstance(default, int):
            value = int(value) if num(value) is not None else default
        else:
            value = float(value) if num(value) is not None else default
        st.session_state[name] = value
    return name


def _normalize_holdings(entries) -> list[dict]:
    out, seen = [], set()
    for h in entries if isinstance(entries, list) else []:
        if not isinstance(h, dict):
            continue
        t = str(h.get("ticker", "")).strip().upper().replace(".", "-")
        if not TICKER_RE.match(t) or t in seen:
            continue
        shares, cost = num(h.get("shares")), num(h.get("cost"))
        out.append({"ticker": t, "shares": shares if shares is not None and shares >= 0 else None,
                    "cost": cost if cost is not None and cost >= 0 else None})
        seen.add(t)
    return out


def holdings() -> list[dict]:
    """The shared holdings list (Feed + Portfolio), loaded once per session from browser storage."""
    if "holdings" not in st.session_state:
        st.session_state["holdings"] = _normalize_holdings(stored_json(HOLDINGS_ITEM, None) or
                                                           [dict(h) for h in DEFAULT_HOLDINGS])
    return st.session_state["holdings"]


def add_holding(ticker: str, shares=None, cost=None) -> str | None:
    """Add (or update) a holding. Returns an error message, or None on success."""
    t = (ticker or "").strip().upper().replace(".", "-")
    if not TICKER_RE.match(t):
        return f"“{ticker}” doesn't look like a ticker symbol."
    hl = holdings()
    for h in hl:
        if h["ticker"] == t:
            if shares is not None:
                h["shares"] = shares
                st.session_state.pop(f"pf_sh_{t}", None)
            if cost is not None:
                h["cost"] = cost
                st.session_state.pop(f"pf_cost_{t}", None)
            return None
    hl.append({"ticker": t, "shares": shares, "cost": cost})
    return None


def remove_holding(ticker: str) -> None:
    st.session_state["holdings"] = [h for h in holdings() if h["ticker"] != ticker]
    for k in (f"pf_sh_{ticker}", f"pf_cost_{ticker}"):
        st.session_state.pop(k, None)


def global_writes(items: dict) -> dict:
    """Browser-storage writes for the calculator inputs and the holdings list (only when changed)."""
    writes: dict = {}
    if _RUN.get("calc"):
        saved = stored_json(CALC_ITEM, {})
        saved = saved if isinstance(saved, dict) else {}
        values = {k: st.session_state.get(k, saved.get(k, d)) for k, d in CALC_DEFAULTS.items()}
        if all(_same(values[k], d) for k, d in CALC_DEFAULTS.items()):
            if CALC_ITEM in items:
                writes[CALC_ITEM] = None
        elif set(saved) != set(values) or not all(_same(values[k], saved[k]) for k in values):
            writes[CALC_ITEM] = json.dumps(values)
    if _RUN.get("pf_editor") and PF_EDITOR_KEY in st.session_state:
        saved = stored_json(UI_ITEM, {})
        saved = saved if isinstance(saved, dict) else {}
        is_open = bool(st.session_state[PF_EDITOR_KEY])
        if is_open != bool(saved.get(PF_EDITOR_KEY, False)):
            writes[UI_ITEM] = json.dumps({**saved, PF_EDITOR_KEY: is_open})
    if "holdings" in st.session_state:
        current = st.session_state["holdings"]
        stored = stored_json(HOLDINGS_ITEM, None)
        if stored is None:
            if current != DEFAULT_HOLDINGS:
                writes[HOLDINGS_ITEM] = json.dumps(current)
        elif _normalize_holdings(stored) != current:
            writes[HOLDINGS_ITEM] = json.dumps(current)
    return writes


# --------------------------------------------------------------------------- #
# Stock-based compensation (SBC): history, projection methods, backtest
# --------------------------------------------------------------------------- #
SBC_METHODS = ["Trend", "Flat", "Dollar growth"]
SBC_MAX_MARKET_CAP = 20e9
SBC_GATE_NOTE = "SBC view is shown only for software companies under $20B market cap, where dilution is typically material."


def sbc_gate(m: dict) -> bool:
    """SBC system only for software / SaaS companies (provider industry contains "Software") under $20B."""
    cap = pos(m.get("market_cap"))
    return cap is not None and cap < SBC_MAX_MARKET_CAP and "software" in (m.get("industry") or "").lower()
SBC_RECENCY = {0: 3.0, 1: 2.0, 2: 1.5}  # weight by age (0 = latest year); older years 0.5


def sbc_history(history: pd.DataFrame, m: dict) -> list[dict]:
    """Up to 5 fiscal years of SBC and revenue: EDGAR ShareBasedCompensation, else the provider cash-flow line."""
    rows = []
    for label, row in history.iterrows():
        end, rev = pd.Timestamp(row["Period End"]), num(row.get("Revenue"))
        sbc, source = num(row.get("Stock-Based Compensation")), "SEC EDGAR"
        if sbc is None:
            for a in m.get("annual") or []:
                try:
                    near = abs((pd.Timestamp(a["date"]) - end).days) <= 21
                except (KeyError, ValueError):
                    continue
                if near and a.get("sbc") is not None:
                    sbc, source = num(a["sbc"]), f"{a['source']} cash-flow statement"
                    break
        if pos(rev) and sbc is not None and sbc >= 0:
            rows.append({"label": label, "end": end, "sbc": sbc, "revenue": rev, "pct": sbc / rev * 100,
                         "source": source})
    for i, r in enumerate(rows):
        r["yoy"] = (r["sbc"] / rows[i - 1]["sbc"] - 1) * 100 if i and rows[i - 1]["sbc"] > 0 else None
    return rows


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def sbc_weights(rows: list[dict]) -> tuple[list[float], int | None]:
    """Recency weights (latest 3 years 3 / 2 / 1.5, older 0.5) with at most one outlier year halved.

    Outlier rule (needs 4+ years): the year whose SBC % of revenue deviates most from the median of the
    other years is flagged if that deviation exceeds 10 percentage points."""
    n = len(rows)
    w = [SBC_RECENCY.get(n - 1 - i, 0.5) for i in range(n)]
    outlier = None
    if n >= 4:
        devs = [abs(r["pct"] - _median([o["pct"] for j, o in enumerate(rows) if j != i])) for i, r in enumerate(rows)]
        i = max(range(n), key=lambda k: devs[k])
        if devs[i] > 10:
            outlier, w[i] = i, w[i] * 0.5
    return w, outlier


def sbc_slope(rows: list[dict], w: list[float]) -> float:
    """Weighted least-squares slope of SBC % of revenue, in percentage points per year."""
    if len(rows) < 2:
        return 0.0
    xs, ys, sw = range(len(rows)), [r["pct"] for r in rows], sum(w)
    xm = sum(wi * x for wi, x in zip(w, xs)) / sw
    ym = sum(wi * y for wi, y in zip(w, ys)) / sw
    sxx = sum(wi * (x - xm) ** 2 for wi, x in zip(w, xs))
    sxy = sum(wi * (x - xm) * (y - ym) for wi, x, y in zip(w, xs, ys))
    return sxy / sxx if sxx else 0.0


def sbc_trend(rows: list[dict], steps: int = YEARS) -> tuple[list[float], float]:
    """Trend: start at the latest actual %, move by the weighted slope with its effect halving each year,
    a one-year change capped at 20% of the prior year's %, and never below 0."""
    slope = sbc_slope(rows, sbc_weights(rows)[0])
    p, out = rows[-1]["pct"], []
    for k in range(steps):
        change = min(max(slope * 0.5 ** k, -0.2 * p), max(1.0, 0.2 * p))
        p = max(0.0, p + change)
        out.append(p)
    return out, slope


def sbc_flat(rows: list[dict], steps: int = YEARS) -> list[float]:
    w, _ = sbc_weights(rows)
    return [sum(wi * r["pct"] for wi, r in zip(w, rows)) / sum(w)] * steps


def sbc_dollar(rows: list[dict], base_revenue: float, proj: list[dict]) -> tuple[list[float] | None, float | None]:
    """Dollar growth: SBC dollar CAGR (outlier excluded) dampened to 80%, capped at each year's revenue growth."""
    _, outlier = sbc_weights(rows)
    usable = [r for i, r in enumerate(rows) if i != outlier and r["sbc"] > 0]
    if len(usable) < 2:
        return None, None
    years = (usable[-1]["end"] - usable[0]["end"]).days / 365.25
    if years <= 0:
        return None, None
    cagr = (usable[-1]["sbc"] / usable[0]["sbc"]) ** (1 / years) - 1
    sbc, prev, out = rows[-1]["sbc"], base_revenue, []
    for p in proj:
        sbc *= 1 + min(0.8 * cagr, p["revenue"] / prev - 1)
        out.append(max(0.0, sbc / p["revenue"] * 100))
        prev = p["revenue"]
    return out, cagr


def sbc_backtest(rows: list[dict]) -> dict | None:
    """Project the latest year's SBC % with Trend, using only data through two fiscal years earlier."""
    if len(rows) < 4:
        return None
    sub = rows[:-2]
    predicted = sbc_trend(sub, steps=2)[0][1]
    actual = rows[-1]["pct"]
    return {"first": sub[0]["label"], "through": sub[-1]["label"], "target": rows[-1]["label"],
            "predicted": predicted, "actual": actual,
            "error": (predicted - actual) / actual * 100 if actual else None}


def apply_sbc_method(ticker: str, values: dict) -> None:
    """Method picker callback: fill the five SBC % boxes with the chosen method's projection."""
    method = st.session_state[input_key(ticker, "sbc_method")]
    for i, v in enumerate(values[method]):
        st.session_state[input_key(ticker, f"s{i}")] = v


def render_sbc(ticker: str, base: dict, years: list[int], proj: list[dict], rows: list[dict],
               market_cap) -> list[float] | None:
    """SBC projection boxes and the history/basis panel. Returns projected SBC $ per year, or None."""
    st.subheader("Stock-Based Compensation (SBC)")
    if not rows:
        with st.expander("SBC History & Projection Basis"):
            st.info("No usable stock-based compensation history was found for this company (SEC EDGAR and the "
                    "provider's cash-flow statement report none). The SBC rows and the after-SBC DCF are skipped.")
        return None

    trend, slope = sbc_trend(rows)
    flat = sbc_flat(rows)
    dollar, cagr = sbc_dollar(rows, base["revenue"], proj)
    methods = {"Trend": trend, "Flat": flat, "Dollar growth": dollar or flat}
    rounded = {k: [round(v, 1) for v in vals] for k, vals in methods.items()}
    for i in range(YEARS):
        seed_input(ticker, f"s{i}", rounded["Trend"][i], lo=0.0)
    method_key = seed_input(ticker, "sbc_method", "Trend", options=SBC_METHODS)

    with st.container(border=True, key="sbc_proj"):
        st.markdown("**SBC % of revenue — projection**",
                    help="Projected SBC $ = projected revenue × SBC %. Pick a method to refill the boxes, or edit them.")
        st.radio("Projection method", SBC_METHODS, horizontal=True, key=method_key,
                 index=SBC_METHODS.index(st.session_state[method_key]),
                 on_change=apply_sbc_method, args=(ticker, rounded))
        with st.container(key="grid_s"):
            for i, col in enumerate(st.columns(YEARS)):
                col.number_input(f"{years[i]}E", min_value=0.0, step=0.5, format="%.1f",
                                 **bound(input_key(ticker, f"s{i}")))
        pcts = [st.session_state[input_key(ticker, f"s{i}")] for i in range(YEARS)]
        sbc = [p["revenue"] * pct / 100 for p, pct in zip(proj, pcts)]
        total = sum(sbc)
        share = safe_div(total, market_cap)
        st.caption(md_escape(
            f"Projected SBC {years[0]}–{years[-1]}: {fmt_money(total)} in total"
            + (f" — {share:.1%} of today's market cap ({fmt_money(market_cap)}), the approximate shareholder "
               "dilution cost." if share is not None else ".")))

    with st.expander("SBC History & Projection Basis"):
        w, outlier = sbc_weights(rows)
        st.dataframe(pd.DataFrame([{
            "Year": r["label"], "SBC": fmt_money(r["sbc"]), "Revenue": fmt_money(r["revenue"]),
            "SBC % of revenue": f"{r['pct']:.1f}%",
            "YoY SBC growth": fmt_pct(r["yoy"] / 100, signed=True) if r["yoy"] is not None else "—",
            "Weight": f"{wi:.2f}" + (" (outlier, halved)" if i == outlier else ""), "Source": r["source"],
        } for i, (r, wi) in enumerate(zip(rows, w))]).set_index("Year"), width="stretch")
        if outlier is not None:
            r = rows[outlier]
            others = _median([o["pct"] for j, o in enumerate(rows) if j != outlier])
            st.warning(f"Outlier: {r['label']} SBC was {r['pct']:.1f}% of revenue, {abs(r['pct'] - others):.1f} "
                       f"percentage points from the median of the other years ({others:.1f}%). Its weight is halved "
                       "in the Trend and Flat calculations, and it is not used as the starting year of the "
                       "Dollar-growth CAGR.")
        elif len(rows) >= 4:
            st.caption("No outlier: no year's SBC % of revenue is more than 10 percentage points from the median "
                       "of the other years.")
        else:
            st.caption("Fewer than 4 years of history, so the outlier rule is not applied.")
        if rows[-1]["end"] != base["end"]:
            st.caption(f"Latest SBC year is {rows[-1]['label']}, earlier than the model's base year {base['label']}.")
        fmt_list = lambda vals: ", ".join(f"{v:.1f}%" for v in vals)
        st.markdown(md_escape(
            f"- **Trend (default):** recency-weighted slope {slope:+.1f} pp/yr (latest 3 years weighted 3 / 2 / 1.5, "
            f"older 0.5), applied from the latest {rows[-1]['pct']:.1f}% with its effect halving each year, a "
            f"one-year move capped at 20% of the prior year, never below 0 → {fmt_list(trend)}\n"
            f"- **Flat:** recency-weighted average → {flat[0]:.1f}% every year\n"
            + (f"- **Dollar growth:** SBC dollar CAGR {cagr:+.1%} × 80% = {0.8 * cagr:+.1%} a year, capped each year "
               f"at projected revenue growth → {fmt_list(dollar)}" if dollar else
               "- **Dollar growth:** not enough SBC history for a dollar CAGR, so it uses the Flat values")))
        bt = sbc_backtest(rows)
        if bt:
            err = f" — error {bt['error']:+.0f}%" if bt["error"] is not None else ""
            st.markdown(f"**Backtest:** using only {bt['first']}–{bt['through']}, the Trend method projected "
                        f"{bt['target']} SBC at {bt['predicted']:.1f}% of revenue vs {bt['actual']:.1f}% actual{err} "
                        f"({bt['predicted'] - bt['actual']:+.1f} pp).")
        else:
            st.markdown("**Backtest:** not available — it needs at least 4 years of SBC history (2 to fit the trend, "
                        "plus the 2 years it projects).")
    return sbc


def render_sbc_dcf(proj, sbc, terminal, wacc, net_debt, shares, implied, price) -> None:
    """DCF (after SBC): the standard DCF run on FCF − projected SBC, shown beside the standard result."""
    r = run_dcf([p["fcf"] - s for p, s in zip(proj, sbc)], terminal, wacc, net_debt, shares)
    st.subheader("Standard DCF vs DCF (after SBC)")
    with st.container(border=True, key="sbc_dcf"):
        st.markdown(f"**Shares used: {shares / 1e6:,.1f}M (shares outstanding — SBC already deducted above)**")
        cols = st.columns(3)
        cols[0].metric("Standard DCF implied price", fmt_price(implied))
        if r is None:
            cols[1].metric("DCF (after SBC) implied price", "N/A")
            st.caption("Final-year FCF after SBC is not positive, so the after-SBC terminal value is not meaningful.")
        else:
            up = upside(r["price"], price)
            cols[1].metric("DCF (after SBC) implied price", fmt_price(r["price"]),
                           delta=fmt_pct(up, signed=True) + " vs current" if up is not None else None)
            cols[2].metric("SBC drag", f"{fmt_price(implied - r['price'])} per share",
                           help="Standard DCF implied price − DCF (after SBC) implied price.")
        st.caption("DCF (after SBC) discounts FCF after SBC (FCF − projected SBC) with the same WACC, terminal growth, "
                   "net debt and share count as the standard DCF. The share count is not also grown for SBC: "
                   "deducting SBC and diluting the shares would count the same cost twice.")


# --------------------------------------------------------------------------- #
# Compounding calculator + goal solver
# --------------------------------------------------------------------------- #
def simulate_growth(start: float, monthly: float, annual_return: float, years: int, increase: float) -> pd.DataFrame:
    """Monthly compounding, contributions at month end; the monthly contribution rises once a year."""
    i = annual_return / 100 / 12
    balance = contributed = float(start)
    pmt, rows = float(monthly), []
    for year in range(1, int(years) + 1):
        if year > 1:
            pmt *= 1 + increase / 100
        for _ in range(12):
            balance = balance * (1 + i) + pmt
            contributed += pmt
        rows.append({"Year": year, "Contributed": contributed, "Growth": balance - contributed, "End balance": balance})
    return pd.DataFrame(rows)


def required_monthly(target: float, years: int, start: float, annual_return: float) -> tuple[float, float]:
    """Monthly contribution needed to reach the target (future value of an ordinary annuity).
    Returns (required monthly, what the starting amount alone grows to)."""
    n, i = int(years) * 12, annual_return / 100 / 12
    grown = start * (1 + i) ** n
    if grown >= target:
        return 0.0, grown
    if i == 0:
        return (target - start) / n, grown
    return (target - grown) * i / ((1 + i) ** n - 1), grown


def _money_full(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def render_compounding() -> None:
    _RUN["calc"] = True
    for k in CALC_DEFAULTS:
        seed_calc(k)
    mode = st.radio("Mode", CALC_MODES, horizontal=True, key="calc_mode",
                    index=CALC_MODES.index(st.session_state["calc_mode"]))
    if mode == "Calculator":
        c = st.columns(5)
        start = c[0].number_input("Starting amount ($)", min_value=0.0, step=100.0, format="%.2f", **bound("calc_start"))
        monthly = c[1].number_input("Monthly contribution ($)", min_value=0.0, step=25.0, format="%.2f",
                                    **bound("calc_monthly"))
        ret = c[2].number_input("Expected annual return (%)", min_value=-50.0, max_value=50.0, step=0.5,
                                format="%.2f", **bound("calc_return"))
        years = c[3].number_input("Years", min_value=1, max_value=60, step=1, **bound("calc_years"))
        inc = c[4].number_input("Annual contribution increase (%)", min_value=0.0, max_value=50.0, step=0.5,
                                format="%.2f", **bound("calc_increase"))
        df = simulate_growth(start, monthly, ret, years, inc)
        final, contributed = df["End balance"].iloc[-1], df["Contributed"].iloc[-1]
        st.html(f'<div class="big-label">Final balance after {int(years)} years</div>'
                f'<div class="big-figure">{html.escape(_money_full(final))}</div>')
        st.html(tile_grid([("Total contributed (incl. starting amount)", _money_full(contributed)),
                           ("Total growth", _money_full(final - contributed))]))
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_scatter(x=df["Year"], y=df["Contributed"], name="Contributions", stackgroup="one",
                            line=dict(color="#6b7280"), hovertemplate="Year %{x}: $%{y:,.0f}<extra>Contributions</extra>")
            fig.add_scatter(x=df["Year"], y=df["Growth"], name="Growth", stackgroup="one",
                            line=dict(color=GOLD), hovertemplate="Year %{x}: $%{y:,.0f}<extra>Growth</extra>")
            fig.update_layout(height=360, margin=dict(t=10, b=8, l=6, r=6), dragmode=False,
                              xaxis=dict(title="Year", fixedrange=True), yaxis=dict(tickprefix="$", fixedrange=True),
                              legend=dict(orientation="h", x=0, y=-0.18, yanchor="top"))
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key="calc_chart")
        except ImportError:
            pass
        st.dataframe(pd.DataFrame({
            "Year": df["Year"], "Contributed": df["Contributed"].map(_money_full),
            "Growth": df["Growth"].map(_money_full), "End balance": df["End balance"].map(_money_full),
        }).set_index("Year"), width="stretch")
        st.caption("Monthly compounding at the annual return ÷ 12, contributions at the end of each month; the "
                   "monthly contribution rises once a year by the increase %. Contributed includes the starting amount.")
    else:
        c = st.columns(4)
        target = c[0].number_input("Target amount ($)", min_value=0.0, step=1000.0, format="%.2f", **bound("goal_target"))
        years = c[1].number_input("Years", min_value=1, max_value=60, step=1, **bound("goal_years"))
        start = c[2].number_input("Starting amount ($)", min_value=0.0, step=100.0, format="%.2f", **bound("goal_start"))
        ret = c[3].number_input("Expected annual return (%)", min_value=-50.0, max_value=50.0, step=0.5, format="%.2f",
                                **bound("goal_return"))
        need, grown = required_monthly(target, years, start, ret)
        st.html(f'<div class="big-label">Required monthly contribution</div>'
                f'<div class="big-figure">{html.escape(f"${need:,.2f}")}</div>')
        if need == 0:
            st.success(f"Goal already covered: {_money_full(start)} growing at {ret:g}% a year for {int(years)} years "
                       f"reaches {_money_full(grown)}, which meets the {_money_full(target)} target without monthly "
                       "contributions.".replace("$", r"\$"))
        df = simulate_growth(start, need, ret, years, 0.0)
        st.dataframe(pd.DataFrame({
            "Year": df["Year"], "Contributed": df["Contributed"].map(_money_full),
            "Growth": df["Growth"].map(_money_full), "End balance": df["End balance"].map(_money_full),
        }).set_index("Year"), width="stretch")
        st.caption("Solved with the future value of an ordinary annuity, monthly compounding: required = (target − "
                   "start × (1+i)ⁿ) × i ÷ ((1+i)ⁿ − 1), with i = annual return ÷ 12 and n = years × 12. At 0% return, "
                   "required = (target − start) ÷ months.")


# --------------------------------------------------------------------------- #
# My Daily Feed: news (yfinance), posts (StockTwits), filings (SEC EDGAR)
# --------------------------------------------------------------------------- #
FEED_TTL = 30 * 60


def _epoch(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        return ts.timestamp()
    except (ValueError, TypeError):
        return None


def _http_url(url) -> str | None:
    return url if isinstance(url, str) and url.startswith(("https://", "http://")) else None


@st.cache_data(ttl=FEED_TTL, show_spinner=False)
def feed_news(ticker: str) -> dict:
    import yfinance as yf

    fetched = time.time()
    try:
        raw = []
        try:
            raw = yf.Ticker(ticker).get_news(count=15) or []
        except Exception:
            raw = []
        if not raw:
            raw = yf.Search(ticker, news_count=15).news or []
        items = []
        for n in raw:
            c = n.get("content") if isinstance(n.get("content"), dict) else n
            related = n.get("relatedTickers")
            if isinstance(related, list) and related and ticker not in [str(r).upper() for r in related]:
                continue
            url = None
            for k in ("canonicalUrl", "clickThroughUrl"):
                if isinstance(c.get(k), dict):
                    url = url or _http_url(c[k].get("url"))
            url = url or _http_url(c.get("link"))
            provider = c.get("provider")
            publisher = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher")
            if c.get("title") and url:
                items.append({"source": "News", "ticker": ticker, "title": c["title"], "meta": publisher or "",
                              "ts": _epoch(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")),
                              "url": url})
        return {"items": items, "error": None, "fetched_at": fetched}
    except Exception as exc:
        return {"items": [], "error": type(exc).__name__, "fetched_at": fetched}


SOCIAL_DAILY_CAP = 8   # social posts per ticker per day in the merged feed
NEWS_CAP = 10          # news items per ticker; filings are not capped
CASHTAG_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9.\-]{0,9}")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
LEADING_CASHTAGS_RE = re.compile(r"^(?:\s*\$[A-Za-z][A-Za-z0-9.\-]{0,9})*\s*")


def _empty_stats() -> dict:
    return {"fetched": 0, "replies": 0, "reshares": 0, "spam": {}}


def social_filter(posts: list[dict]) -> tuple[list[dict], dict]:
    """Keep original posts only, then drop spam. A check that cannot run (missing field) keeps the post.

    Not original: replies (reply metadata, or text that opens with an @mention after any leading
    cashtags) and reshares/retweets. Spam: more than 4 cashtags; no words beyond cashtags, links,
    mentions or emoji; identical text from any author; near-identical text (90%+) from the same author."""
    stats = _empty_stats()
    stats["fetched"] = len(posts)
    kept, seen_text, by_author = [], set(), {}
    for p in posts:
        try:
            if p.get("is_reshare"):
                stats["reshares"] += 1
                continue
            if p.get("is_reply") or LEADING_CASHTAGS_RE.sub("", p["body"]).startswith("@"):
                stats["replies"] += 1
                continue
        except Exception:
            pass
        reason = None
        try:
            body = p["body"]
            words = MENTION_RE.sub(" ", CASHTAG_RE.sub(" ", URL_RE.sub(" ", body)))
            norm = " ".join(re.sub(r"[^\w\s]", " ", URL_RE.sub(" ", body.lower())).split())
            if len(CASHTAG_RE.findall(body)) > 4:
                reason = "more than 4 cashtags"
            elif not re.search(r"[A-Za-z0-9]", words):
                reason = "no text beyond cashtags/links"
            elif norm in seen_text:
                reason = "duplicate text"
            elif any(difflib.SequenceMatcher(None, norm, prev).ratio() >= 0.9 for prev in by_author.get(p.get("user"), [])):
                reason = "near-duplicate from the same author"
            else:
                seen_text.add(norm)
                by_author.setdefault(p.get("user"), []).append(norm)
        except Exception:
            reason = None
        if reason:
            stats["spam"][reason] = stats["spam"].get(reason, 0) + 1
            continue
        kept.append(p)
    return kept, stats


def _post_item(ticker: str, p: dict) -> dict:
    body = p["body"]
    return {"source": "Posts", "ticker": ticker, "title": body[:280] + ("…" if len(body) > 280 else ""),
            "meta": p["meta"], "sentiment": p.get("sentiment"), "ts": p.get("ts"), "url": p["url"],
            "engagement": p.get("engagement")}


@st.cache_data(ttl=FEED_TTL, show_spinner=False)
def feed_posts(ticker: str) -> dict:
    """StockTwits symbol stream, filtered to original, non-spam posts."""
    fetched = time.time()
    try:
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
                         headers={"User-Agent": "Mozilla/5.0 (compatible; JD-DCF-Model/1.0)"}, timeout=HTTP_TIMEOUT)
        if r.status_code == 404:
            return {"items": [], "error": None, "fetched_at": fetched, "stats": _empty_stats()}
        if r.status_code != 200:
            return {"items": [], "error": f"HTTP {r.status_code}", "fetched_at": fetched, "stats": _empty_stats()}
        posts = []
        for msg in r.json().get("messages") or []:
            try:
                user = (msg.get("user") or {}).get("username") or "unknown"
                body = (msg.get("body") or "").strip()
                if not body:
                    continue
                likes = (msg.get("likes") or {}).get("total")
                reshared = (msg.get("reshares") or {}).get("reshared_count")
                posts.append({
                    "body": body, "user": user, "meta": f"@{user}",
                    "is_reply": bool((msg.get("conversation") or {}).get("in_reply_to_message_id")),
                    "is_reshare": bool(msg.get("reshare_message")),
                    "engagement": None if likes is None and reshared is None else (likes or 0) + (reshared or 0),
                    "sentiment": ((msg.get("entities") or {}).get("sentiment") or {}).get("basic"),
                    "ts": _epoch(msg.get("created_at")),
                    "url": f"https://stocktwits.com/{user}/message/{msg.get('id')}",
                })
            except Exception:
                continue
        kept, stats = social_filter(posts)
        return {"items": [_post_item(ticker, p) for p in kept], "error": None, "fetched_at": fetched, "stats": stats}
    except Exception as exc:
        return {"items": [], "error": type(exc).__name__, "fetched_at": fetched, "stats": _empty_stats()}


@st.cache_data(ttl=FEED_TTL, show_spinner=False)
def feed_x_posts(ticker: str) -> dict:
    """X (Twitter) recent search — only when an X_BEARER_TOKEN is configured. Replies and retweets are
    excluded in the query itself, and the same spam filters apply."""
    fetched = time.time()
    token = get_secret("X_BEARER_TOKEN")
    if not token:
        return {"items": [], "error": None, "fetched_at": fetched, "stats": None}
    try:
        r = requests.get("https://api.twitter.com/2/tweets/search/recent", timeout=HTTP_TIMEOUT,
                         headers={"Authorization": f"Bearer {token}"},
                         params={"query": f"${ticker} -is:reply -is:retweet", "max_results": 50,
                                 "tweet.fields": "created_at,public_metrics,in_reply_to_user_id,referenced_tweets",
                                 "expansions": "author_id", "user.fields": "username"})
        if r.status_code != 200:
            return {"items": [], "error": f"X HTTP {r.status_code}", "fetched_at": fetched, "stats": _empty_stats()}
        data = r.json()
        users = {u.get("id"): u.get("username") for u in (data.get("includes") or {}).get("users", [])}
        posts = []
        for t in data.get("data") or []:
            try:
                refs = {ref.get("type") for ref in t.get("referenced_tweets") or []}
                pm = t.get("public_metrics") or {}
                user = users.get(t.get("author_id")) or "unknown"
                posts.append({
                    "body": (t.get("text") or "").strip(), "user": user, "meta": f"@{user} on X",
                    "is_reply": bool(t.get("in_reply_to_user_id")) or "replied_to" in refs,
                    "is_reshare": "retweeted" in refs,
                    "engagement": (pm.get("like_count", 0) + pm.get("retweet_count", 0) + pm.get("reply_count", 0)) if pm else None,
                    "ts": _epoch(t.get("created_at")), "url": f"https://x.com/{user}/status/{t.get('id')}",
                })
            except Exception:
                continue
        kept, stats = social_filter([p for p in posts if p["body"]])
        return {"items": [_post_item(ticker, p) for p in kept], "error": None, "fetched_at": fetched, "stats": stats}
    except Exception as exc:
        return {"items": [], "error": f"X {type(exc).__name__}", "fetched_at": fetched, "stats": _empty_stats()}


def _merge_posts(*results: dict) -> dict:
    """Combine social sources into one Posts result with summed filter statistics."""
    stats, items, errors = _empty_stats(), [], []
    for res in results:
        items += res["items"]
        if res.get("error"):
            errors.append(res["error"])
        if res.get("stats"):
            for k in ("fetched", "replies", "reshares"):
                stats[k] += res["stats"][k]
            for reason, n in res["stats"]["spam"].items():
                stats["spam"][reason] = stats["spam"].get(reason, 0) + n
    return {"items": items, "error": "; ".join(errors) or None, "stats": stats,
            "fetched_at": min(r["fetched_at"] for r in results)}


def cap_posts(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """At most SOCIAL_DAILY_CAP posts per ticker per day; higher engagement wins where it is known."""
    by_day: dict = {}
    for it in items:
        day = dt.datetime.fromtimestamp(it["ts"], dt.timezone.utc).date() if it.get("ts") else None
        by_day.setdefault(day, []).append(it)
    shown, more = [], []
    for day_items in by_day.values():
        ranked = sorted(day_items, key=lambda it: (-(it.get("engagement") or 0), -(it.get("ts") or 0)))
        shown += ranked[:SOCIAL_DAILY_CAP]
        more += ranked[SOCIAL_DAILY_CAP:]
    newest = lambda lst: sorted(lst, key=lambda it: it.get("ts") or 0, reverse=True)
    return newest(shown), newest(more)


@st.cache_data(ttl=FEED_TTL, show_spinner=False)
def feed_filings(ticker: str) -> dict:
    fetched = time.time()
    try:
        cik = lookup_cik(ticker)
        if cik is None:
            return {"items": [], "error": None, "fetched_at": fetched}
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=sec_headers(),
                         timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        recent = r.json()["filings"]["recent"]
        items = []
        for i in range(min(12, len(recent.get("form", [])))):
            accn = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{doc}" if doc else
                   f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}")
            desc = recent.get("primaryDocDescription", [""] * (i + 1))[i] or recent["form"][i]
            items.append({"source": "Filings", "ticker": ticker, "title": f"{recent['form'][i]} — {desc}",
                          "meta": f"filed {recent['filingDate'][i]}",
                          "ts": _epoch(recent.get("acceptanceDateTime", [None] * (i + 1))[i] or recent["filingDate"][i]),
                          "url": url})
        return {"items": items, "error": None, "fetched_at": fetched}
    except Exception as exc:
        return {"items": [], "error": type(exc).__name__, "fetched_at": fetched}


def refresh_feed() -> None:
    feed_news.clear()
    feed_posts.clear()
    feed_x_posts.clear()
    feed_filings.clear()


def rel_time(ts: float | None) -> str:
    if ts is None:
        return "time unknown"
    secs = time.time() - ts
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    if secs < 7 * 86400:
        return f"{int(secs // 86400)}d ago"
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%b %d, %Y")


def _feed_card(item: dict) -> str:
    badge = item["source"].lower()
    sentiment = item.get("sentiment")
    sent_html = (f'<span class="feed-sent {"bull" if sentiment == "Bullish" else "bear"}">{html.escape(sentiment)}</span>'
                 if sentiment in ("Bullish", "Bearish") else "")
    return (f'<div class="feed-card"><div class="feed-meta"><span class="feed-badge {badge}">{item["source"].upper()}</span>'
            f'<b>{html.escape(item["ticker"])}</b><span class="muted">{html.escape(item["meta"])}</span>'
            f'<span class="muted">{rel_time(item["ts"])}</span>{sent_html}</div>'
            f'<div class="feed-title"><a href="{html.escape(item["url"])}" target="_blank" rel="noopener noreferrer">'
            f'{html.escape(item["title"])}</a></div></div>')


def render_feed() -> None:
    hl = holdings()
    st.caption(PRIVACY_NOTE)
    with st.form("feed_add", clear_on_submit=True, border=False):
        c = st.columns([3, 1], vertical_alignment="bottom")
        new = c[0].text_input("Add a ticker to your feed", placeholder="e.g. MSFT")
        if c[1].form_submit_button("Add", width="stretch") and new.strip():
            err = add_holding(new)
            if err:
                st.warning(err)
    if hl:
        st.caption("Tracking — tap ✕ to remove:")
        with st.container(horizontal=True, key="feed_chips"):
            for h in hl:
                st.button(f"✕ {h['ticker']}", key=f"feed_rm_{h['ticker']}", on_click=remove_holding, args=(h["ticker"],))
    if not hl:
        st.info("Your feed is empty — add a ticker above to follow its news, posts and filings.")
        return

    tickers = [h["ticker"] for h in hl]
    with st.spinner("Loading your feed…"):
        results = {t: {"News": feed_news(t), "Posts": _merge_posts(feed_posts(t), feed_x_posts(t)),
                       "Filings": feed_filings(t)} for t in tickers}
    oldest = min(r["fetched_at"] for by in results.values() for r in by.values())
    c = st.columns([3, 1], vertical_alignment="center")
    c[0].caption(f"Last refreshed {dt.datetime.fromtimestamp(oldest, dt.timezone.utc).strftime('%b %d, %H:%M UTC')} "
                 f"({rel_time(oldest)}). Feed data is cached for 30 minutes.")
    c[1].button("↻ Refresh", key="feed_refresh", on_click=refresh_feed, width="stretch")

    options = ["All", *tickers]
    if st.session_state.get("feed_ticker") not in options:
        st.session_state.pop("feed_ticker", None)
    pick = st.pills("Ticker", options, default="All", key="feed_ticker") or "All"
    sources = st.pills("Sources", ["News", "Posts", "Filings"], selection_mode="multi",
                       default=["News", "Posts", "Filings"], key="feed_sources") or []

    # Per ticker: news capped at NEWS_CAP, social posts capped per day (overflow under "Show more"), filings all.
    picked = tickers if pick == "All" else [pick]
    display, more_posts = {}, {}
    for t in picked:
        news = sorted(results[t]["News"]["items"], key=lambda it: it["ts"] or 0, reverse=True)
        shown_posts, more_posts[t] = cap_posts(results[t]["Posts"]["items"])
        display[t] = {"News": news[:NEWS_CAP], "Posts": shown_posts, "Filings": results[t]["Filings"]["items"]}
        ps = results[t]["Posts"]["stats"]
        spam = sum(ps["spam"].values())
        spam_detail = ", ".join(f"{n} {why}" for why, n in ps["spam"].items())
        notes = [f"{src} unavailable right now ({res['error']})" for src, res in results[t].items() if res["error"]]
        st.caption(
            f"**{t}** — {len(display[t]['News'])} news (of {len(news)}) · posts: {ps['fetched']} fetched, "
            f"{ps['replies']} replies and {ps['reshares']} reshares removed (original posts only), {spam} spam removed"
            + (f" ({spam_detail})" if spam else "") + f", {len(shown_posts)} shown"
            + (f" (+{len(more_posts[t])} under Show more)" if more_posts[t] else "")
            + f" · {len(display[t]['Filings'])} filings" + (f" · {'; '.join(notes)}" if notes else ""))
        if not any(results[t][src]["items"] for src in results[t]):
            st.info(f"No recent news, posts or filings found for {t}. The other tickers are unaffected.")

    seen, items = set(), []
    for t in picked:
        for src in sources:
            for it in display[t][src]:
                key = it["url"].lower() if it["url"] else it["title"].strip().lower()
                # Filings always show: distinct filings can share a title (e.g. several Form 4s), so use the URL only.
                tkey = key if it["source"] == "Filings" else it["title"].strip().lower()
                if key in seen or tkey in seen:
                    continue
                seen.update({key, tkey})
                items.append(it)
    items.sort(key=lambda it: it["ts"] or 0, reverse=True)
    if not sources:
        st.info("Pick at least one source above to see items.")
    elif not items:
        st.info("Nothing to show for this filter yet.")
    else:
        st.html('<div class="feed-list">' + "".join(_feed_card(it) for it in items[:150]) + "</div>")
    if "Posts" in sources:
        for t in picked:
            extra = [it for it in more_posts[t] if it["url"].lower() not in seen]
            if extra:
                with st.expander(f"Show {len(extra)} more {t} posts"):
                    st.html('<div class="feed-list">' + "".join(_feed_card(it) for it in extra) + "</div>")


# --------------------------------------------------------------------------- #
# My Portfolio
# --------------------------------------------------------------------------- #
QUOTE_TTL = 5 * 60


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def live_quote(ticker: str) -> dict:
    """Last price and previous close from yfinance (cached 5 minutes)."""
    import yfinance as yf

    price = prev = None
    try:
        fi = yf.Ticker(ticker).fast_info
        price, prev = pos(fi.get("lastPrice")), pos(fi.get("previousClose"))
    except Exception:
        pass
    return {"price": price, "prev": prev, "fetched_at": time.time()}


def open_in_valuation(ticker: str) -> None:
    st.session_state["ticker_query"] = ticker
    st.session_state["view"] = "valuation"


def _add_from_form() -> None:
    t = st.session_state.get("pf_new_ticker", "")
    err = add_holding(t, st.session_state.get("pf_new_shares"), st.session_state.get("pf_new_cost")) if t.strip() else None
    st.session_state["_pf_error"] = err


def render_portfolio() -> None:
    hl = holdings()
    st.caption(PRIVACY_NOTE)
    # The editor is collapsed by default; the user's open/closed choice is remembered in this browser.
    if PF_EDITOR_KEY not in st.session_state:
        saved = stored_json(UI_ITEM, {})
        st.session_state[PF_EDITOR_KEY] = bool(saved.get(PF_EDITOR_KEY)) if isinstance(saved, dict) else False
    _RUN["pf_editor"] = True
    with st.expander("✏️ Edit holdings", expanded=st.session_state[PF_EDITOR_KEY], key=PF_EDITOR_KEY,
                     on_change="rerun"):
        render_holdings_editor(hl)
    if not hl:
        st.info("No holdings yet — open ✏️ Edit holdings above to add a ticker.")
        return

    quotes = {h["ticker"]: live_quote(h["ticker"]) for h in hl}
    oldest = min(q["fetched_at"] for q in quotes.values())
    c = st.columns([3, 1], vertical_alignment="center")
    c[0].caption(f"Prices as of {dt.datetime.fromtimestamp(oldest, dt.timezone.utc).strftime('%b %d, %H:%M UTC')} "
                 "(yfinance, refreshed every 5 minutes).")
    c[1].button("↻ Refresh prices", key="pf_refresh", on_click=live_quote.clear, width="stretch")
    _portfolio_stats(hl, quotes)


def render_holdings_editor(hl: list[dict]) -> None:
    """Add/remove holdings and edit shares and average cost (inside the collapsible editor)."""
    with st.form("pf_add", clear_on_submit=True, border=False):
        c = st.columns([1.2, 1, 1, 0.8], vertical_alignment="bottom")
        c[0].text_input("Ticker", key="pf_new_ticker", placeholder="e.g. MSFT")
        c[1].number_input("Shares (optional)", min_value=0.0, value=None, step=1.0, key="pf_new_shares")
        c[2].number_input("Average cost $ (optional)", min_value=0.0, value=None, step=1.0, format="%.2f", key="pf_new_cost")
        c[3].form_submit_button("Add holding", on_click=_add_from_form, width="stretch")
    if st.session_state.pop("_pf_error", None):
        st.warning("That doesn't look like a ticker symbol.")
    if not hl:
        return
    st.markdown("**Holdings** — tap a ticker to open it in Valuation; shares and cost are optional.")
    with st.container(key="pf_rows"):
        for h in hl:
            t = h["ticker"]
            for key, field in ((f"pf_sh_{t}", "shares"), (f"pf_cost_{t}", "cost")):
                if key not in st.session_state:
                    st.session_state[key] = h[field]
            cols = st.columns([1.1, 1, 1, 0.45], vertical_alignment="bottom")
            cols[0].button(f"{t} ↗", key=f"pf_go_{t}", on_click=open_in_valuation, args=(t,), width="stretch",
                           help=f"Open {t} in Valuation")
            cols[1].number_input(f"{t} shares", min_value=0.0, step=1.0, placeholder="Shares", **bound(f"pf_sh_{t}"))
            cols[2].number_input(f"{t} avg cost ($)", min_value=0.0, step=1.0, format="%.2f", placeholder="Avg cost",
                                 **bound(f"pf_cost_{t}"))
            cols[3].button("✕", key=f"pf_rm_{t}", on_click=remove_holding, args=(t,), help=f"Remove {t}",
                           width="stretch")
            h["shares"], h["cost"] = st.session_state[f"pf_sh_{t}"], st.session_state[f"pf_cost_{t}"]


def _portfolio_stats(hl: list[dict], quotes: dict) -> None:
    """Summary tiles, holdings table, tap-through row and allocation chart."""
    rows, counted = [], []
    for h in hl:
        q = quotes[h["ticker"]]
        price, prev = q["price"], q["prev"]
        day_pct = price / prev - 1 if price and prev else None
        row = {"Ticker": h["ticker"], "Shares": h["shares"], "Avg cost": h["cost"], "Price": price,
               "Day change %": day_pct, "Market value": None, "Unrealized $": None, "Unrealized %": None,
               "day_dollars": None}
        if price and h["shares"] and h["cost"] is not None:
            mv, basis = h["shares"] * price, h["shares"] * h["cost"]
            row.update({"Market value": mv, "Unrealized $": mv - basis,
                        "Unrealized %": (mv - basis) / basis if basis else None,
                        "day_dollars": h["shares"] * (price - prev) if prev else 0.0, "basis": basis})
            counted.append(row)
        rows.append(row)

    total_mv = sum(r["Market value"] for r in counted)
    if counted:
        total_basis = sum(r["basis"] for r in counted)
        unreal = total_mv - total_basis
        day = sum(r["day_dollars"] for r in counted)
        day_pct = day / (total_mv - day) if total_mv - day else None
        sign = lambda x: ("+" if x >= 0 else "−") + f"${abs(x):,.2f}"
        st.html(tile_grid([
            ("Total market value", f"${total_mv:,.2f}"), ("Total cost", f"${total_basis:,.2f}"),
            ("Unrealized gain/loss", f"{sign(unreal)} ({unreal / total_basis:+.2%})" if total_basis else sign(unreal)),
            ("Today's change", f"{sign(day)} ({day_pct:+.2%})" if day_pct is not None else sign(day)),
        ], grid_cls="cols4"))
    else:
        st.info("Enter shares and average cost for a holding to see portfolio totals. Ticker-only entries show "
                "price and day change only.")

    fmt = lambda v, f: "—" if v is None else f(v)
    st.dataframe(pd.DataFrame([{
        "Ticker": r["Ticker"], "Shares": fmt(r["Shares"], lambda v: f"{v:,.4g}"),
        "Avg cost": fmt(r["Avg cost"], lambda v: f"${v:,.2f}"),
        "Price": fmt(r["Price"], lambda v: f"${v:,.2f}") if r["Price"] else "unavailable",
        "Market value": fmt(r["Market value"], lambda v: f"${v:,.2f}"),
        "Day change %": fmt(r["Day change %"], lambda v: f"{v:+.2%}"),
        "Unrealized $": fmt(r["Unrealized $"], lambda v: f"{'+' if v >= 0 else '−'}${abs(v):,.2f}"),
        "Unrealized %": fmt(r["Unrealized %"], lambda v: f"{v:+.2%}"),
        "Weight %": fmt(r["Market value"], lambda v: f"{v / total_mv:.1%}" if total_mv else "—"),
    } for r in rows]).set_index("Ticker"), width="stretch")
    if any(q["price"] is None for q in quotes.values()):
        bad = ", ".join(t for t, q in quotes.items() if q["price"] is None)
        st.caption(f"No price available for {bad} — check the symbol. Other holdings are unaffected.")
    # Tap-through to Valuation stays available while the editor is collapsed.
    with st.container(horizontal=True, key="pf_open_row", vertical_alignment="center"):
        st.caption("Open in Valuation:")
        for h in hl:
            st.button(f"{h['ticker']} ↗", key=f"pf_open_{h['ticker']}", on_click=open_in_valuation, args=(h["ticker"],),
                      type="tertiary")

    if counted:
        try:
            import plotly.graph_objects as go

            alloc = sorted(counted, key=lambda r: r["Market value"])
            fig = go.Figure(go.Bar(x=[r["Market value"] for r in alloc], y=[r["Ticker"] for r in alloc], orientation="h",
                                   marker_color=GOLD, text=[f"{r['Market value'] / total_mv:.1%}" for r in alloc],
                                   textposition="auto",
                                   hovertemplate="%{y}: $%{x:,.2f}<extra></extra>"))
            fig.update_layout(height=80 + 44 * len(alloc), margin=dict(t=30, b=8, l=6, r=6), dragmode=False,
                              title=dict(text="Allocation by market value", font=dict(size=13)),
                              xaxis=dict(tickprefix="$", fixedrange=True), yaxis=dict(fixedrange=True))
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key="pf_alloc")
        except ImportError:
            pass


storage_slot = st.container()  # fixed position for the (invisible) browser-storage component
shown_ticker = None
try:
    shown_ticker = main()
except Exception as exc:  # last-resort guard: never show a traceback
    st.error(f"Something went wrong while building this view: {type(exc).__name__}: {exc}")
finally:
    try:
        persist_inputs(shown_ticker, storage_slot)
    except Exception:
        pass  # saving is best-effort; never break the page
    render_footer()
