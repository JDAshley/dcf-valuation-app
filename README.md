# DCF Valuation App

A Streamlit app that values a US-listed company with a 5-year discounted cash flow model and several
multiple-based price targets, using free data sources.

> **For education only. Not financial advice.**

## What it does

1. **Ticker search.** Enter a ticker (`MSFT`, `BRK.B`) or a company name (`coca-cola`). The header shows the
   company name, sector, price and market cap.
2. **Valuation snapshot.** Trailing and forward P/E, PEG, P/S, P/B, EV/Sales, EV/EBITDA, FCF yield and
   dividend yield. A multiple shows N/A when it is negative or unavailable.
3. **Financial model.** One Actual year (the latest fiscal year from SEC EDGAR) plus 5 projected years.
   You can edit every assumption in the highlighted input boxes: revenue growth % and EBITDA margin % for
   each year, and one FCF conversion % (FCF ÷ EBITDA). Defaults come from analyst consensus where it exists
   (FMP, or yfinance revenue estimates) and from the latest actual results otherwise. A caption always says
   which source was used, and "Reset to consensus" restores the defaults. A table and a grouped bar chart
   (Actual vs Projected) show revenue, EBITDA and FCF.
4. **DCF.** The DCF discounts the 5 projected FCFs from the model. A Gordon Growth terminal value is taken
   on year-5 FCF, then the app goes from enterprise value through net debt to an implied share price.
   Sliders set WACC and terminal growth.
5. **Price target setter.** The method is picked automatically and you can override it:
   P/E for profitable companies, P/S for unprofitable ones, EV/EBITDA when year-1 EBITDA is positive, and
   P/FCF when year-1 FCF is positive. P/S, EV/EBITDA and P/FCF use year-1 figures from your model.
6. **Summary.** Every applicable method's target and upside, plus a blended target (simple average).
7. **Financials warning.** Banks and insurers get a warning that a firm-level DCF does not apply to them.
   The DCF is hidden by default for these companies.

## Saved inputs

Every input (ticker, assumption boxes, WACC, terminal growth, target multiples and the price target method)
is saved automatically in **your own browser's localStorage**, under one key per ticker
(`dcf_inputs_<TICKER>`) plus `dcf_last_ticker`. Nothing is stored on the server, so this works on Streamlit
Community Cloud and on phones. When you return, the last ticker and its inputs are restored, with a
"Restored your saved assumptions" note. **Reset to consensus** deletes that ticker's saved inputs.

Notes:
- Saved inputs are per browser and per site address. `localhost:8501` and `localhost:8502` count as
  different sites, so they keep separate saves. The same goes for a phone and a laptop.
- Only inputs you have changed are stored. A ticker you only viewed keeps following the live consensus
  defaults.
- The storage component lives in `browser_store/index.html`. Deploy it together with `app.py`.

## Data sources

| Data | Source |
|---|---|
| Historical fundamentals | [SEC EDGAR Company Facts API](https://www.sec.gov/edgar/sec-api-documentation) (no key needed) |
| Profile, price, ratios, analyst estimates | [Financial Modeling Prep](https://site.financialmodelingprep.com/) free API (optional key) |
| Fallback (no FMP key, or gaps in FMP's free tier) | [yfinance](https://github.com/ranaroussi/yfinance) |

All responses are cached for 24 hours with `st.cache_data`.

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

### Optional: API key and SEC contact

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
FMP_API_KEY = "your-fmp-api-key-here"
SEC_USER_AGENT = "Your Name your.email@example.com"
```

You can also set these as environment variables (`FMP_API_KEY`, `SEC_USER_AGENT`). Without an FMP key the
app uses yfinance. `secrets.toml` is git-ignored, so your key is never committed.

## Limitations

- EDGAR fundamentals cover US-GAAP filers only. For foreign IFRS filers (20-F) the historicals are
  unavailable, and the DCF falls back to market-data FCF where it can.
- Total debt is approximate: long-term debt (including the current portion) plus short-term borrowings.
  Cash excludes marketable securities.
- The DCF uses the latest fiscal-year FCF as its base, so a single unusual year (a large one-off tax
  payment, for example) moves the result a lot.
