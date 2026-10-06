"""
FX Treasury Copilot - 7-day dynamic FX projection for corporate cash-flow management.

Hugging Face pipelines used (course requirement: at least two):
  1. text-classification : fine-tuned sentiment model          (Student A, Model 1)
  2. summarization       : fine-tuned news-briefing model       (Student B, Model 2)
  3. time-series forecast: amazon/chronos-bolt-small            (pre-trained, zero-shot)

Business logic layers on top of the model forecast:
  - Live market anchoring  : latest 5 daily closes from yfinance
  - Volatility adjustment  : ^VIX (20), ^V2TX (20), ^VFTSE (15) and Eurozone ESI (100)
  - Macro event simulation : +0.2% volatility from day 2 for EUR, GBP, INR
  - Weekend transaction fee: x0.995 on Saturday/Sunday projections (HSBC alignment)
  - Execution advice       : best / worst day to convert, given the trade direction

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# TODO: after pushing your fine-tuned models to the Hub (see the two fine-tuning
# notebooks), replace the two ids below. Course rule: the models used here must be
# exactly the models produced by your notebooks.
SENTIMENT_MODEL_ID = "chena2339/finbert-fx-sentiment"   # Student A - Model 1
SUMMARY_MODEL_ID   = "chena2339/t5-fx-briefing"         # Student B - Model 2
CHRONOS_MODEL_ID   = "amazon/chronos-bolt-small"              # pre-trained, NOT fine-tuned

FX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "HKD/USD": "HKDUSD=X",
    "INR/USD": "INRUSD=X",
}

# Volatility indices and their neutral baselines.
VOL_BENCHMARKS = {"^VIX": 20.0, "^V2TX": 20.0, "^VFTSE": 15.0}
ESI_BASELINE = 100.0  # Eurozone Economic Sentiment Indicator, long-run average = 100

# Google News RSS queries used to collect pair-specific headlines (no API key needed).
NEWS_QUERIES = {
    "EUR/USD": "euro dollar ECB exchange rate",
    "GBP/USD": "pound sterling dollar Bank of England",
    "HKD/USD": "Hong Kong dollar peg HKMA",
    "INR/USD": "rupee dollar RBI exchange rate",
}

FORECAST_DAYS   = 7
WEEKEND_FACTOR  = 0.995   # assumed 0.5% weekend transaction fee
EVENT_SPIKE     = 0.002   # +0.2% volatility spike for the macro-event simulation
EVENT_PAIRS     = {"EUR/USD", "GBP/USD", "INR/USD"}   # HKD is pegged -> excluded
SENTIMENT_WEIGHT = 0.0015  # max ~ +/-0.15% daily drift from news sentiment
VOL_WEIGHT       = 0.0005  # daily drift per relative volatility-index deviation
ESI_WEIGHT       = 0.0005  # EUR-specific daily drift per relative ESI deviation

st.set_page_config(page_title="FX Treasury Copilot", layout="wide")


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_fx_history(ticker: str, period: str = "6mo") -> pd.Series:
    """Download daily closes with multi-source fallback (yfinance -> free API -> static default)."""
    import yfinance as yf
    import requests

    # 1. Primary Source: yfinance
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) > 0:
                close.index = pd.to_datetime(close.index).tz_localize(None)
                return close
    except Exception:
        pass

    # 2. Secondary Source: Free open.er-api.com (No API key required)
    try:
        base_curr = ticker[:3]  # e.g. "EUR" from "EURUSD=X"
        url = f"https://open.er-api.com/v6/latest/{base_curr}"
        resp = requests.get(url, timeout=5).json()
        if resp.get("result") == "success":
            rate = resp["rates"]["USD"]
            dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq="D")
            return pd.Series(rate, index=dates)
    except Exception:
        pass

    # 3. Final Fallback: Static defaults if all network calls fail
    defaults = {
        "EURUSD=X": 1.0850,
        "GBPUSD=X": 1.2700,
        "HKDUSD=X": 0.1280,
        "INRUSD=X": 0.0120,
    }
    base_price = defaults.get(ticker, 1.0000)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq="D")
    return pd.Series(base_price, index=dates)

@st.cache_data(ttl=3600, show_spinner=False)
def load_vol_indices() -> dict:
    """Latest close of each volatility index; missing indices are returned as None."""
    values = {}
    for ticker in VOL_BENCHMARKS:
        try:
            s = load_fx_history(ticker, period="5d")
            values[ticker] = float(s.iloc[-1]) if len(s) else None
        except Exception:
            values[ticker] = None
    return values


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_headlines(query: str, n: int) -> list:
    """Latest headlines from Google News RSS for one query."""
    import urllib.parse

    import feedparser

    url = ("https://news.google.com/rss/search?q="
           + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en")
    feed = feedparser.parse(url)
    return [e.title for e in feed.entries[:n]]


# ---------------------------------------------------------------------------
# Hugging Face pipelines (cached resources -> loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the forecasting model (Chronos-Bolt) ...")
def load_forecaster():
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(CHRONOS_MODEL_ID, device_map="cpu")


@st.cache_resource(show_spinner="Loading the fine-tuned sentiment model ...")
def load_sentiment_model(model_id: str):
    from transformers import pipeline

    # HF pipeline #1: fine-tuned by Student A.
    return pipeline("text-classification", model=model_id, truncation=True, max_length=128)


@st.cache_resource(show_spinner="Loading the fine-tuned briefing model ...")
def load_summary_model(model_id: str):
    from transformers import pipeline

    # HF pipeline #2: fine-tuned by Student B.
    return pipeline("summarization", model=model_id)


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------
def chronos_forecast(pipe, history: pd.Series, days: int = FORECAST_DAYS):
    """7-day forecast. Returns (median, p10, p90). Never calls Chronos twice on failure."""
    import torch

    values = pd.to_numeric(history, errors="coerce").dropna().to_numpy(dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("No closes available for Chronos.")
    last_spot = float(values[-1])

    # Bolt patch size is 16. Left-pad with NaN so unfold always sees a full patch.
    min_length = 32
    if values.size < min_length:
        pad = np.full(min_length - values.size, np.nan, dtype=np.float32)
        values = np.concatenate([pad, values])
    context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # (1, time)

    try:
        quantiles, _ = pipe.predict_quantiles(
            context, prediction_length=days, quantile_levels=[0.1, 0.5, 0.9]
        )
        q = quantiles[0].detach().float().cpu().numpy()  # (days, 3): p10, p50, p90
        p10, median, p90 = q[:, 0], q[:, 1], q[:, 2]
    except Exception as exc:
        st.warning(f"Chronos fallback used ({type(exc).__name__}).")
        drift = float(np.log(values[-1] / values[-2])) if np.isfinite(values[-2]) else 0.0
        steps = np.arange(1, days + 1)
        median = last_spot * np.exp(drift * steps)
        p10, p90 = median * 0.99, median * 1.01

    fill = lambda a: np.nan_to_num(np.asarray(a, dtype=float), nan=last_spot)
    return fill(median), fill(p10), fill(p90)


def score_headlines(clf, headlines: list):
    """Score headlines with the fine-tuned sentiment model.

    Returns (signed_score, label_counts, detail_rows):
      signed_score in [-1, 1] = mean( P(positive) - P(negative) ) over all headlines.
    """
    if not headlines:
        return 0.0, {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}, []
    dists = clf(headlines, top_k=None, batch_size=16)    # full 3-class distribution
    signed, counts, rows = [], {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}, []
    for headline, dist in zip(headlines, dists):
        d = {x["label"].upper(): x["score"] for x in dist}
        signed.append(d.get("POSITIVE", 0.0) - d.get("NEGATIVE", 0.0))
        top = max(dist, key=lambda x: x["score"])
        counts[top["label"].upper()] += 1
        rows.append({"Headline": headline,
                     "Sentiment": top["label"].title(),
                     "Confidence": round(float(top["score"]), 3)})
    return float(np.mean(signed)), counts, rows


def generate_briefing(summarizer, headlines: list, max_items: int = 8) -> str:
    """Compress the top headlines into one executive briefing (T5-small, fine-tuned)."""
    if not headlines:
        return "No headlines available."
    text = " ".join(headlines[:max_items])[:3000]        # stay inside the 512-token window
    return summarizer(text, max_length=64, min_length=15)[0]["summary_text"]


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------
def compute_daily_vol_adjustment(pair: str, vol_vals: dict, esi: float) -> float:
    """Daily drift adjustment (decimal, e.g. 0.001 = +0.1%/day) from vol indices and ESI.

    Sign convention: pairs are quoted XXX/USD, so risk-off (index above baseline)
    strengthens the dollar and pushes the pair DOWN.
    """
    adj = 0.0
    vix = vol_vals.get("^VIX")
    if vix is not None:                                   # global risk sentiment -> all pairs
        adj += -VOL_WEIGHT * (vix - VOL_BENCHMARKS["^VIX"]) / VOL_BENCHMARKS["^VIX"]
    if pair == "EUR/USD":
        v2tx = vol_vals.get("^V2TX")
        if v2tx is not None:
            adj += -VOL_WEIGHT * (v2tx - VOL_BENCHMARKS["^V2TX"]) / VOL_BENCHMARKS["^V2TX"]
        adj += ESI_WEIGHT * (esi - ESI_BASELINE) / ESI_BASELINE   # eurozone sentiment
    elif pair == "GBP/USD":
        vftse = vol_vals.get("^VFTSE")
        if vftse is not None:
            adj += -VOL_WEIGHT * (vftse - VOL_BENCHMARKS["^VFTSE"]) / VOL_BENCHMARKS["^VFTSE"]
    elif pair == "HKD/USD":
        adj *= 0.2   # HKD is pegged to USD; global risk sentiment barely moves it
    return adj


def build_projection(history, median, p10, p90, sentiment_score, daily_vol_adj,
                     pair, event_on, weekend_fee_on, seed=7) -> pd.DataFrame:
    """Apply the corporate-finance rules on top of the raw model forecast."""
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1),
                          periods=FORECAST_DAYS, freq="D")
    rng = np.random.default_rng(seed)   # fixed seed -> reproducible event simulation

    rows, cum_adj = [], 0.0
    for t in range(FORECAST_DAYS):
        day_no = t + 1
        # 1) news-sentiment drift, decaying over the horizon
        cum_adj += SENTIMENT_WEIGHT * sentiment_score * (0.9 ** t)
        # 2) volatility-index / ESI drift
        cum_adj += daily_vol_adj
        # 3) macro-event simulation: +0.2% volatility from day 2 (EUR, GBP, INR only)
        band_widen = 0.0
        if event_on and pair in EVENT_PAIRS and day_no >= 2:
            band_widen = EVENT_SPIKE
            cum_adj += rng.normal(0.0, EVENT_SPIKE / 4)   # small zero-mean event noise

        rate = median[t] * (1 + cum_adj)
        low = p10[t] * (1 + cum_adj) * (1 - band_widen)
        high = p90[t] * (1 + cum_adj) * (1 + band_widen)

        # 4) weekend transaction fee (HSBC alignment): 0.5% off on Sat/Sun
        note = ""
        if weekend_fee_on and dates[t].weekday() >= 5:
            rate, low, high = rate * WEEKEND_FACTOR, low * WEEKEND_FACTOR, high * WEEKEND_FACTOR
            note = "Weekend fee applied"

        rows.append({"Date": dates[t].date(), pair: round(float(rate), 4),
                     "Low (P10)": round(float(low), 4),
                     "High (P90)": round(float(high), 4), "Note": note})
    return pd.DataFrame(rows)


def recommend_days(proj: pd.DataFrame, pair: str, direction: str):
    """Best / avoid execution day given the trade direction; weekends excluded (fee)."""
    weekday_mask = ~pd.to_datetime(proj["Date"]).dt.weekday.isin([5, 6])
    tradable = proj[weekday_mask] if weekday_mask.any() else proj
    if direction.startswith("Buy"):      # buying foreign currency -> want the pair LOW
        return (tradable.loc[tradable[pair].idxmin()],
                tradable.loc[tradable[pair].idxmax()])
    return (tradable.loc[tradable[pair].idxmax()],   # selling -> want the pair HIGH
            tradable.loc[tradable[pair].idxmin()])


def upload_to_gsheet(df: pd.DataFrame, sheet_url: str, tab: str):
    """Upload the projection matrix to a Google Sheet tab.

    Requires a service-account key stored in .streamlit/secrets.toml as
    [gcp_service_account]. The Sheet must be shared with the service-account email.
    """
    import gspread

    creds = dict(st.secrets["gcp_service_account"])
    gc = gspread.service_account_from_dict(creds)
    ws = gc.open_by_url(sheet_url).worksheet(tab)
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("FX Treasury Copilot")
st.caption("7-day dynamic FX projection for corporate cash-flow management - "
           "deep-learning forecast + fine-tuned news sentiment + executive briefings.")

with st.sidebar:
    st.header("Settings")
    sentiment_id = st.text_input("Sentiment model id (Hugging Face)", SENTIMENT_MODEL_ID)
    summary_id = st.text_input("Briefing model id (Hugging Face)", SUMMARY_MODEL_ID)
    selected_pairs = st.multiselect("Currency pairs", list(FX_PAIRS),
                                    default=["EUR/USD", "GBP/USD"])
    direction = st.radio("Your cash-flow position",
                         ["Buy foreign currency with USD (accounts payable)",
                          "Sell foreign currency for USD (accounts receivable)"])
    esi = st.slider("Eurozone ESI (baseline 100)", 90.0, 110.0, 100.0, 0.1,
                    help="Economic Sentiment Indicator, published monthly by the EU Commission.")
    event_on = st.checkbox("Macro event risk next week (+0.2% vol from day 2)", value=True)
    weekend_fee_on = st.checkbox("Apply 0.5% weekend transaction fee", value=True)
    n_headlines = st.slider("Headlines per pair", 5, 20, 10)
    st.divider()
    with st.expander("Google Sheets upload (optional)"):
        gs_enable = st.checkbox("Upload projection to Google Sheet", value=False)
        gs_url = st.text_input("Sheet URL")
        gs_tab = st.text_input("Tab name", value="FX_Projection")

if not selected_pairs:
    st.warning("Select at least one currency pair in the sidebar.")
    st.stop()

for label, mid in [("Sentiment", sentiment_id), ("Briefing", summary_id)]:
    if "your-username" in mid:
        st.error(f"{label} model id is still a placeholder. Fine-tune the model with the "
                 "corresponding notebook, push it to the Hugging Face Hub, then paste your "
                 "model id in the sidebar (or edit the constants at the top of app.py).")
        st.stop()

# --- Live market anchoring -------------------------------------------------
with st.spinner("Fetching live market data ..."):
    histories, errors = {}, []
    for pair in selected_pairs:
        try:
            histories[pair] = load_fx_history(FX_PAIRS[pair])
        except Exception as exc:
            errors.append(f"{pair}: {exc}")
    vol_vals = load_vol_indices()

for e in errors:
    st.error("Failed to load " + e)
if not histories:
    st.stop()

st.subheader("Market snapshot - last 5 daily closes")

# Extract the latest 5 daily closes and align them by date
history_data = {}
for pair, ticker in FX_PAIRS.items():
    s = load_fx_history(ticker, period="10d")
    # Convert index to date-only to fix outer-join issues caused by time components
    s.index = s.index.date
    # Deduplicate keeping the last price per day, then select the latest 5 trading days
    history_data[pair] = s.groupby(s.index).last().tail(5)

# Combine into a DataFrame for display
snapshot_df = pd.DataFrame(history_data)
st.dataframe(snapshot_df, use_container_width=True)

vol_display = {t: (f"{v:.2f}" if v is not None else "n/a") for t, v in vol_vals.items()}
st.caption("Volatility indices (baseline): " +
           ", ".join(f"{t} = {v} ({VOL_BENCHMARKS[t]:.0f})" for t, v in vol_display.items()) +
           f"  |  ESI = {esi:.1f} (100)")

# --- Forecast ---------------------------------------------------------------
forecaster = load_forecaster()
sentiment_clf = load_sentiment_model(sentiment_id)
summarizer = load_summary_model(summary_id)

projections, sentiments, briefings, headline_rows = {}, {}, {}, {}
with st.spinner("Running forecast + news analysis ..."):
    for pair in histories:
        median, p10, p90 = chronos_forecast(forecaster, histories[pair])
        headlines = fetch_headlines(NEWS_QUERIES[pair], n_headlines)
        score, counts, rows = score_headlines(sentiment_clf, headlines)
        sentiments[pair] = (score, counts)
        headline_rows[pair] = rows
        briefings[pair] = generate_briefing(summarizer, headlines)
        vol_adj = compute_daily_vol_adjustment(pair, vol_vals, esi)
        projections[pair] = build_projection(histories[pair], median, p10, p90,
                                             score, vol_adj, pair, event_on, weekend_fee_on)

# --- Per-pair dashboards -----------------------------------------------------
import plotly.graph_objects as go

tabs = st.tabs(selected_pairs)
for tab, pair in zip(tabs, selected_pairs):
    with tab:
        proj, hist = projections[pair], histories[pair]
        score, counts = sentiments[pair]
        spot = float(hist.iloc[-1])
        end_rate = float(proj[pair].iloc[-1])
        best, avoid = recommend_days(proj, pair, direction)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spot", f"{spot:.4f}")
        c2.metric("7-day projection", f"{end_rate:.4f}", f"{(end_rate / spot - 1) * 100:+.2f}%")
        c3.metric("Best execution day", str(best["Date"]), f"{best[pair]:.4f}")
        c4.metric("Day to avoid", str(avoid["Date"]), f"{avoid[pair]:.4f}")
        st.caption("Weekends are excluded from the recommendation because of the 0.5% fee. "
                   f"News sentiment score: {score:+.3f} "
                   f"(+ = {counts['POSITIVE']} pos / {counts['NEUTRAL']} neu / {counts['NEGATIVE']} neg)")

        # Chart: recent history + forecast with P10-P90 band.
        hist_view = hist.tail(30)
        xs = [hist_view.index[-1]] + pd.to_datetime(proj["Date"]).tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_view.index, y=hist_view.values,
                                 name="History", line=dict(color="#9aa0a6")))
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=[spot] + proj["High (P90)"].tolist() + [spot] + proj["Low (P10)"].tolist()[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0), name="P10-P90 band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=[spot] + proj[pair].tolist(),
                                 name="Forecast", line=dict(color="#1f77b4", width=3)))
        fig.update_layout(title=f"{pair} - last 30 days + 7-day projection",
                          yaxis_title=pair, height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily executive briefing (fine-tuned T5-small)")
        st.info(briefings[pair])

        with st.expander("Projection table"):
            st.dataframe(proj, use_container_width=True, hide_index=True)
        with st.expander("News sentiment details"):
            st.dataframe(pd.DataFrame(headline_rows[pair]),
                         use_container_width=True, hide_index=True)

# --- Combined projection matrix ---------------------------------------------
st.subheader("Combined 7-day projection matrix")
combined = projections[selected_pairs[0]][["Date", selected_pairs[0]]]
for pair in selected_pairs[1:]:
    combined = combined.merge(projections[pair][["Date", pair]], on="Date", how="outer")
combined = combined.sort_values("Date").ffill().round(4)   # NaN safety
st.dataframe(combined, use_container_width=True, hide_index=True)
st.download_button("Download projection (CSV)", combined.to_csv(index=False),
                   "fx_projection_7d.csv", "text/csv")

if gs_enable:
    if not gs_url:
        st.warning("Enter a Google Sheet URL in the sidebar first.")
    else:
        try:
            upload_to_gsheet(combined, gs_url, gs_tab)
            st.success(f"Projection uploaded to tab '{gs_tab}'.")
        except Exception as exc:
            st.error("Google Sheets upload failed. Check st.secrets['gcp_service_account'] "
                     f"and that the sheet is shared with the service account. Details: {exc}")

st.caption("Models: " + sentiment_id + " | " + summary_id + " | " + CHRONOS_MODEL_ID +
           ". Educational project - not investment advice.")
