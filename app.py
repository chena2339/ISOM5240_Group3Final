"""HSBC 7-day FX conversion screen for ISOM5240.

Pipeline A: fine-tuned FinBERT text classifier (appreciation / depreciation / neutral).
Pipeline B: fine-tuned DistilBERT event classifier
(central_bank / inflation_print / jobs_data / no_macro_event).

Both pipelines must load the weights produced by the Colab notebooks.
A base checkpoint is not a silent substitute: the course penalizes a mismatch
between the fine-tuned file and the Streamlit model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fx_engine import (
    fetch_market_anchor,
    fetch_vol_indices,
    project_seven_days,
    recommend,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_A = APP_DIR / "models" / "finbert_fx_direction"
MODEL_B = APP_DIR / "models" / "distilbert_macro_event"
DIRECTION_LABELS = ["appreciation", "depreciation", "neutral"]
EVENT_LABELS = ["central_bank", "inflation_print", "jobs_data", "no_macro_event"]

SAMPLE_HEADLINES = {
    "EUR": "Dealers mark EUR stronger versus USD into the London fix. CPI is scheduled and EUR dealers cut risk into the print.",
    "GBP": "Sterling slips versus the dollar as offshore funding tightens. Traders reprice sterling before the central bank rate decision.",
    "HKD": "The Hong Kong dollar stays inside the convertibility band versus USD. No scheduled macro release is driving Hong Kong dollar this session.",
    "INR": "Stop-loss selling extends the INR decline against USD. Labour-market figures are ahead and Indian rupee flow turns cautious.",
}


@st.cache_resource(show_spinner="Loading fine-tuned pipelines")
def load_pipelines(path_a: str, path_b: str):
    from transformers import pipeline

    direction = pipeline("text-classification", model=path_a, tokenizer=path_a)
    event = pipeline("text-classification", model=path_b, tokenizer=path_b)
    return direction, event


def score_headline(pipe, text: str) -> tuple[str, float]:
    # Truncate to the model context. FinBERT and DistilBERT both use 512 tokens.
    result = pipe(text[:1500], truncation=True, top_k=1)
    row = result[0] if isinstance(result, list) else result
    return str(row["label"]).lower(), float(row["score"])


def main() -> None:
    st.set_page_config(page_title="HSBC FX 7-day screen", layout="wide")
    st.title("HSBC corporate FX conversion screen")
    st.caption("Company: HSBC · https://www.hsbc.com.hk · Two Hugging Face pipelines, then a 7-day path.")

    with st.sidebar:
        st.header("Model folders")
        path_a = st.text_input("Pipeline A folder", str(MODEL_A))
        path_b = st.text_input("Pipeline B folder", str(MODEL_B))
        esi = st.number_input("Eurozone ESI (manual, baseline 100)", value=100.0, step=0.5)
        side = st.selectbox(
            "Conversion side",
            options=["sell_fc", "buy_fc"],
            format_func=lambda x: "Sell foreign currency for USD" if x == "sell_fc" else "Buy foreign currency with USD",
        )
        st.caption("Weekend rows use an assumed 0.5% conversion cost (x0.995). Weekdays do not.")

    missing = [p for p in (path_a, path_b) if not Path(p).exists()]
    if missing:
        st.error(
            "Fine-tuned model folder not found. Run the Colab notebooks and copy the saved "
            "folders to GitHub_App_Files/models before deploying. Missing: " + ", ".join(missing)
        )
        st.stop()

    direction_pipe, event_pipe = load_pipelines(path_a, path_b)
    st.subheader("Headlines scored by the two pipelines")
    texts = {}
    cols = st.columns(2)
    for i, code in enumerate(("EUR", "GBP", "HKD", "INR")):
        with cols[i % 2]:
            texts[code] = st.text_area(f"{code} headline", SAMPLE_HEADLINES[code], height=90)

    if st.button("Build 7-day projection", type="primary"):
        signals = {}
        score_rows = []
        for code, text in texts.items():
            direction, d_score = score_headline(direction_pipe, text)
            event, e_score = score_headline(event_pipe, text)
            signals[code] = {"direction": direction, "confidence": d_score, "event": event}
            score_rows.append(
                {
                    "Currency": code,
                    "Direction": direction,
                    "Direction confidence": round(d_score, 4),
                    "Event": event,
                    "Event confidence": round(e_score, 4),
                }
            )
        spots, drifts, anchor_note = fetch_market_anchor(5)
        vol = fetch_vol_indices()
        forecast = project_seven_days(spots, drifts, signals, vol, esi=esi)
        advice = recommend(forecast, side=side)

        st.info(anchor_note)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VIX", f"{vol['VIX']:.2f}", f"{vol['VIX'] - 20:.2f} vs 20")
        m2.metric("V2TX", f"{vol['V2TX']:.2f}", f"{vol['V2TX'] - 20:.2f} vs 20")
        m3.metric("VFTSE", f"{vol['VFTSE']:.2f}", f"{vol['VFTSE'] - 20:.2f} vs 20")
        m4.metric("ESI input", f"{esi:.1f}", f"{esi - 100:.1f} vs 100")

        st.subheader("Pipeline scores")
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True)
        st.subheader("7-day quotes")
        show_cols = ["Date", "Weekday", "HKD/USD", "INR/USD", "GBP/USD", "EUR/USD"]
        st.dataframe(forecast[show_cols], use_container_width=True)
        st.subheader("Best day and day to avoid")
        st.dataframe(advice, use_container_width=True)
        st.caption(
            "Day 2 onward applies a 0.2% event shock to EUR, GBP and INR only when Pipeline B "
            "is not no_macro_event. HKD drift is damped for the peg. NaNs are forward-filled, then replaced with 0."
        )


if __name__ == "__main__":
    main()

