"""
Stock Analytics & Forecasting
Author: Prince Panwar
Major Project — CSE @ SRM University
Description: NIFTY 50 stock forecasting using LSTM, Random Forest, SVM, and ARIMA.
             Interactive Streamlit dashboard for analytics and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from data_loader import download_stock_data, NIFTY50_TICKERS
from models.lstm_model import run_lstm
from models.rf_model import run_random_forest
from models.svm_model import run_svm
from models.arima_model import run_arima
from analytics import compute_indicators, compute_returns_stats

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analytics & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#0f4c81; }
    .metric-card { background:#f0f4ff; border-radius:8px; padding:12px; margin:4px; }
    .alert-up    { color:#16a34a; font-weight:600; }
    .alert-down  { color:#dc2626; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/NSE_logo.svg/1200px-NSE_logo.svg.png", width=180)
    st.markdown("## ⚙️ Settings")

    ticker = st.selectbox("Select Stock", NIFTY50_TICKERS, index=0)
    period = st.selectbox("Historical Period", ["1y","2y","3y","5y"], index=1)
    forecast_days = st.slider("Forecast Days", 7, 60, 30)

    st.markdown("---")
    st.markdown("### Models to Run")
    run_lstm_flag  = st.checkbox("LSTM (Deep Learning)",    value=True)
    run_rf_flag    = st.checkbox("Random Forest",           value=True)
    run_svm_flag   = st.checkbox("Support Vector Machine",  value=True)
    run_arima_flag = st.checkbox("ARIMA",                   value=True)

    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Author:** Prince Panwar  \n**Project:** Major CSE Project")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📈 Stock Analytics & Forecasting</p>', unsafe_allow_html=True)
st.markdown(f"**Ticker:** `{ticker}` | **Period:** {period} | **Forecast:** {forecast_days} days")

if not run_btn:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to begin.")
    st.markdown("""
    ### How it works
    1. **Data** is fetched from Yahoo Finance (NIFTY 50 stocks)
    2. **Technical indicators** (RSI, MACD, Bollinger Bands, EMA) are computed
    3. **Four ML/DL models** generate price forecasts:
        - 🧠 LSTM — captures long-term sequence patterns
        - 🌲 Random Forest — ensemble tree-based regression
        - 🔵 SVM — support vector regression
        - 📊 ARIMA — classical statistical time-series model
    4. Results are compared side-by-side with performance metrics (RMSE, MAE, R²)
    """)
    st.stop()

# ─── Data Loading ─────────────────────────────────────────────────────────────
with st.spinner(f"Downloading {ticker} data..."):
    df = download_stock_data(ticker, period)

if df is None or df.empty:
    st.error("Failed to download data. Check your internet connection or try another ticker.")
    st.stop()

df = compute_indicators(df)
close = df['Close'].values
dates = df.index

# ─── Price Chart ──────────────────────────────────────────────────────────────
st.subheader("📊 Historical Price & Volume")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3], vertical_spacing=0.05)

fig.add_trace(go.Candlestick(x=dates, open=df['Open'], high=df['High'],
                              low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=df['EMA_20'], name="EMA 20",
                          line=dict(color='orange', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=df['BB_Upper'], name="BB Upper",
                          line=dict(color='purple', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=df['BB_Lower'], name="BB Lower",
                          line=dict(color='purple', width=1, dash='dot'),
                          fill='tonexty', fillcolor='rgba(128,0,128,0.05)'), row=1, col=1)
fig.add_trace(go.Bar(x=dates, y=df['Volume'], name="Volume",
                      marker_color='lightblue'), row=2, col=1)
fig.update_layout(height=550, xaxis_rangeslider_visible=False, template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# ─── Key Metrics ──────────────────────────────────────────────────────────────
st.subheader("📋 Key Statistics")
stats = compute_returns_stats(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"₹{close[-1]:,.2f}")
c2.metric("52W High",      f"₹{df['High'].max():,.2f}")
c3.metric("52W Low",       f"₹{df['Low'].min():,.2f}")
c4.metric("Avg Daily Ret", f"{stats['mean_return']*100:.3f}%")
c5.metric("Volatility",    f"{stats['volatility']*100:.2f}%")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**RSI (14)**")
    rsi_val = df['RSI'].iloc[-1]
    rsi_color = "🔴 Overbought" if rsi_val > 70 else ("🟢 Oversold" if rsi_val < 30 else "🟡 Neutral")
    st.write(f"{rsi_val:.2f} — {rsi_color}")
with col2:
    st.markdown("**MACD Signal**")
    macd_diff = df['MACD'].iloc[-1] - df['Signal'].iloc[-1]
    st.write(f"{'🟢 Bullish' if macd_diff > 0 else '🔴 Bearish'} ({macd_diff:.2f})")

# ─── Run Models ───────────────────────────────────────────────────────────────
st.subheader("🤖 Forecasting Models")
results = {}

if run_lstm_flag:
    with st.spinner("Running LSTM..."):
        results['LSTM'] = run_lstm(close, forecast_days)

if run_rf_flag:
    with st.spinner("Running Random Forest..."):
        results['Random Forest'] = run_random_forest(df, forecast_days)

if run_svm_flag:
    with st.spinner("Running SVM..."):
        results['SVM'] = run_svm(df, forecast_days)

if run_arima_flag:
    with st.spinner("Running ARIMA..."):
        results['ARIMA'] = run_arima(close, forecast_days)

if not results:
    st.warning("No models selected.")
    st.stop()

# ─── Forecast Chart ───────────────────────────────────────────────────────────
forecast_dates = pd.date_range(dates[-1] + timedelta(days=1), periods=forecast_days, freq='B')
colors = {'LSTM': '#e63946', 'Random Forest': '#2a9d8f', 'SVM': '#e76f51', 'ARIMA': '#457b9d'}

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dates[-90:], y=close[-90:], name="Historical",
                           line=dict(color='black', width=2)))
for model, res in results.items():
    if res and 'forecast' in res:
        fig2.add_trace(go.Scatter(
            x=forecast_dates, y=res['forecast'], name=model,
            line=dict(color=colors.get(model, 'gray'), width=2, dash='dash')
        ))
fig2.update_layout(title="Price Forecast Comparison", height=420,
                    xaxis_title="Date", yaxis_title="Price (₹)", template='plotly_white')
st.plotly_chart(fig2, use_container_width=True)

# ─── Model Performance ────────────────────────────────────────────────────────
st.subheader("📐 Model Performance Metrics")
metrics_data = []
for model, res in results.items():
    if res and 'metrics' in res:
        m = res['metrics']
        metrics_data.append({
            "Model": model,
            "RMSE":  f"{m.get('rmse', 0):.4f}",
            "MAE":   f"{m.get('mae', 0):.4f}",
            "R²":    f"{m.get('r2', 0):.4f}",
            "Forecast (Next Day)": f"₹{res['forecast'][0]:,.2f}" if res.get('forecast') else "—"
        })
if metrics_data:
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

# ─── Technical Indicators ─────────────────────────────────────────────────────
with st.expander("📉 Technical Indicators"):
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=("RSI (14)", "MACD"))
    fig3.add_trace(go.Scatter(x=dates, y=df['RSI'], name="RSI",
                               line=dict(color='purple')), row=1, col=1)
    fig3.add_hline(y=70, line_dash="dot", line_color="red", row=1, col=1)
    fig3.add_hline(y=30, line_dash="dot", line_color="green", row=1, col=1)
    fig3.add_trace(go.Scatter(x=dates, y=df['MACD'],   name="MACD",   line=dict(color='blue')),  row=2, col=1)
    fig3.add_trace(go.Scatter(x=dates, y=df['Signal'], name="Signal", line=dict(color='orange')),row=2, col=1)
    fig3.update_layout(height=450, template='plotly_white')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("Data sourced from Yahoo Finance via yfinance. For educational purposes only. Not financial advice.")
