"""
data_loader.py — Yahoo Finance data downloader
"""
import yfinance as yf
import streamlit as st

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
    "BAJFINANCE.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "ONGC.NS", "NESTLEIND.NS", "POWERGRID.NS", "TECHM.NS", "NTPC.NS",
    "DIVISLAB.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "BPCL.NS",
    "GRASIM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS", "UPL.NS",
    "TATACONSUM.NS", "APOLLOHOSP.NS", "M&M.NS", "LTIM.NS", "HINDALCO.NS"
]

@st.cache_data(ttl=3600)
def download_stock_data(ticker: str, period: str = "2y"):
    """Download historical OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] Download failed for {ticker}: {e}")
        return None
