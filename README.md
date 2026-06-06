# Stock Analytics & Forecasting — Major Project

An interactive NIFTY 50 stock forecasting dashboard using LSTM, Random Forest, SVM, and ARIMA.

## Features
- **Data**: Fetches real OHLCV data from Yahoo Finance for all 50 NIFTY stocks
- **Technical Analysis**: RSI, MACD, Bollinger Bands, EMA 20/50
- **4 Forecasting Models**:
  - 🧠 **LSTM** — 2-layer LSTM with dropout (TensorFlow/Keras)
  - 🌲 **Random Forest** — 200-tree ensemble with lag features
  - 🔵 **SVM (SVR)** — RBF kernel with recursive prediction
  - 📊 **ARIMA** — Auto ARIMA with pmdarima / statsmodels fallback
- **Metrics**: RMSE, MAE, R² for all models
- **Charts**: Candlestick, Bollinger Bands, RSI, MACD, Volume (Plotly)
- **Forecast Chart**: Side-by-side forecast comparison for up to 60 days

## Installation

```bash
git clone https://github.com/PrincePanwar/stock-analytics-forecasting.git
cd stock-analytics-forecasting
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## Project Structure
```
stock-analytics/
├── app.py            # Streamlit dashboard
├── data_loader.py    # Yahoo Finance downloader
├── analytics.py      # Technical indicators
├── models/
│   ├── lstm_model.py
│   ├── rf_model.py
│   ├── svm_model.py
│   └── arima_model.py
└── requirements.txt
```

## Graceful Degradation
All models have fallbacks — if TensorFlow is not installed, LSTM falls back to linear regression. If pmdarima is missing, ARIMA uses statsmodels. The app always runs.

## Tech Stack
| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Charts | Plotly |
| Data | yfinance |
| LSTM | TensorFlow / Keras |
| Random Forest | scikit-learn |
| SVM | scikit-learn |
| ARIMA | pmdarima / statsmodels |

## Disclaimer
This project is for educational purposes only. Not financial advice.

## Author
**Prince Panwar** — CSE @ SRM University | AWS Certified Cloud Practitioner (1000/1000)
