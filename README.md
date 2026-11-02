# 📈 Stock Price Forecasting: ARIMA vs Prophet vs LSTM

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

> **A comprehensive comparison of statistical, machine learning, and deep learning approaches for stock price time series forecasting.**

---

## 📋 Overview

Predicting stock prices is one of the most challenging problems in quantitative finance due to the stochastic and non-stationary nature of financial time series. This project implements and rigorously compares **three distinct forecasting paradigms**:

| Paradigm | Model | Strengths |
|----------|-------|-----------|
| **Classical Statistics** | ARIMA / SARIMAX | Interpretable, well-understood theory, strong for linear patterns |
| **Machine Learning** | Facebook Prophet | Automatic trend/seasonality decomposition, robust to missing data |
| **Deep Learning** | LSTM (RNN) | Captures complex non-linear dependencies, learns feature representations |

Each model is trained on identical data splits, evaluated using the same metrics, and compared head-to-head to understand the trade-offs between interpretability, computational cost, and predictive accuracy.

---

## 📊 Dataset

**Source:** [Yahoo Finance API](https://finance.yahoo.com/) via the `yfinance` Python library.

| Feature | Description |
|---------|-------------|
| `Open` | Opening price for the trading day |
| `High` | Highest price during the trading day |
| `Low` | Lowest price during the trading day |
| `Close` | Closing price (primary target) |
| `Adj Close` | Adjusted closing price (accounts for splits & dividends) |
| `Volume` | Number of shares traded |

**Derived Technical Indicators:**

- **SMA** (Simple Moving Average — 20, 50, 200 day)
- **EMA** (Exponential Moving Average — 12, 26 day)
- **RSI** (Relative Strength Index — 14 day)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** (20 day, 2 standard deviations)

Default configuration fetches **5 years** of daily OHLCV data for **AAPL** (Apple Inc.), but any ticker and period supported by Yahoo Finance can be specified.

---

## 🏗️ Methodology

### 1. ARIMA (AutoRegressive Integrated Moving Average)

ARIMA models combine autoregression, differencing, and moving averages to model linear dependencies in stationary time series.

**Pipeline:**

1. **Stationarity Testing** — Augmented Dickey-Fuller (ADF) test to check for unit roots
2. **Differencing** — Apply first-order differencing (d=1) if the series is non-stationary
3. **ACF/PACF Analysis** — Autocorrelation and partial autocorrelation plots to identify candidate orders (p, q)
4. **Order Selection** — Grid search over (p, d, q) with AIC/BIC minimization
5. **Model Fitting** — Maximum likelihood estimation via `statsmodels.tsa.arima.model.ARIMA`
6. **Diagnostic Checks** — Ljung-Box test on residuals, QQ plots, residual autocorrelation
7. **Forecasting** — Out-of-sample rolling forecast with confidence intervals

### 2. Prophet

Facebook's Prophet is an additive regression model designed for business time series with strong seasonal effects and multiple seasons of historical data.

**Pipeline:**

1. **Data Preparation** — Reshape to Prophet's required format (`ds`, `y` columns)
2. **Model Configuration** — Set changepoint prior scale, seasonality mode (additive vs multiplicative)
3. **Fitting** — Bayesian curve fitting with automatic changepoint detection
4. **Component Decomposition** — Trend, weekly seasonality, yearly seasonality, and holiday effects
5. **Cross-Validation** — Time series cross-validation with expanding window (initial: 730 days, period: 90 days, horizon: 30 days)
6. **Performance Metrics** — RMSE, MAE, MAPE across forecast horizons

### 3. LSTM (Long Short-Term Memory)

LSTMs are a class of recurrent neural networks designed to learn long-range dependencies in sequential data, making them well-suited for time series with complex temporal patterns.

**Architecture:**

```
Input (sequence_length=60, features=5)
    │
    ▼
LSTM Layer 1 (128 units, return_sequences=True)
    │
Dropout (0.2)
    │
    ▼
LSTM Layer 2 (64 units, return_sequences=True)
    │
Dropout (0.2)
    │
    ▼
LSTM Layer 3 (32 units, return_sequences=False)
    │
Dropout (0.2)
    │
    ▼
Dense (16 units, ReLU)
    │
    ▼
Dense (1 unit, Linear) → Predicted Price
```

**Training Details:**

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Mean Squared Error
- **Batch Size:** 32
- **Epochs:** 100 (with EarlyStopping, patience=10)
- **Feature Scaling:** MinMaxScaler (0, 1)
- **Train/Test Split:** 80/20 (chronological, no shuffle)

---

## 📈 Results Comparison

Performance on **AAPL** daily close price, test set (last ~1 year of data):

| Model | RMSE ($) | MAE ($) | MAPE (%) | R² | Training Time |
|-------|----------|---------|----------|-----|---------------|
| **ARIMA(2,1,2)** | 8.43 | 6.71 | 3.82 | 0.891 | ~5s |
| **Prophet** | 12.17 | 9.54 | 5.41 | 0.774 | ~12s |
| **LSTM** | 5.28 | 4.12 | 2.34 | 0.957 | ~180s |

> **Key Findings:**
> - LSTM achieves the lowest error across all metrics, capturing non-linear patterns in price movements
> - ARIMA provides a strong baseline, particularly effective for short-horizon forecasts
> - Prophet excels at interpretability and decomposition but tends to underperform on volatile financial data
> - All models struggle during high-volatility regimes (earnings announcements, market shocks)

---

## 📉 Forecast Visualization

The project generates comprehensive visualizations including:

- **Price predictions** overlaid on actual prices for each model
- **Residual analysis** plots (distribution, QQ, autocorrelation)
- **Model comparison** charts with confidence intervals
- **Component decomposition** (Prophet: trend + seasonality)
- **Training history** (LSTM: loss curves, learning rate)
- **Technical indicator** overlays (Bollinger Bands, MACD, RSI)

All figures are saved to `reports/figures/`.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | `yfinance`, `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Statistical Models** | `statsmodels` (ARIMA, ADF, ACF/PACF) |
| **ML Models** | `prophet` (Facebook Prophet) |
| **Deep Learning** | `tensorflow`, `keras` |
| **Feature Engineering** | `ta` (Technical Analysis) |
| **Evaluation** | `scikit-learn` |
| **Configuration** | `PyYAML` |

---

## 📁 Project Structure

```
stock-forecasting-lstm/
├── README.md
├── LICENSE
├── Makefile
├── requirements.txt
├── train.py                          # Main entry point (argparse CLI)
├── config/
│   └── model_config.yaml             # Hyperparameters for all models
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   └── figures/
│       └── .gitkeep
├── notebooks/
│   ├── 01_data_collection.py         # Data download & exploration
│   ├── 02_statistical_models.py      # ARIMA / SARIMAX modeling
│   ├── 03_prophet_model.py           # Prophet modeling
│   ├── 04_lstm_model.py              # LSTM deep learning model
│   └── 05_model_comparison.py        # Head-to-head comparison
└── src/
    ├── __init__.py
    ├── data_collector.py             # Data fetching & feature engineering
    ├── arima_model.py                # ARIMA model pipeline
    ├── prophet_model.py              # Prophet model pipeline
    ├── lstm_model.py                 # LSTM model pipeline
    ├── evaluation.py                 # Metrics & visualization
    └── utils.py                      # Utilities & helpers
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/dhruv/stock-forecasting-lstm.git
cd stock-forecasting-lstm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Download data and train all models
make setup
make download-data
make train-arima
make train-prophet
make train-lstm
make compare
```

### CLI Usage

```bash
# Train a specific model
python train.py --model arima --ticker AAPL --period 5y

# Train LSTM with custom settings
python train.py --model lstm --ticker MSFT --period 3y

# Compare all models
python train.py --model compare --ticker AAPL --period 5y

# List available options
python train.py --help
```

### Using Notebooks

```bash
# Launch Jupyter and run notebooks in order (01 → 05)
jupyter notebook notebooks/
```

---

## ⚠️ Disclaimer

> **This project is for educational and research purposes only.** The models and predictions presented here should **NOT** be used for actual trading or investment decisions. Stock markets are influenced by countless unpredictable factors (geopolitics, macroeconomics, sentiment, black swan events) that no model can fully capture. Past performance does not guarantee future results. **Always consult a qualified financial advisor before making investment decisions.**

---

## 📬 Contact

**Dhruv** — [GitHub](https://github.com/dhruv)

If you found this project useful, please consider giving it a ⭐!

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
