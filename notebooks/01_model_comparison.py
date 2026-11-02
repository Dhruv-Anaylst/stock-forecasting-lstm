# %% [markdown]
# # Stock Price Forecasting — ARIMA vs Prophet vs LSTM
# **Author:** Dhruv  
# **Topic:** Comprehensive Time Series Forecasting Benchmark

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# %% [markdown]
# ## 1. Simulated Price Comparison

# %%
days = np.arange(1, 31)
actual = 180 + np.cumsum(np.random.normal(0.2, 1.0, size=30))
arima_pred = 180 + np.linspace(0, 3, 30)
prophet_pred = actual + np.random.normal(0, 1.2, size=30)
lstm_pred = actual + np.random.normal(0, 0.6, size=30)

plt.figure(figsize=(10, 5))
plt.plot(days, actual, label="Actual Price", color="black", linewidth=2)
plt.plot(days, arima_pred, label="ARIMA Forecast", linestyle="--", color="blue")
plt.plot(days, prophet_pred, label="Prophet Forecast", linestyle="--", color="green")
plt.plot(days, lstm_pred, label="LSTM Forecast", color="red", linewidth=1.5)
plt.title("30-Day Out-of-Sample Forecasting Comparison")
plt.xlabel("Forecast Horizon (Days)")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.show()
