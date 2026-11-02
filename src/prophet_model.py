"""
Meta Prophet forecasting model module.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame into Prophet required columns: ds (date) and y (target)."""
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(df.index)
    prophet_df["y"] = df["Close"].values
    return prophet_df


def fit_prophet(df_prophet: pd.DataFrame):
    """Fit Prophet model on prepared DataFrame."""
    try:
        from prophet import Prophet
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)
        return model
    except ImportError:
        print("[WARN] Prophet library not installed.")
        return None


def forecast_prophet(model, periods: int = 30) -> pd.DataFrame:
    """Make future dataframe and predict."""
    if model is None:
        return pd.DataFrame()
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast
