"""
Deep Learning LSTM Sequence Forecasting Model using TensorFlow/Keras.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data: np.ndarray, seq_length: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Convert time series array into sliding window input/target sequences."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: tuple[int, int], units: list[int] = [64, 32], dropout: float = 0.2):
    """Build multi-layer LSTM model architecture with Dropout regularization."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model
