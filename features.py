import numpy as np
import pandas as pd

def compute_features(df: pd.DataFrame, lags=(1,2,3)):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["vol_20"] = df["ret"].rolling(20).std()
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(window=14).mean()
    down = -delta.clip(upper=0).rolling(window=14).mean()
    df["rsi_14"] = 100 - (100/(1 + (up / (down.replace(0, np.nan)))))
    df["rsi_14"].fillna(50, inplace=True)
    for k in lags:
        df[f"ret_lag{k}"] = df["ret"].shift(k)
    df["y"] = df["ret"].shift(-1)
    df = df.dropna()
    feature_cols = ["ret", "ret_lag1", "ret_lag2", "ret_lag3", "sma_5", "sma_20", "rsi_14", "vol_20"]
    X = df[feature_cols].values
    y = df["y"].values
    return X, y, df
