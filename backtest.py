import numpy as np
import pandas as pd

def signals_from_preds(preds, threshold=0.0):
    signals = np.zeros_like(preds)
    signals[preds > threshold] = 1
    signals[preds < -threshold] = -1
    return signals

def simple_backtest(df_feat, preds, initial_cap=1.0):
    df = df_feat.iloc[:len(preds)].copy()
    df["pred"] = preds
    df["signal"] = signals_from_preds(df["pred"].values)
    df["strategy_ret"] = df["signal"].shift(0) * df["y"]
    df["strategy_ret"].fillna(0, inplace=True)
    df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod() * initial_cap
    df["cum_buyhold"] = (1 + df["y"]).cumprod() * initial_cap
    metrics = {
        "final_strategy": df["cum_strategy"].iloc[-1],
        "final_buyhold": df["cum_buyhold"].iloc[-1],
        "sharpe_strategy": df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-9) * np.sqrt(252)
    }
    return df, metrics
