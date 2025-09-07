import yfinance as yf
import pandas as pd
from .config import DEFAULT_TICKER, YEARS

def load_data(ticker: str = DEFAULT_TICKER, years: int = YEARS) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    df = df[["close", "volume"]].dropna()
    return df
