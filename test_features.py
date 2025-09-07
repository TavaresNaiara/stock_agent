import pandas as pd
from app.features import compute_features

def test_features_basic():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    close = pd.Series(range(30), index=dates)
    df = pd.DataFrame({"close": close, "volume": 1000}, index=dates)
    X, y, df_feat = compute_features(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 8
