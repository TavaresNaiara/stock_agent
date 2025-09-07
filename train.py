#!/usr/bin/env python3
from app.data_loader import load_data
from app.features import compute_features
from app.model import train_model, evaluate_model, save_model
from app.config import DEFAULT_TICKER, YEARS
import pprint

def main(ticker=DEFAULT_TICKER, years=5):
    df = load_data(ticker, years)
    X, y, df_feat = compute_features(df)
    model = train_model(X, y, n_splits=5)
    metrics = evaluate_model(model, X, y, n_splits=5)
    path = save_model(model)
    print("Model saved to:", path)
    pprint.pprint(metrics)

if __name__ == "__main__":
    main()
