import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODEL_DIR
from pathlib import Path

MODEL_PATH = Path(MODEL_DIR) / "ridge_model.joblib"

def train_model(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RidgeCV(alphas=np.logspace(-4, 4, 30), cv=tscv)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds, truths = [], []
    for tr_idx, te_idx in tscv.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict(X[te_idx])
        preds.extend(p); truths.extend(y[te_idx])
    mae = mean_absolute_error(truths, preds)
    rmse = mean_squared_error(truths, preds, squared=False)
    r2 = r2_score(truths, preds)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    return path

def load_model(path=MODEL_PATH):
    return joblib.load(path)
