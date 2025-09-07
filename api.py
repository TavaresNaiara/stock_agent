from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import load_model
from .data_loader import load_data
from .features import compute_features

app = FastAPI(title="Stock Agent - Regression API")
_model = None

class PredictRequest(BaseModel):
    ticker: str = "PETR4.SA"
    years: int = 1

@app.on_event("startup")
def startup_event():
    global _model
    try:
        _model = load_model()
    except Exception:
        _model = None

@app.get("/")
def root():
    return {"status":"ok","model_loaded": _model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    df = load_data(req.ticker, years=req.years)
    X, y, df_feat = compute_features(df)
    if X.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Not enough data to compute features")
    model = _model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")
    preds = model.predict(X)
    last_pred = float(preds[-1])
    return {
        "ticker": req.ticker,
        "last_index": str(df_feat.index[-1]),
        "prediction_t_plus_1": last_pred
    }
