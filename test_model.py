from app.model import train_model

def test_train_model():
    import numpy as np
    X = np.random.rand(100, 8)
    y = np.random.rand(100)
    model = train_model(X, y, n_splits=3)
    assert hasattr(model, 'predict')
