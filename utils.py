def prepare_for_prediction(df):
    from .features import compute_features
    X, y, df_feat = compute_features(df)
    return X, y, df_feat
