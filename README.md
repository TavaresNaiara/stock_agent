# Stock Agent - Regressão Linear para Análise de Ações

Este projeto implementa um agente de IA baseado em Regressão Linear para prever retornos de ações do mercado financeiro e gerar sinais de compra/venda.

## Estrutura do Projeto

```
stock_agent/
├── README.md
├── requirements.txt
├── Dockerfile
├── app/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── backtest.py
│   ├── api.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   └── run_api.sh
├── models/
└── tests/
    ├── test_features.py
    └── test_model.py
```

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Treine o modelo:
   ```bash
   python scripts/train.py
   ```

3. Rode a API:
   ```bash
   bash scripts/run_api.sh
   ```

   Endpoints disponíveis:
   - `GET /` → status da API
   - `POST /predict` com JSON `{"ticker":"PETR4.SA","years":1}` → previsão do próximo retorno

4. Backtest manual (exemplo em Python REPL):
   ```python
   from app.data_loader import load_data
   from app.features import compute_features
   from app.model import load_model
   from app.backtest import simple_backtest

   df = load_data("PETR4.SA", years=5)
   X, y, df_feat = compute_features(df)
   model = load_model()
   preds = model.predict(X)
   df_bt, metrics = simple_backtest(df_feat, preds)
   print(metrics)
   df_bt[['cum_strategy','cum_buyhold']].plot()
   ```

## Observações

- Este projeto é **apenas acadêmico** e não constitui recomendação de investimento.
- Expansões possíveis: incluir normalização de features, ensembles, ML avançado, custos de transação no backtest.
