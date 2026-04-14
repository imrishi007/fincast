# FinCast

FinCast is a multimodal forecasting system for 30 technology stocks. It predicts realized volatility and direction by combining price history, SEC filings, sector graph structure, macro signals, and earnings surprise features.

Current headline metrics from the latest model are Vol R2 = 0.9315 and Direction AUC = 0.5925.

## Why This Repository Is Useful

1. It contains a full research pipeline from data processing to model training and evaluation.
2. It includes trained checkpoints and result files for immediate inspection.
3. It ships with a web application for interactive inference.

## Model In Plain Terms

1. Price branch: CNN plus BiLSTM over 60 day windows.
2. Text branch: FinBERT encoding for SEC filings.
3. Graph branch: GAT over sector relations.
4. Macro branch: compact MLP encoder.
5. Fusion: gated multimodal trunk with dual heads for volatility and direction.
6. Key addition: HAR RV skip path that improves volatility accuracy.

## What Is In The Repository

1. `src/` contains models, datasets, training code, and evaluation logic.
2. `scripts/` contains entry points for training, backtesting, and experiments.
3. `data/` contains raw inputs, targets, and embedding artifacts.
4. `models/` contains checkpoints and result files.
5. `notebooks/` keeps the latest analysis notebooks.
6. `webapp/` contains backend and frontend for interactive usage.
7. `reports/` contains compact summaries and benchmark outputs.

## Quick Start

### 1. Install

```bash
git clone https://github.com/imrishi007/fincast.git
cd fincast
pip install -r requirements.txt
```

### 2. Run Latest Training Pipeline

```bash
python scripts/run_phase15_experiments.py
```

### 3. Read Latest Results

```python
import json

with open("models/phase15_experiment_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print(results)
```

## Run The Web Application

### 1. Backend

```bash
cd webapp/backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd webapp/frontend
npm install
npm run dev
```

## Important Files To Check First

1. `models/best_model.pt`
2. `models/phase15_experiment_results.json`
3. `reports/phase4_to_phase14_results_summary.md`
4. `notebooks/17_phase15_results.ipynb`
5. `scripts/run_phase15_experiments.py`

## References

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.
2. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
3. Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., & Li, H. (2007). Learning to rank: From pairwise approach to listwise approach. Proceedings of the 24th International Conference on Machine Learning, 129-136.
4. Corsi, F. (2009). A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics, 7(2), 174-196.
5. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.
6. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. The Journal of Finance, 25(2), 383-417.
7. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.
8. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223-2273.
9. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations.
10. Markowitz, H. (1952). Portfolio selection. The Journal of Finance, 7(1), 77-91.
11. Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246-256.
12. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. International Conference on Learning Representations.
13. Yang, Y., Uy, M. C. S., & Huang, A. (2020). FinBERT: A pretrained language model for financial communications. arXiv:2006.08097.
