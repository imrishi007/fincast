# Volatility Forecasting Web Application

Interactive web UI for the Phase 15 Multimodal Volatility Prediction Model.

## Overview

This application provides a modern web interface to explore volatility predictions for 30 NASDAQ tech stocks using our deep learning model that achieves **R² = 0.9315**.

## Architecture

```
webapp/
├── backend/          # FastAPI Python server (port 8000)
│   ├── main.py       # API endpoints & model inference
│   └── requirements.txt
└── frontend/         # React + Vite app (port 5173)
    ├── src/
    │   ├── App.jsx   # Main application component
    │   └── App.css   # Styling
    └── package.json
```

## Features

- **Stock Selection**: Choose from 30 NASDAQ tech stocks across 6 sectors
- **Real-time Predictions**: Volatility forecasts using live market data from Yahoo Finance
- **Interactive Charts**: 
  - Volatility forecast timeline (historical + predicted)
  - Modality gate weights pie chart
  - HAR-RV feature bar chart
  - Price & returns history
- **Direction Signal**: Bullish/Bearish prediction with probability
- **Model Info**: Architecture details and performance metrics

## Quick Start

### 1. Start Backend (port 8000)
```bash
cd webapp/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend (port 5173)
```bash
cd webapp/frontend
npm install
npm run dev
```

### 3. Open Browser
Navigate to: **http://localhost:5173**

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check & status |
| `GET /stocks` | List all available stocks |
| `GET /predict/{ticker}` | Get volatility prediction for a stock |
| `GET /historical/{ticker}` | Get historical price/volatility data |
| `GET /model-info` | Get model architecture & metrics |

## Model Details

- **Architecture**: Multimodal fusion (Price CNN-BiLSTM, FinBERT text, GAT graph, Macro MLP)
- **Test R²**: 0.9315 (Phase 15 Deeper Vol Head)
- **Improvements**: +1.1% vs Phase 14, +19.6% vs HAR-RV baseline

## Stock Universe

| Sector | Stocks |
|--------|--------|
| Core Tech | AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, INTC, TSLA, ORCL |
| Semiconductors | QCOM, TXN, AVGO, MRVL, KLAC |
| Cloud/SaaS | CRM, ADBE, NOW, SNOW, DDOG |
| Internet | NFLX, UBER, PYPL, SNAP |
| Hardware | DELL, AMAT, LRCX |
| Diversified | IBM, CSCO, HPE |

## Screenshots

The UI features:
- Dark theme with gradient background
- Responsive card-based layout
- Real-time API status indicator
- Interactive stock selector grouped by sector
- Recharts-powered visualizations

## Tech Stack

- **Backend**: FastAPI, PyTorch, uvicorn, yfinance
- **Frontend**: React 18, Vite, Recharts, Lucide Icons, Axios
- **Model**: Phase 15 Deeper Vol Head (635K parameters)

---
*Multimodal Volatility Forecasting • Dept. of CS (AI-ML) • Adani University • 2025*
