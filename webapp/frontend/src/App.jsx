import { useState, useEffect } from "react";
import axios from "axios";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [historical, setHistorical] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    checkApiStatus();
    fetchStocks();
    fetchModelInfo();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/`);
      setApiStatus(response.data.model_loaded ? "online" : "model-loading");
    } catch {
      setApiStatus("offline");
    }
  };

  const fetchStocks = async () => {
    try {
      const response = await axios.get(`${API_URL}/stocks`);
      setStocks(response.data);
    } catch (err) {
      console.error("Failed to fetch stocks:", err);
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-info`);
      setModelInfo(response.data);
    } catch (err) {
      console.error("Failed to fetch model info:", err);
    }
  };

  const handleStockSelect = async (ticker) => {
    if (!ticker) {
      setSelectedStock("");
      setPrediction(null);
      setHistorical(null);
      return;
    }
    setSelectedStock(ticker);
    setLoading(true);
    setError(null);

    try {
      const [predResponse, histResponse] = await Promise.all([
        axios.get(`${API_URL}/predict/${ticker}`),
        axios.get(`${API_URL}/historical/${ticker}?days=60`),
      ]);
      setPrediction(predResponse.data);
      setHistorical(histResponse.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch prediction data");
    } finally {
      setLoading(false);
    }
  };

  // Group stocks by sector for the dropdown optgroups
  const groupedStocks = stocks.reduce((acc, stock) => {
    if (!acc[stock.sector]) acc[stock.sector] = [];
    acc[stock.sector].push(stock);
    return acc;
  }, {});

  // Gate weights as bar chart data
  const gateData = prediction
    ? [
        { name: "Price", weight: +(prediction.gate_weights.price * 100).toFixed(1) },
        { name: "Text", weight: +(prediction.gate_weights.text * 100).toFixed(1) },
        { name: "Graph", weight: +(prediction.gate_weights.graph * 100).toFixed(1) },
        { name: "Macro", weight: +(prediction.gate_weights.macro * 100).toFixed(1) },
      ]
    : [];

  // HAR-RV bar data
  const harData = prediction
    ? [
        { name: "1-Day", rv: +(prediction.har_rv_features.daily_rv * 100).toFixed(2) },
        { name: "5-Day", rv: +(prediction.har_rv_features.weekly_rv * 100).toFixed(2) },
        { name: "22-Day", rv: +(prediction.har_rv_features.monthly_rv * 100).toFixed(2) },
      ]
    : [];

  // Radar data for modalities
  const radarData = prediction
    ? [
        { modality: "Price", weight: prediction.gate_weights.price * 100, fullMark: 100 },
        { modality: "Text", weight: prediction.gate_weights.text * 100, fullMark: 100 },
        { modality: "Graph", weight: prediction.gate_weights.graph * 100, fullMark: 100 },
        { modality: "Macro", weight: prediction.gate_weights.macro * 100, fullMark: 100 },
      ]
    : [];

  // Compute rolling return stats from historical data
  const computeStats = () => {
    if (!historical || !historical.data || historical.data.length < 2) return null;
    const returns = historical.data
      .map((d) => d.return)
      .filter((r) => r !== 0 && r !== undefined);
    if (returns.length === 0) return null;
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance =
      returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;
    const std = Math.sqrt(variance);
    const maxRet = Math.max(...returns);
    const minRet = Math.min(...returns);
    const positive = returns.filter((r) => r > 0).length;
    return {
      meanReturn: mean.toFixed(3),
      stdReturn: std.toFixed(3),
      maxReturn: maxRet.toFixed(2),
      minReturn: minRet.toFixed(2),
      positiveRatio: ((positive / returns.length) * 100).toFixed(1),
      nDays: returns.length,
    };
  };

  const stats = computeStats();

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>Volatility Forecasting</h1>
            <p>Multimodal Gated Fusion Model</p>
          </div>
          <div className="header-right">
            <div className={`api-status ${apiStatus}`}>
              <span className="status-dot"></span>
              <span>
                {apiStatus === "online"
                  ? "API ONLINE"
                  : apiStatus === "offline"
                    ? "API OFFLINE"
                    : "CONNECTING"}
              </span>
            </div>
            {modelInfo && (
              <div className="model-r2">R2 = {modelInfo.metrics.test_r2}</div>
            )}
          </div>
        </div>
      </header>

      <main className="main">
        {/* Stock Selector */}
        <section className="stock-selector">
          <label>Select Equity</label>
          <div className="stock-dropdown">
            <select
              value={selectedStock}
              onChange={(e) => handleStockSelect(e.target.value)}
              disabled={loading}
            >
              <option value="">-- Choose a stock --</option>
              {Object.entries(groupedStocks).map(([sector, sectorStocks]) => (
                <optgroup key={sector} label={sector}>
                  {sectorStocks.map((stock) => (
                    <option key={stock.ticker} value={stock.ticker}>
                      {stock.ticker}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            <span className="dropdown-arrow">&#9660;</span>
            {selectedStock && prediction && (
              <span className="selected-info">
                {prediction.sector} / ${prediction.current_price}
              </span>
            )}
          </div>
        </section>

        {/* Loading */}
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <span>Fetching {selectedStock} data...</span>
          </div>
        )}

        {/* Error */}
        {error && <div className="error-message">{error}</div>}

        {/* ─── PREDICTION RESULTS ─── */}
        {prediction && !loading && (
          <div className="results">
            {/* Metric Cards */}
            <div className="section-title">Prediction Summary</div>
            <div className="metric-cards">
              <div className="metric-card">
                <div className="label">Predicted Volatility</div>
                <div className="value">{prediction.predicted_volatility_pct}%</div>
                <div className="sub">60-day annualised</div>
              </div>
              <div className="metric-card">
                <div className="label">Direction Signal</div>
                <div className="value">{prediction.direction}</div>
                <div className="sub">
                  {(prediction.direction_probability * 100).toFixed(1)}% probability
                </div>
              </div>
              <div className="metric-card">
                <div className="label">Confidence</div>
                <div className="value">{prediction.confidence}</div>
                <div className="sub">Based on gate entropy</div>
              </div>
              <div className="metric-card">
                <div className="label">Current Price</div>
                <div className="value">${prediction.current_price}</div>
                <div className="sub">{prediction.ticker}</div>
              </div>
            </div>

            {/* Volatility Forecast Chart — full width */}
            <div className="section-title">Volatility Forecast</div>
            <div className="charts-grid">
              <div className="chart-panel full-width">
                <div className="chart-title">Historical and Predicted Volatility</div>
                <ResponsiveContainer width="100%" height={320}>
                  <AreaChart data={prediction.historical_vol}>
                    <defs>
                      <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#111" stopOpacity={0.12} />
                        <stop offset="95%" stopColor="#111" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11, fill: "#999" }}
                      tickFormatter={(v) => v.slice(5)}
                      stroke="#ddd"
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#999" }}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      stroke="#ddd"
                    />
                    <Tooltip
                      formatter={(v) => [`${(v * 100).toFixed(2)}%`, "Volatility"]}
                      labelFormatter={(l) => `Date: ${l}`}
                    />
                    <Area
                      type="monotone"
                      dataKey="volatility"
                      stroke="#111"
                      fill="url(#volGrad)"
                      strokeWidth={2}
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
                <div className="chart-note">
                  Historical 20-day rolling volatility. Final point is the model's
                  predicted volatility.
                </div>
              </div>
            </div>

            {/* HAR-RV Detail Row */}
            <div className="section-title">HAR-RV Components</div>
            <div className="vol-detail">
              <div className="vol-detail-item">
                <div className="label">Daily RV (1-Day)</div>
                <div className="value">
                  {(prediction.har_rv_features.daily_rv * 100).toFixed(2)}%
                </div>
              </div>
              <div className="vol-detail-item">
                <div className="label">Weekly RV (5-Day)</div>
                <div className="value">
                  {(prediction.har_rv_features.weekly_rv * 100).toFixed(2)}%
                </div>
              </div>
              <div className="vol-detail-item">
                <div className="label">Monthly RV (22-Day)</div>
                <div className="value">
                  {(prediction.har_rv_features.monthly_rv * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            {/* Gate Weights + Radar + HAR-RV Chart */}
            <div className="section-title">Modality Analysis</div>
            <div className="charts-grid">
              {/* Gate Weights Bar */}
              <div className="chart-panel">
                <div className="chart-title">Gate Weights (Per-Sample)</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={gateData} layout="vertical">
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#eee"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      tick={{ fontSize: 11, fill: "#999" }}
                      tickFormatter={(v) => `${v}%`}
                      stroke="#ddd"
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fontSize: 13, fill: "#333", fontWeight: 600 }}
                      stroke="#ddd"
                      width={55}
                    />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Bar dataKey="weight" fill="#333" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="chart-note">
                  Sigmoid gating weights for this prediction
                </div>
              </div>

              {/* Radar Chart */}
              <div className="chart-panel">
                <div className="chart-title">Modality Contribution Radar</div>
                <ResponsiveContainer width="100%" height={220}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#ddd" />
                    <PolarAngleAxis
                      dataKey="modality"
                      tick={{ fontSize: 12, fill: "#333", fontWeight: 600 }}
                    />
                    <PolarRadiusAxis
                      angle={90}
                      domain={[0, 100]}
                      tick={{ fontSize: 10, fill: "#999" }}
                    />
                    <Radar
                      name="Weight"
                      dataKey="weight"
                      stroke="#111"
                      fill="#111"
                      fillOpacity={0.15}
                      strokeWidth={2}
                    />
                    <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
                  </RadarChart>
                </ResponsiveContainer>
                <div className="chart-note">
                  Visual representation of how the model weighs each data source
                </div>
              </div>

              {/* HAR-RV Bar Chart */}
              <div className="chart-panel">
                <div className="chart-title">Realised Volatility by Horizon</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={harData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 12, fill: "#333", fontWeight: 600 }}
                      stroke="#ddd"
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#999" }}
                      tickFormatter={(v) => `${v}%`}
                      stroke="#ddd"
                    />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Bar dataKey="rv" fill="#666" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="chart-note">
                  Annualised RV at 1, 5, and 22 trading day windows
                </div>
              </div>

              {/* Return Distribution */}
              {historical && (
                <div className="chart-panel">
                  <div className="chart-title">Daily Returns Distribution</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart
                      data={historical.data.filter(
                        (d) => d.return !== 0 && d.return !== undefined
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: "#999" }}
                        tickFormatter={(v) => v.slice(8)}
                        stroke="#ddd"
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: "#999" }}
                        tickFormatter={(v) => `${v}%`}
                        stroke="#ddd"
                      />
                      <Tooltip
                        formatter={(v) => [`${v}%`, "Return"]}
                        labelFormatter={(l) => `Date: ${l}`}
                      />
                      <Bar
                        dataKey="return"
                        fill="#999"
                        radius={[2, 2, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="chart-note">
                    Daily percentage returns over the observation window
                  </div>
                </div>
              )}
            </div>

            {/* Gate Weights Table */}
            <div className="section-title">Gate Weight Breakdown</div>
            <table className="gate-table">
              <thead>
                <tr>
                  <th>Modality</th>
                  <th>Encoder</th>
                  <th>Weight</th>
                  <th>Contribution</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Price</td>
                  <td>CNN-BiLSTM (256-d)</td>
                  <td className="mono">
                    {(prediction.gate_weights.price * 100).toFixed(1)}%
                  </td>
                  <td>OHLCV time-series, technical indicators, rolling volatility</td>
                </tr>
                <tr>
                  <td>Text</td>
                  <td>FinBERT (768-d)</td>
                  <td className="mono">
                    {(prediction.gate_weights.text * 100).toFixed(1)}%
                  </td>
                  <td>SEC 10-K filings, risk disclosures, financial language</td>
                </tr>
                <tr>
                  <td>Graph</td>
                  <td>GAT (256-d)</td>
                  <td className="mono">
                    {(prediction.gate_weights.graph * 100).toFixed(1)}%
                  </td>
                  <td>30-node inter-firm graph, 7 semantic edge types</td>
                </tr>
                <tr>
                  <td>Macro</td>
                  <td>MLP (32-d)</td>
                  <td className="mono">
                    {(prediction.gate_weights.macro * 100).toFixed(1)}%
                  </td>
                  <td>VIX, yields, credit spreads, market momentum</td>
                </tr>
              </tbody>
            </table>

            {/* Statistics Insights */}
            {stats && (
              <>
                <div className="section-title">Return Statistics ({stats.nDays}-Day Window)</div>
                <div className="insights-grid">
                  <div className="insight-card">
                    <div className="insight-label">Mean Daily Return</div>
                    <div className="insight-value">{stats.meanReturn}%</div>
                    <div className="insight-desc">
                      Average daily percentage return over the observation window
                    </div>
                  </div>
                  <div className="insight-card">
                    <div className="insight-label">Return Std Dev</div>
                    <div className="insight-value">{stats.stdReturn}%</div>
                    <div className="insight-desc">
                      Standard deviation of daily returns (realised risk)
                    </div>
                  </div>
                  <div className="insight-card">
                    <div className="insight-label">Best / Worst Day</div>
                    <div className="insight-value">
                      +{stats.maxReturn}% / {stats.minReturn}%
                    </div>
                    <div className="insight-desc">
                      Maximum single-day gain and loss in the window
                    </div>
                  </div>
                  <div className="insight-card">
                    <div className="insight-label">Positive Days</div>
                    <div className="insight-value">{stats.positiveRatio}%</div>
                    <div className="insight-desc">
                      Percentage of trading days with positive returns
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Price History Chart */}
            {historical && (
              <>
                <div className="section-title">Price and Volatility History</div>
                <div className="charts-grid">
                  <div className="chart-panel full-width">
                    <div className="chart-title">
                      {prediction.ticker} — Price (left axis) vs Volatility (right
                      axis)
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={historical.data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 11, fill: "#999" }}
                          tickFormatter={(v) => v.slice(5)}
                          stroke="#ddd"
                        />
                        <YAxis
                          yAxisId="left"
                          tick={{ fontSize: 11, fill: "#999" }}
                          stroke="#ddd"
                        />
                        <YAxis
                          yAxisId="right"
                          orientation="right"
                          tick={{ fontSize: 11, fill: "#999" }}
                          stroke="#ddd"
                        />
                        <Tooltip />
                        <Legend />
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="price"
                          stroke="#111"
                          dot={false}
                          strokeWidth={2}
                          name="Price ($)"
                        />
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="volatility"
                          stroke="#aaa"
                          dot={false}
                          strokeWidth={1.5}
                          strokeDasharray="5 3"
                          name="Volatility"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ─── MODEL INFO — landing state ─── */}
        {modelInfo && !selectedStock && (
          <>
            <section className="model-info">
              <h2>Model Architecture and Performance</h2>
              <div className="info-grid">
                <div className="info-card">
                  <h3>Architecture</h3>
                  <ul>
                    <li>
                      <strong>Price</strong>
                      <span>{modelInfo.architecture.price_encoder}</span>
                    </li>
                    <li>
                      <strong>Text</strong>
                      <span>{modelInfo.architecture.text_encoder}</span>
                    </li>
                    <li>
                      <strong>Graph</strong>
                      <span>{modelInfo.architecture.graph_encoder}</span>
                    </li>
                    <li>
                      <strong>Macro</strong>
                      <span>{modelInfo.architecture.macro_encoder}</span>
                    </li>
                    <li>
                      <strong>Fusion</strong>
                      <span>{modelInfo.architecture.fusion}</span>
                    </li>
                    <li>
                      <strong>Vol Head</strong>
                      <span>{modelInfo.architecture.vol_head}</span>
                    </li>
                  </ul>
                </div>
                <div className="info-card">
                  <h3>Test Performance</h3>
                  <ul>
                    <li>
                      <strong>Volatility R2</strong>
                      <span>{modelInfo.metrics.test_r2}</span>
                    </li>
                    <li>
                      <strong>RMSE</strong>
                      <span>{modelInfo.metrics.test_rmse}</span>
                    </li>
                    <li>
                      <strong>Direction AUC</strong>
                      <span>{modelInfo.metrics.test_auc}</span>
                    </li>
                    <li>
                      <strong>Parameters</strong>
                      <span>{modelInfo.metrics.parameters?.toLocaleString()}</span>
                    </li>
                  </ul>
                </div>
                <div className="info-card">
                  <h3>Training</h3>
                  <ul>
                    <li>
                      <strong>Epochs</strong>
                      <span>{modelInfo.training.epochs}</span>
                    </li>
                    <li>
                      <strong>Best Val R2</strong>
                      <span>{modelInfo.training.best_val_r2}</span>
                    </li>
                    <li>
                      <strong>Loss Function</strong>
                      <span>{modelInfo.training.loss}</span>
                    </li>
                    <li>
                      <strong>Optimizer</strong>
                      <span>{modelInfo.training.optimizer}</span>
                    </li>
                  </ul>
                </div>
              </div>
            </section>

            <section className="about-section">
              <div className="section-title">About This Model</div>
              <div className="about-grid">
                <div className="about-card">
                  <h3>Overview</h3>
                  <p>
                    This model predicts 60-day realised volatility for 30 NASDAQ
                    technology stocks using four data modalities: price time-series
                    (CNN-BiLSTM), SEC 10-K filings (FinBERT), inter-firm graph
                    relationships (GAT), and macroeconomic indicators (MLP). A learned
                    sigmoid gate fuses these modalities with a HAR-RV skip connection
                    that preserves classical autoregressive structure.
                  </p>
                </div>
                <div className="about-card">
                  <h3>Stock Universe</h3>
                  <ul>
                    <li>Core Tech: AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, INTC, TSLA, ORCL</li>
                    <li>Semiconductors: QCOM, TXN, AVGO, MRVL, KLAC</li>
                    <li>Cloud/SaaS: CRM, ADBE, NOW, SNOW, DDOG</li>
                    <li>Internet: NFLX, UBER, PYPL, SNAP</li>
                    <li>Hardware: DELL, AMAT, LRCX</li>
                    <li>Diversified: IBM, CSCO, HPE</li>
                  </ul>
                </div>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
