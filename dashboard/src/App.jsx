import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchCurrentRegime, fetchRegimeHistory, fetchHealth } from './api';
import { REGIME_VOLATILITY } from './utils';
import StatusCards from './components/StatusCards';
import RegimeHistoryChart from './components/RegimeHistoryChart';
import VolatilityTrendChart from './components/VolatilityTrendChart';
import ConfidenceTrendChart from './components/ConfidenceTrendChart';
import Header from './components/Header';
import './App.css';

const REFRESH_INTERVAL = 30_000; // 30 seconds

export default function App() {
  // ── State ──────────────────────────────────────────────
  const [currentRegime, setCurrentRegime] = useState(null);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [countdown, setCountdown] = useState(REFRESH_INTERVAL / 1000);
  const timerRef = useRef(null);
  const countdownRef = useRef(null);

  // ── Data fetching ──────────────────────────────────────
  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [regimeData, historyData, healthData] = await Promise.all([
        fetchCurrentRegime(),
        fetchRegimeHistory(50),
        fetchHealth(),
      ]);

      setCurrentRegime(regimeData);
      setHealth(healthData);

      // Build history array with derived volatility scores (newest first from API)
      const enriched = historyData.predictions.map((entry) => ({
        ...entry,
        volatility: REGIME_VOLATILITY[entry.current_regime] ?? 0.5,
        time: new Date(entry.timestamp).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      }));
      // Reverse so chart reads left-to-right chronologically
      setHistory(enriched.reverse());
      setLastRefresh(new Date());
      setCountdown(REFRESH_INTERVAL / 1000);
    } catch (err) {
      console.error('Failed to load data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Auto-refresh every 30s ─────────────────────────────
  useEffect(() => {
    loadData();
    timerRef.current = setInterval(loadData, REFRESH_INTERVAL);
    return () => clearInterval(timerRef.current);
  }, [loadData]);

  // ── Countdown timer ────────────────────────────────────
  useEffect(() => {
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => (prev <= 1 ? REFRESH_INTERVAL / 1000 : prev - 1));
    }, 1000);
    return () => clearInterval(countdownRef.current);
  }, []);

  // ── Manual refresh ─────────────────────────────────────
  const handleRefresh = () => {
    clearInterval(timerRef.current);
    loadData();
    timerRef.current = setInterval(loadData, REFRESH_INTERVAL);
  };

  return (
    <div className="app">
      <Header
        health={health}
        countdown={countdown}
        lastRefresh={lastRefresh}
        onRefresh={handleRefresh}
        loading={loading}
      />

      {error && (
        <div className="error-banner" id="error-banner">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
          <button className="error-dismiss" onClick={() => setError(null)}>
            ✕
          </button>
        </div>
      )}

      <main className="dashboard">
        <section className="cards-section" id="status-cards">
          <StatusCards regime={currentRegime} loading={loading && !currentRegime} />
        </section>

        <section className="charts-section" id="charts">
          <div className="chart-card" id="regime-history-chart">
            <h2 className="chart-title">Regime History</h2>
            <RegimeHistoryChart data={history} loading={loading && history.length === 0} />
          </div>

          <div className="charts-row">
            <div className="chart-card" id="volatility-trend-chart">
              <h2 className="chart-title">Volatility Trend</h2>
              <VolatilityTrendChart data={history} loading={loading && history.length === 0} />
            </div>

            <div className="chart-card" id="confidence-trend-chart">
              <h2 className="chart-title">Confidence Trend</h2>
              <ConfidenceTrendChart data={history} loading={loading && history.length === 0} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
