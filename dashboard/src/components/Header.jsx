import { formatDateTime } from '../utils';

export default function Header({ health, countdown, lastRefresh, onRefresh, loading }) {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="header" id="dashboard-header">
      <div className="header-left">
        <h1 className="header-title">Market Regime Detector</h1>
        <p className="header-subtitle">
          Live BTCUSDT regime analysis · Auto-refreshes every 30s
        </p>
      </div>

      <div className="header-right">
        {health && (
          <div className="status-badge" id="health-status">
            <span className={`status-dot ${isHealthy ? '' : 'degraded'}`} />
            <span>{isHealthy ? 'Model Healthy' : 'Degraded'}</span>
            <span style={{ color: 'var(--text-muted)' }}>
              · {health.model_algorithm}
            </span>
          </div>
        )}

        <span className="countdown-badge" id="countdown-timer">
          Next refresh in {countdown}s
        </span>

        <button
          className="refresh-btn"
          id="refresh-button"
          onClick={onRefresh}
          disabled={loading}
          title={
            lastRefresh
              ? `Last refresh: ${formatDateTime(lastRefresh.toISOString())}`
              : 'Refresh now'
          }
        >
          <span className={`refresh-icon ${loading ? 'spinning' : ''}`}>⟳</span>
          Refresh
        </button>
      </div>
    </header>
  );
}
