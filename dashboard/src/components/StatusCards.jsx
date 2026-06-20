import { getRegimeColor, REGIME_ICONS, formatDateTime } from '../utils';

export default function StatusCards({ regime, loading }) {
  if (loading || !regime) {
    return (
      <>
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </>
    );
  }

  const regimeColor = getRegimeColor(regime.current_regime);
  const regimeIcon = REGIME_ICONS[regime.current_regime] || '❓';
  const confidencePct = (regime.confidence * 100).toFixed(1);

  return (
    <>
      {/* Current Regime Card */}
      <div
        className="stat-card"
        id="card-current-regime"
        style={{ '--card-accent': regimeColor }}
      >
        <span className="stat-label">Current Regime</span>
        <span className="stat-value">
          <span className="regime-icon">{regimeIcon}</span>
          <span style={{ color: regimeColor }}>{regime.current_regime}</span>
        </span>
        <span className="stat-description">{regime.regime_description}</span>
      </div>

      {/* Confidence Card */}
      <div
        className="stat-card"
        id="card-confidence"
        style={{ '--card-accent': 'var(--accent-violet)' }}
      >
        <span className="stat-label">Confidence</span>
        <span className="stat-value">{confidencePct}%</span>
        <div className="confidence-bar-track">
          <div
            className="confidence-bar-fill"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
        <span className="stat-description">
          Cluster #{regime.cluster_id} · {regime.symbol}
        </span>
      </div>

      {/* Last Update Card */}
      <div
        className="stat-card"
        id="card-last-update"
        style={{ '--card-accent': 'var(--accent-emerald)' }}
      >
        <span className="stat-label">Last Update</span>
        <span className="stat-value" style={{ fontSize: 'var(--font-lg)' }}>
          {formatDateTime(regime.timestamp)}
        </span>
        <span className="stat-description">UTC prediction timestamp</span>
      </div>
    </>
  );
}

function SkeletonCard() {
  return (
    <div className="stat-card">
      <div className="skeleton skeleton-text" style={{ width: '40%' }} />
      <div className="skeleton skeleton-value" style={{ width: '70%', marginTop: '8px' }} />
      <div className="skeleton skeleton-text" style={{ width: '90%', marginTop: '12px' }} />
    </div>
  );
}
