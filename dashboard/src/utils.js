/**
 * Regime-related utilities — color mapping, labels, icons.
 */

/** Map regime label → CSS color variable value */
export const REGIME_COLORS = {
  'High Volatility': '#f43f5e',
  'Low Volatility': '#06b6d4',
  'Trending Up': '#10b981',
  'Trending Down': '#ef4444',
  'Mean Reverting': '#f59e0b',
  'Transitional': '#8b5cf6',
  'Noise': '#64748b',
  'Unknown': '#475569',
};

/** Map regime label → emoji icon */
export const REGIME_ICONS = {
  'High Volatility': '🔥',
  'Low Volatility': '❄️',
  'Trending Up': '📈',
  'Trending Down': '📉',
  'Mean Reverting': '🔄',
  'Transitional': '⚡',
  'Noise': '🌫️',
  'Unknown': '❓',
};

/**
 * Map regime label to a volatility score (0–1) for charting.
 * This is a heuristic — gives the Volatility Trend chart
 * a numeric axis even though the backend doesn't expose raw volatility.
 */
export const REGIME_VOLATILITY = {
  'High Volatility': 0.95,
  'Transitional': 0.7,
  'Mean Reverting': 0.5,
  'Trending Down': 0.45,
  'Trending Up': 0.4,
  'Low Volatility': 0.15,
  'Noise': 0.5,
  'Unknown': 0.5,
};

/** Get color for a regime, with fallback */
export function getRegimeColor(regime) {
  return REGIME_COLORS[regime] || REGIME_COLORS['Unknown'];
}

/** Format ISO timestamp to readable local string */
export function formatTimestamp(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/** Format ISO timestamp to date + time */
export function formatDateTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}
