/**
 * API service — fetches data from the Phase 5 FastAPI backend.
 *
 * In development: Vite proxy rewrites /api → http://127.0.0.1:10000
 * In production:  calls the Render backend URL directly via VITE_API_URL
 */

const BASE = import.meta.env.VITE_API_URL || '/api';

/**
 * GET /health
 */
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * GET /regime/current?symbol=...
 */
export async function fetchCurrentRegime(symbol = 'BTCUSDT') {
  const res = await fetch(`${BASE}/regime/current?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error(`Regime fetch failed: ${res.status}`);
  return res.json();
}

/**
 * GET /regime/history?limit=...
 */
export async function fetchRegimeHistory(limit = 50) {
  const res = await fetch(`${BASE}/regime/history?limit=${limit}`);
  if (!res.ok) throw new Error(`History fetch failed: ${res.status}`);
  return res.json();
}
