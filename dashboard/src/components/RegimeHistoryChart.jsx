import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { getRegimeColor, REGIME_ICONS } from '../utils';

const REGIME_LABELS = [
  'High Volatility',
  'Low Volatility',
  'Trending Up',
  'Trending Down',
  'Mean Reverting',
  'Transitional',
  'Noise',
  'Unknown',
];

/** Map regime label to a numeric index for the Y-axis */
function regimeToIndex(regime) {
  const idx = REGIME_LABELS.indexOf(regime);
  return idx >= 0 ? idx : REGIME_LABELS.length - 1;
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <div className="tooltip-label">{d.time}</div>
      <div className="tooltip-value" style={{ color: getRegimeColor(d.current_regime) }}>
        {REGIME_ICONS[d.current_regime] || '❓'} {d.current_regime}
      </div>
      <div className="tooltip-label" style={{ marginTop: 4 }}>
        Confidence: {(d.confidence * 100).toFixed(1)}%
      </div>
    </div>
  );
}

export default function RegimeHistoryChart({ data, loading }) {
  if (loading || data.length === 0) {
    return <div className="skeleton skeleton-chart" />;
  }

  const chartData = data.map((d) => ({
    ...d,
    regimeIndex: regimeToIndex(d.current_regime) + 1,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(148,163,184,0.07)"
          vertical={false}
        />
        <XAxis
          dataKey="time"
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
        />
        <YAxis
          domain={[0, REGIME_LABELS.length]}
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => REGIME_LABELS[v - 1] || ''}
          width={100}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(148,163,184,0.05)' }} />
        <Bar dataKey="regimeIndex" radius={[4, 4, 0, 0]} maxBarSize={40}>
          {chartData.map((entry, idx) => (
            <Cell
              key={idx}
              fill={getRegimeColor(entry.current_regime)}
              fillOpacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
