import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <div className="tooltip-label">{d.time}</div>
      <div className="tooltip-value" style={{ color: '#8b5cf6' }}>
        Confidence: {(d.confidence * 100).toFixed(1)}%
      </div>
      <div className="tooltip-label" style={{ marginTop: 4 }}>
        Regime: {d.current_regime}
      </div>
    </div>
  );
}

export default function ConfidenceTrendChart({ data, loading }) {
  if (loading || data.length === 0) {
    return <div className="skeleton skeleton-chart" />;
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.02} />
          </linearGradient>
        </defs>
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
          domain={[0, 1]}
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          width={48}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="confidence"
          stroke="#8b5cf6"
          strokeWidth={2.5}
          dot={{ r: 4, fill: '#8b5cf6', stroke: '#1a2035', strokeWidth: 2 }}
          activeDot={{ r: 6, fill: '#8b5cf6', stroke: '#fff', strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
