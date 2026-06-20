import {
  AreaChart,
  Area,
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
      <div className="tooltip-value" style={{ color: '#f59e0b' }}>
        Volatility: {(d.volatility * 100).toFixed(0)}%
      </div>
      <div className="tooltip-label" style={{ marginTop: 4 }}>
        Regime: {d.current_regime}
      </div>
    </div>
  );
}

export default function VolatilityTrendChart({ data, loading }) {
  if (loading || data.length === 0) {
    return <div className="skeleton skeleton-chart" />;
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="volatilityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.35} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
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
        <Area
          type="monotone"
          dataKey="volatility"
          stroke="#f59e0b"
          strokeWidth={2}
          fill="url(#volatilityGradient)"
          dot={{ r: 3, fill: '#f59e0b', stroke: '#1a2035', strokeWidth: 2 }}
          activeDot={{ r: 5, fill: '#f59e0b', stroke: '#fff', strokeWidth: 2 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
