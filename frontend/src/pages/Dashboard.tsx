import { useQuery } from '@tanstack/react-query'
import { fetchAlerts } from '../utils/api'
import { AlertTriangle, Shield, Clock, TrendingUp } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts'
import { format, parseISO } from 'date-fns'

const RISK_COLORS: Record<string, string> = {
  critical: '#f43f5e',
  high: '#f97316',
  medium: '#f59e0b',
  low: '#22c55e',
}

export default function Dashboard() {
  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => fetchAlerts(200),
    refetchInterval: 5000,
  })

  const criticalCount = alerts.filter((a) => a.risk_level === 'critical').length
  const highCount = alerts.filter((a) => a.risk_level === 'high').length
  const avgLatency = alerts.length
    ? (alerts.reduce((s, a) => s + (a.latency_ms ?? 0), 0) / alerts.length).toFixed(1)
    : '—'
  const blocked = alerts.filter((a) => a.action === 'block').length

  // Time-series data grouped by hour
  const byHour = alerts.reduce<Record<string, number>>((acc, a) => {
    const h = a.timestamp ? format(parseISO(a.timestamp), 'HH:00') : 'unknown'
    acc[h] = (acc[h] ?? 0) + 1
    return acc
  }, {})
  const timeData = Object.entries(byHour)
    .slice(-24)
    .map(([hour, count]) => ({ hour, count }))

  // Risk distribution
  const riskDist = ['critical', 'high', 'medium', 'low'].map((level) => ({
    level,
    count: alerts.filter((a) => a.risk_level === level).length,
  }))

  const stats = [
    { label: 'Critical Alerts', value: criticalCount, icon: AlertTriangle, color: 'text-fraud-500' },
    { label: 'High Risk', value: highCount, icon: Shield, color: 'text-orange-400' },
    { label: 'Avg Latency (ms)', value: avgLatency, icon: Clock, color: 'text-blue-400' },
    { label: 'Transactions Blocked', value: blocked, icon: TrendingUp, color: 'text-warn-500' },
  ]

  if (isLoading) return <div className="p-6 text-gray-400">Loading…</div>

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Dashboard</h2>

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="card flex items-center gap-4">
            <Icon className={`w-8 h-8 ${color}`} />
            <div>
              <div className="text-2xl font-bold text-gray-100">{value}</div>
              <div className="text-xs text-gray-400">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Alerts over Time</h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={timeData}>
              <defs>
                <linearGradient id="alertGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="hour" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#111827', border: 'none' }} />
              <Area type="monotone" dataKey="count" stroke="#f43f5e" fill="url(#alertGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={riskDist} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis dataKey="level" type="category" tick={{ fill: '#6b7280', fontSize: 10 }} width={60} />
              <Tooltip contentStyle={{ background: '#111827', border: 'none' }} />
              <Bar dataKey="count" radius={4}>
                {riskDist.map((entry) => (
                  <Cell key={entry.level} fill={RISK_COLORS[entry.level]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent alerts table */}
      <div className="card">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Recent Fraud Alerts</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b border-gray-800">
                {['Transaction ID', 'Customer', 'Score', 'Risk', 'Action', 'Latency (ms)'].map((h) => (
                  <th key={h} className="pb-2 pr-4 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {alerts.slice(0, 20).map((a) => (
                <tr key={a.alert_id} className="hover:bg-gray-800/40">
                  <td className="py-2 pr-4 font-mono text-xs text-gray-400">{a.transaction_id?.slice(0, 8)}…</td>
                  <td className="py-2 pr-4 font-mono text-xs">{a.customer_id?.slice(0, 8)}…</td>
                  <td className="py-2 pr-4">
                    <span className={`font-bold ${a.ensemble_score >= 0.8 ? 'text-fraud-500' : a.ensemble_score >= 0.6 ? 'text-orange-400' : 'text-gray-300'}`}>
                      {typeof a.ensemble_score === 'number' ? a.ensemble_score.toFixed(3) : '—'}
                    </span>
                  </td>
                  <td className="py-2 pr-4">
                    <span className={`badge-${a.risk_level}`}>{a.risk_level}</span>
                  </td>
                  <td className="py-2 pr-4 capitalize">{a.action}</td>
                  <td className="py-2 pr-4 text-gray-400">
                    <span className={a.latency_ms > 150 ? 'text-fraud-400' : 'text-safe-400'}>
                      {a.latency_ms?.toFixed(1)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
