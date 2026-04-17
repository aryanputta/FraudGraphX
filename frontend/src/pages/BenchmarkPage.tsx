import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchBenchmark, runBenchmark } from '../utils/api'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import { Play } from 'lucide-react'

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: '#60a5fa',
  xgboost: '#f59e0b',
  gnn: '#a78bfa',
}

export default function BenchmarkPage() {
  const qc = useQueryClient()
  const { data: results = [] } = useQuery({
    queryKey: ['benchmark'],
    queryFn: fetchBenchmark,
    refetchInterval: 10_000,
  })

  const { mutate: startBenchmark, isPending } = useMutation({
    mutationFn: runBenchmark,
    onSuccess: () => setTimeout(() => qc.invalidateQueries({ queryKey: ['benchmark'] }), 3000),
  })

  const radarData = ['precision', 'recall', 'f1', 'roc_auc'].map((metric) => ({
    metric,
    ...Object.fromEntries(results.map((r: any) => [r.model_name, (r[metric] ?? 0) * 100])),
  }))

  const latencyData = results.map((r: any) => ({
    name: r.model_name?.replace('_', ' '),
    avg: r.avg_latency_ms,
    p99: r.p99_latency_ms,
  }))

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-100">Model Benchmark</h2>
        <button
          className="flex items-center gap-2 bg-fraud-500 hover:bg-fraud-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          onClick={() => startBenchmark()}
          disabled={isPending}
        >
          <Play className="w-4 h-4" />
          {isPending ? 'Training…' : 'Run Benchmark'}
        </button>
      </div>

      {/* Metrics table */}
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-500 border-b border-gray-800">
              {['Model', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Avg Lat (ms)', 'P99 Lat (ms)', 'FP Rate'].map((h) => (
                <th key={h} className="pb-3 pr-6 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {results.map((r: any) => (
              <tr key={r.model_name} className="hover:bg-gray-800/40">
                <td className="py-3 pr-6">
                  <span className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ background: MODEL_COLORS[r.model_name] ?? '#6b7280' }}
                    />
                    {r.model_name?.replace(/_/g, ' ')}
                  </span>
                </td>
                {['precision', 'recall', 'f1', 'roc_auc'].map((m) => (
                  <td key={m} className="py-3 pr-6">
                    <span className={`font-mono font-bold ${(r[m] ?? 0) >= 0.8 ? 'text-safe-400' : (r[m] ?? 0) >= 0.6 ? 'text-warn-400' : 'text-fraud-400'}`}>
                      {(r[m] ?? 0).toFixed(4)}
                    </span>
                  </td>
                ))}
                <td className="py-3 pr-6 font-mono text-blue-400">{r.avg_latency_ms?.toFixed(2)}</td>
                <td className="py-3 pr-6 font-mono text-blue-300">{r.p99_latency_ms?.toFixed(2)}</td>
                <td className="py-3 pr-6 font-mono text-orange-400">{(r.false_positive_rate ?? 0).toFixed(4)}</td>
              </tr>
            ))}
            {results.length === 0 && (
              <tr><td colSpan={8} className="py-8 text-center text-gray-500">No benchmark data – click Run Benchmark</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Radar chart */}
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Model Comparison (Radar)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1f2937" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#6b7280', fontSize: 11 }} />
              {results.map((r: any) => (
                <Radar
                  key={r.model_name}
                  name={r.model_name}
                  dataKey={r.model_name}
                  stroke={MODEL_COLORS[r.model_name]}
                  fill={MODEL_COLORS[r.model_name]}
                  fillOpacity={0.15}
                />
              ))}
              <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 11 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Latency chart */}
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Inference Latency (ms)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={latencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#111827', border: 'none' }} />
              <Bar dataKey="avg" name="Avg (ms)" fill="#60a5fa" radius={4} />
              <Bar dataKey="p99" name="P99 (ms)" fill="#f59e0b" radius={4} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
