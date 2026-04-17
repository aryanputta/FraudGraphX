import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchAlerts, submitFeedback, type Alert } from '../utils/api'
import { CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react'

function RiskBadge({ level }: { level: string }) {
  return <span className={`badge-${level}`}>{level}</span>
}

function SHAPBar({ feature, value }: { feature: string; value: number }) {
  const pct = Math.min(Math.abs(value) * 300, 100)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-36 text-gray-400 truncate">{feature}</span>
      <div className="flex-1 bg-gray-800 rounded h-2">
        <div
          className={`h-2 rounded ${value > 0 ? 'bg-fraud-500' : 'bg-safe-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`w-14 text-right font-mono ${value > 0 ? 'text-fraud-400' : 'text-safe-400'}`}>
        {value.toFixed(4)}
      </span>
    </div>
  )
}

function AlertRow({ alert }: { alert: Alert }) {
  const [expanded, setExpanded] = useState(false)
  const qc = useQueryClient()
  const feedback = useMutation({
    mutationFn: (actual: boolean) =>
      submitFeedback({ alert_id: alert.alert_id, predicted_fraud: true, actual_fraud: actual }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
  })

  return (
    <>
      <tr
        className="hover:bg-gray-800/50 cursor-pointer"
        onClick={() => setExpanded((e) => !e)}
      >
        <td className="py-2 pr-4 font-mono text-xs text-gray-400">{alert.transaction_id?.slice(0, 12)}…</td>
        <td className="py-2 pr-4 font-mono text-xs">{alert.customer_id?.slice(0, 12)}…</td>
        <td className="py-2 pr-4">
          <span className={`font-bold ${alert.ensemble_score >= 0.8 ? 'text-fraud-500' : 'text-gray-200'}`}>
            {alert.ensemble_score?.toFixed(3)}
          </span>
        </td>
        <td className="py-2 pr-4"><RiskBadge level={alert.risk_level} /></td>
        <td className="py-2 pr-4 capitalize">{alert.action}</td>
        <td className="py-2 pr-4">
          <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
            <button
              className="p-1 text-fraud-400 hover:text-fraud-300"
              title="Confirm fraud"
              onClick={() => feedback.mutate(true)}
            ><CheckCircle className="w-4 h-4" /></button>
            <button
              className="p-1 text-safe-400 hover:text-safe-300"
              title="Mark as false positive"
              onClick={() => feedback.mutate(false)}
            ><XCircle className="w-4 h-4" /></button>
          </div>
        </td>
        <td className="py-2 pr-4 text-gray-500">
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </td>
      </tr>

      {expanded && (
        <tr>
          <td colSpan={7} className="pb-4 px-4">
            <div className="grid grid-cols-2 gap-4 p-4 bg-gray-900 rounded-lg">
              {/* SHAP Values */}
              {alert.shap_values && (
                <div>
                  <h4 className="text-xs font-semibold text-gray-400 mb-2">SHAP Feature Importance</h4>
                  <div className="space-y-1">
                    {Object.entries(alert.shap_values)
                      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                      .slice(0, 8)
                      .map(([f, v]) => <SHAPBar key={f} feature={f} value={v} />)}
                  </div>
                </div>
              )}

              {/* Graph Path */}
              <div>
                {alert.graph_path && alert.graph_path.length > 0 && (
                  <>
                    <h4 className="text-xs font-semibold text-gray-400 mb-2">Suspicious Graph Path</h4>
                    <div className="flex flex-wrap gap-1 items-center text-xs">
                      {alert.graph_path.map((node, i) => (
                        <span key={i} className="flex items-center gap-1">
                          <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-300 font-mono">{node}</span>
                          {i < alert.graph_path!.length - 1 && <span className="text-gray-600">→</span>}
                        </span>
                      ))}
                    </div>
                  </>
                )}

                {/* LLM Notes */}
                {alert.investigator_notes && (
                  <div className="mt-3">
                    <h4 className="text-xs font-semibold text-gray-400 mb-2">AI Investigation Notes</h4>
                    <pre className="text-xs text-gray-300 bg-gray-800 p-3 rounded whitespace-pre-wrap font-sans">
                      {alert.investigator_notes}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function AlertsPage() {
  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => fetchAlerts(100),
    refetchInterval: 5000,
  })

  if (isLoading) return <div className="p-6 text-gray-400">Loading alerts…</div>

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-100">Fraud Alerts</h2>
        <span className="text-sm text-gray-400">{alerts.length} alerts</span>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-500 border-b border-gray-800">
              {['Transaction', 'Customer', 'Score', 'Risk', 'Action', 'Feedback', ''].map((h) => (
                <th key={h} className="pb-3 pr-4 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {alerts.map((alert) => (
              <AlertRow key={alert.alert_id} alert={alert} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
