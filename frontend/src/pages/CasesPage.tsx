import { useState } from 'react'
import { ShieldAlert, Clock, User } from 'lucide-react'

const MOCK_CASES = [
  { case_id: 'c1', customer_id: 'cust-abc123', status: 'open', risk_level: 'critical', fraud_type: 'fraud_ring', fraud_amount: 15000, opened_at: '2024-01-15T10:23:00Z' },
  { case_id: 'c2', customer_id: 'cust-def456', status: 'under_review', risk_level: 'high', fraud_type: 'mule_account', fraud_amount: 42000, opened_at: '2024-01-14T08:12:00Z' },
  { case_id: 'c3', customer_id: 'cust-ghi789', status: 'confirmed_fraud', risk_level: 'critical', fraud_type: 'money_laundering', fraud_amount: 125000, opened_at: '2024-01-13T16:45:00Z' },
]

const STATUS_COLORS: Record<string, string> = {
  open: 'bg-blue-900 text-blue-200',
  under_review: 'bg-warn-900 text-warn-100',
  confirmed_fraud: 'bg-fraud-900 text-fraud-100',
  cleared: 'bg-safe-900 text-safe-100',
  escalated: 'bg-purple-900 text-purple-100',
}

export default function CasesPage() {
  const [filter, setFilter] = useState('all')
  const cases = filter === 'all' ? MOCK_CASES : MOCK_CASES.filter((c) => c.status === filter)

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-100">Case Management</h2>
        <select
          className="bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-1.5 text-sm"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          {['all', 'open', 'under_review', 'confirmed_fraud', 'cleared', 'escalated'].map((s) => (
            <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>

      <div className="space-y-3">
        {cases.map((c) => (
          <div key={c.case_id} className="card flex items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-fraud-500 mt-0.5 shrink-0" />
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono text-sm text-gray-200">{c.case_id}</span>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${STATUS_COLORS[c.status]}`}>
                    {c.status.replace(/_/g, ' ')}
                  </span>
                  <span className={`badge-${c.risk_level}`}>{c.risk_level}</span>
                </div>
                <div className="text-xs text-gray-400 space-y-0.5">
                  <div className="flex items-center gap-1.5">
                    <User className="w-3 h-3" />
                    {c.customer_id}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-3 h-3" />
                    {new Date(c.opened_at).toLocaleString()}
                  </div>
                  <div>Type: <span className="text-gray-200">{c.fraud_type.replace(/_/g, ' ')}</span></div>
                </div>
              </div>
            </div>
            <div className="text-right shrink-0">
              <div className="text-lg font-bold text-fraud-400">${c.fraud_amount.toLocaleString()}</div>
              <div className="text-xs text-gray-500">fraud exposure</div>
              <button className="mt-2 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 px-3 py-1 rounded-lg transition-colors">
                Investigate
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
