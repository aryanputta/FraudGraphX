import axios from 'axios'

export const api = axios.create({ baseURL: '/v1', timeout: 5000 })

export interface Alert {
  alert_id: string
  customer_id: string
  transaction_id: string
  timestamp: string
  ensemble_score: number
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  fraud_type: string
  action: string
  latency_ms: number
  shap_values?: Record<string, number>
  graph_path?: string[]
  investigator_notes?: string
}

export interface Case {
  case_id: string
  alert_id: string
  customer_id: string
  status: string
  risk_level: string
  fraud_type: string
  opened_at: string
  fraud_amount: number
  notes: string[]
}

export interface BenchmarkResult {
  model_name: string
  precision: number
  recall: number
  f1: number
  roc_auc: number
  avg_latency_ms: number
  p99_latency_ms: number
  false_positive_rate: number
}

export const fetchAlerts = (limit = 50) =>
  api.get<Alert[]>('/alerts', { params: { limit } }).then((r) => r.data)

export const fetchAlert = (id: string) =>
  api.get<Alert>(`/alerts/${id}`).then((r) => r.data)

export const fetchCustomerAlerts = (customerId: string) =>
  api.get<Alert[]>(`/customers/${customerId}/alerts`).then((r) => r.data)

export const fetchCustomerGraph = (customerId: string) =>
  api.get(`/customers/${customerId}/graph`).then((r) => r.data)

export const fetchCustomerRisk = (customerId: string) =>
  api.get(`/customers/${customerId}/risk`).then((r) => r.data)

export const fetchBenchmark = () =>
  api.get<BenchmarkResult[]>('/benchmark').then((r) => r.data)

export const runBenchmark = () =>
  api.post('/benchmark/run').then((r) => r.data)

export const submitFeedback = (payload: {
  alert_id: string
  predicted_fraud: boolean
  actual_fraud: boolean
  analyst_notes?: string
}) => api.post('/feedback', payload)
