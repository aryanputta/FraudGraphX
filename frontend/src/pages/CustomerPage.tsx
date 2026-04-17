import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchCustomerRisk, fetchCustomerGraph } from '../utils/api'
import cytoscape from 'cytoscape'

function GraphViewer({ data }: { data: { nodes: any[]; edges: any[] } }) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current || !data) return
    const cy = cytoscape({
      container: containerRef.current,
      elements: [
        ...data.nodes.map((n) => ({
          data: {
            id: n.node_id,
            label: n.node_type,
            risk: n.risk_score,
          },
        })),
        ...data.edges.map((e, i) => ({
          data: {
            id: `e${i}`,
            source: e.source_id,
            target: e.target_id,
            label: e.relationship,
          },
        })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele: any) => {
              const risk = ele.data('risk') ?? 0
              return risk > 0.7 ? '#f43f5e' : risk > 0.4 ? '#f59e0b' : '#22c55e'
            },
            label: 'data(label)',
            color: '#e5e7eb',
            'font-size': 10,
            width: 30,
            height: 30,
          },
        },
        {
          selector: 'edge',
          style: {
            'line-color': '#374151',
            'target-arrow-color': '#374151',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            width: 1,
          },
        },
      ],
      layout: { name: 'cose', animate: false },
    })
    return () => cy.destroy()
  }, [data])

  return <div ref={containerRef} className="w-full h-80 bg-gray-900 rounded-lg" />
}

export default function CustomerPage() {
  const { customerId: paramId } = useParams()
  const [customerId, setCustomerId] = useState(paramId ?? '')
  const [lookupId, setLookupId] = useState(paramId ?? '')

  const { data: risk } = useQuery({
    queryKey: ['customer-risk', lookupId],
    queryFn: () => fetchCustomerRisk(lookupId),
    enabled: !!lookupId,
  })

  const { data: graph } = useQuery({
    queryKey: ['customer-graph', lookupId],
    queryFn: () => fetchCustomerGraph(lookupId),
    enabled: !!lookupId,
  })

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Customer Investigation</h2>

      <div className="card flex gap-3">
        <input
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-fraud-500"
          placeholder="Enter customer ID…"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setLookupId(customerId)}
        />
        <button
          className="bg-fraud-500 hover:bg-fraud-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          onClick={() => setLookupId(customerId)}
        >
          Look up
        </button>
      </div>

      {risk && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">Risk Summary</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Customer ID</span>
                <span className="font-mono text-gray-200">{risk.customer_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Recent Alerts</span>
                <span className="text-fraud-400 font-bold">{risk.recent_alerts?.length ?? 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Ring Members</span>
                <span className="text-orange-400 font-bold">{risk.ring_member_count ?? 0}</span>
              </div>
            </div>
          </div>

          {risk.potential_ring_members?.length > 0 && (
            <div className="card">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Fraud Ring Members</h3>
              <div className="space-y-1">
                {risk.potential_ring_members.map((id: string) => (
                  <button
                    key={id}
                    className="block w-full text-left font-mono text-xs text-blue-400 hover:text-blue-300 px-2 py-1 hover:bg-gray-800 rounded"
                    onClick={() => { setCustomerId(id); setLookupId(id) }}
                  >
                    {id}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {graph && graph.nodes?.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Fraud Graph ({graph.nodes.length} nodes, {graph.edges.length} edges)
            {graph.fraud_ring_detected && (
              <span className="ml-2 badge-critical">FRAUD RING DETECTED</span>
            )}
          </h3>
          <GraphViewer data={graph} />
        </div>
      )}
    </div>
  )
}
