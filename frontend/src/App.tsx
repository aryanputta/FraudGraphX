import { Routes, Route, NavLink } from 'react-router-dom'
import { AlertTriangle, BarChart2, ShieldAlert, Users, Activity } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import AlertsPage from './pages/AlertsPage'
import CasesPage from './pages/CasesPage'
import CustomerPage from './pages/CustomerPage'
import BenchmarkPage from './pages/BenchmarkPage'

const NAV = [
  { to: '/', icon: Activity, label: 'Dashboard' },
  { to: '/alerts', icon: AlertTriangle, label: 'Alerts' },
  { to: '/cases', icon: ShieldAlert, label: 'Cases' },
  { to: '/customers', icon: Users, label: 'Customers' },
  { to: '/benchmark', icon: BarChart2, label: 'Benchmark' },
]

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-fraud-500 font-bold text-xl tracking-tight">FraudGraphX</h1>
          <p className="text-gray-500 text-xs mt-0.5">Fraud Detection Platform</p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-fraud-500/10 text-fraud-400'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-3 border-t border-gray-800 text-xs text-gray-600">v1.0.0</div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto bg-gray-950">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/cases" element={<CasesPage />} />
          <Route path="/customers/:customerId?" element={<CustomerPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
        </Routes>
      </main>
    </div>
  )
}
