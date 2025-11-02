import React, { useEffect, useMemo, useState } from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";

const API_BASE = "/dashboard/api";
const REFRESH_INTERVAL_MS = 15000;

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.00%";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function useDashboardData() {
  const [state, setState] = useState({
    loading: true,
    error: null,
    ledger: [],
    summary: { total: 0, lanes: {} },
    metrics: {},
    updatedAt: null,
  });

  useEffect(() => {
    let cancelled = false;

    async function fetchPayload(endpoint) {
      const response = await fetch(`${API_BASE}${endpoint}`);
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }
      return await response.json();
    }

    async function load() {
      try {
        const [summary, metrics, ledger] = await Promise.all([
          fetchPayload("/summary"),
          fetchPayload("/metrics"),
          fetchPayload("/ledger?limit=50"),
        ]);
        if (!cancelled) {
          setState({
            loading: false,
            error: null,
            summary,
            metrics,
            ledger: ledger.records ?? [],
            updatedAt: new Date().toISOString(),
          });
        }
      } catch (error) {
        if (!cancelled) {
          setState((prev) => ({
            ...prev,
            loading: false,
            error: error.message,
            updatedAt: new Date().toISOString(),
          }));
        }
      }
    }

    load();
    const interval = setInterval(load, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return state;
}

function StatusPill({ lane }) {
  const normalised = (lane ?? "unknown").toLowerCase();
  const tone = normalised === "autonomic" || normalised === "deliberative" ? "ok" : "alert";
  return <span className={`status-pill ${tone}`}>{lane ?? "unknown"}</span>;
}

function MetricsGrid({ summary, metrics }) {
  const items = useMemo(() => {
    const entries = [];
    entries.push({
      label: "Ledger Events",
      value: summary.total,
      delta: `${Object.keys(summary.lanes || {}).length} governance lanes",
    });
    if (metrics) {
      entries.push({
        label: "Epistemic Integrity",
        value: formatPercent(metrics.epistemic_integrity ?? 0),
        delta: "Target ≥ 85%",
      });
      entries.push({
        label: "Epistemic Drift",
        value: formatPercent(metrics.epistemic_drift ?? 0),
        delta: "Lower is better",
      });
      entries.push({
        label: "Adversarial Resilience",
        value: formatPercent(metrics.adversarial_resilience ?? 0),
        delta: "Target ≥ 90%",
      });
    }
    return entries;
  }, [summary, metrics]);

  return (
    <div className="metrics-grid">
      {items.map((item) => (
        <div key={item.label} className="metric-card">
          <h2>{item.label}</h2>
          <span className="value">{item.value}</span>
          <span className="delta">{item.delta}</span>
        </div>
      ))}
    </div>
  );
}

function LedgerTable({ records }) {
  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Lane</th>
            <th>Stability</th>
            <th>Claims</th>
          </tr>
        </thead>
        <tbody>
          {records.map((record, index) => (
            <tr key={`${record.timestamp}-${index}`}>
              <td>{record.timestamp}</td>
              <td><StatusPill lane={record.governance_lane} /></td>
              <td>{(record.stability_score ?? 0).toFixed(3)}</td>
              <td>
                <pre>{JSON.stringify(record.claims ?? [], null, 2)}</pre>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DashboardApp() {
  const { loading, error, ledger, summary, metrics, updatedAt } = useDashboardData();

  return (
    <div className="dashboard">
      <h1>Tessrax Governance Control Tower</h1>
      <p className="subtitle">Cluster health, ledger telemetry, and epistemic integrity in real time.</p>
      {error ? <div className="error-banner">⚠️ {error}</div> : null}
      <MetricsGrid summary={summary} metrics={metrics} />
      <h2>Ledger Activity</h2>
      {loading && ledger.length === 0 ? (
        <p>Loading ledger telemetry…</p>
      ) : (
        <LedgerTable records={ledger} />
      )}
      <p className="updated-at">Last updated: {updatedAt ? new Date(updatedAt).toLocaleString() : "Pending"}</p>
    </div>
  );
}

const container = document.getElementById("root");
if (!container) {
  throw new Error("Dashboard root element missing");
}

createRoot(container).render(<DashboardApp />);
