import { useEffect, useState } from 'react';
import { Key, Plus, Trash2, Copy, Check, AlertTriangle, ShieldCheck } from 'lucide-react';
import { useApiKeyStore } from '../stores/apiKeyStore';
import type { ApiKeyCreatedResponse } from '../api/types';

// ─── Create Key Dialog ────────────────────────────────────────────────────────

function CreateKeyDialog({ onClose, onCreated }: {
  onClose: () => void;
  onCreated: (result: ApiKeyCreatedResponse) => void;
}) {
  const [name, setName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const { createKey } = useApiKeyStore();

  const handleCreate = async () => {
    if (!name.trim()) return;
    setIsCreating(true);
    try {
      const result = await createKey(name.trim());
      onCreated(result);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-card border border-border rounded-lg p-6 w-full max-w-md shadow-lg">
        <h2 className="text-lg font-semibold mb-4">Create API Key</h2>
        <label className="block text-sm font-medium mb-1">Name</label>
        <input
          className="w-full border border-border rounded-md px-3 py-2 text-sm bg-background mb-4 focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="e.g., My App"
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleCreate()}
          autoFocus
        />
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 text-sm rounded-md hover:bg-accent">Cancel</button>
          <button
            onClick={handleCreate}
            disabled={!name.trim() || isCreating}
            className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {isCreating ? 'Creating...' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Show Key Dialog (one-time reveal) ───────────────────────────────────────

function ShowKeyDialog({ result, onClose }: { result: ApiKeyCreatedResponse; onClose: () => void }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(result.key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-card border border-border rounded-lg p-6 w-full max-w-lg shadow-lg">
        <h2 className="text-lg font-semibold mb-1">API Key Created</h2>
        <p className="text-sm text-amber-500 flex items-center gap-1 mb-4">
          <AlertTriangle className="w-4 h-4" />
          Copy this key now. It will not be shown again.
        </p>
        <div className="flex items-center gap-2 bg-muted rounded-md px-3 py-2 font-mono text-sm mb-4 break-all">
          <span className="flex-1 select-all">{result.key}</span>
          <button onClick={copy} className="shrink-0 p-1 rounded hover:bg-accent" title="Copy">
            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
          </button>
        </div>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Stats Panel ──────────────────────────────────────────────────────────────

function StatsPanel() {
  const { stats, statsDays, fetchStats } = useApiKeyStore();

  useEffect(() => { fetchStats(statsDays); }, [statsDays, fetchStats]);

  const dayOptions = [1, 7, 30];

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">Request Statistics</h3>
        <div className="flex gap-1">
          {dayOptions.map(d => (
            <button
              key={d}
              onClick={() => fetchStats(d)}
              className={`px-2 py-0.5 text-xs rounded ${statsDays === d ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>
      {stats ? (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-2xl font-bold">{stats.totalRequests}</p>
              <p className="text-xs text-muted-foreground">Total</p>
            </div>
            <div>
              <p className="text-2xl font-bold">{(stats.errorRate * 100).toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">Error Rate</p>
            </div>
            <div>
              <p className="text-2xl font-bold">{Math.round(stats.avgDurationMs)}ms</p>
              <p className="text-xs text-muted-foreground">Avg Latency</p>
            </div>
          </div>
          {stats.requestsByEndpoint.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Top Endpoints</p>
              <div className="space-y-1">
                {stats.requestsByEndpoint.map(e => (
                  <div key={e.path} className="flex items-center gap-2 text-xs">
                    <span className="flex-1 font-mono truncate text-muted-foreground">{e.path}</span>
                    <span className="font-medium">{e.count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No data</p>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export function ApiKeys() {
  const { keys, isLoading, fetchKeys, deleteKey } = useApiKeyStore();
  const [showCreate, setShowCreate] = useState(false);
  const [createdKey, setCreatedKey] = useState<ApiKeyCreatedResponse | null>(null);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete API key "${name}"? This cannot be undone.`)) return;
    await deleteKey(id);
  };

  const handleCreated = (result: ApiKeyCreatedResponse) => {
    setShowCreate(false);
    setCreatedKey(result);
  };

  const formatDate = (iso: string) =>
    new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Key className="w-5 h-5" />
          <h1 className="text-xl font-semibold">API Keys</h1>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <Plus className="w-4 h-4" />
          Create Key
        </button>
      </div>

      {/* Status banner */}
      {!isLoading && (
        keys.length === 0 ? (
          <div className="flex items-center gap-2 px-4 py-3 rounded-lg border border-amber-200 bg-amber-50 text-amber-800 text-sm">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            No API keys — all requests are allowed without authentication. Create a key to enable access control.
          </div>
        ) : (
          <div className="flex items-center gap-2 px-4 py-3 rounded-lg border border-green-200 bg-green-50 text-green-800 text-sm">
            <ShieldCheck className="w-4 h-4 shrink-0" />
            {keys.length} API key{keys.length > 1 ? 's' : ''} active — all requests require a valid Bearer token.
          </div>
        )
      )}

      {/* Keys table */}
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : keys.length > 0 ? (
        <div className="border border-border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-muted-foreground">
              <tr>
                <th className="text-left px-4 py-2 font-medium">Name</th>
                <th className="text-left px-4 py-2 font-medium">Key</th>
                <th className="text-left px-4 py-2 font-medium">Created</th>
                <th className="text-left px-4 py-2 font-medium">Last Used</th>
                <th className="text-right px-4 py-2 font-medium">Requests</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {keys.map(k => (
                <tr key={k.id} className="hover:bg-muted/30">
                  <td className="px-4 py-3 font-medium">{k.name}</td>
                  <td className="px-4 py-3 font-mono text-muted-foreground">{k.keyPrefix}****</td>
                  <td className="px-4 py-3 text-muted-foreground">{formatDate(k.createdAt)}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {k.lastUsedAt ? formatDate(k.lastUsedAt) : '—'}
                  </td>
                  <td className="px-4 py-3 text-right">{k.totalRequests.toLocaleString()}</td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => handleDelete(k.id, k.name)}
                      className="p-1.5 rounded hover:bg-destructive/10 text-destructive"
                      title="Delete key"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {/* Stats — only show when keys exist */}
      {keys.length > 0 && <StatsPanel />}

      {/* Dialogs */}
      {showCreate && <CreateKeyDialog onClose={() => setShowCreate(false)} onCreated={handleCreated} />}
      {createdKey && <ShowKeyDialog result={createdKey} onClose={() => setCreatedKey(null)} />}
    </div>
  );
}
