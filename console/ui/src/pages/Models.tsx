import { useEffect, useState } from 'react';
import { useSystemStore } from '../stores/systemStore';
import { formatBytes, formatDate } from '../lib/utils';
import { Trash2, RefreshCw, Power, Download, Search, Loader2 } from 'lucide-react';
import { api } from '../api/client';
import type { ModelCheckResult } from '../api/types';

export function Models() {
  const {
    cachedModels,
    loadedModels,
    downloadingModels,
    isLoading,
    fetchModels,
    fetchLoadedModels,
    deleteModel,
    unloadModel,
    startDownload,
    isDownloading: checkIsDownloading,
  } = useSystemStore();

  // Download form state
  const [repoId, setRepoId] = useState('');
  const [checkResult, setCheckResult] = useState<ModelCheckResult | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [checkError, setCheckError] = useState<string | null>(null);

  // Get current download state for the form input
  const currentDownload = downloadingModels.get(repoId);
  const isDownloading = currentDownload !== undefined && !currentDownload.error && currentDownload.progress?.status !== 'Completed';
  const downloadProgress = currentDownload?.progress ?? null;
  const downloadError = currentDownload?.error ?? null;

  useEffect(() => {
    fetchModels();
    fetchLoadedModels();
  }, [fetchModels, fetchLoadedModels]);

  const handleRefresh = () => {
    fetchModels();
    fetchLoadedModels();
  };

  const handleDelete = async (repoId: string) => {
    if (confirm(`Delete model "${repoId}"? This cannot be undone.`)) {
      await deleteModel(repoId);
    }
  };

  const handleUnload = async (key: string) => {
    await unloadModel(key);
  };

  const handleCheck = async () => {
    if (!repoId.trim()) return;
    setIsChecking(true);
    setCheckResult(null);
    setCheckError(null);
    try {
      const result = await api.checkModel(repoId.trim());
      setCheckResult(result);
    } catch (err) {
      setCheckError((err as Error).message);
    } finally {
      setIsChecking(false);
    }
  };

  const handleDownload = async () => {
    if (!repoId.trim()) return;
    startDownload(repoId.trim());
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Model Management</h1>
        <button
          onClick={handleRefresh}
          className="px-3 py-2 bg-secondary rounded-lg flex items-center gap-2 hover:bg-secondary/80"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Download Section */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Download Model from HuggingFace</h2>
        <div className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">Repository ID</label>
              <input
                type="text"
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
                placeholder="e.g., BAAI/bge-small-en-v1.5"
                className="w-full px-3 py-2 bg-muted border border-border rounded"
                disabled={isDownloading}
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                onClick={handleCheck}
                disabled={isChecking || isDownloading || !repoId.trim()}
                className="px-4 py-2 bg-secondary rounded flex items-center gap-2 disabled:opacity-50"
              >
                {isChecking ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                Check
              </button>
              <button
                onClick={handleDownload}
                disabled={isChecking || isDownloading || !repoId.trim()}
                className="px-4 py-2 bg-primary text-primary-foreground rounded flex items-center gap-2 disabled:opacity-50"
              >
                {isDownloading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                Download
              </button>
            </div>
          </div>

          {/* Check Result */}
          {checkResult && (
            <div
              className={`p-3 rounded ${
                checkResult.exists
                  ? 'bg-green-500/10 border border-green-500/30'
                  : 'bg-destructive/10 border border-destructive/30'
              }`}
            >
              {checkResult.exists ? (
                <div className="space-y-1">
                  <p className="font-medium text-green-500">✓ Model found</p>
                  <p className="text-sm">
                    Type: <span className="font-medium">{checkResult.detectedType}</span> |{' '}
                    Files: {checkResult.fileCount} |{' '}
                    Size: {formatBytes(checkResult.totalSizeBytes)}
                  </p>
                </div>
              ) : (
                <p className="text-destructive">✗ Model not found: {checkResult.error}</p>
              )}
            </div>
          )}

          {/* Download Progress */}
          {downloadProgress && isDownloading && (
            <div className="p-3 bg-muted rounded space-y-2">
              <div className="flex justify-between text-sm">
                <span className="truncate max-w-md">{downloadProgress.fileName}</span>
                <span>{downloadProgress.percentComplete.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-secondary rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${downloadProgress.percentComplete}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {formatBytes(downloadProgress.bytesDownloaded)} / {formatBytes(downloadProgress.totalBytes)}
              </p>
            </div>
          )}

          {/* Download Complete */}
          {downloadProgress?.status === 'Completed' && !isDownloading && (
            <div className="p-3 bg-green-500/10 border border-green-500/30 rounded">
              <p className="text-green-500 font-medium">✓ Download completed successfully</p>
            </div>
          )}

          {/* Error */}
          {(checkError || downloadError) && (
            <div className="p-3 bg-destructive/10 border border-destructive/30 rounded">
              <p className="text-destructive">Error: {checkError || downloadError}</p>
            </div>
          )}
        </div>
      </div>

      {/* Loaded Models */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">
          Loaded Models ({loadedModels.length})
        </h2>
        {loadedModels.length === 0 ? (
          <p className="text-muted-foreground">No models currently loaded</p>
        ) : (
          <div className="space-y-2">
            {loadedModels.map((model) => (
              <div
                key={`${model.modelType}:${model.modelId}`}
                className="flex items-center justify-between p-3 bg-muted rounded-lg"
              >
                <div>
                  <p className="font-medium">{model.modelId}</p>
                  <p className="text-sm text-muted-foreground">
                    Type: {model.modelType} | Last used:{' '}
                    {formatDate(model.lastUsedAt)}
                  </p>
                </div>
                <button
                  onClick={() => handleUnload(`${model.modelType}:${model.modelId}`)}
                  className="p-2 hover:bg-destructive/20 rounded text-destructive"
                  title="Unload model"
                >
                  <Power className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Downloading Models */}
      {downloadingModels.size > 0 && (
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Loader2 className="w-5 h-5 animate-spin text-primary" />
            Downloading ({downloadingModels.size})
          </h2>
          <div className="space-y-3">
            {Array.from(downloadingModels.values()).map((dl) => (
              <div key={dl.repoId} className="p-3 bg-muted rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">{dl.repoId}</span>
                  {dl.error ? (
                    <span className="text-destructive text-sm">Failed</span>
                  ) : dl.progress?.status === 'Completed' ? (
                    <span className="text-green-500 text-sm">✓ Completed</span>
                  ) : (
                    <span className="text-primary text-sm">
                      {dl.progress?.percentComplete?.toFixed(1) ?? 0}%
                    </span>
                  )}
                </div>
                {dl.error ? (
                  <p className="text-sm text-destructive">{dl.error}</p>
                ) : dl.progress && dl.progress.status !== 'Completed' ? (
                  <>
                    <div className="w-full bg-secondary rounded-full h-2 mb-1">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ width: `${dl.progress.percentComplete}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground truncate">
                      {dl.progress.fileName} - {formatBytes(dl.progress.bytesDownloaded)} / {formatBytes(dl.progress.totalBytes)}
                    </p>
                  </>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cached Models */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">
          Cached Models ({cachedModels.length})
        </h2>
        {cachedModels.length === 0 ? (
          <p className="text-muted-foreground">
            No models in cache. Download models using the LMSupply library.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="p-2 font-medium">Repository</th>
                  <th className="p-2 font-medium">Type</th>
                  <th className="p-2 font-medium">Size</th>
                  <th className="p-2 font-medium">Files</th>
                  <th className="p-2 font-medium">Status</th>
                  <th className="p-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {cachedModels.map((model) => {
                  const downloading = checkIsDownloading(model.repoId);
                  const loaded = loadedModels.some(m => m.modelId === model.repoId);
                  return (
                    <tr key={model.repoId} className="border-b border-border/50">
                      <td className="p-2">
                        <p className="font-medium">{model.repoId}</p>
                        <p className="text-xs text-muted-foreground truncate max-w-xs">
                          {model.localPath}
                        </p>
                      </td>
                      <td className="p-2">
                        <span className="px-2 py-1 bg-muted rounded text-xs">
                          {model.detectedType}
                        </span>
                      </td>
                      <td className="p-2 text-sm">{formatBytes(model.sizeBytes)}</td>
                      <td className="p-2 text-sm">{model.fileCount}</td>
                      <td className="p-2">
                        {downloading ? (
                          <span className="px-2 py-1 bg-blue-500/20 text-blue-500 rounded text-xs flex items-center gap-1 w-fit">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Downloading
                          </span>
                        ) : loaded ? (
                          <span className="px-2 py-1 bg-green-500/20 text-green-500 rounded text-xs">
                            Loaded
                          </span>
                        ) : (
                          <span className="px-2 py-1 bg-muted text-muted-foreground rounded text-xs">
                            Ready
                          </span>
                        )}
                      </td>
                      <td className="p-2">
                        <button
                          onClick={() => handleDelete(model.repoId)}
                          disabled={downloading || loaded}
                          className="p-2 hover:bg-destructive/20 rounded text-destructive disabled:opacity-30 disabled:cursor-not-allowed"
                          title={downloading ? "Cannot delete while downloading" : loaded ? "Unload model first" : "Delete model"}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
