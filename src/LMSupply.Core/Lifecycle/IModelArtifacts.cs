namespace LMSupply.Lifecycle;

/// <summary>
/// What a host asks about one kind of model before it loads one: is everything a load needs already on this machine, how
/// much is still to download, and the download itself. Each model package provides one (for example the generator's and the
/// embedder's), so a host that manages several models asks them all the same way.
/// </summary>
/// <remarks>
/// «Available offline» means a load with downloads disabled (<c>DisableAutoDownload</c>) opens cached files only and makes no
/// network request — the model's files and, for a GGUF model, the llama-server it runs on. <see cref="DownloadAsync"/> fetches
/// exactly that, so after it the model is available offline. Nothing here loads a model or starts a server.
/// </remarks>
public interface IModelArtifacts
{
    /// <summary>The kind of model this provides for, such as <c>"generator"</c> or <c>"embedder"</c>.</summary>
    string Kind { get; }

    /// <summary>
    /// Whether a load of <paramref name="modelId"/> with downloads disabled would find everything it needs in the cache.
    /// Reads the cache only; never errs toward true.
    /// </summary>
    Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default);

    /// <summary>
    /// What <see cref="DownloadAsync"/> of <paramref name="modelId"/> would still fetch on this machine: the model's files and,
    /// apart, the runtimes its load runs on. Both 0 when it is available offline. May read the repository and release
    /// listings; downloads nothing.
    /// </summary>
    Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Downloads what a load of <paramref name="modelId"/> needs — the model's files and, for a GGUF model, llama-server — so
    /// that it is available offline afterwards. Already cached files are not fetched again.
    /// </summary>
    Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Deletes what LMSupply downloaded of <paramref name="modelId"/>'s files, so a load downloads them again (or, with downloads
    /// disabled, fails). Files another tool put in the shared cache, a local path, and runtimes such as llama-server are kept.
    /// </summary>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="IOException">A file is open, for example by a running server (Windows).</exception>
    bool Delete(string modelId);
}
