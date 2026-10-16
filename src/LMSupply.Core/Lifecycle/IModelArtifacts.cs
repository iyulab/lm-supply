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
    /// Bytes of model files a load of <paramref name="modelId"/> would still download (0 when they are cached). Runtimes such as
    /// llama-server are not counted. May read the repository listing.
    /// </summary>
    Task<long> GetRemainingDownloadBytesAsync(string modelId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Downloads what a load of <paramref name="modelId"/> needs — the model's files and, for a GGUF model, llama-server — so
    /// that it is available offline afterwards. Already cached files are not fetched again.
    /// </summary>
    Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default);
}
