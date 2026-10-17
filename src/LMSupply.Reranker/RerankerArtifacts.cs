using LMSupply.Exceptions;
using LMSupply.Hardware;
using LMSupply.Lifecycle;
using LMSupply.Llama.Server;

namespace LMSupply.Reranker;

/// <summary>
/// <see cref="IModelArtifacts"/> for rerankers: whether a <see cref="LocalReranker.LoadAsync"/> with downloads disabled would
/// find its model file — and, for a GGUF model, llama-server — in the cache, and the download that makes it so. The options
/// are the ones the load will use (provider and cache directory decide which files that is).
/// </summary>
public sealed class RerankerArtifacts : IModelArtifacts
{
    private readonly RerankerOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public RerankerArtifacts(RerankerOptions? options = null)
    {
        _options = options?.Clone() ?? new RerankerOptions();
    }

    /// <inheritdoc />
    public string Kind => "reranker";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!LocalReranker.IsModelDownloaded(modelId, _options.CacheDirectory))
            return false;
        if (!LocalReranker.LoadsAsGguf(modelId))
            return true;

        try
        {
            await Server().AcquireServerAsync(Backend(), localFilesOnly: true, progress: null, cancellationToken);
            return true;
        }
        catch (ModelNotFoundException)
        {
            return false;
        }
    }

    /// <inheritdoc />
    public Task<long> GetRemainingDownloadBytesAsync(string modelId, CancellationToken cancellationToken = default)
        => LocalReranker.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken);

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        await LocalReranker.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        if (LocalReranker.LoadsAsGguf(modelId))
            await Server().AcquireServerAsync(Backend(), localFilesOnly: false, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalReranker.DeleteModel(modelId, _options.CacheDirectory);

    private LlamaServerUpdateService Server() => LlamaServerUpdateService.Resolve(_options.ServerUpdateOptions);

    private LlamaServerBackend Backend() =>
        Llama.LlamaBackendSelector.MapProvider(_options.Provider, HardwareProfile.For(_options.Provider).GpuInfo);
}
