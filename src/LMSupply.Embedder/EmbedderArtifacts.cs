using LMSupply.Exceptions;
using LMSupply.Hardware;
using LMSupply.Lifecycle;
using LMSupply.Llama.Server;

namespace LMSupply.Embedder;

/// <summary>
/// <see cref="IModelArtifacts"/> for embedders: whether a <see cref="LocalEmbedder.LoadAsync"/> with downloads disabled would
/// find its model files — and, for a GGUF model, llama-server — in the cache, and the download that makes it so. The options
/// are the ones the load will use (provider and cache directory decide which files that is).
/// </summary>
public sealed class EmbedderArtifacts : IModelArtifacts
{
    private readonly EmbedderOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public EmbedderArtifacts(EmbedderOptions? options = null)
    {
        _options = options?.Clone() ?? new EmbedderOptions();
    }

    /// <inheritdoc />
    public string Kind => "embedder";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!LocalEmbedder.IsModelDownloaded(modelId, _options.CacheDirectory))
            return false;
        if (!LocalEmbedder.IsGgufModel(modelId))
            return await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

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
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        var model = await LocalEmbedder.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken);
        return new RemainingDownload(model, LocalEmbedder.IsGgufModel(modelId)
            ? await Server().GetRemainingDownloadBytesAsync(Backend(), cancellationToken)
            : await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));
    }

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        await LocalEmbedder.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        if (LocalEmbedder.IsGgufModel(modelId))
            await Server().AcquireServerAsync(Backend(), localFilesOnly: false, progress, cancellationToken);
        else
            await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalEmbedder.DeleteModel(modelId, _options.CacheDirectory);

    private LlamaServerUpdateService Server() => LlamaServerUpdateService.Resolve(_options.ServerUpdateOptions);

    private LlamaServerBackend Backend() =>
        Llama.LlamaBackendSelector.MapProvider(_options.Provider, HardwareProfile.For(_options.Provider).GpuInfo);
}
