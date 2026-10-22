using LMSupply.Hardware;
using LMSupply.Lifecycle;
using LMSupply.Llama.Server;

namespace LMSupply.Decider;

/// <summary>
/// <see cref="IModelArtifacts"/> for deciders: whether a <see cref="LocalDecider"/> load with downloads disabled would find
/// its GGUF file and llama-server in the cache, what is left to download, the download that makes it so, and the delete.
/// The options are the ones the load will use (provider, quantization hint and cache directory decide which files that is).
/// </summary>
public sealed class DeciderArtifacts : IModelArtifacts
{
    private readonly DeciderOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public DeciderArtifacts(DeciderOptions? options = null)
    {
        _options = options?.Clone() ?? new DeciderOptions();
    }

    /// <inheritdoc />
    public string Kind => "decider";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        if (!LocalDecider.IsModelDownloaded(modelId, _options))
            return false;

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
        => new(await LocalDecider.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await Server().GetRemainingDownloadBytesAsync(Backend(), cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalDecider.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await Server().AcquireServerAsync(Backend(), localFilesOnly: false, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalDecider.DeleteModel(modelId, _options);

    private LlamaServerUpdateService Server() => LlamaServerUpdateService.Resolve(_options.ServerUpdateOptions);

    private LlamaServerBackend Backend() =>
        Llama.LlamaBackendSelector.MapProvider(_options.Provider, HardwareProfile.For(_options.Provider).GpuInfo);
}
