using LMSupply.Exceptions;
using LMSupply.Hardware;
using LMSupply.Lifecycle;
using LMSupply.Llama.Server;

namespace LMSupply.Generator;

/// <summary>
/// <see cref="IModelArtifacts"/> for text generators: whether a <see cref="LocalGenerator.LoadAsync(string, GeneratorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>
/// with downloads disabled would find its weights — and, for a GGUF model, llama-server — in the cache, and the download that
/// makes it so. The options are the ones the load will use (provider and cache directory decide which files that is).
/// </summary>
public sealed class GeneratorArtifacts : IModelArtifacts
{
    private readonly GeneratorOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public GeneratorArtifacts(GeneratorOptions? options = null)
    {
        _options = options?.Clone() ?? new GeneratorOptions();
    }

    /// <inheritdoc />
    public string Kind => "generator";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!LocalGenerator.IsModelDownloaded(modelId, _options))
            return false;
        if (!LocalGenerator.IsGgufTarget(modelId, _options.Clone()))
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
        => LocalGenerator.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken);

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        await LocalGenerator.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        if (LocalGenerator.IsGgufTarget(modelId, _options.Clone()))
            await Server().AcquireServerAsync(Backend(), localFilesOnly: false, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalGenerator.DeleteModel(modelId, _options);

    private LlamaServerUpdateService Server() => LlamaServerUpdateService.Resolve(_options.ServerUpdateOptions);

    private LlamaServerBackend Backend() =>
        Llama.LlamaBackendSelector.MapProvider(_options.Provider, HardwareProfile.For(_options.Provider).GpuInfo);
}
