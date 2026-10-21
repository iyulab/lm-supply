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
            return OnnxBackend() is { } onnx && await onnx.IsRuntimeAvailableOfflineAsync(_options.Provider, cancellationToken);

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
        var model = await LocalGenerator.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken);
        return new RemainingDownload(model, LocalGenerator.IsGgufTarget(modelId, _options.Clone())
            ? await Server().GetRemainingDownloadBytesAsync(Backend(), cancellationToken)
            : OnnxBackend() is { } onnx ? await onnx.GetRuntimeDownloadBytesAsync(_options.Provider, cancellationToken) : null);
    }

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        await LocalGenerator.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        if (LocalGenerator.IsGgufTarget(modelId, _options.Clone()))
            await Server().AcquireServerAsync(Backend(), localFilesOnly: false, progress, cancellationToken);
        else if (OnnxBackend() is { } onnx)
            await onnx.DownloadRuntimeAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalGenerator.DeleteModel(modelId, _options);

    // The ONNX backend a load of an ONNX model runs on; without one registered such a load fails, so there is no runtime to count
    private static Abstractions.IOnnxGeneratorBackend? OnnxBackend() =>
        OnnxGeneratorBackendRegistry.IsRegistered ? OnnxGeneratorBackendRegistry.Require() : null;

    private LlamaServerUpdateService Server() => LlamaServerUpdateService.Resolve(_options.ServerUpdateOptions);

    private LlamaServerBackend Backend() =>
        Llama.LlamaBackendSelector.MapProvider(_options.Provider, HardwareProfile.For(_options.Provider).GpuInfo);
}
