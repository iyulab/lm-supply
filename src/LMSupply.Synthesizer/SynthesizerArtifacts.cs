using LMSupply.Lifecycle;

namespace LMSupply.Synthesizer;

/// <summary>
/// <see cref="IModelArtifacts"/> for speech synthesizers: whether a <see cref="LocalSynthesizer"/> load with downloads
/// disabled would find the voice's model and config and the ONNX Runtime it runs on in the cache, what is left to download,
/// the download that makes it so, and the delete — of this voice only, though the catalog's voices share a repository. The
/// options are the ones the load will use (provider and cache directory decide which files that is).
/// </summary>
public sealed class SynthesizerArtifacts : IModelArtifacts
{
    private readonly SynthesizerOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public SynthesizerArtifacts(SynthesizerOptions? options = null)
    {
        _options = options?.Clone() ?? new SynthesizerOptions();
    }

    /// <inheritdoc />
    public string Kind => "synthesizer";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => await LocalSynthesizer.IsModelDownloadedAsync(modelId, _options, cancellationToken)
           && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalSynthesizer.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalSynthesizer.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalSynthesizer.DeleteModel(modelId, _options);
}
