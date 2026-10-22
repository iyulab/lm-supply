using LMSupply.Lifecycle;

namespace LMSupply.Translator;

/// <summary>
/// <see cref="IModelArtifacts"/> for translators: whether a <see cref="LocalTranslator"/> load with downloads disabled would
/// find the model files and the ONNX Runtime it runs on in the cache, what is left to download, the download that makes it
/// so, and the delete. The options are the ones the load will use (provider, quantization hint and cache directory decide
/// which files that is).
/// </summary>
public sealed class TranslatorArtifacts : IModelArtifacts
{
    private readonly TranslatorOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public TranslatorArtifacts(TranslatorOptions? options = null)
    {
        _options = options?.Clone() ?? new TranslatorOptions();
    }

    /// <inheritdoc />
    public string Kind => "translator";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => await LocalTranslator.IsModelDownloadedAsync(modelId, _options, cancellationToken)
           && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalTranslator.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalTranslator.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalTranslator.DeleteModel(modelId, _options);
}
