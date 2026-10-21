using LMSupply.Lifecycle;

namespace LMSupply.Ocr;

/// <summary>
/// <see cref="IModelArtifacts"/> for OCR, keyed by language: the model id is an ISO language code (e.g. <c>"en"</c>, <c>"ko"</c>),
/// as for <see cref="LocalOcr.LoadForLanguageAsync"/> — whether that load with downloads disabled would find the detection model
/// and the language's recognizer in the cache, and the download that makes it so. The options are the ones the load will use.
/// </summary>
/// <remarks>
/// A delete removes the language's recognizer and keeps the detection model every language shares; languages of one script
/// share their recognizer, so deleting one leaves the others not available offline either.
/// </remarks>
public sealed class OcrArtifacts : IModelArtifacts
{
    private readonly OcrOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public OcrArtifacts(OcrOptions? options = null)
    {
        _options = options?.Clone() ?? new OcrOptions();
    }

    /// <inheritdoc />
    public string Kind => "ocr";

    /// <inheritdoc />
    public Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => LocalOcr.IsLanguageDownloadedAsync(modelId, _options, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalOcr.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken), RuntimeBytes: null);

    /// <inheritdoc />
    public Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        => LocalOcr.DownloadForLanguageAsync(modelId, _options, progress, cancellationToken);

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalOcr.DeleteLanguageModel(modelId, _options);
}
