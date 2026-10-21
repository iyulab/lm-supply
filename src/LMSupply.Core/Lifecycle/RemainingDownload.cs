namespace LMSupply.Lifecycle;

/// <summary>
/// What <see cref="IModelArtifacts.DownloadAsync"/> would still fetch for one model on this machine, in bytes: the model's files
/// and the runtimes its load runs on, apart, so a consent prompt can state both.
/// </summary>
/// <param name="ModelBytes">The model's files not in the cache at the length the repository lists; 0 when all are.</param>
/// <param name="RuntimeBytes">
/// The runtimes a load needs that are not in the cache — llama-server (and, for CUDA, its runtime companion) for a GGUF model,
/// the ONNX Runtime native package (and ONNX Runtime GenAI for an ONNX generator) for an ONNX one; 0 when they are cached or
/// supplied by the application. Null when that
/// cannot be told: the release listing or package feed it is read from cannot be reached, or a load would not download it
/// (runtime downloads disabled in <c>RuntimeManagerOptions</c>, or an application runtime directory that lacks it).
/// </param>
public readonly record struct RemainingDownload(long ModelBytes, long? RuntimeBytes)
{
    /// <summary>Everything still to fetch; null when <see cref="RuntimeBytes"/> is unknown.</summary>
    public long? TotalBytes => RuntimeBytes is { } runtime ? ModelBytes + runtime : null;
}
