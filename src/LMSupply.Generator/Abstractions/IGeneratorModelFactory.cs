namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Factory for loading generator model instances.
/// </summary>
public interface IGeneratorModelFactory
{
    /// <summary>
    /// Loads a generator model asynchronously.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "microsoft/Phi-3.5-mini-instruct-onnx").</param>
    /// <param name="options">Optional model configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The loaded generator model.</returns>
    Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if a model is available locally (already downloaded).
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if the model is available locally.</returns>
    bool IsModelAvailable(string modelId);

    /// <summary>
    /// Downloads a model if not already available.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task DownloadModelAsync(
        string modelId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Progress information for model downloads.
/// </summary>
/// <param name="TotalBytes">Total bytes to download.</param>
/// <param name="DownloadedBytes">Bytes downloaded so far.</param>
/// <param name="FileName">Current file being downloaded.</param>
public sealed record ModelDownloadProgress(
    long TotalBytes,
    long DownloadedBytes,
    string? FileName)
{
    /// <summary>
    /// Gets the download progress as a percentage (0-100).
    /// </summary>
    public double ProgressPercent =>
        TotalBytes > 0 ? (double)DownloadedBytes / TotalBytes * 100 : 0;
}
