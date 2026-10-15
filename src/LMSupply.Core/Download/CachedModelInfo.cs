namespace LMSupply.Download;

/// <summary>
/// Information about a cached model in the HuggingFace cache directory.
/// </summary>
public sealed record CachedModelInfo
{
    /// <summary>
    /// HuggingFace repository ID (e.g., "microsoft/Phi-4-mini-instruct-onnx").
    /// </summary>
    public required string RepoId { get; init; }

    /// <summary>
    /// Local filesystem path to the model snapshot.
    /// </summary>
    public required string LocalPath { get; init; }

    /// <summary>
    /// Total size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Total size in megabytes.
    /// </summary>
    public double SizeMB => SizeBytes / (1024.0 * 1024.0);

    /// <summary>
    /// Number of files in the model directory.
    /// </summary>
    public int FileCount { get; init; }

    /// <summary>
    /// Detected model type based on file patterns and repository ID.
    /// </summary>
    public ModelType DetectedType { get; init; }

    /// <summary>
    /// Whether this library downloaded anything of the repository — exactly when <see cref="CacheManager.DeleteModel"/>
    /// would delete something (<see cref="CacheManager.HasOwnFiles"/>). False for a repository only other tools wrote into
    /// the shared hub cache: a listing can offer Delete where it acts and say «downloaded by another application» elsewhere.
    /// </summary>
    public bool HasOwnFiles { get; init; }

    /// <summary>
    /// Last modification time of the model directory.
    /// </summary>
    public DateTime LastModified { get; init; }

    /// <summary>
    /// List of file names in the model directory.
    /// </summary>
    public IReadOnlyList<string> Files { get; init; } = [];

    /// <summary>
    /// Minimum size threshold for a model to be considered complete (1MB).
    /// </summary>
    public const long MinimumCompleteSize = 1024 * 1024;

    /// <summary>
    /// Returns true if the model appears to be completely downloaded.
    /// </summary>
    public bool IsComplete => SizeBytes >= MinimumCompleteSize;

    /// <summary>
    /// Optional metadata from HuggingFace Hub (populated if .metadata.json exists).
    /// </summary>
    public ModelMetadata? Metadata { get; init; }

    /// <summary>
    /// Returns true if HuggingFace metadata is available.
    /// </summary>
    public bool HasMetadata => Metadata is not null;
}
