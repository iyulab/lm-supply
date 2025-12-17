namespace LMSupply.Core.Download;

/// <summary>
/// Result of automatic model file discovery from a HuggingFace repository.
/// </summary>
public sealed class ModelDiscoveryResult
{
    /// <summary>
    /// The HuggingFace repository ID.
    /// </summary>
    public required string RepoId { get; init; }

    /// <summary>
    /// The detected subfolder containing ONNX files, or null if in root.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// List of ONNX model files to download (relative paths).
    /// </summary>
    public required IReadOnlyList<string> OnnxFiles { get; init; }

    /// <summary>
    /// List of external data files (.onnx_data) associated with the model.
    /// </summary>
    public IReadOnlyList<string> ExternalDataFiles { get; init; } = [];

    /// <summary>
    /// List of configuration and tokenizer files.
    /// </summary>
    public IReadOnlyList<string> ConfigFiles { get; init; } = [];

    /// <summary>
    /// All available model variants organized by category.
    /// </summary>
    public ModelVariants? AvailableVariants { get; init; }

    /// <summary>
    /// Whether the model has external data files that must be downloaded together.
    /// </summary>
    public bool HasExternalData => ExternalDataFiles.Count > 0;

    /// <summary>
    /// Gets all files that need to be downloaded.
    /// </summary>
    public IEnumerable<string> GetAllFiles()
    {
        foreach (var file in OnnxFiles)
            yield return file;

        foreach (var file in ExternalDataFiles)
            yield return file;

        foreach (var file in ConfigFiles)
            yield return file;
    }

    /// <summary>
    /// Gets just the file names without subfolder prefix.
    /// </summary>
    public IEnumerable<string> GetFileNames()
    {
        return GetAllFiles().Select(f =>
        {
            var lastSlash = f.LastIndexOf('/');
            return lastSlash >= 0 ? f[(lastSlash + 1)..] : f;
        });
    }
}

/// <summary>
/// Categorized model variants available in the repository.
/// </summary>
public sealed class ModelVariants
{
    /// <summary>
    /// Default/full precision variants.
    /// </summary>
    public IReadOnlyList<string> Default { get; init; } = [];

    /// <summary>
    /// FP16 (half precision) variants.
    /// </summary>
    public IReadOnlyList<string> Fp16 { get; init; } = [];

    /// <summary>
    /// INT8 quantized variants.
    /// </summary>
    public IReadOnlyList<string> Int8 { get; init; } = [];

    /// <summary>
    /// INT4 quantized variants.
    /// </summary>
    public IReadOnlyList<string> Int4 { get; init; } = [];

    /// <summary>
    /// CPU-optimized variants.
    /// </summary>
    public IReadOnlyList<string> Cpu { get; init; } = [];

    /// <summary>
    /// CUDA/GPU-optimized variants.
    /// </summary>
    public IReadOnlyList<string> Cuda { get; init; } = [];
}
