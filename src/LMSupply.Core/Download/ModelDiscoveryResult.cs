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
    /// The detected model architecture type.
    /// </summary>
    public ModelArchitecture Architecture { get; init; } = ModelArchitecture.Unknown;

    /// <summary>
    /// List of encoder model files for encoder-decoder architectures.
    /// </summary>
    public IReadOnlyList<string> EncoderFiles { get; init; } = [];

    /// <summary>
    /// List of decoder model files for encoder-decoder architectures.
    /// </summary>
    public IReadOnlyList<string> DecoderFiles { get; init; } = [];

    /// <summary>
    /// The detected decoder variant type.
    /// </summary>
    public DecoderVariant DetectedDecoderVariant { get; init; } = DecoderVariant.Standard;

    /// <summary>
    /// Byte length of every file the repository listed, keyed by repository path — the length a download
    /// of each of <see cref="GetAllFiles"/> must produce. Empty when the listing carried no sizes.
    /// </summary>
    internal IReadOnlyDictionary<string, long> FileSizes { get; init; } = new Dictionary<string, long>(StringComparer.Ordinal);

    /// <summary>
    /// Gets the primary encoder file, or null if not an encoder-decoder model.
    /// </summary>
    public string? PrimaryEncoderFile => EncoderFiles.Count > 0 ? EncoderFiles[0] : null;

    /// <summary>
    /// Gets the primary decoder file, or null if not an encoder-decoder model.
    /// </summary>
    public string? PrimaryDecoderFile => DecoderFiles.Count > 0 ? DecoderFiles[0] : null;

    /// <summary>
    /// Whether this is an encoder-decoder model.
    /// </summary>
    public bool IsEncoderDecoder => Architecture == ModelArchitecture.EncoderDecoder;

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

    /// <summary>
    /// Gets the directory path where ONNX model files are located.
    /// This accounts for subfolder structures in HuggingFace repositories.
    /// </summary>
    /// <param name="baseModelDir">The base model directory from cache.</param>
    /// <returns>The full path to the directory containing ONNX files.</returns>
    public string GetOnnxDirectory(string baseModelDir)
    {
        if (string.IsNullOrEmpty(Subfolder))
            return baseModelDir;

        return Path.Combine(baseModelDir, Subfolder.Replace('/', Path.DirectorySeparatorChar));
    }

    /// <summary>
    /// Gets the full path to a specific file within the model directory.
    /// Handles both subfolder and root-level file locations.
    /// </summary>
    /// <param name="baseModelDir">The base model directory from cache.</param>
    /// <param name="relativePath">The relative path of the file (may include subfolder).</param>
    /// <returns>The full local path to the file.</returns>
    public static string GetFilePath(string baseModelDir, string relativePath)
    {
        return Path.Combine(baseModelDir, relativePath.Replace('/', Path.DirectorySeparatorChar));
    }

    /// <summary>
    /// Finds the actual path to an encoder file, searching in discovered encoder files.
    /// Returns null if no encoder file is found.
    /// </summary>
    /// <param name="baseModelDir">The base model directory from cache.</param>
    /// <returns>The full path to the encoder file, or null if not found.</returns>
    public string? GetEncoderPath(string baseModelDir)
    {
        if (EncoderFiles.Count == 0)
            return null;

        return GetFilePath(baseModelDir, EncoderFiles[0]);
    }

    /// <summary>
    /// Finds the actual path to a decoder file, searching in discovered decoder files.
    /// Returns null if no decoder file is found.
    /// </summary>
    /// <param name="baseModelDir">The base model directory from cache.</param>
    /// <returns>The full path to the decoder file, or null if not found.</returns>
    public string? GetDecoderPath(string baseModelDir)
    {
        if (DecoderFiles.Count == 0)
            return null;

        return GetFilePath(baseModelDir, DecoderFiles[0]);
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
    /// 8-bit quantized variants.
    /// </summary>
    public IReadOnlyList<string> Quant8 { get; init; } = [];

    /// <summary>
    /// 4-bit quantized variants.
    /// </summary>
    public IReadOnlyList<string> Quant4 { get; init; } = [];

    /// <summary>
    /// CPU-optimized variants.
    /// </summary>
    public IReadOnlyList<string> Cpu { get; init; } = [];

    /// <summary>
    /// CUDA/GPU-optimized variants.
    /// </summary>
    public IReadOnlyList<string> Cuda { get; init; } = [];
}
