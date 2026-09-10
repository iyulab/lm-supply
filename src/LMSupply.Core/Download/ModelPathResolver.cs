using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Download;

/// <summary>
/// Provides centralized model path resolution for all domain packages.
/// Handles local paths, HuggingFace downloads, and subfolder discovery.
/// </summary>
public sealed class ModelPathResolver : IDisposable
{
    private readonly HuggingFaceDownloader _downloader;
    private bool _disposed;

    /// <summary>
    /// Creates a new model path resolver with the specified cache directory.
    /// </summary>
    /// <param name="cacheDirectory">The directory to cache downloaded models.</param>
    public ModelPathResolver(string? cacheDirectory = null)
    {
        cacheDirectory ??= CacheManager.GetDefaultCacheDirectory();
        _downloader = new HuggingFaceDownloader(cacheDirectory);
    }

    /// <summary>
    /// Result of model path resolution.
    /// </summary>
    public sealed class ResolveResult
    {
        /// <summary>
        /// The resolved path to the model file.
        /// </summary>
        public required string ModelPath { get; init; }

        /// <summary>
        /// The base model directory (for tokenizer/config files).
        /// </summary>
        public required string BaseDirectory { get; init; }

        /// <summary>
        /// The directory containing ONNX model files (may differ from BaseDirectory for subfolder repos).
        /// </summary>
        public required string OnnxDirectory { get; init; }

        /// <summary>
        /// Discovery result if downloaded from HuggingFace (null for local paths).
        /// </summary>
        public ModelDiscoveryResult? Discovery { get; init; }

        /// <summary>
        /// The file the caller asked for, when it was absent and a different one was used instead;
        /// <c>null</c> when the caller got what it named.
        /// </summary>
        /// <remarks>
        /// Substitution is legitimate for a caller that guessed - the fallback for an arbitrary repository
        /// names <c>model.onnx</c> without knowing whether that repository uses it. It is not legitimate
        /// silently: a caller that named an exact file and received a different one is running a different
        /// model, and the only evidence used to be the timings.
        /// </remarks>
        public string? SubstitutedForMissingFile { get; init; }
    }

    /// <summary>
    /// Resolves a single ONNX model file path from a model ID or local path.
    /// </summary>
    /// <param name="modelIdOrPath">Model ID (HuggingFace) or local path.</param>
    /// <param name="expectedOnnxFile">Expected ONNX filename (e.g., "model.onnx").</param>
    /// <param name="preferences">Download preferences for HuggingFace models.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Resolution result containing paths and discovery info.</returns>
    public async Task<ResolveResult> ResolveModelAsync(
        string modelIdOrPath,
        string expectedOnnxFile = "model.onnx",
        ModelPreferences? preferences = null,
        CancellationToken cancellationToken = default)
    {
        // Check if it's a local file path
        if (File.Exists(modelIdOrPath))
        {
            var directory = Path.GetDirectoryName(modelIdOrPath) ?? modelIdOrPath;
            return new ResolveResult
            {
                ModelPath = modelIdOrPath,
                BaseDirectory = directory,
                OnnxDirectory = directory,
                Discovery = null
            };
        }

        // Check if it's a local directory path
        if (Directory.Exists(modelIdOrPath))
        {
            // Validate for critical issues (.part files, LFS pointers) before resolution
            var validation = ModelDirectoryValidator.Validate(modelIdOrPath);

            // Only block on critical failures (.part files, LFS pointers, manifest mismatches).
            // Allow "No model files found" to fall through — model may be in subdirectories.
            if (!validation.IsValid
                && validation.Reason != null
                && !validation.Reason.StartsWith("No model files found", StringComparison.Ordinal))
            {
                throw new ModelLoadException(
                    $"Model directory validation failed: {validation.Reason}",
                    modelIdOrPath);
            }

            var localPath = Path.Combine(modelIdOrPath, expectedOnnxFile);
            if (File.Exists(localPath))
            {
                return new ResolveResult
                {
                    ModelPath = localPath,
                    BaseDirectory = modelIdOrPath,
                    OnnxDirectory = modelIdOrPath,
                    Discovery = null
                };
            }

            // Try to find any ONNX file in the directory (including subdirectories)
            var onnxFiles = Directory.GetFiles(modelIdOrPath, "*.onnx", SearchOption.AllDirectories);
            if (onnxFiles.Length > 0)
            {
                var firstOnnx = onnxFiles[0];
                var onnxDir = Path.GetDirectoryName(firstOnnx) ?? modelIdOrPath;
                WarnSubstitution(expectedOnnxFile, Path.GetFileName(firstOnnx), modelIdOrPath);
                return new ResolveResult
                {
                    ModelPath = firstOnnx,
                    BaseDirectory = modelIdOrPath,
                    OnnxDirectory = onnxDir,
                    Discovery = null,
                    SubstitutedForMissingFile = expectedOnnxFile
                };
            }

            throw new FileNotFoundException(
                $"No ONNX model found in directory: {modelIdOrPath}",
                expectedOnnxFile);
        }

        // Strip variant suffix from HuggingFace repo IDs (e.g., "owner/repo:variant" → "owner/repo")
        // Variant selects a specific file within the repo; the file is identified by expectedOnnxFile.
        var repoId = StripVariantSuffix(modelIdOrPath);

        // Download from HuggingFace using discovery for proper subfolder handling
        preferences ??= ModelPreferences.ForCurrentHardware();

        // The caller named a file, so ask for that file. Without this the quantization preference derived
        // from the machine's hardware tier decides instead, and on a repository publishing int8 alongside
        // the plain weights it fetches the int8 and never the file that was named - after which the
        // fallback below hands that over as though nothing had happened.
        preferences = WithRequestedFile(preferences, expectedOnnxFile);

        var (modelDir, discovery) = await _downloader.DownloadWithDiscoveryAsync(
            repoId,
            preferences: preferences,
            cancellationToken: cancellationToken);

        // Use discovery result to find ONNX file in correct directory
        var onnxDirectory = discovery.GetOnnxDirectory(modelDir);
        var modelPath = Path.Combine(onnxDirectory, expectedOnnxFile);
        string? substituted = null;

        // If the named file really is not in this repository, fall back to what was discovered - a caller
        // resolving an arbitrary repository guesses "model.onnx" and is often wrong - but say so.
        if (!File.Exists(modelPath) && discovery.OnnxFiles.Count > 0)
        {
            modelPath = ModelDiscoveryResult.GetFilePath(modelDir, discovery.OnnxFiles[0]);
            substituted = expectedOnnxFile;
            WarnSubstitution(expectedOnnxFile, Path.GetFileName(modelPath), repoId);
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"ONNX model not found in {modelDir}. Expected: {expectedOnnxFile}. " +
                $"Discovered ONNX files: [{string.Join(", ", discovery.OnnxFiles)}]",
                modelPath);
        }

        return new ResolveResult
        {
            ModelPath = modelPath,
            BaseDirectory = modelDir,
            OnnxDirectory = onnxDirectory,
            Discovery = discovery,
            SubstitutedForMissingFile = substituted
        };
    }

    /// <summary>
    /// Resolves encoder-decoder model paths from a model ID or local path.
    /// </summary>
    /// <param name="modelIdOrPath">Model ID (HuggingFace) or local path.</param>
    /// <param name="expectedEncoderFile">Expected encoder filename.</param>
    /// <param name="expectedDecoderFile">Expected decoder filename.</param>
    /// <param name="preferences">Download preferences for HuggingFace models.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Resolution result with encoder/decoder paths.</returns>
    public async Task<EncoderDecoderResolveResult> ResolveEncoderDecoderAsync(
        string modelIdOrPath,
        string expectedEncoderFile = "encoder_model.onnx",
        string expectedDecoderFile = "decoder_model_merged.onnx",
        ModelPreferences? preferences = null,
        CancellationToken cancellationToken = default)
    {
        // Check if it's a local directory path
        if (Directory.Exists(modelIdOrPath))
        {
            var encoderPath = Path.Combine(modelIdOrPath, expectedEncoderFile);
            var decoderPath = Path.Combine(modelIdOrPath, expectedDecoderFile);

            if (File.Exists(encoderPath) && File.Exists(decoderPath))
            {
                return new EncoderDecoderResolveResult
                {
                    EncoderPath = encoderPath,
                    DecoderPath = decoderPath,
                    BaseDirectory = modelIdOrPath,
                    OnnxDirectory = modelIdOrPath,
                    Discovery = null
                };
            }
        }

        // Strip variant suffix from HuggingFace repo IDs
        var repoId = StripVariantSuffix(modelIdOrPath);

        // Download from HuggingFace using discovery
        preferences ??= ModelPreferences.ForCurrentHardware();

        var (modelDir, discovery) = await _downloader.DownloadWithDiscoveryAsync(
            repoId,
            preferences: preferences,
            cancellationToken: cancellationToken);

        // Use discovery result for path resolution
        var onnxDirectory = discovery.GetOnnxDirectory(modelDir);

        // Try discovery-based paths first
        var encoderPath2 = discovery.GetEncoderPath(modelDir)
            ?? Path.Combine(onnxDirectory, expectedEncoderFile);
        var decoderPath2 = discovery.GetDecoderPath(modelDir)
            ?? Path.Combine(onnxDirectory, expectedDecoderFile);

        if (!File.Exists(encoderPath2))
        {
            throw new FileNotFoundException(
                $"Encoder model not found. Expected: {expectedEncoderFile}. " +
                $"Discovered encoder files: [{string.Join(", ", discovery.EncoderFiles)}]",
                encoderPath2);
        }

        if (!File.Exists(decoderPath2))
        {
            throw new FileNotFoundException(
                $"Decoder model not found. Expected: {expectedDecoderFile}. " +
                $"Discovered decoder files: [{string.Join(", ", discovery.DecoderFiles)}]",
                decoderPath2);
        }

        return new EncoderDecoderResolveResult
        {
            EncoderPath = encoderPath2,
            DecoderPath = decoderPath2,
            BaseDirectory = modelDir,
            OnnxDirectory = onnxDirectory,
            Discovery = discovery
        };
    }

    /// <summary>
    /// Result of encoder-decoder model path resolution.
    /// </summary>
    public sealed class EncoderDecoderResolveResult
    {
        /// <summary>
        /// Path to the encoder model file.
        /// </summary>
        public required string EncoderPath { get; init; }

        /// <summary>
        /// Path to the decoder model file.
        /// </summary>
        public required string DecoderPath { get; init; }

        /// <summary>
        /// The base model directory (for tokenizer/config files).
        /// </summary>
        public required string BaseDirectory { get; init; }

        /// <summary>
        /// The directory containing ONNX model files.
        /// </summary>
        public required string OnnxDirectory { get; init; }

        /// <summary>
        /// Discovery result if downloaded from HuggingFace.
        /// </summary>
        public ModelDiscoveryResult? Discovery { get; init; }
    }

    /// <summary>
    /// Strips variant suffix from a model ID (e.g., "owner/repo:variant" → "owner/repo").
    /// Only strips when the colon appears after the last '/' in a HuggingFace-style repo ID.
    /// Preserves Windows drive letters (e.g., "D:\path"), local paths, and plain aliases.
    /// </summary>
    internal static string StripVariantSuffix(string modelIdOrPath)
    {
        // Must contain '/' to be a HuggingFace repo ID
        if (!modelIdOrPath.Contains('/'))
            return modelIdOrPath;

        var colonIdx = modelIdOrPath.LastIndexOf(':');
        if (colonIdx <= 0 || colonIdx == modelIdOrPath.Length - 1)
            return modelIdOrPath;

        // Only strip if the colon comes after the last '/' (variant part of repo ID)
        var lastSlash = modelIdOrPath.LastIndexOf('/');
        if (colonIdx > lastSlash)
            return modelIdOrPath[..colonIdx];

        return modelIdOrPath;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _downloader.Dispose();
    }

    /// <summary>
    /// Returns preferences that ask for <paramref name="requestedFile"/> by name, leaving an explicit
    /// caller-supplied file list alone.
    /// </summary>
    internal static ModelPreferences WithRequestedFile(ModelPreferences preferences, string requestedFile)
    {
        ArgumentNullException.ThrowIfNull(preferences);

        if (string.IsNullOrWhiteSpace(requestedFile) || preferences.PreferredOnnxFiles.Count > 0)
            return preferences;

        return new ModelPreferences
        {
            PreferLowMemory = preferences.PreferLowMemory,
            QuantizationPriority = preferences.QuantizationPriority,
            DecoderVariantPriority = preferences.DecoderVariantPriority,
            PreferredOnnxFiles = [requestedFile]
        };
    }

    private static void WarnSubstitution(string requested, string used, string source)
    {
        if (string.Equals(requested, used, StringComparison.OrdinalIgnoreCase))
            return;

        Trace.TraceWarning(
            $"[ModelPathResolver] '{source}' does not contain '{requested}'; loading '{used}' instead. " +
            "This is a different model - neither its speed nor its accuracy is the one that was asked for.");
    }
}
