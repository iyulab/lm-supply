using LMSupply.Download;

namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Backend seam for ONNX Runtime GenAI-based text generation, implemented by the optional
/// LMSupply.Generator.Onnx package. LMSupply.Generator itself has no compile-time dependency on
/// Microsoft.ML.OnnxRuntimeGenAI, so a consumer using only the GGUF/llama-server backend never
/// pulls that native runtime in. Register an implementation via
/// <see cref="LMSupply.Generator.OnnxGeneratorBackendRegistry.Register"/> before loading an ONNX
/// model (the LMSupply.Generator.Onnx package exposes <c>OnnxGeneratorBackend.Register()</c> for
/// this).
/// </summary>
public interface IOnnxGeneratorBackend
{
    /// <summary>
    /// Constructs a loaded generator model from an already-resolved local ONNX model directory.
    /// </summary>
    IGeneratorModel CreateModel(
        string modelId,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        string? configBasePath = null);

    /// <summary>
    /// Creates a factory for resolving, downloading, and loading ONNX models by HuggingFace
    /// model ID.
    /// </summary>
    IOnnxGeneratorModelFactory CreateFactory(string cacheDirectory, ExecutionProvider provider);

    /// <summary>
    /// Ensures the ONNX Runtime GenAI native binaries are downloaded for the given provider.
    /// </summary>
    Task EnsureRuntimeAsync(
        ExecutionProvider provider,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken);
}
