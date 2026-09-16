using LMSupply.Download;
using LMSupply.Generator;
using LMSupply.Generator.Abstractions;
using LMSupply.Runtime;

namespace LMSupply.Generator.Onnx;

/// <summary>
/// Entry point for enabling ONNX Runtime GenAI model loading in LMSupply.Generator. Call
/// <see cref="Register"/> once at startup (before loading an ONNX model) after adding a
/// PackageReference to this package. The "auto" model selection never picks this backend -- it is
/// GGUF on every host -- so registering it only matters for explicit ONNX model ids.
/// </summary>
public static class OnnxGeneratorBackend
{
    /// <summary>
    /// Registers the ONNX Runtime GenAI backend with LMSupply.Generator. Idempotent — safe to
    /// call more than once (e.g. from multiple startup paths in the same process).
    /// </summary>
    public static void Register() => OnnxGeneratorBackendRegistry.Register(new Backend());

    private sealed class Backend : IOnnxGeneratorBackend
    {
        public IGeneratorModel CreateModel(
            string modelId,
            string modelPath,
            IChatFormatter chatFormatter,
            GeneratorOptions options,
            string? configBasePath = null)
            => new Internal.OnnxGeneratorModel(modelId, modelPath, chatFormatter, options, configBasePath);

        public IOnnxGeneratorModelFactory CreateFactory(string cacheDirectory, ExecutionProvider provider)
            => new OnnxGeneratorModelFactory(cacheDirectory, provider);

        public async Task EnsureRuntimeAsync(
            ExecutionProvider provider,
            IProgress<DownloadProgress>? progress,
            CancellationToken cancellationToken)
        {
            ExecutionProviderSupport.ThrowIfUnsupported(provider);

            // Initialize RuntimeManager to detect hardware
            await RuntimeManager.Instance.InitializeAsync(cancellationToken);

            // Resolve Auto to actual provider
            var actualProvider = provider == ExecutionProvider.Auto
                ? RuntimeManager.Instance.RecommendedProvider
                : provider;

            // Map provider to string for manifest lookup
            var providerString = actualProvider switch
            {
                ExecutionProvider.Cuda => RuntimeManager.Instance.GetDefaultProvider(), // cuda11 or cuda12
                ExecutionProvider.CoreML => "cpu", // CoreML uses CPU binaries
                _ => "cpu"
            };

            // Download base onnxruntime binaries first (genai depends on these)
            await RuntimeManager.Instance.EnsureRuntimeAsync(
                "onnxruntime",
                provider: providerString,
                progress: progress,
                cancellationToken: cancellationToken);

            // Download GenAI runtime binaries
            await RuntimeManager.Instance.EnsureRuntimeAsync(
                "onnxruntime-genai",
                provider: providerString,
                progress: progress,
                cancellationToken: cancellationToken);
        }
    }
}
