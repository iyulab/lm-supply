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

    private const string GenAiManagedPackage = "Microsoft.ML.OnnxRuntimeGenAI.Managed";
    private const string GenAiVersionMetadataKey = "OnnxRuntimeGenAIVersion";

    /// <summary>
    /// The version of the managed ONNX Runtime GenAI package the application runs with. Its native runtime is provisioned
    /// at exactly this version: GenAI is 0.x and its native interface changes between minors. The managed assembly
    /// carries no version of its own (0.0.0.0), so this is the package version the application resolved (its deps.json),
    /// falling back to the version this package was built against.
    /// </summary>
    internal static string GenAiVersion { get; } = ReadResolvedGenAiVersion() ?? ReadBuildGenAiVersion();

    /// <summary>The GenAI managed package version in the application's deps.json, or null without one.</summary>
    internal static string? ReadResolvedGenAiVersion()
    {
        if (AppContext.GetData("APP_CONTEXT_DEPS_FILES") is not string files)
            return null;

        foreach (var file in files.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(file));
                if (!document.RootElement.TryGetProperty("libraries", out var libraries))
                    continue;

                foreach (var library in libraries.EnumerateObject())
                {
                    var slash = library.Name.IndexOf('/');
                    if (slash > 0 && library.Name.AsSpan(0, slash).Equals(GenAiManagedPackage, StringComparison.OrdinalIgnoreCase))
                        return library.Name[(slash + 1)..];
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
            {
                // An unreadable deps file says nothing about the version; the build-time version stands in
            }
        }

        return null;
    }

    /// <summary>The GenAI version this package was built against (stamped from the central package pin).</summary>
    internal static string ReadBuildGenAiVersion() =>
        typeof(OnnxGeneratorBackend).Assembly
            .GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
            .OfType<System.Reflection.AssemblyMetadataAttribute>()
            .FirstOrDefault(a => a.Key == GenAiVersionMetadataKey)?.Value
        ?? throw new InvalidOperationException($"LMSupply.Generator.Onnx was built without its {GenAiVersionMetadataKey} metadata.");

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
            if (provider == ExecutionProvider.OpenVino)
                throw new NotSupportedException(ExecutionProviderSupport.OpenVinoGenerationMessage);

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

            // Download GenAI runtime binaries at the managed assembly's version (not the NuGet latest)
            await RuntimeManager.Instance.EnsureRuntimeAsync(
                "onnxruntime-genai",
                version: GenAiVersion,
                provider: providerString,
                progress: progress,
                cancellationToken: cancellationToken);
        }
    }
}
