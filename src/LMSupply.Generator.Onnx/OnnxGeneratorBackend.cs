using LMSupply.Lifecycle;
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

        // The host joins the deps files with ';' on every platform (not Path.PathSeparator, which is ':' on Unix)
        foreach (var file in files.Split(';', StringSplitOptions.RemoveEmptyEntries))
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
            // Base onnxruntime binaries first (genai depends on these), then GenAI at the managed assembly's version (not
            // the NuGet latest)
            foreach (var package in await RuntimePackagesAsync(RuntimeManager.Instance, provider, cancellationToken))
            {
                await RuntimeManager.Instance.EnsureRuntimeAsync(
                    package.Package,
                    version: package.Version,
                    provider: package.Slot,
                    progress: progress,
                    cancellationToken: cancellationToken);
            }
        }

        public async Task<bool> IsRuntimeAvailableOfflineAsync(ExecutionProvider provider, CancellationToken cancellationToken)
            => await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(
                await RuntimePackagesAsync(OnnxRuntimeArtifacts.Manager, provider, cancellationToken), cancellationToken);

        public async Task<long?> GetRuntimeDownloadBytesAsync(ExecutionProvider provider, CancellationToken cancellationToken)
            => await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(
                await RuntimePackagesAsync(OnnxRuntimeArtifacts.Manager, provider, cancellationToken), cancellationToken);

        public async Task DownloadRuntimeAsync(ExecutionProvider provider, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
            => await OnnxRuntimeArtifacts.DownloadAsync(
                await RuntimePackagesAsync(OnnxRuntimeArtifacts.Manager, provider, cancellationToken), progress, cancellationToken);

        // The runtime packages a GenAI session of the provider runs on: ONNX Runtime and GenAI, in the provider's slot
        private static async Task<IReadOnlyList<OnnxRuntimeArtifacts.RuntimePackage>> RuntimePackagesAsync(
            RuntimeManager manager, ExecutionProvider provider, CancellationToken cancellationToken)
        {
            ExecutionProviderSupport.ThrowIfUnsupported(provider);
            if (provider == ExecutionProvider.OpenVino)
                throw new NotSupportedException(ExecutionProviderSupport.OpenVinoGenerationMessage);

            // Initialize RuntimeManager to detect hardware
            await manager.InitializeAsync(cancellationToken);

            // Resolve Auto to actual provider
            var actualProvider = provider == ExecutionProvider.Auto
                ? manager.RecommendedProvider
                : provider;

            // Map provider to string for manifest lookup
            var providerString = actualProvider switch
            {
                ExecutionProvider.Cuda => manager.GetDefaultProvider(), // cuda11 or cuda12
                ExecutionProvider.CoreML => "cpu", // CoreML uses CPU binaries
                _ => "cpu"
            };

            return [new("onnxruntime", providerString), new("onnxruntime-genai", providerString, GenAiVersion)];
        }
    }
}
