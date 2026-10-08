namespace LMSupply.Runtime;

/// <summary>
/// Registry for ONNX Runtime package configurations.
/// Provides a centralized, extensible mapping of runtime types and providers to NuGet packages.
/// </summary>
public static class RuntimePackageRegistry
{
    /// <summary>
    /// Known runtime package types.
    /// </summary>
    public static class PackageTypes
    {
        public const string OnnxRuntime = "onnxruntime";
        public const string OnnxRuntimeGenAI = "onnxruntime-genai";

        /// <summary>
        /// An execution provider delivered as an ONNX Runtime plugin library: provisioned next to the base runtime
        /// (which still has to be loaded) and handed to ONNX Runtime by path, never loaded through P/Invoke.
        /// </summary>
        public const string ExecutionProviderPlugin = "onnxruntime-ep";
    }

    /// <summary>
    /// Known execution providers.
    /// </summary>
    public static class Providers
    {
        public const string Cpu = "cpu";
        public const string Cuda = "cuda";
        public const string Cuda11 = "cuda11";
        public const string Cuda12 = "cuda12";
        public const string CoreML = "coreml";
        public const string OpenVino = "openvino";
    }

    /// <summary>
    /// Package configuration for a specific runtime/provider combination.
    /// </summary>
    public sealed record PackageConfig(
        string PackageId,
        string NativeLibraryName,
        string[] AdditionalLibraries = default!)
    {
        public string[] AdditionalLibraries { get; init; } = AdditionalLibraries ?? [];

        /// <summary>
        /// The one version of the package this library serves, or <see langword="null"/> when the version follows the
        /// loaded ONNX Runtime assembly. A plugin package versions independently of ONNX Runtime, so its version is
        /// fixed here rather than resolved: "the latest on the feed" would be a native binary no test has run.
        /// </summary>
        public string? PinnedVersion { get; init; }
    }

    // ONNX Runtime package mappings
    // Note: CUDA packages use platform-specific overrides via CudaPackagesByPlatform,
    // but entries here are needed for GetSupportedProviders() to return cuda11/cuda12
    //
    // No DirectML entry (0.67.0): Microsoft.ML.OnnxRuntime.DirectML ends at 1.24.4 and the runtime
    // version is taken from the loaded ONNX Runtime assembly (1.30.0), so the entry could only ever
    // ask nuget.org for a package that does not exist. The GenAI registry below has none either --
    // Microsoft.ML.OnnxRuntimeGenAI.DirectML is still published, but OnnxGeneratorBackend provisions
    // the base "onnxruntime" package for the provider first, so a DirectML request dies there before
    // GenAI is consulted. See ExecutionProviderSupport.
    private static readonly Dictionary<string, PackageConfig> OnnxRuntimePackages = new(StringComparer.OrdinalIgnoreCase)
    {
        [Providers.Cpu] = new("Microsoft.ML.OnnxRuntime", "onnxruntime"),
        [Providers.Cuda] = new("Microsoft.ML.OnnxRuntime.Gpu", "onnxruntime", ["onnxruntime_providers_cuda", "onnxruntime_providers_shared"]),
        [Providers.Cuda11] = new("Microsoft.ML.OnnxRuntime.Gpu", "onnxruntime", ["onnxruntime_providers_cuda", "onnxruntime_providers_shared"]),
        [Providers.Cuda12] = new("Microsoft.ML.OnnxRuntime.Gpu", "onnxruntime", ["onnxruntime_providers_cuda", "onnxruntime_providers_shared"]),
    };

    // ONNX Runtime GenAI package mappings
    private static readonly Dictionary<string, PackageConfig> GenAiPackages = new(StringComparer.OrdinalIgnoreCase)
    {
        [Providers.Cpu] = new("Microsoft.ML.OnnxRuntimeGenAI", "onnxruntime-genai"),
        [Providers.Cuda] = new("Microsoft.ML.OnnxRuntimeGenAI.Cuda", "onnxruntime-genai", ["onnxruntime-genai-cuda"]),
        [Providers.Cuda11] = new("Microsoft.ML.OnnxRuntimeGenAI.Cuda", "onnxruntime-genai", ["onnxruntime-genai-cuda"]),
        [Providers.Cuda12] = new("Microsoft.ML.OnnxRuntimeGenAI.Cuda", "onnxruntime-genai", ["onnxruntime-genai-cuda"]),
    };

    /// <summary>The OpenVINO plugin version this library is tested with.</summary>
    public const string OpenVinoPluginVersion = "1.7.0";

    // Execution provider plugin packages (win-x64 and linux-x64 natives only)
    private static readonly Dictionary<string, PackageConfig> ExecutionProviderPluginPackages = new(StringComparer.OrdinalIgnoreCase)
    {
        [Providers.OpenVino] = new("Intel.ML.OnnxRuntime.EP.OpenVINO", "onnxruntime_providers_openvino_plugin")
        {
            PinnedVersion = OpenVinoPluginVersion,
        },
    };

    // Platform-specific CUDA package overrides (ONNX Runtime only)
    private static readonly Dictionary<string, string> CudaPackagesByPlatform = new(StringComparer.OrdinalIgnoreCase)
    {
        ["win-x64"] = "Microsoft.ML.OnnxRuntime.Gpu.Windows",
        ["win-arm64"] = "Microsoft.ML.OnnxRuntime.Gpu.Windows",
        ["linux-x64"] = "Microsoft.ML.OnnxRuntime.Gpu",
        ["linux-arm64"] = "Microsoft.ML.OnnxRuntime.Gpu",
    };

    /// <summary>
    /// Gets the package configuration for a runtime type and provider.
    /// </summary>
    /// <param name="packageType">The package type (e.g., "onnxruntime", "onnxruntime-genai").</param>
    /// <param name="provider">The execution provider (e.g., "cpu", "cuda12").</param>
    /// <param name="runtimeIdentifier">Optional RID for platform-specific package selection.</param>
    /// <returns>The package configuration, or null if not found.</returns>
    public static PackageConfig? GetPackageConfig(
        string packageType,
        string provider,
        string? runtimeIdentifier = null)
    {
        var normalizedProvider = NormalizeProvider(provider);

        var registry = RegistryFor(packageType);

        // Handle CUDA platform-specific packages for standard ONNX Runtime
        if (IsCudaProvider(normalizedProvider) &&
            packageType.Equals(PackageTypes.OnnxRuntime, StringComparison.OrdinalIgnoreCase) &&
            !string.IsNullOrEmpty(runtimeIdentifier) &&
            CudaPackagesByPlatform.TryGetValue(runtimeIdentifier, out var cudaPackageId))
        {
            return new PackageConfig(cudaPackageId, "onnxruntime", ["onnxruntime_providers_cuda", "onnxruntime_providers_shared"]);
        }

        // Look up in the registry
        if (registry.TryGetValue(normalizedProvider, out var config))
        {
            return config;
        }

        // A plugin is its provider: there is no CPU package to stand in for it.
        if (registry == ExecutionProviderPluginPackages)
        {
            return null;
        }

        // Fallback to CPU
        return registry.GetValueOrDefault(Providers.Cpu);
    }

    /// <summary>
    /// Gets all supported providers for a package type.
    /// </summary>
    public static IEnumerable<string> GetSupportedProviders(string packageType)
    {
        var registry = RegistryFor(packageType);

        return registry.Keys;
    }

    private static Dictionary<string, PackageConfig> RegistryFor(string packageType)
    {
        if (packageType.Equals(PackageTypes.OnnxRuntimeGenAI, StringComparison.OrdinalIgnoreCase))
            return GenAiPackages;
        if (packageType.Equals(PackageTypes.ExecutionProviderPlugin, StringComparison.OrdinalIgnoreCase))
            return ExecutionProviderPluginPackages;
        return OnnxRuntimePackages;
    }

    /// <summary>
    /// Checks if a provider is CUDA-based.
    /// </summary>
    public static bool IsCudaProvider(string provider)
    {
        return provider.Equals(Providers.Cuda, StringComparison.OrdinalIgnoreCase) ||
               provider.Equals(Providers.Cuda11, StringComparison.OrdinalIgnoreCase) ||
               provider.Equals(Providers.Cuda12, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Normalizes provider names (e.g., "CUDA" → "cuda").
    /// </summary>
    private static string NormalizeProvider(string provider)
    {
        return provider.ToLowerInvariant() switch
        {
            "auto" => Providers.Cpu, // Auto resolves elsewhere; default to CPU for package lookup
            _ => provider.ToLowerInvariant()
        };
    }

    /// <summary>
    /// Gets the native library filename with platform-specific extension.
    /// </summary>
    public static string GetNativeLibraryFileName(string libraryName, PlatformInfo platform)
    {
        if (platform.IsWindows)
            return $"{libraryName}.dll";
        if (platform.IsMacOS)
            return $"lib{libraryName}.dylib";
        return $"lib{libraryName}.so";
    }
}
