using LMSupply.Runtime;

namespace LMSupply.Generator;

/// <summary>
/// Detects hardware capabilities and provides recommendations for LLM inference.
/// </summary>
public static class HardwareDetector
{
    /// <summary>
    /// Gets hardware recommendation for LLM inference.
    /// </summary>
    public static HardwareRecommendation GetRecommendation()
    {
        // Use the cached HardwareProfile so GPU detection and the provider decision are made once
        // and stay consistent with the rest of the system (single source of truth).
        var profile = Hardware.HardwareProfile.Current;
        return CreateRecommendation(profile);
    }

    /// <summary>
    /// Gets the hardware recommendation for a load with <paramref name="provider"/>. An explicit
    /// <see cref="ExecutionProvider.Cpu"/> is answered from system memory without probing the GPU.
    /// </summary>
    public static HardwareRecommendation GetRecommendation(ExecutionProvider provider)
    {
        var profile = Hardware.HardwareProfile.For(provider);
        return CreateRecommendation(profile);
    }

    /// <summary>
    /// Gets the best available execution provider.
    /// Delegates to <see cref="Hardware.HardwareProfile"/> — the single source of truth for the
    /// VRAM-aware provider recommendation — instead of duplicating the decision here.
    /// </summary>
    public static ExecutionProvider GetBestProvider()
        => Hardware.HardwareProfile.Current.RecommendedProvider;

    /// <summary>
    /// Resolves Auto execution provider to the best available option.
    /// </summary>
    public static ExecutionProvider ResolveProvider(ExecutionProvider provider)
    {
        return provider == ExecutionProvider.Auto ? GetBestProvider() : provider;
    }

    private static HardwareRecommendation CreateRecommendation(Hardware.HardwareProfile profile)
    {
        var gpuInfo = profile.GpuInfo;
        var gpuMemoryGB = (gpuInfo.TotalMemoryBytes ?? 0) / (1024.0 * 1024 * 1024);
        var systemMemoryGB = profile.SystemMemoryBytes / (1024.0 * 1024 * 1024);

        // The provider the profile recommends: the detected one for Current, CPU for the CPU-only profile.
        var provider = profile.RecommendedProvider;

        // Determine recommended model size based on available memory
        var (maxModelParams, quantization, maxContext) = (gpuMemoryGB, systemMemoryGB, provider) switch
        {
            // GPU scenarios
            (>= 16, _, ExecutionProvider.Cuda) => (14_000_000_000L, "FP16", 16384),
            (>= 8, _, ExecutionProvider.Cuda) => (7_000_000_000L, "Quant8", 8192),
            (>= 4, _, ExecutionProvider.Cuda) => (3_000_000_000L, "Quant4", 4096),

            // CPU-only scenarios
            (_, >= 32, ExecutionProvider.Cpu) => (7_000_000_000L, "Quant4", 8192),
            (_, >= 16, ExecutionProvider.Cpu) => (3_000_000_000L, "Quant4", 4096),
            (_, >= 8, ExecutionProvider.Cpu) => (1_000_000_000L, "Quant4", 2048),

            // Minimal requirements
            _ => (1_000_000_000L, "Quant4", 2048)
        };

        // Recommend specific models
        var recommendedModels = GetRecommendedModels(maxModelParams, quantization);

        return new HardwareRecommendation
        {
            Provider = provider,
            GpuInfo = gpuInfo,
            SystemMemoryBytes = profile.SystemMemoryBytes,
            MaxModelParameters = maxModelParams,
            RecommendedQuantization = quantization,
            MaxContextLength = maxContext,
            RecommendedModels = recommendedModels
        };
    }

    private static List<string> GetRecommendedModels(long maxParams, string quantization)
    {
        var models = new List<string>();

        // Always recommend Phi-3.5-mini (3.8B) for Quant4
        if (maxParams >= 3_000_000_000)
        {
            models.Add("microsoft/Phi-3.5-mini-instruct-onnx");
        }

        // Add Llama-3.2-1B for smaller systems
        if (maxParams >= 1_000_000_000)
        {
            models.Add("onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX");
        }

        // Add Llama-3.2-3B for medium systems
        if (maxParams >= 3_000_000_000)
        {
            models.Add("onnx-community/Llama-3.2-3B-Instruct-GENAI-ONNX");
        }

        // Add Phi-4 for larger systems
        if (maxParams >= 7_000_000_000)
        {
            models.Add("microsoft/phi-4-onnx");
        }

        return models;
    }
}

/// <summary>
/// Hardware recommendation for LLM inference.
/// </summary>
public sealed record HardwareRecommendation
{
    /// <summary>
    /// Recommended execution provider.
    /// </summary>
    public required ExecutionProvider Provider { get; init; }

    /// <summary>
    /// Detected GPU information.
    /// </summary>
    public required GpuInfo GpuInfo { get; init; }

    /// <summary>
    /// Total system memory in bytes.
    /// </summary>
    public required long SystemMemoryBytes { get; init; }

    /// <summary>
    /// Maximum recommended model parameter count.
    /// </summary>
    public required long MaxModelParameters { get; init; }

    /// <summary>
    /// Recommended quantization level (FP16, Quant8, Quant4).
    /// </summary>
    public required string RecommendedQuantization { get; init; }

    /// <summary>
    /// Maximum recommended context length.
    /// </summary>
    public required int MaxContextLength { get; init; }

    /// <summary>
    /// List of recommended models for this hardware.
    /// </summary>
    public required IReadOnlyList<string> RecommendedModels { get; init; }

    /// <summary>
    /// Gets a human-readable summary of the recommendation.
    /// </summary>
    public string GetSummary()
    {
        var gpuDesc = GpuInfo.Vendor == GpuVendor.Unknown && GpuInfo.DeviceName == "CPU Only"
            ? "CPU only"
            : $"{GpuInfo.DeviceName} ({GpuInfo.TotalMemoryBytes / (1024.0 * 1024 * 1024):F1}GB)";

        var sysMemGB = SystemMemoryBytes / (1024.0 * 1024 * 1024);

        return $"""
            Hardware: {gpuDesc}
            System Memory: {sysMemGB:F1}GB
            Provider: {Provider}
            Quantization: {RecommendedQuantization}
            Max Context: {MaxContextLength}
            Recommended Models: {string.Join(", ", RecommendedModels.Take(2))}
            """;
    }
}
