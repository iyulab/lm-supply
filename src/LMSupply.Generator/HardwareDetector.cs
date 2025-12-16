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
        var gpuInfo = GpuDetector.DetectPrimaryGpu();
        var systemMemory = GetSystemMemoryBytes();

        return CreateRecommendation(gpuInfo, systemMemory);
    }

    /// <summary>
    /// Gets the best available execution provider.
    /// </summary>
    public static ExecutionProvider GetBestProvider()
    {
        var gpuInfo = GpuDetector.DetectPrimaryGpu();

        // CUDA has best performance for NVIDIA GPUs
        if (gpuInfo.Vendor == GpuVendor.Nvidia && gpuInfo.TotalMemoryBytes >= 4L * 1024 * 1024 * 1024)
        {
            return ExecutionProvider.Cuda;
        }

        // DirectML for Windows with compatible GPU
        if (gpuInfo.DirectMLSupported)
        {
            return ExecutionProvider.DirectML;
        }

        // CoreML for Apple Silicon
        if (gpuInfo.CoreMLSupported)
        {
            return ExecutionProvider.CoreML;
        }

        return ExecutionProvider.Cpu;
    }

    /// <summary>
    /// Resolves Auto execution provider to the best available option.
    /// </summary>
    public static ExecutionProvider ResolveProvider(ExecutionProvider provider)
    {
        return provider == ExecutionProvider.Auto ? GetBestProvider() : provider;
    }

    private static HardwareRecommendation CreateRecommendation(GpuInfo gpuInfo, long systemMemoryBytes)
    {
        var gpuMemoryGB = (gpuInfo.TotalMemoryBytes ?? 0) / (1024.0 * 1024 * 1024);
        var systemMemoryGB = systemMemoryBytes / (1024.0 * 1024 * 1024);

        // Determine best execution provider
        var provider = GetBestProvider();

        // Determine recommended model size based on available memory
        var (maxModelParams, quantization, maxContext) = (gpuMemoryGB, systemMemoryGB, provider) switch
        {
            // GPU scenarios
            (>= 16, _, ExecutionProvider.Cuda) => (14_000_000_000L, "FP16", 16384),
            (>= 8, _, ExecutionProvider.Cuda) => (7_000_000_000L, "INT8", 8192),
            (>= 4, _, ExecutionProvider.Cuda or ExecutionProvider.DirectML) => (3_000_000_000L, "INT4", 4096),

            // CPU-only scenarios
            (_, >= 32, ExecutionProvider.Cpu) => (7_000_000_000L, "INT4", 8192),
            (_, >= 16, ExecutionProvider.Cpu) => (3_000_000_000L, "INT4", 4096),
            (_, >= 8, ExecutionProvider.Cpu) => (1_000_000_000L, "INT4", 2048),

            // Minimal requirements
            _ => (1_000_000_000L, "INT4", 2048)
        };

        // Recommend specific models
        var recommendedModels = GetRecommendedModels(maxModelParams, quantization);

        return new HardwareRecommendation
        {
            Provider = provider,
            GpuInfo = gpuInfo,
            SystemMemoryBytes = systemMemoryBytes,
            MaxModelParameters = maxModelParams,
            RecommendedQuantization = quantization,
            MaxContextLength = maxContext,
            RecommendedModels = recommendedModels
        };
    }

    private static IReadOnlyList<string> GetRecommendedModels(long maxParams, string quantization)
    {
        var models = new List<string>();

        // Always recommend Phi-3.5-mini (3.8B) for INT4
        if (maxParams >= 3_000_000_000)
        {
            models.Add("microsoft/Phi-3.5-mini-instruct-onnx");
        }

        // Add Llama-3.2-1B for smaller systems
        if (maxParams >= 1_000_000_000)
        {
            models.Add("onnx-community/Llama-3.2-1B-Instruct-ONNX");
        }

        // Add Llama-3.2-3B for medium systems
        if (maxParams >= 3_000_000_000)
        {
            models.Add("onnx-community/Llama-3.2-3B-Instruct-ONNX");
        }

        // Add Phi-4 for larger systems
        if (maxParams >= 7_000_000_000)
        {
            models.Add("microsoft/phi-4-onnx");
        }

        return models;
    }

    private static long GetSystemMemoryBytes()
    {
        try
        {
            return (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        }
        catch
        {
            // Fallback: assume 8GB
            return 8L * 1024 * 1024 * 1024;
        }
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
    /// Recommended quantization level (FP16, INT8, INT4).
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
