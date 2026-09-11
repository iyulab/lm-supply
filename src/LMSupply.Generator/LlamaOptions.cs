using LMSupply.Hardware;
using LMSupply.Llama.Server;
using LMSupply.Runtime;

namespace LMSupply.Generator;

/// <summary>
/// Advanced configuration options for GGUF model loading and inference via llama-server.
/// These options provide fine-grained control over llama.cpp backend behavior.
/// </summary>
public sealed class LlamaOptions
{
    /// <summary>
    /// Gets or sets the number of layers to offload to GPU.
    /// -1 = All layers on GPU (default for GPU providers)
    /// 0 = CPU only
    /// N = Offload N layers to GPU
    /// </summary>
    /// <remarks>
    /// If <see cref="GpuOffloadRatio"/> is set, it takes precedence over this value.
    /// </remarks>
    public int? GpuLayerCount { get; set; }

    /// <summary>
    /// Gets or sets the GPU offload ratio (0.0 to 1.0).
    /// 0.0 = CPU only (equivalent to GpuLayerCount = 0)
    /// 1.0 = All layers on GPU (equivalent to GpuLayerCount = -1)
    /// 0.5 = 50% of layers on GPU
    /// </summary>
    /// <remarks>
    /// This provides a more intuitive way to control GPU usage.
    /// When set, this value takes precedence over <see cref="GpuLayerCount"/>.
    /// </remarks>
    public float? GpuOffloadRatio { get; set; }

    /// <summary>
    /// Gets or sets the batch size for prompt processing.
    /// Higher values improve throughput but use more memory.
    /// Defaults to 512.
    /// </summary>
    public uint? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the RoPE frequency base for context extension.
    /// Use with RoPE-scaling-aware models for extended context lengths.
    /// </summary>
    public float? RopeFrequencyBase { get; set; }

    /// <summary>
    /// Gets or sets the RoPE frequency scale factor.
    /// Use with RoPE-scaling-aware models for extended context lengths.
    /// </summary>
    public float? RopeFrequencyScale { get; set; }

    /// <summary>
    /// Gets or sets whether to use Flash Attention for improved performance.
    /// Requires compatible GPU (CUDA compute capability 7.0+).
    /// Defaults to false for compatibility.
    /// </summary>
    public bool? FlashAttention { get; set; }

    /// <summary>
    /// Gets or sets whether to use memory mapping for model loading.
    /// Enables faster model loading and reduced memory usage when the same model
    /// is loaded multiple times. Defaults to true.
    /// </summary>
    public bool? UseMemoryMap { get; set; }

    /// <summary>
    /// Gets or sets whether to lock model memory to prevent swapping.
    /// Improves inference latency but may require elevated privileges.
    /// Defaults to false.
    /// </summary>
    public bool? UseMemoryLock { get; set; }

    /// <summary>
    /// Gets or sets the main GPU index for multi-GPU systems.
    /// 0-based index of the GPU to use as the primary device.
    /// </summary>
    public int? MainGpu { get; set; }

    /// <summary>
    /// Gets or sets the number of threads for CPU computation.
    /// Defaults to llama-server's own detection (<c>--threads</c> not passed).
    /// </summary>
    /// <remarks>
    /// llama-server's default is the number of <em>physical</em> cores — it deliberately leaves out
    /// SMT siblings and efficiency cores, which is the better choice on real desktop hardware, so
    /// this library does not override it with <see cref="Environment.ProcessorCount"/>. The
    /// detection reads the CPU topology the OS exposes, and a virtual machine can under-report it:
    /// a 4-vCPU shared CI runner has been observed to start with <c>n_threads = 1</c>. When the
    /// host is a VM or container whose topology you know, set this explicitly (for example to
    /// <see cref="Environment.ProcessorCount"/>).
    /// </remarks>
    public int? Threads { get; set; }

    /// <summary>
    /// Gets or sets the absolute limit on llama-server startup, counted from process launch
    /// regardless of activity. <c>null</c> keeps <see cref="LlamaServerConfig.DefaultStartupTimeout"/>
    /// (10 minutes). Must be at least <see cref="StartupStallTimeout"/>.
    /// </summary>
    /// <remarks>
    /// This is the far-out cap; the working limit is <see cref="StartupStallTimeout"/>. A legitimate
    /// start can take minutes — a 4 GB model read through mmap from a cold page cache on a shared CI
    /// runner or a laptop HDD — and the process is busy the whole time, so a fixed budget cannot tell
    /// slow from stuck. The cap only bounds a process that keeps showing activity without ever
    /// answering <c>/health</c>.
    /// </remarks>
    public TimeSpan? StartupTimeout { get; set; }

    /// <summary>
    /// Gets or sets how long the starting llama-server may go without any observable activity
    /// (a new stderr line, working-set growth, or CPU time consumed) before it is declared stuck.
    /// <c>null</c> keeps <see cref="LlamaServerConfig.DefaultStartupStallTimeout"/> (120 seconds).
    /// </summary>
    public TimeSpan? StartupStallTimeout { get; set; }

    /// <summary>
    /// Gets or sets the quantization type for KV cache keys.
    /// Auto selects Q8_0 for CUDA/Metal/Hip (Vulkan b8500+), F16 for CPU/SYCL.
    /// </summary>
    public KvCacheQuantizationType TypeK { get; set; } = KvCacheQuantizationType.Auto;

    /// <summary>
    /// Gets or sets the quantization type for KV cache values.
    /// Auto selects Q8_0 for CUDA/Metal/Hip (Vulkan b8500+), F16 for CPU/SYCL.
    /// </summary>
    public KvCacheQuantizationType TypeV { get; set; } = KvCacheQuantizationType.Auto;

    /// <summary>
    /// Gets or sets the physical batch size for prompt processing.
    /// Smaller than BatchSize, controls memory usage during processing.
    /// Defaults to 512.
    /// </summary>
    public uint? UBatchSize { get; set; }

    #region Speculative Decoding

    /// <summary>
    /// Gets or sets the speculative decoding strategy.
    /// Auto enables Ngram mode when the server build supports it (≥ b8500).
    /// Ngram mode provides 1.5–3× speedup with no quality loss and requires no draft model.
    /// </summary>
    public SpeculativeDecodingMode SpeculativeDecoding { get; set; } = SpeculativeDecodingMode.Auto;

    /// <summary>
    /// Gets or sets the path to a draft model for DraftModel speculation mode.
    /// Only used when SpeculativeDecoding = DraftModel.
    /// </summary>
    public string? DraftModelPath { get; set; }

    #endregion

    #region YaRN RoPE Scaling

    /// <summary>
    /// Gets or sets the RoPE scaling mode for context extension.
    /// Default passes through to model metadata. YaRN requires YarnOriginalContext.
    /// </summary>
    public RopeScalingMode RopeScaling { get; set; } = RopeScalingMode.Default;

    /// <summary>
    /// Gets or sets the original context size the model was trained with (YaRN only).
    /// Example: 4096 for most 7B models, 8192 for extended variants.
    /// </summary>
    public uint? YarnOriginalContext { get; set; }

    /// <summary>
    /// Gets or sets the YaRN extrapolation mix factor (-1 = model default).
    /// </summary>
    public float? YarnExtensionFactor { get; set; }

    /// <summary>
    /// Gets or sets the YaRN attention magnitude scaling factor.
    /// </summary>
    public float? YarnAttentionFactor { get; set; }

    /// <summary>
    /// Gets or sets the YaRN low-frequency ramp parameter (beta-fast).
    /// </summary>
    public float? YarnBetaFast { get; set; }

    /// <summary>
    /// Gets or sets the YaRN high-frequency ramp parameter (beta-slow).
    /// </summary>
    public float? YarnBetaSlow { get; set; }

    #endregion

    #region Phase 3: Multimodal Support

    /// <summary>
    /// Gets or sets the path to a multimodal projector file (mmproj).
    /// Required for vision models like LLaVA, Qwen-VL, InternVL, etc.
    /// The projector bridges visual features with the language model.
    /// </summary>
    public string? MultimodalProjector { get; set; }

    #endregion

    #region Phase 3: LoRA Support

    /// <summary>
    /// Gets or sets the path to a LoRA adapter file.
    /// Allows loading fine-tuned adapters without modifying base model weights.
    /// </summary>
    public string? LoraPath { get; set; }

    /// <summary>
    /// Gets or sets the LoRA adapter scale factor.
    /// Controls the strength of LoRA adaptation. Default: 1.0.
    /// Lower values reduce the adapter's effect, higher values increase it.
    /// </summary>
    public float? LoraScale { get; set; }

    #endregion

    /// <summary>
    /// Gets or sets additional raw command-line arguments passed through to llama-server.
    /// Escape hatch for flags not modeled by the properties above (e.g. "--verbose" for
    /// per-request prompt/completion logging). Appended after the arguments this type
    /// already derives from its other properties.
    /// </summary>
    public IReadOnlyList<string>? AdditionalArgs { get; set; }

    /// <summary>
    /// Gets optimal LlamaOptions based on actual model size and available VRAM.
    /// Uses VramFitResult for precise GPU offload and batch settings.
    /// </summary>
    public static LlamaOptions GetOptimalForHardware(
        GpuInfo gpu,
        long modelSizeBytes,
        int totalLayers = 32)
    {
        var fitResult = VramFitResult.Evaluate(
            VramBudget.GetAvailableBytes(gpu),
            modelSizeBytes,
            totalLayers);

        var profile = HardwareProfile.Current;
        var flashAttention = fitResult.GpuLayerCount != 0 && ShouldEnableFlashAttention(profile);

        // The count alone: it is exact for these layers, and a ratio derived from it would take precedence
        // and be turned back into a count — rounding can lose a layer on the way (20/33 -> 19).
        return new LlamaOptions
        {
            GpuLayerCount = fitResult.GpuLayerCount,
            BatchSize = fitResult.RecommendedBatchSize,
            UBatchSize = Math.Min(fitResult.RecommendedBatchSize, 512),
            FlashAttention = flashAttention,
            UseMemoryMap = true,
            UseMemoryLock = false,
            TypeK = fitResult.RecommendedKvQuantType,
            TypeV = fitResult.RecommendedKvQuantType
        };
    }

    /// <summary>
    /// Gets optimal LlamaOptions based on current hardware profile.
    /// </summary>
    /// <returns>LlamaOptions configured for optimal performance on detected hardware.</returns>
    public static LlamaOptions GetOptimalForHardware()
    {
        var profile = HardwareProfile.Current;

        // FlashAttention support depends on GPU vendor and tier
        // - NVIDIA CUDA: Compute capability 7.0+ (Volta and newer)
        // - Apple Metal: Supported on Apple Silicon
        // - Vulkan/Hip: Not reliably supported, disable by default
        var flashAttention = ShouldEnableFlashAttention(profile);

        return profile.Tier switch
        {
            PerformanceTier.Ultra => new LlamaOptions
            {
                GpuLayerCount = -1,          // All layers on GPU
                BatchSize = 4096,            // Large batch for fast prompt processing (TTFT)
                UBatchSize = 1024,           // Physical batch size
                FlashAttention = flashAttention,
                UseMemoryMap = true,
                UseMemoryLock = false,       // Don't require elevated privileges
                TypeK = KvCacheQuantizationType.Q8_0,  // Quantized KV cache for memory savings
                TypeV = KvCacheQuantizationType.Q8_0
            },
            PerformanceTier.High => new LlamaOptions
            {
                GpuLayerCount = -1,
                BatchSize = 2048,            // Larger batch for faster first token
                UBatchSize = 512,
                FlashAttention = flashAttention,
                UseMemoryMap = true,
                UseMemoryLock = false,
                TypeK = KvCacheQuantizationType.Q8_0,
                TypeV = KvCacheQuantizationType.Q8_0
            },
            PerformanceTier.Medium => new LlamaOptions
            {
                GpuLayerCount = -1,
                BatchSize = 1024,            // Increased for better TTFT
                UBatchSize = 512,
                FlashAttention = false,      // Conservative for mid-range
                UseMemoryMap = true,
                UseMemoryLock = false,
                TypeK = KvCacheQuantizationType.Q4_0,  // More aggressive quantization for memory
                TypeV = KvCacheQuantizationType.Q4_0
            },
            _ => new LlamaOptions              // Low tier
            {
                GpuLayerCount = 0,            // CPU only for safety
                BatchSize = 512,             // Reasonable batch for prompt processing
                UBatchSize = 256,
                FlashAttention = false,
                UseMemoryMap = true,
                UseMemoryLock = false
                // No KV cache quantization for CPU (default F16)
            }
        };
    }

    /// <summary>
    /// Determines if FlashAttention should be enabled based on hardware.
    /// </summary>
    private static bool ShouldEnableFlashAttention(HardwareProfile profile)
    {
        // Only enable for high-performance tiers
        if (profile.Tier < PerformanceTier.High)
            return false;

        var gpuInfo = profile.GpuInfo;

        return gpuInfo.Vendor switch
        {
            // NVIDIA: Enable for compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
            GpuVendor.Nvidia when gpuInfo.CudaComputeCapabilityMajor >= 7 => true,

            // Apple Silicon: Metal supports FlashAttention
            GpuVendor.Apple => true,

            // AMD/Intel/Others: Vulkan and Hip backends don't reliably support FlashAttention
            // Wait for llama.cpp to improve Vulkan FlashAttention support
            _ => false
        };
    }

    /// <summary>
    /// Calculates the effective GPU layer count, considering both GpuLayerCount and GpuOffloadRatio.
    /// </summary>
    /// <param name="totalLayers">Total number of layers in the model.</param>
    /// <returns>The number of layers to offload to GPU.</returns>
    public int GetEffectiveGpuLayerCount(int totalLayers)
    {
        // GpuOffloadRatio takes precedence if set
        if (GpuOffloadRatio.HasValue)
        {
            var ratio = Math.Clamp(GpuOffloadRatio.Value, 0f, 1f);

            return ratio switch
            {
                0f => 0,                           // CPU only
                >= 1f => -1,                       // All layers on GPU
                _ => (int)(totalLayers * ratio)   // Partial offload
            };
        }

        // Fall back to GpuLayerCount
        return GpuLayerCount ?? -1; // Default to all layers on GPU
    }

    /// <summary>
    /// Creates options with the specified GPU offload ratio.
    /// </summary>
    /// <param name="ratio">Offload ratio: 0.0 (CPU) to 1.0 (full GPU).</param>
    /// <returns>A new LlamaOptions instance.</returns>
    public static LlamaOptions WithGpuRatio(float ratio)
    {
        return new LlamaOptions
        {
            GpuOffloadRatio = Math.Clamp(ratio, 0f, 1f)
        };
    }

    /// <summary>
    /// Creates CPU-only options.
    /// </summary>
    public static LlamaOptions CpuOnly => new() { GpuOffloadRatio = 0f };

    /// <summary>
    /// Creates full GPU offload options.
    /// </summary>
    public static LlamaOptions FullGpu => new() { GpuOffloadRatio = 1f };
}
