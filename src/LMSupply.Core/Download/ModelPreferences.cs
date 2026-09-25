using LMSupply.Hardware;

namespace LMSupply.Core.Download;

/// <summary>
/// User preferences for model selection during discovery.
/// </summary>
public sealed class ModelPreferences
{
    /// <summary>
    /// Default preferences: balanced between quality and memory.
    /// </summary>
    public static ModelPreferences Default { get; } = new();

    /// <summary>
    /// Preferences optimized for low memory usage.
    /// </summary>
    public static ModelPreferences LowMemory { get; } = new()
    {
        PreferLowMemory = true,
        QuantizationPriority = [Quantization.Quant4, Quantization.Quant8, Quantization.Fp16, Quantization.Default]
    };

    /// <summary>
    /// Preferences optimized for quality/accuracy.
    /// </summary>
    public static ModelPreferences HighQuality { get; } = new()
    {
        PreferLowMemory = false,
        QuantizationPriority = [Quantization.Default, Quantization.Fp16, Quantization.Quant8, Quantization.Quant4]
    };

    /// <summary>
    /// Creates preferences adapted to the given performance tier.
    /// Ultra  → Default (FP32) first — maximum quality, ample VRAM
    /// High   → FP16 first          — high quality, 8-16GB VRAM
    /// Medium → Quant8 first        — balanced, 4-8GB VRAM
    /// Low    → Quant4 first        — minimal footprint, under 4GB VRAM
    /// </summary>
    public static ModelPreferences ForTier(PerformanceTier tier) => tier switch
    {
        PerformanceTier.Ultra => new ModelPreferences
        {
            PreferLowMemory = false,
            QuantizationPriority = [Quantization.Default, Quantization.Fp16,
                                    Quantization.Quant8, Quantization.Quant4]
        },
        PerformanceTier.High => new ModelPreferences
        {
            PreferLowMemory = false,
            QuantizationPriority = [Quantization.Fp16, Quantization.Default,
                                    Quantization.Quant8, Quantization.Quant4]
        },
        PerformanceTier.Medium => new ModelPreferences
        {
            PreferLowMemory = false,
            QuantizationPriority = [Quantization.Quant8, Quantization.Fp16,
                                    Quantization.Quant4, Quantization.Default]
        },
        PerformanceTier.Low => new ModelPreferences
        {
            PreferLowMemory = true,
            QuantizationPriority = [Quantization.Quant4, Quantization.Quant8,
                                    Quantization.Fp16, Quantization.Default]
        },
        _ => throw new ArgumentOutOfRangeException(nameof(tier), tier, null)
    };

    /// <summary>
    /// Creates preferences optimized for the current machine's hardware.
    /// </summary>
    public static ModelPreferences ForCurrentHardware() =>
        ForTier(HardwareProfile.Current.Tier);

    /// <summary>
    /// Creates preferences for a load with <paramref name="provider"/>, from <see cref="HardwareProfile.For"/>: an
    /// explicit <see cref="ExecutionProvider.Cpu"/> ranks quantizations by the CPU tier (system memory) and does not
    /// probe the GPU; every other provider uses the current hardware, like <see cref="ForCurrentHardware"/>.
    /// </summary>
    public static ModelPreferences ForProvider(ExecutionProvider provider) =>
        ForTier(HardwareProfile.For(provider).Tier);

    /// <summary>
    /// Creates preferences from a quantization hint string (e.g., "fp16", "int8", "q4", "bnb4").
    /// The hinted quantization level is prioritized first, with other levels as fallbacks.
    /// </summary>
    public static ModelPreferences ForQuantizationHint(string? hint)
    {
        if (string.IsNullOrWhiteSpace(hint))
            return ForCurrentHardware();

        var primary = ParseQuantizationHint(hint);
        if (primary == Quantization.Default)
            return new ModelPreferences
            {
                QuantizationPriority = [Quantization.Default, Quantization.Fp16, Quantization.Quant8, Quantization.Quant4]
            };

        // Put the hinted level first, then fill remaining in quality-descending order
        var all = new[] { Quantization.Default, Quantization.Fp16, Quantization.Quant8, Quantization.Quant4 };
        var priority = new List<Quantization> { primary };
        priority.AddRange(all.Where(q => q != primary));

        return new ModelPreferences
        {
            PreferLowMemory = primary is Quantization.Quant4,
            QuantizationPriority = priority
        };
    }

    /// <summary>
    /// Parses a user-facing quantization hint to a Quantization enum value.
    /// </summary>
    public static Quantization ParseQuantizationHint(string hint)
    {
        var normalized = hint.Trim().ToLowerInvariant().TrimStart('_');
        return normalized switch
        {
            "fp16" or "half" or "float16" => Quantization.Fp16,
            "int8" or "uint8" or "q8" or "quantized" => Quantization.Quant8,
            "int4" or "q4" or "bnb4" or "q4f16" or "4bit" => Quantization.Quant4,
            "fp32" or "default" or "full" or "float32" => Quantization.Default,
            _ => Quantization.Default
        };
    }

    /// <summary>
    /// Whether to prefer smaller/quantized models for lower memory usage.
    /// </summary>
    public bool PreferLowMemory { get; init; }

    /// <summary>
    /// Preferred execution provider (CPU, CUDA, etc.).
    /// </summary>
    public ExecutionProvider PreferredProvider { get; init; } = ExecutionProvider.Cpu;

    /// <summary>
    /// Priority order for quantization selection.
    /// First matching variant will be selected.
    /// </summary>
    public IReadOnlyList<Quantization> QuantizationPriority { get; init; } =
        [Quantization.Default, Quantization.Fp16, Quantization.Quant8, Quantization.Quant4];

    /// <summary>
    /// Specific subfolder to use. If set, overrides auto-detection.
    /// </summary>
    public string? PreferredSubfolder { get; init; }

    /// <summary>
    /// Specific ONNX file names to look for (e.g., "model.onnx", "encoder_model.onnx").
    /// If empty, all ONNX files in the selected subfolder are included.
    /// </summary>
    public IReadOnlyList<string> PreferredOnnxFiles { get; init; } = [];

    /// <summary>
    /// Priority order for decoder variant selection in encoder-decoder models.
    /// First matching variant will be selected.
    /// </summary>
    public IReadOnlyList<DecoderVariant> DecoderVariantPriority { get; init; } =
        [DecoderVariant.Merged, DecoderVariant.Standard, DecoderVariant.WithPast];

    /// <summary>
    /// Explicit encoder file to use (overrides auto-detection).
    /// For edge cases with non-standard naming.
    /// </summary>
    public string? ExplicitEncoderFile { get; init; }

    /// <summary>
    /// Explicit decoder file to use (overrides auto-detection).
    /// For edge cases with non-standard naming.
    /// </summary>
    public string? ExplicitDecoderFile { get; init; }

    /// <summary>
    /// Whether encoder and decoder should have matching quantization levels.
    /// </summary>
    public bool RequireMatchedQuantization { get; init; } = true;
}

/// <summary>
/// Model quantization levels.
/// </summary>
public enum Quantization
{
    /// <summary>Default/full precision (FP32).</summary>
    Default,

    /// <summary>Half precision (FP16).</summary>
    Fp16,

    /// <summary>8-bit integer quantization.</summary>
    Quant8,

    /// <summary>4-bit integer quantization.</summary>
    Quant4
}
