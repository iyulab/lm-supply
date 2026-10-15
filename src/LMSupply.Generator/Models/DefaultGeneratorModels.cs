using LMSupply.Core.Download;
namespace LMSupply.Generator;

/// <summary>
/// Provides definitions for built-in supported generator models.
/// Updated: 2025-01 based on HuggingFace ONNX availability and benchmarks.
/// </summary>
public static class DefaultGeneratorModels
{
    /// <summary>
    /// Gets the default model (MIT license, best balance).
    /// </summary>
    public static GeneratorModelEntry Default => Phi4Mini;

    /// <summary>
    /// Gets the fast model. Currently same as Default since Phi4Mini is the smallest FC-capable ONNX model.
    /// </summary>
    public static GeneratorModelEntry Fast => Phi4Mini;

    // ===== Tier 1: MIT License - No restrictions =====

    /// <summary>
    /// Microsoft Phi-4 Mini - Default balanced model (MIT license).
    /// 3.8B parameters, 16K context, excellent reasoning for its size.
    /// Released: 2025-01, successor to Phi-3.5 Mini.
    /// </summary>
    public static GeneratorModelEntry Phi4Mini { get; } = new()
    {
        ModelId = "microsoft/Phi-4-mini-instruct-onnx",
        AliasName = "phi-4-mini",
        DisplayName = "Phi-4 Mini",
        Description = "Default: Phi-4 Mini, 3.8B params, MIT, 16K context",
        ParameterCount = 3_800_000_000,
        License = LicenseTier.MIT,
        LicenseName = "MIT",
        ChatFormat = "phi3",
        DefaultQuantization = Quantization.Quant4,
        RecommendedContextLength = 16384,
        NumLayers = 32,
        HiddenSize = 3072,
        Subfolder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    };

    /// <summary>
    /// Microsoft Phi-3.5 Mini - Legacy MIT model.
    /// 3.8B parameters, 128K context support, predecessor to Phi-4 Mini.
    /// </summary>
    public static GeneratorModelEntry Phi35Mini { get; } = new()
    {
        ModelId = "microsoft/Phi-3.5-mini-instruct-onnx",
        AliasName = "phi-3.5-mini",
        DisplayName = "Phi-3.5 Mini",
        Description = "Phi-3.5 Mini, 3.8B params, MIT, 128K context support",
        ParameterCount = 3_800_000_000,
        License = LicenseTier.MIT,
        LicenseName = "MIT",
        ChatFormat = "phi3",
        DefaultQuantization = Quantization.Quant4,
        RecommendedContextLength = 4096,
        NumLayers = 32,
        HiddenSize = 3072,
        Subfolder = "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
    };

    /// <summary>
    /// Microsoft Phi-4 - Quality model (MIT license).
    /// 14B parameters, 16K context, highest quality reasoning.
    /// Released: 2024-12, state-of-the-art for its size.
    /// </summary>
    public static GeneratorModelEntry Phi4 { get; } = new()
    {
        ModelId = "microsoft/phi-4-onnx",
        AliasName = "quality",
        DisplayName = "Phi-4",
        Description = "Quality: Phi-4, 14B params, MIT, highest quality reasoning",
        ParameterCount = 14_000_000_000,
        License = LicenseTier.MIT,
        LicenseName = "MIT",
        ChatFormat = "phi3",
        DefaultQuantization = Quantization.Quant4,
        RecommendedContextLength = 8192,
        NumLayers = 40,
        HiddenSize = 5120,
        Subfolder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    };

    /// <summary>
    /// Gets all built-in ONNX models. Only FC-capable Phi-4 series models are included.
    /// </summary>
    public static IReadOnlyList<GeneratorModelEntry> All { get; } =
    [
        Phi4Mini,   // default
        Phi35Mini,  // phi-3.5-mini (legacy)
        Phi4,       // quality
    ];
}
