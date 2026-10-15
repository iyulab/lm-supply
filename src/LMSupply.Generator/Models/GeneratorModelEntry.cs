using LMSupply.Hardware;
using LMSupply.Core.Download;

namespace LMSupply.Generator;

/// <summary>
/// License tier classification.
/// </summary>
public enum LicenseTier
{
    /// <summary>MIT license - no restrictions, commercial use allowed.</summary>
    MIT = 0,

    /// <summary>Conditional license - restrictions apply (MAU limits, usage policies).</summary>
    Conditional = 1,

    /// <summary>Research only - not suitable for production use.</summary>
    ResearchOnly = 2
}

/// <summary>
/// Model metadata and configuration.
/// </summary>
public sealed record GeneratorModelEntry : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Unique model identifier (HuggingFace format: org/model-name).
    /// </summary>
    public required string ModelId { get; init; }

    // IModelInfoBase implementation
    string IModelInfoBase.Id => ModelId;
    int? IModelInfoBase.ContextLength => RecommendedContextLength;

    /// <summary>
    /// Gets the alias name for this model.
    /// Set by the registry for system aliases (e.g., "default", "fast").
    /// Defaults to ModelId if not explicitly set.
    /// </summary>
    public string AliasName { get; init; } = null!;

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Human-friendly display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Total parameter count.
    /// </summary>
    public required long ParameterCount { get; init; }

    /// <summary>
    /// License classification tier.
    /// </summary>
    public required LicenseTier License { get; init; }

    /// <summary>
    /// License name (e.g., "MIT", "Llama 3.2 Community License").
    /// </summary>
    public required string LicenseName { get; init; }

    /// <summary>
    /// Description of any usage restrictions. Null for unrestricted licenses.
    /// </summary>
    public string? LicenseRestrictions { get; init; }

    /// <summary>
    /// Chat format identifier for prompt formatting.
    /// </summary>
    public required string ChatFormat { get; init; }

    /// <summary>
    /// Default quantization level for this model.
    /// </summary>
    public required Quantization DefaultQuantization { get; init; }

    /// <summary>
    /// Recommended maximum context length.
    /// </summary>
    public required int RecommendedContextLength { get; init; }

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    public required int NumLayers { get; init; }

    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    public required int HiddenSize { get; init; }

    /// <summary>
    /// Subfolder within the repository containing the ONNX files.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Gets memory configuration for this model.
    /// </summary>
    public ModelMemoryConfig GetMemoryConfig(int? contextLength = null) => new()
    {
        ParameterCount = ParameterCount,
        NumLayers = NumLayers,
        HiddenSize = HiddenSize,
        ContextLength = contextLength ?? RecommendedContextLength,
        Quantization = DefaultQuantization
    };

    /// <summary>
    /// Checks if this model has usage restrictions.
    /// </summary>
    public bool HasRestrictions => License != LicenseTier.MIT;

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => null;
    long IModelMemoryInfo.ParameterCount => ParameterCount;
    string? IModelMemoryInfo.QuantizationType => null;
}
