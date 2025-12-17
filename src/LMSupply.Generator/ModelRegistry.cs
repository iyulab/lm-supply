namespace LMSupply.Generator;

/// <summary>
/// Registry of supported ONNX models with license and configuration information.
/// </summary>
public static class ModelRegistry
{
    private static readonly Dictionary<string, ModelInfo> _models = new(StringComparer.OrdinalIgnoreCase)
    {
        // Tier 1: MIT License - No restrictions
        ["microsoft/Phi-4-mini-instruct-onnx"] = new ModelInfo
        {
            ModelId = "microsoft/Phi-4-mini-instruct-onnx",
            DisplayName = "Phi-4 Mini",
            ParameterCount = 3_800_000_000,
            License = LicenseTier.MIT,
            LicenseName = "MIT",
            ChatFormat = "phi3",
            DefaultQuantization = Quantization.INT4,
            RecommendedContextLength = 16384,
            NumLayers = 32,
            HiddenSize = 3072,
            Subfolder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
        },
        ["microsoft/Phi-3.5-mini-instruct-onnx"] = new ModelInfo
        {
            ModelId = "microsoft/Phi-3.5-mini-instruct-onnx",
            DisplayName = "Phi-3.5 Mini",
            ParameterCount = 3_800_000_000,
            License = LicenseTier.MIT,
            LicenseName = "MIT",
            ChatFormat = "phi3",
            DefaultQuantization = Quantization.INT4,
            RecommendedContextLength = 4096,
            NumLayers = 32,
            HiddenSize = 3072,
            Subfolder = "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
        },
        ["microsoft/phi-4-onnx"] = new ModelInfo
        {
            ModelId = "microsoft/phi-4-onnx",
            DisplayName = "Phi-4",
            ParameterCount = 14_000_000_000,
            License = LicenseTier.MIT,
            LicenseName = "MIT",
            ChatFormat = "phi3",
            DefaultQuantization = Quantization.INT4,
            RecommendedContextLength = 8192,
            NumLayers = 40,
            HiddenSize = 5120,
            Subfolder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
        },

        // Tier 2: Conditional - Usage restrictions apply
        ["onnx-community/Llama-3.2-1B-Instruct-ONNX"] = new ModelInfo
        {
            ModelId = "onnx-community/Llama-3.2-1B-Instruct-ONNX",
            DisplayName = "Llama 3.2 1B",
            ParameterCount = 1_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Llama 3.2 Community License",
            LicenseRestrictions = "700M MAU limit for commercial use",
            ChatFormat = "llama3",
            DefaultQuantization = Quantization.INT4,
            RecommendedContextLength = 4096,
            NumLayers = 16,
            HiddenSize = 2048,
            Subfolder = "onnx"
        },
        ["onnx-community/Llama-3.2-3B-Instruct-ONNX"] = new ModelInfo
        {
            ModelId = "onnx-community/Llama-3.2-3B-Instruct-ONNX",
            DisplayName = "Llama 3.2 3B",
            ParameterCount = 3_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Llama 3.2 Community License",
            LicenseRestrictions = "700M MAU limit for commercial use",
            ChatFormat = "llama3",
            DefaultQuantization = Quantization.INT4,
            RecommendedContextLength = 4096,
            NumLayers = 28,
            HiddenSize = 3072,
            Subfolder = "onnx"
        },
        ["google/gemma-2-2b-it-onnx"] = new ModelInfo
        {
            ModelId = "google/gemma-2-2b-it-onnx",
            DisplayName = "Gemma 2 2B",
            ParameterCount = 2_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Gemma Terms of Use",
            LicenseRestrictions = "Prohibited Use Policy applies",
            ChatFormat = "gemma",
            DefaultQuantization = Quantization.INT4,
            RecommendedContextLength = 4096,
            NumLayers = 26,
            HiddenSize = 2304,
            Subfolder = "onnx"
        }
    };

    /// <summary>
    /// Gets all registered models.
    /// </summary>
    public static IReadOnlyList<ModelInfo> GetAllModels() =>
        _models.Values.ToList();

    /// <summary>
    /// Gets models filtered by license tier.
    /// </summary>
    public static IReadOnlyList<ModelInfo> GetModelsByLicense(LicenseTier tier) =>
        _models.Values.Where(m => m.License == tier).ToList();

    /// <summary>
    /// Gets MIT-licensed models (no usage restrictions).
    /// </summary>
    public static IReadOnlyList<ModelInfo> GetUnrestrictedModels() =>
        GetModelsByLicense(LicenseTier.MIT);

    /// <summary>
    /// Gets model information by ID.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "microsoft/Phi-3.5-mini-instruct-onnx").</param>
    /// <returns>Model information if found, null otherwise.</returns>
    public static ModelInfo? GetModel(string modelId) =>
        _models.GetValueOrDefault(modelId);

    /// <summary>
    /// Checks if a model is registered.
    /// </summary>
    public static bool IsRegistered(string modelId) =>
        _models.ContainsKey(modelId);

    /// <summary>
    /// Gets models that fit within available memory.
    /// </summary>
    /// <param name="availableMemoryBytes">Available memory in bytes.</param>
    /// <param name="contextLength">Desired context length.</param>
    /// <returns>List of models that can fit in memory.</returns>
    public static IReadOnlyList<ModelInfo> GetModelsForMemory(long availableMemoryBytes, int contextLength = 4096)
    {
        return _models.Values
            .Where(m => CanFitInMemory(m, availableMemoryBytes, contextLength))
            .OrderByDescending(m => m.ParameterCount)
            .ToList();
    }

    /// <summary>
    /// Gets the default recommended model based on hardware.
    /// </summary>
    public static ModelInfo GetDefaultModel()
    {
        var recommendation = HardwareDetector.GetRecommendation();
        var availableMemory = recommendation.GpuInfo.TotalMemoryBytes
            ?? recommendation.SystemMemoryBytes;

        // Prefer MIT-licensed models first
        var candidates = GetModelsForMemory(availableMemory)
            .OrderBy(m => m.License) // MIT first (lower enum value)
            .ThenByDescending(m => m.ParameterCount);

        return candidates.FirstOrDefault() ?? _models["microsoft/Phi-3.5-mini-instruct-onnx"];
    }

    private static bool CanFitInMemory(ModelInfo model, long availableMemoryBytes, int contextLength)
    {
        var config = new ModelMemoryConfig
        {
            ParameterCount = model.ParameterCount,
            NumLayers = model.NumLayers,
            HiddenSize = model.HiddenSize,
            ContextLength = contextLength,
            Quantization = model.DefaultQuantization
        };

        return MemoryEstimator.CanFitInMemory(config, availableMemoryBytes);
    }
}

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
public sealed record ModelInfo
{
    /// <summary>
    /// Unique model identifier (HuggingFace format: org/model-name).
    /// </summary>
    public required string ModelId { get; init; }

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
}
