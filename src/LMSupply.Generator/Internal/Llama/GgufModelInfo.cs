using LMSupply.Hardware;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Metadata for a registered GGUF model.
/// </summary>
public sealed record GgufModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// HuggingFace repository ID (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF").
    /// </summary>
    public required string RepoId { get; init; }

    /// <summary>
    /// Short alias name for this model (set by registry).
    /// </summary>
    public string AliasName { get; internal init; } = string.Empty;

    /// <summary>
    /// Human-friendly display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description => DisplayName;

    // IModelInfoBase explicit implementation
    string IModelInfoBase.Id => RepoId;

    /// <summary>
    /// Default GGUF file name to download if not specified.
    /// </summary>
    public required string DefaultFile { get; init; }

    /// <summary>
    /// Chat template format identifier (e.g., "llama3", "chatml", "gemma").
    /// </summary>
    public required string ChatFormat { get; init; }

    /// <summary>
    /// Maximum supported context length.
    /// </summary>
    public required int ContextLength { get; init; }

    /// <summary>
    /// Approximate parameter count (for memory estimation).
    /// </summary>
    public long ParameterCount { get; init; }

    /// <summary>
    /// License tier classification.
    /// </summary>
    public required LicenseTier License { get; init; }

    /// <summary>
    /// License name (e.g., "MIT", "Llama 3.2 Community License").
    /// </summary>
    public string? LicenseName { get; init; }

    /// <summary>
    /// Description of usage restrictions, if any.
    /// </summary>
    public string? LicenseRestrictions { get; init; }

    /// <summary>
    /// Optional subfolder within the repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Size in bytes of the registry's file (the quantization <see cref="QuantizationType"/> names), for VRAM budget
    /// calculations. Not always the download: when that file does not fit the machine's memory budget, the downloader takes
    /// a smaller quantization of the same model.
    /// </summary>
    public long? EstimatedSizeBytes { get; init; }

    /// <summary>
    /// Quantization type string (e.g., "Q4_K_M", "Q8_0").
    /// </summary>
    public string? QuantizationType { get; init; }

    /// <summary>
    /// Number of split shards for this model.
    /// Null or 1 = single file. Greater than 1 = split GGUF.
    /// When set, DefaultFile should point to the first shard (-00001-of-NNNNN).
    /// llama-server will auto-load remaining shards from the same directory.
    /// </summary>
    public int? ShardCount { get; init; }

    /// <summary>
    /// Number of transformer layers. Used for KV cache size estimation.
    /// Populate from the model's config.json / HF model card.
    /// </summary>
    public int NumLayers { get; init; }

    /// <summary>
    /// Hidden dimension size (d_model). Used for KV cache size estimation.
    /// Populate from the model's config.json / HF model card.
    /// </summary>
    public int HiddenSize { get; init; }

    /// <summary>
    /// Known issues for this model. Tags from <see cref="GgufModelKnownIssues"/>.
    /// Empty by default; set for models with known compatibility issues.
    /// </summary>
    public IReadOnlyList<string> KnownIssues { get; init; } = [];

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => EstimatedSizeBytes;
    long IModelMemoryInfo.ParameterCount => ParameterCount;
    string? IModelMemoryInfo.QuantizationType => QuantizationType;
}

/// <summary>
/// Well-known issue tags for GGUF models.
/// </summary>
public static class GgufModelKnownIssues
{
    /// <summary>Tool-use may produce empty or malformed results at Q4_K_M quantization.</summary>
    public const string ToolUseUnreliableQ4 = "tool-use-unreliable-q4";

    /// <summary>Instruction-following may be unreliable at Q4_K_M quantization.</summary>
    public const string InstructionFollowingUnreliableQ4 = "instruction-following-unreliable-q4";

    /// <summary>Model generates &lt;think&gt;...&lt;/think&gt; blocks by default.
    /// Set FilterReasoningTokens = true if think blocks should not appear in output.</summary>
    public const string ThinkingEnabledByDefault = "thinking-enabled-by-default";
}

/// <summary>
/// GGUF quantization types ordered by quality (higher = better quality, larger size).
/// </summary>
public enum GgufQuantization
{
    /// <summary>2-bit quantization (smallest, lowest quality)</summary>
    Q2_K = 0,

    /// <summary>3-bit quantization</summary>
    Q3_K_S = 10,
    Q3_K_M = 11,
    Q3_K_L = 12,

    /// <summary>4-bit quantization (recommended balance)</summary>
    Q4_K_S = 20,
    Q4_K_M = 21,  // Default recommended
    Q4_0 = 22,
    Q4_1 = 23,

    /// <summary>5-bit quantization</summary>
    Q5_K_S = 30,
    Q5_K_M = 31,
    Q5_0 = 32,
    Q5_1 = 33,

    /// <summary>6-bit quantization</summary>
    Q6_K = 40,

    /// <summary>8-bit quantization (highest quality)</summary>
    Q8_0 = 50,

    /// <summary>16-bit float (full precision)</summary>
    F16 = 100
}
