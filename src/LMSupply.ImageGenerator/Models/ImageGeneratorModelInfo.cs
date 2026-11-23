using LMSupply.Core;

namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Information about a loaded image generator model.
/// </summary>
public sealed class ImageGeneratorModelInfo : IModelInfoBase
{
    /// <summary>
    /// Model identifier (HuggingFace repo ID or local path).
    /// </summary>
    public required string ModelId { get; init; }

    // IModelInfoBase implementation
    string IModelInfoBase.Id => ModelId;

    /// <summary>
    /// Gets the alias name for this model.
    /// Defaults to ModelName or ModelId if not explicitly set.
    /// </summary>
    public string AliasName { get; init; } = null!;

    /// <summary>
    /// The rung this model holds on the domain's ladder (<see cref="ModelTier"/>), or <see langword="null"/> when it is
    /// off the ladder — a variant, a voice or a preset. The rung's name is an alias of the model.
    /// </summary>
    public ModelTier? Tier { get; init; }

    /// <summary>
    /// Friendly name of the model.
    /// </summary>
    public string? ModelName { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Model architecture type.
    /// </summary>
    public required string Architecture { get; init; }

    /// <summary>
    /// Execution provider being used.
    /// </summary>
    public required ExecutionProvider Provider { get; init; }

    /// <summary>
    /// Whether FP16 precision is being used.
    /// </summary>
    public required bool IsFp16 { get; init; }

    /// <summary>
    /// Default image width for this model.
    /// </summary>
    public required int DefaultWidth { get; init; }

    /// <summary>
    /// Default image height for this model.
    /// </summary>
    public required int DefaultHeight { get; init; }

    /// <summary>
    /// Recommended number of inference steps.
    /// </summary>
    public required int RecommendedSteps { get; init; }

    /// <summary>
    /// Recommended guidance scale.
    /// </summary>
    public required float RecommendedGuidanceScale { get; init; }

    /// <summary>
    /// Total size of model files in bytes.
    /// </summary>
    public long? ModelSizeBytes { get; init; }

    /// <summary>
    /// Path to the model directory on disk.
    /// </summary>
    public string? ModelPath { get; init; }
}
