namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Registry of well-known image generation models with pre-configured settings.
/// Updated: 2025-12 based on HuggingFace ONNX availability.
/// </summary>
public static class WellKnownImageModels
{
    /// <summary>
    /// Model alias to HuggingFace repository mapping.
    /// All models use Latent Consistency Model (LCM) for fast inference (2-8 steps).
    /// </summary>
    private static readonly Dictionary<string, ModelDefinition> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        // LCM-based models (2-4 steps, optimized for edge inference)
        // LCM-Dreamshaper-V7: 512x512, ~1GB, best balance of speed and quality
        ["default"] = new("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX", "LCM-Dreamshaper-V7", 4, 1.0f),
        ["fast"] = new("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX", "LCM-Dreamshaper-V7", 2, 1.0f),
        ["quality"] = new("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX", "LCM-Dreamshaper-V7", 8, 1.5f),

        // Direct model references
        ["lcm-dreamshaper-v7"] = new("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX", "LCM-Dreamshaper-V7", 4, 1.0f),

        // LCM-SSD-1B: Smaller, faster alternative (1B params)
        // Note: Requires ONNX conversion from original model
        ["lcm-ssd-1b"] = new("segmind/LCM-SSD-1B-onnx", "LCM-SSD-1B", 4, 1.0f),
    };

    /// <summary>
    /// Resolves a model alias or ID to a full model definition.
    /// </summary>
    /// <param name="modelIdOrAlias">Model alias (e.g., "default") or HuggingFace repo ID.</param>
    /// <returns>Resolved model definition.</returns>
    public static ModelDefinition Resolve(string modelIdOrAlias)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrAlias);

        // Check if it's a known alias
        if (Aliases.TryGetValue(modelIdOrAlias, out var definition))
        {
            return definition;
        }

        // Treat as a direct HuggingFace repo ID with default settings
        return new ModelDefinition(modelIdOrAlias, null, 4, 1.0f);
    }

    /// <summary>
    /// Gets all available model aliases.
    /// </summary>
    public static IReadOnlyCollection<string> GetAliases() => Aliases.Keys;

    /// <summary>
    /// Checks if the given string is a known model alias.
    /// </summary>
    public static bool IsAlias(string modelIdOrAlias) =>
        Aliases.ContainsKey(modelIdOrAlias);
}

/// <summary>
/// Definition of a known image generation model.
/// </summary>
/// <param name="RepoId">HuggingFace repository ID.</param>
/// <param name="FriendlyName">Human-readable model name.</param>
/// <param name="RecommendedSteps">Recommended number of inference steps.</param>
/// <param name="RecommendedGuidanceScale">Recommended guidance scale.</param>
public readonly record struct ModelDefinition(
    string RepoId,
    string? FriendlyName,
    int RecommendedSteps,
    float RecommendedGuidanceScale);
