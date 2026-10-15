namespace LMSupply.Generator;

/// <summary>
/// Well-known model identifiers for LMSupply components.
/// Updated: 2025-12 based on MTEB leaderboard and community benchmarks.
/// </summary>
public static class WellKnownModels
{
    /// <summary>
    /// Text generation models (ONNX Runtime GenAI).
    /// </summary>
    public static class Generator
    {
        /// <summary>
        /// Default — routes through <see cref="LocalGenerator.LoadAsync(string, GeneratorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>
        /// hardware-aware auto selection. Resolves to a GGUF model sized to the host on every
        /// platform (CUDA, Metal, Vulkan or CPU).
        /// </summary>
        public const string Default = "default";

        /// <summary>
        /// Fast — Phi-4 Mini (smallest FC-capable ONNX model). Pins the ONNX path explicitly;
        /// use <see cref="Default"/> for hardware-aware selection instead.
        /// </summary>
        public const string Fast = "phi-4-mini";

        /// <summary>
        /// Small — alias for <see cref="Fast"/>.
        /// </summary>
        public const string Small = Fast;

        /// <summary>
        /// Quality model - Microsoft Phi-4 (MIT license).
        /// 14B parameters, 16K context, highest quality reasoning.
        /// Released: 2024-12, state-of-the-art for its size.
        /// </summary>
        public const string Quality = "microsoft/phi-4-onnx";

        /// <summary>
        /// Medium model - Microsoft Phi-3.5 Mini.
        /// 3.8B parameters, 128K context, excellent for long documents.
        /// Note: MIT license, predecessor to Phi-4 Mini.
        /// </summary>
        public const string Medium = "microsoft/Phi-3.5-mini-instruct-onnx";

        /// <summary>
        /// Large/quality model - Microsoft Phi-4.
        /// 14B parameters, highest quality reasoning among ONNX models.
        /// </summary>
        public const string Large = Quality;

        /// <summary>
        /// Legacy default - Phi-3.5 Mini for backward compatibility.
        /// </summary>
        public const string Phi35Mini = "microsoft/Phi-3.5-mini-instruct-onnx";
    }

    /// <summary>
    /// Gets license information for a model.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>License tier classification.</returns>
    public static LicenseTier GetLicenseTier(string modelId)
    {
        GeneratorModelRegistry.Default.TryResolve(modelId, out var info);
        return info?.License ?? LicenseTier.Conditional;
    }

    /// <summary>
    /// Checks if a model has usage restrictions.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if the model has restrictions (non-MIT license).</returns>
    public static bool HasRestrictions(string modelId)
    {
        return GetLicenseTier(modelId) != LicenseTier.MIT;
    }

    /// <summary>
    /// Gets MIT-licensed models only (no usage restrictions).
    /// Uses concrete aliases so registry license lookup succeeds —
    /// the platform-aware <see cref="Generator.Default"/> alias is excluded
    /// because it resolves through the auto path and has no single license tier.
    /// </summary>
    public static IReadOnlyList<string> GetUnrestrictedModels() =>
    [
        Generator.Fast,      // phi-4-mini alias (MIT)
        Generator.Quality    // microsoft/phi-4-onnx (MIT)
    ];
}
