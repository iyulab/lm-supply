namespace LMSupply;

/// <summary>
/// Base class for all LMSupply model options.
/// Provides common configuration properties shared across all packages.
/// </summary>
public abstract class LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the custom cache directory for model files.
    /// <para>Default: null (uses HuggingFace standard cache location: ~/.cache/huggingface/hub)</para>
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the execution provider for inference.
    /// <para>Default: <see cref="ExecutionProvider.Auto"/> (automatically selects the best available provider)</para>
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;
}
