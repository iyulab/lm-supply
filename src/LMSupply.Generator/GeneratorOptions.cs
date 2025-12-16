namespace LMSupply.Generator;

/// <summary>
/// Configuration options for loading a generator model.
/// </summary>
public sealed class GeneratorOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the chat format to use.
    /// If null, the format is auto-detected from the model name.
    /// </summary>
    public string? ChatFormat { get; set; }

    /// <summary>
    /// Gets or sets whether to log detailed information during model loading.
    /// Defaults to false.
    /// </summary>
    public bool Verbose { get; set; }

    /// <summary>
    /// Gets or sets the maximum context length to use.
    /// If null, uses the model's default context length.
    /// </summary>
    public int? MaxContextLength { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of concurrent generation requests.
    /// Used to prevent resource exhaustion during high load.
    /// Defaults to 1 (sequential processing) for stability.
    /// </summary>
    public int MaxConcurrentRequests { get; set; } = 1;
}
