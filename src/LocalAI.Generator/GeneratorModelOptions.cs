namespace LocalAI.Generator;

/// <summary>
/// Configuration options for loading a generator model.
/// </summary>
public sealed class GeneratorModelOptions
{
    /// <summary>
    /// Gets or sets the directory for caching downloaded models.
    /// Defaults to ~/.cache/huggingface/hub/ following HuggingFace Hub standards.
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the execution provider for inference.
    /// Defaults to Auto (automatically selects the best available provider).
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

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
}
