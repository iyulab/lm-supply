namespace LocalAI.Generator.Models;

/// <summary>
/// Configuration options for text generation.
/// </summary>
public sealed class GeneratorOptions
{
    /// <summary>
    /// Gets or sets the maximum number of tokens to generate.
    /// Defaults to 512.
    /// </summary>
    public int MaxTokens { get; set; } = 512;

    /// <summary>
    /// Gets or sets the temperature for sampling.
    /// Higher values produce more random outputs. Range: 0.0 to 2.0.
    /// Defaults to 0.7.
    /// </summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>
    /// Gets or sets the top-p (nucleus) sampling parameter.
    /// Considers tokens with cumulative probability mass up to this value. Range: 0.0 to 1.0.
    /// Defaults to 0.9.
    /// </summary>
    public float TopP { get; set; } = 0.9f;

    /// <summary>
    /// Gets or sets the top-k sampling parameter.
    /// Considers only the top k tokens. Set to 0 to disable.
    /// Defaults to 50.
    /// </summary>
    public int TopK { get; set; } = 50;

    /// <summary>
    /// Gets or sets the repetition penalty.
    /// Values greater than 1.0 discourage repetition.
    /// Defaults to 1.1.
    /// </summary>
    public float RepetitionPenalty { get; set; } = 1.1f;

    /// <summary>
    /// Gets or sets the stop sequences that will terminate generation.
    /// </summary>
    public IReadOnlyList<string>? StopSequences { get; set; }

    /// <summary>
    /// Gets or sets whether to include the input prompt in the output.
    /// Defaults to false.
    /// </summary>
    public bool IncludePromptInOutput { get; set; }

    /// <summary>
    /// Creates a default instance of GeneratorOptions.
    /// </summary>
    public static GeneratorOptions Default => new();

    /// <summary>
    /// Creates options optimized for creative text generation.
    /// </summary>
    public static GeneratorOptions Creative => new()
    {
        Temperature = 0.9f,
        TopP = 0.95f,
        TopK = 100,
        RepetitionPenalty = 1.2f
    };

    /// <summary>
    /// Creates options optimized for deterministic/precise outputs.
    /// </summary>
    public static GeneratorOptions Precise => new()
    {
        Temperature = 0.1f,
        TopP = 0.5f,
        TopK = 10,
        RepetitionPenalty = 1.0f
    };
}
