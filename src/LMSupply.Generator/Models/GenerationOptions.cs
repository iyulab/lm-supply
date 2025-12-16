namespace LMSupply.Generator.Models;

/// <summary>
/// Configuration options for text generation (inference parameters).
/// </summary>
public sealed class GenerationOptions
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
    /// Gets or sets whether to enable random sampling.
    /// When false, uses greedy decoding (always picks highest probability token).
    /// Defaults to true.
    /// </summary>
    public bool DoSample { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of beams for beam search.
    /// Set to 1 to disable beam search.
    /// Note: Beam search disables KV cache sharing for better quality.
    /// Defaults to 1.
    /// </summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to share buffer between past and present KV cache.
    /// Improves memory efficiency but incompatible with beam search (num_beams > 1).
    /// Defaults to true.
    /// </summary>
    public bool PastPresentShareBuffer { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of new tokens to generate (excluding prompt).
    /// If null, limited only by MaxTokens (prompt + generated).
    /// </summary>
    public int? MaxNewTokens { get; set; }

    /// <summary>
    /// Creates a default instance of GenerationOptions.
    /// </summary>
    public static GenerationOptions Default => new();

    /// <summary>
    /// Creates options optimized for creative text generation.
    /// </summary>
    public static GenerationOptions Creative => new()
    {
        Temperature = 0.9f,
        TopP = 0.95f,
        TopK = 100,
        RepetitionPenalty = 1.2f
    };

    /// <summary>
    /// Creates options optimized for deterministic/precise outputs.
    /// </summary>
    public static GenerationOptions Precise => new()
    {
        Temperature = 0.1f,
        TopP = 0.5f,
        TopK = 10,
        RepetitionPenalty = 1.0f
    };
}
