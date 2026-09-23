namespace LMSupply.Generator.Models;

/// <summary>
/// Represents the result of a text generation operation, including usage statistics.
/// </summary>
/// <param name="Content">The generated text content.</param>
/// <param name="Usage">Token usage statistics for the generation.</param>
/// <param name="FinishReason">
/// Why the generation ended, OpenAI-style: <c>"length"</c> when it stopped at
/// <see cref="GenerationOptions.MaxTokens"/> or the model's context, <c>"stop"</c> at the end token or a stop
/// sequence; <c>null</c> when the backend does not report it.
/// </param>
public readonly record struct GenerationResult(
    string Content,
    TokenUsage Usage,
    string? FinishReason = null);

/// <summary>
/// Token usage statistics for a generation operation.
/// </summary>
/// <param name="PromptTokens">Number of tokens in the input prompt.</param>
/// <param name="CompletionTokens">Number of tokens generated.</param>
public readonly record struct TokenUsage(
    int PromptTokens,
    int CompletionTokens)
{
    /// <summary>
    /// Gets the total number of tokens (prompt + completion).
    /// </summary>
    public int TotalTokens => PromptTokens + CompletionTokens;

    /// <summary>
    /// Creates an empty token usage instance.
    /// </summary>
    public static TokenUsage Empty => new(0, 0);

    /// <summary>
    /// Estimates token count from text using a simple heuristic.
    /// Assumes approximately 4 characters per token on average.
    /// </summary>
    /// <param name="text">The text to estimate tokens for.</param>
    /// <returns>Estimated token count.</returns>
    public static int EstimateTokens(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        // Rough estimate: ~4 characters per token for English text
        // This is a common heuristic used by OpenAI and others
        return (int)Math.Ceiling(text.Length / 4.0);
    }
}
