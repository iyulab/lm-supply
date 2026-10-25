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
    string? FinishReason = null)
{
    /// <summary>
    /// How long the backend spent on this generation and how fast it went, as the backend measured it. Null when the
    /// backend does not report timings (the ONNX path does not; llama-server does), and when
    /// <see cref="GenerationOptions.MaxTokens"/> cut the stream client-side.
    /// </summary>
    public GenerationTimings? Timings { get; init; }

    /// <summary>
    /// The log probability of each generated token, in order — reasoning tokens included — when
    /// <see cref="GenerationOptions.TopLogprobs"/> asked for them; null otherwise.
    /// </summary>
    public IReadOnlyList<TokenLogprob>? Logprobs { get; init; }
}

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
    /// True when the counts are an estimate from the text (about four characters per token) because the backend did
    /// not report its own. An estimate cannot see a reasoning model's hidden reasoning, which also consumes the output
    /// budget.
    /// </summary>
    public bool IsEstimated { get; init; }

    /// <summary>
    /// Creates an empty token usage instance.
    /// </summary>
    public static TokenUsage Empty => new(0, 0);

    /// <summary>
    /// Estimates token count from text by the kinds of its characters (Latin letters, Hangul, kana, Han, digits,
    /// punctuation, whitespace runs, other scripts), at ratios fitted to the o200k_base tokenizer.
    /// </summary>
    /// <param name="text">The text to estimate tokens for.</param>
    /// <returns>Estimated token count; 0 for an empty text.</returns>
    public static int EstimateTokens(string text) =>
        TokenEstimation.CharTokenEstimator.Count(text, TokenEstimation.CharTokenRatios.O200k);
}
