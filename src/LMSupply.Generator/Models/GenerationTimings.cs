namespace LMSupply.Generator.Models;

/// <summary>
/// Timings for one generation, as the backend measured them (llama-server's <c>timings</c>). Every value is the
/// server's own. The rates are not derived from the token counts, for two reasons: prompt tokens reused from the
/// server's prompt cache are counted in the usage but not evaluated, and the server measures the generation rate from
/// the first generated token on. Each member is null when the backend omits it.
/// </summary>
public sealed record GenerationTimings
{
    /// <summary>Prompt tokens reused from the backend's prompt cache instead of being evaluated.</summary>
    public int? CachedPromptTokens { get; init; }

    /// <summary>Prompt tokens evaluated for this request (the prompt minus <see cref="CachedPromptTokens"/>).</summary>
    public int? PromptTokensEvaluated { get; init; }

    /// <summary>Time spent evaluating the prompt.</summary>
    public TimeSpan? PromptDuration { get; init; }

    /// <summary>Prompt evaluation rate, in tokens per second.</summary>
    public double? PromptTokensPerSecond { get; init; }

    /// <summary>Time spent generating the completion, reasoning included.</summary>
    public TimeSpan? CompletionDuration { get; init; }

    /// <summary>
    /// Generation rate, in tokens per second, reasoning included. This is the decode speed. A rate computed from the
    /// visible text over wall-clock time since the first visible token overstates it for a reasoning model.
    /// </summary>
    public double? CompletionTokensPerSecond { get; init; }
}
