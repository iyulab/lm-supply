namespace LMSupply.Generator.Models;

/// <summary>
/// How far the backend has read the prompt of a streamed chat completion, before the first token. A long prompt can
/// take minutes to read on a small machine; these are the stream's only items meanwhile, so a host can tell a prompt
/// still being read from a backend that has stopped, and show how far it has got.
/// </summary>
public sealed record PromptProgress
{
    /// <summary>Tokens in the prompt.</summary>
    public required int TotalTokens { get; init; }

    /// <summary>
    /// Prompt tokens reused from the backend's prompt cache. They count as processed from the start: only
    /// <see cref="TotalTokens"/> minus these are read, and only they take time.
    /// </summary>
    public required int CachedTokens { get; init; }

    /// <summary>Prompt tokens processed so far, <see cref="CachedTokens"/> included.</summary>
    public required int ProcessedTokens { get; init; }

    /// <summary>Time since the backend started reading the prompt.</summary>
    public required TimeSpan Elapsed { get; init; }
}
