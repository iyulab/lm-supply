namespace LMSupply.Generator.Models;

/// <summary>
/// A generated token and its log probability, with the most likely candidates at its position when
/// <see cref="GenerationOptions.TopLogprobs"/> asked for them — the shape of an OpenAI <c>logprobs.content</c> entry.
/// </summary>
/// <remarks>
/// The values are the model's raw next-token distribution (a softmax over the logits), taken before sampling:
/// temperature, top-k/top-p/min-p, penalties, logit bias and a grammar or JSON schema do not change them, though they
/// still decide which token is sampled. The sampled token can therefore be one the candidates do not list, and a
/// constrained answer's distribution is the raw one: to compare a fixed set of answers, read their tokens' values and
/// renormalize over that set.
/// </remarks>
public sealed record TokenLogprob
{
    /// <summary>The token's id in the model's vocabulary (see <see cref="Abstractions.IGeneratorModel.TokenizeAsync"/>); null when the backend does not report it.</summary>
    public int? TokenId { get; init; }

    /// <summary>The token's text. Empty for a token with no text, such as an end-of-generation token.</summary>
    public required string Token { get; init; }

    /// <summary>The natural log of the token's probability; <c>Math.Exp</c> of it is the probability.</summary>
    public required double Logprob { get; init; }

    /// <summary>
    /// The token's UTF-8 bytes; null when the backend does not report them. A token can hold part of a character, which
    /// <see cref="Token"/> cannot show.
    /// </summary>
    public IReadOnlyList<byte>? Bytes { get; init; }

    /// <summary>
    /// The most likely tokens at this position, most likely first — at most <see cref="GenerationOptions.TopLogprobs"/>
    /// of them; empty when it is 0.
    /// </summary>
    public IReadOnlyList<TopLogprob> TopLogprobs { get; init; } = [];
}

/// <summary>
/// One of the most likely tokens at a position (<see cref="TokenLogprob.TopLogprobs"/>) — an OpenAI <c>top_logprobs</c>
/// entry.
/// </summary>
public sealed record TopLogprob
{
    /// <summary>The token's id in the model's vocabulary; null when the backend does not report it.</summary>
    public int? TokenId { get; init; }

    /// <summary>The token's text. Empty for a token with no text, such as an end-of-generation token.</summary>
    public required string Token { get; init; }

    /// <summary>The natural log of the token's probability at this position, before sampling.</summary>
    public required double Logprob { get; init; }

    /// <summary>The token's UTF-8 bytes; null when the backend does not report them.</summary>
    public IReadOnlyList<byte>? Bytes { get; init; }
}
