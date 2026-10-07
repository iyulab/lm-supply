namespace LMSupply.Embedder.Utils;

/// <summary>
/// The text prefixes a model was trained with, one per input role: a search query, a stored passage, and text with no
/// retrieval role (similarity, clustering, features). Each is <see langword="null"/> when the model has no convention
/// for that role.
/// </summary>
/// <param name="Query">Applied by <see cref="IEmbeddingModel.EmbedQueryAsync(string, CancellationToken)"/>.</param>
/// <param name="Passage">Applied by <see cref="IEmbeddingModel.EmbedPassageAsync(string, CancellationToken)"/>.</param>
/// <param name="Default">Applied by <see cref="IEmbeddingModel.EmbedAsync(string, CancellationToken)"/>.</param>
internal sealed record PromptPrefixes(string? Query, string? Passage, string? Default)
{
    /// <summary>No convention for any role.</summary>
    public static PromptPrefixes None { get; } = new(null, null, null);

    /// <summary>Whether any role has a prefix.</summary>
    public bool Any => Query is not null || Passage is not null || Default is not null;

    /// <summary>The prefixes a catalog or loaded <see cref="ModelInfo"/> carries.</summary>
    public static PromptPrefixes Of(ModelInfo info) => new(info.QueryPrefix, info.PassagePrefix, info.DefaultPrefix);
}
