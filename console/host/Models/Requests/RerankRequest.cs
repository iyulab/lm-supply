namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// Rerank request
/// </summary>
public sealed record RerankRequest
{
    /// <summary>
    /// Model ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// Search query
    /// </summary>
    public required string Query { get; init; }

    /// <summary>
    /// Documents to rerank
    /// </summary>
    public required IReadOnlyList<string> Documents { get; init; }

    /// <summary>
    /// Return only the top K results (all when null)
    /// </summary>
    public int? TopK { get; init; }
}
