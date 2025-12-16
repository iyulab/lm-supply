namespace LMSupply.Reranker;

/// <summary>
/// Represents a reranked document with its relevance score.
/// </summary>
/// <param name="OriginalIndex">Zero-based index of the document in the original input array.</param>
/// <param name="Score">Relevance score between 0.0 (not relevant) and 1.0 (highly relevant).</param>
/// <param name="Document">The original document text.</param>
public readonly record struct RankedResult(
    int OriginalIndex,
    float Score,
    string Document
) : IComparable<RankedResult>
{
    /// <summary>
    /// Compares results by score in descending order (higher scores first).
    /// </summary>
    public int CompareTo(RankedResult other) => other.Score.CompareTo(Score);

    /// <summary>
    /// Returns a string representation of the result.
    /// </summary>
    public override string ToString()
    {
        var preview = Document.Length > 50 ? Document[..50] + "..." : Document;
        return $"[{Score:F4}] #{OriginalIndex}: {preview}";
    }
}
