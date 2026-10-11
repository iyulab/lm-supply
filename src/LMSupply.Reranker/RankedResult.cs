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
    /// Compares results by <b>rank</b>: a result with a higher score comes first, so sorting yields the ranked order.
    /// The comparison operators follow it and therefore compare rank, not score — to compare scores, compare
    /// <see cref="Score"/>.
    /// </summary>
    public int CompareTo(RankedResult other) => other.Score.CompareTo(Score);

    /// <summary><paramref name="left"/> ranks before <paramref name="right"/> (its score is higher).</summary>
    public static bool operator <(RankedResult left, RankedResult right) => left.CompareTo(right) < 0;

    /// <summary><paramref name="left"/> ranks after <paramref name="right"/> (its score is lower).</summary>
    public static bool operator >(RankedResult left, RankedResult right) => left.CompareTo(right) > 0;

    /// <summary><paramref name="left"/> ranks before or level with <paramref name="right"/> (its score is not lower).</summary>
    public static bool operator <=(RankedResult left, RankedResult right) => left.CompareTo(right) <= 0;

    /// <summary><paramref name="left"/> ranks after or level with <paramref name="right"/> (its score is not higher).</summary>
    public static bool operator >=(RankedResult left, RankedResult right) => left.CompareTo(right) >= 0;

    /// <summary>
    /// Returns a string representation of the result.
    /// </summary>
    public override string ToString()
    {
        var preview = Document.Length > 50 ? Document[..50] + "..." : Document;
        return $"[{Score:F4}] #{OriginalIndex}: {preview}";
    }
}
