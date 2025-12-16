namespace LMSupply.Captioner;

/// <summary>
/// Result of an image captioning operation.
/// </summary>
/// <param name="Caption">The generated caption text.</param>
/// <param name="Confidence">Average log probability of tokens (higher is more confident).</param>
/// <param name="AlternativeCaptions">Alternative captions from beam search (if NumBeams > 1).</param>
public record CaptionResult(
    string Caption,
    float Confidence,
    IReadOnlyList<string> AlternativeCaptions)
{
    /// <summary>
    /// Creates a result with a single caption.
    /// </summary>
    public CaptionResult(string caption, float confidence)
        : this(caption, confidence, [])
    {
    }
}

/// <summary>
/// Result of a visual question answering operation.
/// </summary>
/// <param name="Question">The input question.</param>
/// <param name="Answer">The generated answer.</param>
/// <param name="Confidence">Average log probability of tokens.</param>
public record VqaResult(
    string Question,
    string Answer,
    float Confidence);
