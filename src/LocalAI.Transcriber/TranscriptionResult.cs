namespace LocalAI.Transcriber;

/// <summary>
/// Represents the result of a transcription operation.
/// </summary>
public sealed class TranscriptionResult
{
    /// <summary>
    /// Gets the full transcribed text.
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// Gets the detected or specified language code.
    /// </summary>
    public required string Language { get; init; }

    /// <summary>
    /// Gets the language detection probability (0-1).
    /// </summary>
    public float? LanguageProbability { get; init; }

    /// <summary>
    /// Gets the transcription segments with timestamps.
    /// </summary>
    public IReadOnlyList<TranscriptionSegment> Segments { get; init; } = [];

    /// <summary>
    /// Gets the total audio duration in seconds.
    /// </summary>
    public double? DurationSeconds { get; init; }

    /// <summary>
    /// Gets the inference time in milliseconds.
    /// </summary>
    public double? InferenceTimeMs { get; init; }

    /// <summary>
    /// Gets the real-time factor (audio duration / processing time).
    /// Values greater than 1 indicate faster-than-real-time processing.
    /// </summary>
    public double? RealTimeFactor => DurationSeconds.HasValue && InferenceTimeMs.HasValue
        ? DurationSeconds.Value / (InferenceTimeMs.Value / 1000.0)
        : null;
}

/// <summary>
/// Represents a segment of transcribed audio with timestamps.
/// </summary>
public sealed class TranscriptionSegment
{
    /// <summary>
    /// Gets the segment index (0-based).
    /// </summary>
    public int Id { get; init; }

    /// <summary>
    /// Gets the start time in seconds.
    /// </summary>
    public double Start { get; init; }

    /// <summary>
    /// Gets the end time in seconds.
    /// </summary>
    public double End { get; init; }

    /// <summary>
    /// Gets the transcribed text for this segment.
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// Gets the average log probability of the tokens.
    /// </summary>
    public float? AvgLogProb { get; init; }

    /// <summary>
    /// Gets the probability that this segment contains no speech.
    /// </summary>
    public float? NoSpeechProb { get; init; }

    /// <summary>
    /// Gets the compression ratio of the segment.
    /// </summary>
    public float? CompressionRatio { get; init; }

    /// <summary>
    /// Gets word-level timestamps if available.
    /// </summary>
    public IReadOnlyList<WordTimestamp>? Words { get; init; }

    /// <summary>
    /// Gets the duration of this segment in seconds.
    /// </summary>
    public double Duration => End - Start;

    /// <inheritdoc/>
    public override string ToString() =>
        $"[{TimeSpan.FromSeconds(Start):mm\\:ss\\.ff} --> {TimeSpan.FromSeconds(End):mm\\:ss\\.ff}] {Text}";
}

/// <summary>
/// Represents a word with its timestamp.
/// </summary>
public sealed class WordTimestamp
{
    /// <summary>
    /// Gets the word text.
    /// </summary>
    public required string Word { get; init; }

    /// <summary>
    /// Gets the start time in seconds.
    /// </summary>
    public double Start { get; init; }

    /// <summary>
    /// Gets the end time in seconds.
    /// </summary>
    public double End { get; init; }

    /// <summary>
    /// Gets the probability of this word.
    /// </summary>
    public float? Probability { get; init; }
}
