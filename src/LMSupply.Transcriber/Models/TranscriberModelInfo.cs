namespace LMSupply.Transcriber.Models;

/// <summary>
/// Information about a transcriber model configuration.
/// </summary>
public sealed class TranscriberModelInfo
{
    /// <summary>
    /// Gets or sets the HuggingFace model ID.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets or sets the model alias (e.g., "default", "fast").
    /// </summary>
    public required string Alias { get; init; }

    /// <summary>
    /// Gets or sets the display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets or sets the model architecture (e.g., "Whisper").
    /// </summary>
    public required string Architecture { get; init; }

    /// <summary>
    /// Gets or sets the model size in millions of parameters.
    /// </summary>
    public float ParametersM { get; init; }

    /// <summary>
    /// Gets or sets the model size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Gets or sets the Word Error Rate (WER) on LibriSpeech test-clean.
    /// </summary>
    public float? WerLibriSpeech { get; init; }

    /// <summary>
    /// Gets or sets the maximum audio duration in seconds per chunk.
    /// </summary>
    public int MaxDurationSeconds { get; init; } = 30;

    /// <summary>
    /// Gets or sets the sample rate expected by the model.
    /// </summary>
    public int SampleRate { get; init; } = 16000;

    /// <summary>
    /// Gets or sets the number of mel frequency bins.
    /// </summary>
    public int NumMelBins { get; init; } = 80;

    /// <summary>
    /// Gets or sets the encoder ONNX file name.
    /// </summary>
    public string EncoderFile { get; init; } = "encoder_model.onnx";

    /// <summary>
    /// Gets or sets the decoder ONNX file name.
    /// </summary>
    public string DecoderFile { get; init; } = "decoder_model.onnx";

    /// <summary>
    /// Gets or sets the supported languages (null means multilingual).
    /// </summary>
    public IReadOnlyList<string>? SupportedLanguages { get; init; }

    /// <summary>
    /// Gets or sets whether this is a multilingual model.
    /// </summary>
    public bool IsMultilingual { get; init; } = true;

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Gets or sets the license type.
    /// </summary>
    public string License { get; init; } = "MIT";
}
