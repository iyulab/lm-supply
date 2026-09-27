using LMSupply.Hardware;
using LMSupply.Transcriber.Internal;

namespace LMSupply.Transcriber.Models;

/// <summary>
/// Information about a transcriber model configuration.
/// </summary>
public sealed class TranscriberModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets or sets the HuggingFace model ID.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets or sets the model alias name (e.g., "default", "fast").
    /// </summary>
    public required string AliasName { get; init; }

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
    /// Gets or sets the approximate size in bytes of the model's full-precision export — the figure model selection and
    /// memory estimates are based on. It is <b>not</b> what a load downloads: a Whisper load takes the quantization
    /// <see cref="LMSupplyOptionsBase.QuantizationHint"/> or the hardware tier picks (int8 by default on most machines,
    /// roughly a quarter of this). <see cref="LocalTranscriber.GetDownloadSizeBytesAsync(string, TranscriberOptions?, CancellationToken)"/>
    /// answers the download for the options a load will use.
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
    /// Gets or sets the hidden size (d_model) of the model.
    /// </summary>
    public int HiddenSize { get; init; } = 512;

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
    /// Gets whether this model supports the Whisper translate task (speech → English text).
    /// </summary>
    /// <remarks>
    /// Whisper's translate task is only available on multilingual checkpoints. English-only
    /// variants (e.g., <c>whisper-base.en</c>) do not include the translate token in their
    /// decoder vocabulary and must use the transcribe task exclusively. This is a fixed
    /// capability of the model architecture, not a runtime option.
    /// </remarks>
    public bool IsTranslateSupported => IsMultilingual;

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Gets or sets the license type.
    /// </summary>
    public string License { get; init; } = "MIT";

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => SizeBytes;
    long IModelMemoryInfo.ParameterCount => (long)(ParametersM * 1_000_000);
    string? IModelMemoryInfo.QuantizationType => null;

    /// <summary>
    /// Creates a new instance with parameters overridden from a parsed Whisper config.
    /// Only non-null config values are applied.
    /// </summary>
    internal TranscriberModelInfo WithConfigOverrides(WhisperModelConfig config)
    {
        return new TranscriberModelInfo
        {
            Id = Id,
            AliasName = AliasName,
            DisplayName = DisplayName,
            Architecture = Architecture,
            ParametersM = ParametersM,
            SizeBytes = SizeBytes,
            WerLibriSpeech = WerLibriSpeech,
            MaxDurationSeconds = MaxDurationSeconds,
            SampleRate = SampleRate,
            NumMelBins = config.NumMelBins ?? NumMelBins,
            HiddenSize = config.HiddenSize ?? HiddenSize,
            EncoderFile = EncoderFile,
            DecoderFile = DecoderFile,
            SupportedLanguages = SupportedLanguages,
            IsMultilingual = IsMultilingual,
            Description = Description,
            License = License
        };
    }
}
