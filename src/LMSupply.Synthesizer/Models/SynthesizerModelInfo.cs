using LMSupply.Hardware;

namespace LMSupply.Synthesizer.Models;

/// <summary>
/// Information about a TTS model configuration.
/// </summary>
public sealed record SynthesizerModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets the HuggingFace model ID or local path.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets the friendly alias name for this model.
    /// </summary>
    public required string AliasName { get; init; }

    /// <summary>
    /// Gets the display name for this model.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets the model architecture (e.g., "VITS", "Tacotron2").
    /// </summary>
    public string Architecture { get; init; } = "VITS";

    /// <summary>
    /// Gets the language code (e.g., "en", "ko", "multilingual").
    /// </summary>
    public string Language { get; init; } = "en";

    /// <summary>
    /// Gets the voice name or description.
    /// </summary>
    public string? VoiceName { get; init; }

    /// <summary>
    /// Gets the number of speakers supported (1 for single-speaker models).
    /// </summary>
    public int NumSpeakers { get; init; } = 1;

    /// <summary>
    /// Gets the output sample rate in Hz.
    /// </summary>
    public int SampleRate { get; init; } = 22050;

    /// <summary>
    /// Gets the model file name.
    /// </summary>
    public string ModelFile { get; init; } = "model.onnx";

    /// <summary>
    /// Gets the config file name.
    /// </summary>
    public string ConfigFile { get; init; } = "config.json";

    /// <summary>
    /// Gets the approximate model size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// The licence of the model's weights as curated in the catalog (for a converted build, the licence of the model it
    /// was made from), or <see langword="null"/> when nobody declared one — a user-registered model reports no licence
    /// rather than a guessed default.
    /// </summary>
    public string? License { get; init; }

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => SizeBytes;
    long IModelMemoryInfo.ParameterCount => SizeBytes / 4;
    string? IModelMemoryInfo.QuantizationType => null;
}
