using LMSupply.Hardware;

namespace LMSupply.Synthesizer.Models;

/// <summary>
/// Information about a TTS model configuration.
/// </summary>
public sealed record SynthesizerModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets the HuggingFace repository id or local path the voice is loaded from. One repository can hold many voices,
    /// so this does not name a voice on its own: <see cref="CatalogKey"/> does.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Identifies the voice within the catalog, and loads it: <see cref="Id"/>, followed by <c>:</c> and
    /// <see cref="VoiceName"/> when the voice lives in a folder of its repository
    /// (<c>rhasspy/piper-voices:en/en_US/ljspeech/medium</c>).
    /// </summary>
    public string CatalogKey => string.IsNullOrEmpty(VoiceName) ? Id : $"{Id}:{VoiceName}";

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
    /// Gets the voice's folder inside its repository (<c>en/en_US/ljspeech/medium</c>), which is what tells the voices
    /// of one repository apart; <see langword="null"/> when the model and its config are at the root.
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
