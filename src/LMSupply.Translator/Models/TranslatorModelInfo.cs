using LMSupply.Core.Download;
using LMSupply.Hardware;

namespace LMSupply.Translator.Models;

/// <summary>
/// Information about a translation model.
/// </summary>
public sealed class TranslatorModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets or sets the model identifier (HuggingFace repo or local path).
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets or sets the short alias name for the model.
    /// </summary>
    public required string AliasName { get; init; }

    /// <summary>
    /// Gets or sets the display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets or sets the model architecture (e.g., "MarianMT", "NLLB").
    /// </summary>
    public required string Architecture { get; init; }

    /// <summary>
    /// Gets or sets the source language code.
    /// </summary>
    public required string SourceLanguage { get; init; }

    /// <summary>
    /// Gets or sets the target language code.
    /// </summary>
    public required string TargetLanguage { get; init; }

    /// <summary>
    /// Gets or sets the number of parameters in millions.
    /// </summary>
    public float ParametersM { get; init; }

    /// <summary>
    /// Gets or sets the approximate model size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Gets or sets the BLEU score on the benchmark dataset.
    /// </summary>
    public float BleuScore { get; init; }

    /// <summary>
    /// Gets or sets the maximum sequence length.
    /// </summary>
    public int MaxLength { get; init; } = 512;

    /// <inheritdoc />
    int? IModelInfoBase.ContextLength => MaxLength;

    /// <summary>
    /// Gets or sets the vocabulary size.
    /// </summary>
    public int VocabSize { get; init; }

    /// <summary>
    /// Gets or sets the subfolder within the repository where model files are located.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Gets or sets whether to use auto-discovery for ONNX files.
    /// When true, encoder and decoder files are automatically discovered.
    /// When false, explicit EncoderFile and DecoderFile are used.
    /// </summary>
    public bool UseAutoDiscovery { get; init; } = true;

    /// <summary>
    /// Gets or sets the preferred decoder variant when using auto-discovery.
    /// </summary>
    public DecoderVariant PreferredDecoderVariant { get; init; } = DecoderVariant.Merged;

    /// <summary>
    /// Gets or sets the ONNX encoder file name (used when UseAutoDiscovery is false).
    /// </summary>
    public string? EncoderFile { get; init; }

    /// <summary>
    /// Gets or sets the ONNX decoder file name (used when UseAutoDiscovery is false).
    /// </summary>
    public string? DecoderFile { get; init; }

    /// <summary>
    /// Gets or sets the tokenizer model file name.
    /// </summary>
    public string TokenizerFile { get; init; } = "source.spm";

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string Description { get; init; } = "";

    /// <summary>
    /// Gets or sets the license identifier.
    /// </summary>
    public string License { get; init; } = "Unknown";

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => SizeBytes;
    long IModelMemoryInfo.ParameterCount => (long)(ParametersM * 1_000_000);
    string? IModelMemoryInfo.QuantizationType => null;
}
