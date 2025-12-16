using LMSupply.Vision;

namespace LMSupply.Captioner.Models;

/// <summary>
/// Metadata about a captioning model.
/// </summary>
/// <param name="RepoId">HuggingFace repository ID.</param>
/// <param name="Alias">Short alias for the model.</param>
/// <param name="DisplayName">Human-readable display name.</param>
/// <param name="EncoderFile">ONNX file name for the vision encoder.</param>
/// <param name="DecoderFile">ONNX file name for the text decoder.</param>
/// <param name="TokenizerType">Type of tokenizer used.</param>
/// <param name="PreprocessProfile">Image preprocessing profile.</param>
/// <param name="SupportsVqa">Whether the model supports VQA.</param>
/// <param name="VocabSize">Size of the vocabulary.</param>
/// <param name="BosTokenId">Beginning of sequence token ID.</param>
/// <param name="EosTokenId">End of sequence token ID.</param>
/// <param name="PadTokenId">Padding token ID.</param>
public record ModelInfo(
    string RepoId,
    string Alias,
    string DisplayName,
    string EncoderFile,
    string DecoderFile,
    TokenizerType TokenizerType,
    PreprocessProfile PreprocessProfile,
    bool SupportsVqa,
    int VocabSize,
    int BosTokenId,
    int EosTokenId,
    int PadTokenId)
{
    /// <summary>
    /// Optional subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Additional ONNX files required by the model.
    /// </summary>
    public IReadOnlyList<string> AdditionalFiles { get; init; } = [];
}

/// <summary>
/// Type of tokenizer used by the model.
/// </summary>
public enum TokenizerType
{
    /// <summary>GPT-2 style BPE tokenizer.</summary>
    Gpt2,

    /// <summary>BERT WordPiece tokenizer.</summary>
    Bert,

    /// <summary>SentencePiece tokenizer.</summary>
    SentencePiece,

    /// <summary>Llama tokenizer.</summary>
    Llama
}
