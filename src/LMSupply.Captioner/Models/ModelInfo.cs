using LMSupply.Vision;

namespace LMSupply.Captioner.Models;

/// <summary>
/// Metadata about a captioning model.
/// </summary>
/// <param name="RepoId">HuggingFace repository ID.</param>
/// <param name="AliasName">Short alias name for the model.</param>
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
    string AliasName,
    string DisplayName,
    string EncoderFile,
    string DecoderFile,
    TokenizerType TokenizerType,
    PreprocessProfile PreprocessProfile,
    bool SupportsVqa,
    int VocabSize,
    int BosTokenId,
    int EosTokenId,
    int PadTokenId) : IModelInfoBase
{
    /// <summary>
    /// Gets the model description.
    /// Defaults to DisplayName if not explicitly set.
    /// </summary>
    public string? Description { get; init; }

    // IModelInfoBase explicit implementation
    string IModelInfoBase.Id => RepoId;
    /// <summary>
    /// Optional subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// The licence of the model's weights as the catalog curates it — an SPDX identifier where one exists. For a
    /// conversion repository (an ONNX export of another model) it is the licence of the model it converts, which the
    /// conversion's own card often leaves out. <see langword="null"/> when not curated (a model loaded from its own files).
    /// </summary>
    public string? License { get; init; }

    /// <summary>
    /// Additional ONNX files required by the model.
    /// </summary>
    public IReadOnlyList<string> AdditionalFiles { get; init; } = [];

    /// <summary>
    /// The model family, which decides how the files are run. Defaults to <see cref="CaptionerArchitecture.VitGpt2"/>.
    /// </summary>
    public CaptionerArchitecture Architecture { get; init; } = CaptionerArchitecture.VitGpt2;

    /// <summary>
    /// The token the decoder starts from, when it is not <see cref="BosTokenId"/> (BART-style decoders start from
    /// <c>&lt;/s&gt;</c> and force <c>&lt;s&gt;</c> as the first generated token).
    /// </summary>
    public int? DecoderStartTokenId { get; init; }

    /// <summary>
    /// Whether the repository publishes every ONNX file in quantized variants (<c>_fp16</c>, <c>_quantized</c>,
    /// <c>_q4</c>) beside the full-precision one. When true, a load picks the variant from
    /// <see cref="LMSupplyOptionsBase.QuantizationHint"/> or, without one, the hardware tier — so the file names above
    /// are the full-precision names, not necessarily what is downloaded.
    /// </summary>
    public bool HasQuantizationVariants { get; init; }
}

/// <summary>
/// Captioning model families <see cref="LocalCaptioner"/> can run.
/// </summary>
public enum CaptionerArchitecture
{
    /// <summary>A vision encoder with a GPT-2 decoder that reads its hidden states (ViT-GPT2).</summary>
    VitGpt2,

    /// <summary>
    /// Florence-2: a vision encoder, a text encoder over the image features and a task prompt, and a BART-style
    /// decoder. The caption detail is a task the prompt selects (<see cref="CaptionerOptions.Detail"/>).
    /// </summary>
    Florence2
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
