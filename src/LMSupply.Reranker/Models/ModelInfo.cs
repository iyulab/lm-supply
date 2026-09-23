using LMSupply.Hardware;

namespace LMSupply.Reranker.Models;

/// <summary>
/// Contains metadata and configuration for a reranker model.
/// </summary>
public sealed record ModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets the unique identifier for the model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets the short alias name for the model (e.g., "default", "quality").
    /// </summary>
    public required string AliasName { get; init; }

    /// <summary>
    /// Gets the human-readable display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets the number of model parameters.
    /// </summary>
    public required long Parameters { get; init; }

    /// <summary>
    /// Gets the maximum input sequence length supported by the model.
    /// </summary>
    public required int MaxSequenceLength { get; init; }

    /// <inheritdoc />
    int? IModelInfoBase.ContextLength => MaxSequenceLength;

    /// <summary>
    /// Gets the approximate model size in bytes.
    /// </summary>
    public required long SizeBytes { get; init; }

    /// <summary>
    /// Gets the relative path to the ONNX model file in the repository.
    /// </summary>
    public required string OnnxFile { get; init; }

    /// <summary>
    /// Gets the relative path to the external weights file the ONNX graph loads its tensors from, or <c>null</c> when
    /// the graph holds its own weights. A model over the 2 GB protobuf limit ships <see cref="OnnxFile"/> as a small
    /// graph shell plus this file; without it the session fails at initialization, so it is downloaded with the graph
    /// and a cache holding the graph alone is not a cached model.
    /// </summary>
    public string? OnnxDataFile { get; init; }

    /// <summary>
    /// Gets the repository the tokenizer files (<see cref="TokenizerFile"/>, <c>vocab.txt</c>, <c>sentencepiece.bpe.model</c>)
    /// are downloaded from, or <c>null</c> for the model's own repository. An ONNX export can omit the SentencePiece model a
    /// Unigram tokenizer needs; the original repository still publishes it.
    /// </summary>
    public string? TokenizerRepoId { get; init; }

    /// <summary>
    /// Gets the relative path to the tokenizer configuration file.
    /// </summary>
    public required string TokenizerFile { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// Gets whether this model supports multiple languages.
    /// </summary>
    public bool IsMultilingual { get; init; }

    /// <summary>
    /// Gets the model architecture type.
    /// </summary>
    public ModelArchitecture Architecture { get; init; } = ModelArchitecture.Bert;

    /// <summary>
    /// Gets the expected output tensor shape type.
    /// </summary>
    public OutputShape OutputShape { get; init; } = OutputShape.SingleLogit;

    /// <summary>
    /// Gets the approximate model size in megabytes.
    /// </summary>
    public double SizeMB => SizeBytes / (1024.0 * 1024.0);

    /// <summary>
    /// Returns a string representation of the model.
    /// </summary>
    public override string ToString() =>
        $"{DisplayName} ({SizeMB:F0}MB, {MaxSequenceLength} tokens)";

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => SizeBytes;
    long IModelMemoryInfo.ParameterCount => Parameters;
    string? IModelMemoryInfo.QuantizationType => null;
}

/// <summary>
/// Model architecture types.
/// </summary>
public enum ModelArchitecture
{
    /// <summary>
    /// BERT-based architecture with WordPiece tokenization.
    /// </summary>
    Bert,

    /// <summary>
    /// RoBERTa-based architecture with BPE tokenization.
    /// </summary>
    Roberta,

    /// <summary>
    /// XLM-RoBERTa for multilingual support.
    /// </summary>
    XlmRoberta,

    /// <summary>
    /// Custom JinaBERT architecture with ALiBi attention.
    /// </summary>
    JinaBert
}

/// <summary>
/// Output tensor shape types.
/// </summary>
public enum OutputShape
{
    /// <summary>
    /// Single logit value per input pair [batch_size, 1].
    /// </summary>
    SingleLogit,

    /// <summary>
    /// Two logits for binary classification [batch_size, 2].
    /// </summary>
    BinaryClassification,

    /// <summary>
    /// Raw logit without extra dimension [batch_size].
    /// </summary>
    FlatLogit
}
