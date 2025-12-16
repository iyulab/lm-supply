namespace LMSupply.Reranker.Models;

/// <summary>
/// Contains metadata and configuration for a reranker model.
/// </summary>
public sealed record ModelInfo
{
    /// <summary>
    /// Gets the unique identifier for the model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets the short alias for the model (e.g., "default", "quality").
    /// </summary>
    public required string Alias { get; init; }

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

    /// <summary>
    /// Gets the approximate model size in bytes.
    /// </summary>
    public required long SizeBytes { get; init; }

    /// <summary>
    /// Gets the relative path to the ONNX model file in the repository.
    /// </summary>
    public required string OnnxFile { get; init; }

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
