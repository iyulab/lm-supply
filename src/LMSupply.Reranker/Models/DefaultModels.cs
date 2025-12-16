namespace LMSupply.Reranker.Models;

/// <summary>
/// Provides definitions for built-in supported models.
/// Updated: 2025-01 based on BEIR and MS MARCO benchmarks.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// Gets the default model (balanced speed and quality).
    /// </summary>
    public static ModelInfo Default => MsMarcoMiniLML6V2;

    /// <summary>
    /// MS MARCO MiniLM L6 v2 - Fast and lightweight model.
    /// 22M params, 512 tokens, proven performer for English.
    /// Best speed/quality balance for most use cases.
    /// </summary>
    public static ModelInfo MsMarcoMiniLML6V2 { get; } = new()
    {
        Id = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        Alias = "default",
        DisplayName = "MS MARCO MiniLM L6",
        Parameters = 22_700_000,
        MaxSequenceLength = 512,
        SizeBytes = 90_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Default: Best speed/quality balance for English",
        IsMultilingual = false,
        Architecture = ModelArchitecture.Bert,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// MS MARCO TinyBERT L2 v2 - Ultra-fast lightweight model.
    /// 4.4M params, 512 tokens, fastest inference.
    /// Best for latency-critical applications.
    /// </summary>
    public static ModelInfo MsMarcoTinyBertL2V2 { get; } = new()
    {
        Id = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        Alias = "fast",
        DisplayName = "MS MARCO TinyBERT L2",
        Parameters = 4_400_000,
        MaxSequenceLength = 512,
        SizeBytes = 18_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Fast: Ultra-lightweight for latency-critical apps",
        IsMultilingual = false,
        Architecture = ModelArchitecture.Bert,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// MS MARCO MiniLM L12 v2 - Higher quality English model.
    /// 33M params, 512 tokens, better accuracy than L6.
    /// Legacy quality option, consider BGE Reranker Base for better results.
    /// </summary>
    public static ModelInfo MsMarcoMiniLML12V2 { get; } = new()
    {
        Id = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        Alias = "ms-marco-l12",
        DisplayName = "MS MARCO MiniLM L12",
        Parameters = 33_400_000,
        MaxSequenceLength = 512,
        SizeBytes = 134_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Legacy quality: Better accuracy than L6, English only",
        IsMultilingual = false,
        Architecture = ModelArchitecture.Bert,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// BGE Reranker Base - Quality multilingual model (2024 release).
    /// 278M params, 512 tokens, excellent accuracy.
    /// Recommended quality option with multilingual support.
    /// </summary>
    public static ModelInfo BgeRerankerBase { get; } = new()
    {
        Id = "BAAI/bge-reranker-base",
        Alias = "quality",
        DisplayName = "BGE Reranker Base",
        Parameters = 278_000_000,
        MaxSequenceLength = 512,
        SizeBytes = 440_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Quality: 2024 release, excellent accuracy, multilingual",
        IsMultilingual = true,
        Architecture = ModelArchitecture.XlmRoberta,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// BGE Reranker Large - Highest accuracy model.
    /// 560M params, 512 tokens, best quality.
    /// Best for quality-critical applications.
    /// </summary>
    public static ModelInfo BgeRerankerLarge { get; } = new()
    {
        Id = "BAAI/bge-reranker-large",
        Alias = "large",
        DisplayName = "BGE Reranker Large",
        Parameters = 560_000_000,
        MaxSequenceLength = 512,
        SizeBytes = 1_100_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Large: Highest accuracy, quality-critical apps",
        IsMultilingual = true,
        Architecture = ModelArchitecture.XlmRoberta,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// BGE Reranker v2 M3 - Best multilingual long-context model.
    /// 568M params, 8192 tokens, 100+ languages.
    /// Best for multilingual and long document scenarios.
    /// </summary>
    public static ModelInfo BgeRerankerV2M3 { get; } = new()
    {
        Id = "BAAI/bge-reranker-v2-m3",
        Alias = "multilingual",
        DisplayName = "BGE Reranker v2 M3",
        Parameters = 568_000_000,
        MaxSequenceLength = 8192,
        SizeBytes = 1_100_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Multilingual: 8K context, 100+ languages, long docs",
        IsMultilingual = true,
        Architecture = ModelArchitecture.XlmRoberta,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// Gets all built-in models.
    /// </summary>
    public static IReadOnlyList<ModelInfo> All { get; } =
    [
        MsMarcoMiniLML6V2,
        MsMarcoTinyBertL2V2,
        MsMarcoMiniLML12V2,
        BgeRerankerBase,
        BgeRerankerLarge,
        BgeRerankerV2M3
    ];
}
