namespace LMSupply.Llama.Server;

internal sealed class EmbeddingRequest
{
    /// <summary>
    /// Input text(s) to embed. Can be a single string or array of strings.
    /// </summary>
    public required object Input { get; set; }

    /// <summary>
    /// Model identifier (optional, defaults to loaded model).
    /// </summary>
    public string Model { get; set; } = "default";

    /// <summary>
    /// Encoding format: "float" (default) or "base64".
    /// </summary>
    public string EncodingFormat { get; set; } = "float";
}

internal sealed class EmbeddingResponse
{
    public string Object { get; set; } = "list";
    public List<EmbeddingData> Data { get; set; } = [];
    public string Model { get; set; } = "";
    public EmbeddingUsage Usage { get; set; } = new();
}

internal sealed class EmbeddingData
{
    public string Object { get; set; } = "embedding";
    public float[] Embedding { get; set; } = [];
    public int Index { get; set; }
}

internal sealed class EmbeddingUsage
{
    public int PromptTokens { get; set; }
    public int TotalTokens { get; set; }
}

internal sealed class RerankRequest
{
    /// <summary>
    /// The search query.
    /// </summary>
    public required string Query { get; set; }

    /// <summary>
    /// Documents to rerank.
    /// </summary>
    public required List<string> Documents { get; set; }

    /// <summary>
    /// Maximum number of results to return.
    /// </summary>
    public int TopN { get; set; } = 10;
}

internal sealed class RerankResponse
{
    public List<RerankResult> Results { get; set; } = [];
}

/// <summary>
/// Result from reranking operation.
/// </summary>
public sealed class RerankResult
{
    /// <summary>
    /// Original index of the document in the input list.
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Relevance score (higher is more relevant).
    /// </summary>
    public float RelevanceScore { get; set; }
}

internal sealed class TokenizeRequest
{
    public required string Content { get; init; }

    /// <summary>Whether to add the model's special tokens (BOS); the server's default is not to.</summary>
    public bool? AddSpecial { get; init; }
}

internal sealed class TokenizeResponse
{
    public List<int>? Tokens { get; set; }
}

internal sealed class ApplyTemplateResponse
{
    public string? Prompt { get; set; }
}
