namespace LMSupply.Embedder.Utils;

/// <summary>
/// Provides definitions for built-in supported embedding models.
/// Updated: 2026-05 — BGE-M3 promoted to 'default' for multilingual coverage.
/// </summary>
internal static class DefaultModels
{
    // ===== Alias models =====
    // 'default' and 'quality' both resolve to BGE-M3 (100+ languages, 8K context, ONNX official).
    // For lightweight multilingual use 'fast' (multilingual-e5-small).
    // For high-dimensional dense retrieval use 'large' (multilingual-e5-large).
    // For English-only workloads use 'nomic-embed-text-v1.5' (Matryoshka, fast).

    /// <summary>
    /// Default: BGE-M3, 568M params, 100+ languages, 8K context, SOTA multilingual dense retrieval.
    /// Uses CLS pooling; instruction prefix not required.
    /// </summary>
    public static ModelInfo BgeM3DefaultAlias { get; } = new()
    {
        RepoId = "BAAI/bge-m3",
        AliasName = "default",
        Dimensions = 1024,
        MaxSequenceLength = 8192,
        PoolingMode = PoolingMode.Cls,
        DoLowerCase = false,
        Description = "Default: BGE-M3, 568M params, 100+ languages, 8K context, SOTA multilingual",
        Subfolder = "onnx"
    };

    /// <summary>
    /// Fast: multilingual-e5-small, 118M params, 100+ languages, lightweight.
    /// </summary>
    public static ModelInfo MultilingualE5SmallAlias { get; } = new()
    {
        RepoId = "intfloat/multilingual-e5-small",
        AliasName = "fast",
        Dimensions = 384,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "Fast: multilingual-e5-small, 118M params, 100+ languages, lightweight",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    /// <summary>
    /// Quality: BGE-M3, 568M params, 100+ languages, 8K context, SOTA multilingual, dense+sparse.
    /// Same model as 'default'; exposed separately for pipelines that explicitly request quality tier.
    /// </summary>
    public static ModelInfo BgeM3QualityAlias { get; } = new()
    {
        RepoId = "BAAI/bge-m3",
        AliasName = "quality",
        Dimensions = 1024,
        MaxSequenceLength = 8192,
        PoolingMode = PoolingMode.Cls,
        DoLowerCase = false,
        Description = "Quality: BGE-M3, 568M params, 100+ languages, 8K context, SOTA multilingual",
        Subfolder = "onnx"
    };

    /// <summary>
    /// Large: multilingual-e5-large, 560M params, 100+ languages, highest dense quality.
    /// Note: 512-token context limit. Use BGE-M3 (default) for long-document retrieval.
    /// </summary>
    public static ModelInfo MultilingualE5LargeAlias { get; } = new()
    {
        RepoId = "intfloat/multilingual-e5-large",
        AliasName = "large",
        Dimensions = 1024,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "Large: multilingual-e5-large, 560M params, 100+ languages, highest dense quality",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    // ===== Explicit models (non-alias, registered by short name) =====

    /// <summary>
    /// nomic-embed-text-v1.5, 137M params, English-first, Matryoshka 64–768d, 8K context.
    /// Recommended for English-only RAG pipelines; instruction prefixes applied automatically
    /// via <see cref="ModelInfo.QueryPrefix"/>/<see cref="ModelInfo.PassagePrefix"/>.
    /// </summary>
    public static ModelInfo NomicEmbedTextV15 { get; } = new()
    {
        RepoId = "nomic-ai/nomic-embed-text-v1.5",
        AliasName = "nomic-embed-text-v1.5",
        Dimensions = 768,
        MaxSequenceLength = 8192,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "137M params, English-first, Matryoshka 64–768d, 8K context",
        Subfolder = "onnx",
        QueryPrefix = "search_query: ",
        PassagePrefix = "search_document: "
    };

    /// <summary>
    /// all-mpnet-base-v2, 110M params, legacy quality model, English.
    /// </summary>
    public static ModelInfo AllMpnetBaseV2 { get; } = new()
    {
        RepoId = "sentence-transformers/all-mpnet-base-v2",
        AliasName = "all-mpnet-base-v2",
        Dimensions = 768,
        MaxSequenceLength = 384,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = true,
        Description = "110M params, legacy quality model, English",
        Subfolder = "onnx"
    };

    /// <summary>
    /// bge-base-en-v1.5, 110M params, excellent quality, English.
    /// </summary>
    public static ModelInfo BgeBaseEnV15 { get; } = new()
    {
        RepoId = "BAAI/bge-base-en-v1.5",
        AliasName = "bge-base-en-v1.5",
        Dimensions = 768,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Cls,
        DoLowerCase = true,
        Description = "110M params, excellent quality, English",
        Subfolder = "onnx"
    };

    /// <summary>
    /// bge-large-en-v1.5, 335M params, highest accuracy BGE, English.
    /// </summary>
    public static ModelInfo BgeLargeEnV15 { get; } = new()
    {
        RepoId = "BAAI/bge-large-en-v1.5",
        AliasName = "bge-large-en-v1.5",
        Dimensions = 1024,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Cls,
        DoLowerCase = true,
        Description = "335M params, highest accuracy BGE, English",
        Subfolder = "onnx"
    };

    /// <summary>
    /// e5-small-v2, 33M params, English. Requires the query/passage prefix convention like the
    /// rest of the E5 family (see intfloat/e5-small-v2's model card) — applied automatically via
    /// <see cref="ModelInfo.QueryPrefix"/>/<see cref="ModelInfo.PassagePrefix"/>.
    /// </summary>
    public static ModelInfo E5SmallV2 { get; } = new()
    {
        RepoId = "intfloat/e5-small-v2",
        AliasName = "e5-small-v2",
        Dimensions = 384,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "33M params, English",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    /// <summary>
    /// e5-base-v2, 110M params, excellent retrieval, English.
    /// </summary>
    public static ModelInfo E5BaseV2 { get; } = new()
    {
        RepoId = "intfloat/e5-base-v2",
        AliasName = "e5-base-v2",
        Dimensions = 768,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "110M params, excellent retrieval, English",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    /// <summary>
    /// multilingual-e5-small, 118M params, 100+ languages, compact.
    /// </summary>
    public static ModelInfo MultilingualE5Small { get; } = new()
    {
        RepoId = "intfloat/multilingual-e5-small",
        AliasName = "multilingual-e5-small",
        Dimensions = 384,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "118M params, 100+ languages, compact",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    /// <summary>
    /// multilingual-e5-base, 278M params, 100+ languages, quality.
    /// </summary>
    public static ModelInfo MultilingualE5Base { get; } = new()
    {
        RepoId = "intfloat/multilingual-e5-base",
        AliasName = "multilingual-e5-base",
        Dimensions = 768,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "278M params, 100+ languages, quality",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    /// <summary>
    /// multilingual-e5-large, 560M params, 100+ languages, highest quality.
    /// </summary>
    public static ModelInfo MultilingualE5Large { get; } = new()
    {
        RepoId = "intfloat/multilingual-e5-large",
        AliasName = "multilingual-e5-large",
        Dimensions = 1024,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        Description = "560M params, 100+ languages, highest quality",
        Subfolder = "onnx",
        QueryPrefix = "query: ",
        PassagePrefix = "passage: "
    };

    /// <summary>
    /// gte-large-en-v1.5, 434M params, 8K context, highest accuracy GTE.
    /// </summary>
    public static ModelInfo GteLargeEnV15 { get; } = new()
    {
        RepoId = "Alibaba-NLP/gte-large-en-v1.5",
        AliasName = "gte-large-en-v1.5",
        Dimensions = 1024,
        MaxSequenceLength = 8192,
        PoolingMode = PoolingMode.Cls,
        DoLowerCase = false,
        Description = "434M params, 8K context, highest accuracy GTE",
        Subfolder = "onnx"
    };

    /// <summary>
    /// Gets all built-in models.
    /// </summary>
    public static IReadOnlyList<ModelInfo> All { get; } =
    [
        // Alias models (4 standard aliases)
        BgeM3DefaultAlias,              // default
        MultilingualE5SmallAlias,       // fast
        BgeM3QualityAlias,              // quality
        MultilingualE5LargeAlias,       // large

        // Explicit models (by short name)
        NomicEmbedTextV15,
        AllMpnetBaseV2,
        BgeBaseEnV15,
        BgeLargeEnV15,
        E5SmallV2,
        E5BaseV2,
        MultilingualE5Small,
        MultilingualE5Base,
        MultilingualE5Large,
        GteLargeEnV15,
    ];
}
