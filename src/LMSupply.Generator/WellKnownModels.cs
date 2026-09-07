namespace LMSupply.Generator;

/// <summary>
/// Well-known model identifiers for LMSupply components.
/// Updated: 2025-12 based on MTEB leaderboard and community benchmarks.
/// </summary>
public static class WellKnownModels
{
    /// <summary>
    /// Text generation models (ONNX Runtime GenAI).
    /// </summary>
    public static class Generator
    {
        /// <summary>
        /// Default — routes through <see cref="LocalGenerator.LoadAsync(string, GeneratorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>
        /// hardware-aware auto selection. Resolves to Gemma 4 GGUF on NVIDIA/CPU/macOS/Linux
        /// and Phi-4 Mini ONNX on Windows DirectML + non-NVIDIA.
        /// </summary>
        public const string Default = "default";

        /// <summary>
        /// Fast — Phi-4 Mini (smallest FC-capable ONNX model). Pins the ONNX path explicitly;
        /// use <see cref="Default"/> for hardware-aware selection instead.
        /// </summary>
        public const string Fast = "phi-4-mini";

        /// <summary>
        /// Small — alias for <see cref="Fast"/>.
        /// </summary>
        public const string Small = Fast;

        /// <summary>
        /// Quality model - Microsoft Phi-4 (MIT license).
        /// 14B parameters, 16K context, highest quality reasoning.
        /// Released: 2024-12, state-of-the-art for its size.
        /// </summary>
        public const string Quality = "microsoft/phi-4-onnx";

        /// <summary>
        /// Medium model - Microsoft Phi-3.5 Mini.
        /// 3.8B parameters, 128K context, excellent for long documents.
        /// Note: MIT license, predecessor to Phi-4 Mini.
        /// </summary>
        public const string Medium = "microsoft/Phi-3.5-mini-instruct-onnx";

        /// <summary>
        /// Large/quality model - Microsoft Phi-4.
        /// 14B parameters, highest quality reasoning among ONNX models.
        /// </summary>
        public const string Large = Quality;

        /// <summary>
        /// Legacy default - Phi-3.5 Mini for backward compatibility.
        /// </summary>
        public const string Phi35Mini = "microsoft/Phi-3.5-mini-instruct-onnx";
    }

    /// <summary>
    /// Embedding models (from LMSupply.Embedder).
    /// Updated: 2025-12 based on MTEB leaderboard rankings.
    /// </summary>
    public static class Embedder
    {
        // Every constant in this class must name a model the embedder registry actually carries
        // (LMSupply.Embedder's DefaultModels). A repo id the registry does not know still loads —
        // it is treated as a raw HuggingFace id and downloaded — but it arrives without the
        // registry's tuning (dimensions, pooling mode, subfolder, query/passage prefixes), so the
        // consumer silently gets a worse pipeline than the alias path would have given.
        // WellKnownModelsRegistryTests asserts this, because nothing else does.

        /// <summary>
        /// Default embedding model - BGE-M3.
        /// 568M params, 1024 dims, 8192 tokens, CLS pooling, 100+ languages.
        /// The registry's "default" alias; strongest general-purpose dense retrieval on offer here.
        /// </summary>
        public const string Default = "BAAI/bge-m3";

        /// <summary>
        /// Fast embedding model - multilingual-e5-small.
        /// 118M params, 384 dims, 512 tokens, 100+ languages. Applies the E5 query/passage
        /// prefixes automatically when loaded through the registry.
        /// Best for latency-critical applications.
        /// </summary>
        public const string Fast = "intfloat/multilingual-e5-small";

        /// <summary>
        /// Quality embedding model - GTE Large English v1.5.
        /// 434M params, 1024 dims, 8192 tokens. Long context, English-first, Apache 2.0 license.
        /// </summary>
        public const string Quality = "Alibaba-NLP/gte-large-en-v1.5";

        /// <summary>
        /// Large embedding model - Nomic Embed Text v1.5.
        /// 137M params, 768 dims, 8192 tokens. Long context support.
        /// Excellent for RAG, Apache 2.0 license.
        /// </summary>
        public const string Large = "nomic-ai/nomic-embed-text-v1.5";

        /// <summary>
        /// Multilingual embedding model - E5 Base.
        /// 278M params, 768 dims, 512 tokens. 100+ languages.
        /// Best open-source multilingual option for size.
        /// </summary>
        public const string Multilingual = "intfloat/multilingual-e5-base";

        /// <summary>
        /// Best multilingual embedding - BGE M3.
        /// 568M params, 1024 dims, 8192 tokens. Largest and most accurate.
        /// Supports dense, sparse, and multi-vector retrieval.
        /// </summary>
        public const string MultilingualLarge = "BAAI/bge-m3";

        /// <summary>
        /// Legacy quality - BGE Base English v1.5 for backward compatibility.
        /// </summary>
        public const string BgeBase = "BAAI/bge-base-en-v1.5";
    }

    /// <summary>
    /// Reranking models (from LMSupply.Reranker).
    /// Updated: 2025-12 based on BEIR and MS MARCO benchmarks.
    /// </summary>
    public static class Reranker
    {
        /// <summary>
        /// Default reranker - MS MARCO MiniLM L6.
        /// 22M params, 512 tokens. Best speed/quality balance.
        /// Proven performer on English retrieval tasks.
        /// </summary>
        public const string Default = "cross-encoder/ms-marco-MiniLM-L-6-v2";

        /// <summary>
        /// Fast reranker - MS MARCO TinyBERT L2.
        /// 4.4M params, 512 tokens. Ultra-fast inference.
        /// Best for latency-critical applications.
        /// </summary>
        public const string Fast = "cross-encoder/ms-marco-TinyBERT-L-2-v2";

        /// <summary>
        /// Quality reranker - BGE Reranker Base.
        /// 278M params, 512 tokens. Higher accuracy.
        /// Excellent balance of quality and resource usage.
        /// </summary>
        public const string Quality = "BAAI/bge-reranker-base";

        /// <summary>
        /// Large reranker - BGE Reranker Large.
        /// 560M params, 512 tokens. Highest accuracy.
        /// Best for quality-critical applications.
        /// </summary>
        public const string Large = "BAAI/bge-reranker-large";

        /// <summary>
        /// Multilingual reranker - BGE Reranker v2 M3.
        /// 568M params, 8192 tokens. 100+ languages.
        /// Best for multilingual and long-context reranking.
        /// </summary>
        public const string Multilingual = "BAAI/bge-reranker-v2-m3";

        /// <summary>
        /// Legacy quality - MS MARCO MiniLM L12 for backward compatibility.
        /// </summary>
        public const string MsMarcoL12 = "cross-encoder/ms-marco-MiniLM-L-12-v2";
    }

    /// <summary>
    /// Gets license information for a model.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>License tier classification.</returns>
    public static LicenseTier GetLicenseTier(string modelId)
    {
        GeneratorModelRegistry.Default.TryResolve(modelId, out var info);
        return info?.License ?? LicenseTier.Conditional;
    }

    /// <summary>
    /// Checks if a model has usage restrictions.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if the model has restrictions (non-MIT license).</returns>
    public static bool HasRestrictions(string modelId)
    {
        return GetLicenseTier(modelId) != LicenseTier.MIT;
    }

    /// <summary>
    /// Gets MIT-licensed models only (no usage restrictions).
    /// Uses concrete aliases so registry license lookup succeeds —
    /// the platform-aware <see cref="Generator.Default"/> alias is excluded
    /// because it resolves through the auto path and has no single license tier.
    /// </summary>
    public static IReadOnlyList<string> GetUnrestrictedModels() =>
    [
        Generator.Fast,      // phi-4-mini alias (MIT)
        Generator.Quality    // microsoft/phi-4-onnx (MIT)
    ];
}
