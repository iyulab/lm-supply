namespace LMSupply.Generator;

/// <summary>
/// Well-known model identifiers for LMSupply components.
/// Updated: 2025-01 based on MTEB leaderboard and community benchmarks.
/// </summary>
public static class WellKnownModels
{
    /// <summary>
    /// Text generation models (ONNX Runtime GenAI).
    /// </summary>
    public static class Generator
    {
        /// <summary>
        /// Default balanced model - Microsoft Phi-4 Mini (MIT license).
        /// 3.8B parameters, 16K context, excellent reasoning for its size.
        /// Released: 2025-01, successor to Phi-3.5 Mini.
        /// </summary>
        public const string Default = "microsoft/Phi-4-mini-instruct-onnx";

        /// <summary>
        /// Fast/small model - Meta Llama 3.2 1B.
        /// 1B parameters, fast inference, good for simple tasks.
        /// Note: Llama Community License (700M MAU limit).
        /// </summary>
        public const string Fast = "onnx-community/Llama-3.2-1B-Instruct-ONNX";

        /// <summary>
        /// Small model - same as Fast, alias for clarity.
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
        /// Large model - Meta Llama 3.2 3B.
        /// 3B parameters, good balance of speed and quality.
        /// Note: Llama Community License (700M MAU limit).
        /// </summary>
        public const string Large = "onnx-community/Llama-3.2-3B-Instruct-ONNX";

        /// <summary>
        /// Multilingual model - Google Gemma 2 2B.
        /// 2B parameters, good multilingual support.
        /// Note: Gemma Terms of Use apply.
        /// </summary>
        public const string Multilingual = "google/gemma-2-2b-it-onnx";

        /// <summary>
        /// Legacy default - Phi-3.5 Mini for backward compatibility.
        /// </summary>
        public const string Phi35Mini = "microsoft/Phi-3.5-mini-instruct-onnx";
    }

    /// <summary>
    /// Embedding models (from LMSupply.Embedder).
    /// Updated: 2025-01 based on MTEB leaderboard rankings.
    /// </summary>
    public static class Embedder
    {
        /// <summary>
        /// Default embedding model - BGE Small English v1.5.
        /// 33M params, 384 dims, 512 tokens. MTEB top performer for size.
        /// Best balance of speed, size, and quality for English.
        /// </summary>
        public const string Default = "BAAI/bge-small-en-v1.5";

        /// <summary>
        /// Fast/tiny embedding model - all-MiniLM-L6-v2.
        /// 22M params, 384 dims, 256 tokens. Ultra-lightweight.
        /// Best for latency-critical applications.
        /// </summary>
        public const string Fast = "sentence-transformers/all-MiniLM-L6-v2";

        /// <summary>
        /// Quality embedding model - BGE Base English v1.5.
        /// 110M params, 768 dims, 512 tokens. Higher accuracy.
        /// Good balance of quality and resource usage.
        /// </summary>
        public const string Quality = "BAAI/bge-base-en-v1.5";

        /// <summary>
        /// Large embedding model - Nomic Embed Text v1.5.
        /// 137M params, 768 dims, 8192 tokens. Long context support.
        /// 2024 MTEB top performer, excellent for RAG.
        /// </summary>
        public const string Large = "nomic-ai/nomic-embed-text-v1.5";

        /// <summary>
        /// Multilingual embedding model - E5 Base.
        /// 278M params, 768 dims, 512 tokens. 100+ languages.
        /// Best open-source multilingual option.
        /// </summary>
        public const string Multilingual = "intfloat/multilingual-e5-base";

        /// <summary>
        /// Best multilingual embedding - BGE M3.
        /// 568M params, 1024 dims, 8192 tokens. Largest and most accurate.
        /// Supports dense, sparse, and multi-vector retrieval.
        /// </summary>
        public const string MultilingualLarge = "BAAI/bge-m3";
    }

    /// <summary>
    /// Reranking models (from LMSupply.Reranker).
    /// Updated: 2025-01 based on BEIR and MS MARCO benchmarks.
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
        /// 2024 release, multilingual support.
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
        var info = ModelRegistry.GetModel(modelId);
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
    /// </summary>
    public static IReadOnlyList<string> GetUnrestrictedModels() =>
    [
        Generator.Default,
        Generator.Quality
    ];
}
