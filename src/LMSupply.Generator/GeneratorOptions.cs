using LMSupply.Llama.Server;

namespace LMSupply.Generator;

/// <summary>
/// Configuration options for loading a generator model.
/// </summary>
public sealed class GeneratorOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the chat format to use.
    /// If null, the format is auto-detected from the model name.
    /// </summary>
    public string? ChatFormat { get; set; }

    /// <summary>
    /// Gets or sets whether to log detailed information during model loading.
    /// Defaults to false.
    /// </summary>
    public bool Verbose { get; set; }

    /// <summary>
    /// Gets or sets the maximum context length to use.
    /// If null, uses the model's default context length.
    /// </summary>
    public int? MaxContextLength { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of concurrent generation requests.
    /// Used to prevent resource exhaustion during high load.
    /// Defaults to 1 (sequential processing) for stability.
    /// </summary>
    public int MaxConcurrentRequests { get; set; } = 1;

    /// <summary>
    /// Gets or sets advanced options for GGUF model loading via llama-server.
    /// If null, optimal settings are automatically determined based on hardware.
    /// Only applies to GGUF models (gguf:* aliases or .gguf files).
    /// </summary>
    public LlamaOptions? LlamaOptions { get; set; }

    /// <summary>
    /// When loading with <c>"auto"</c> or <c>"default"</c>, specifies a preferred model ID to try
    /// before hardware-aware auto-selection. Acts as a quality floor: if the preferred model loads
    /// successfully it is returned; if it fails (unavailable, download error, etc.) auto-selection
    /// falls back to hardware-aware selection with a trace warning.
    /// <para>
    /// Accepts the same model ID formats as <see cref="LocalGenerator.LoadAsync"/>
    /// (e.g., <c>"gguf:phi-4-mini"</c>, <c>"gguf:qwen2.5-7b"</c>, a HuggingFace repo ID,
    /// or a local path).
    /// </para>
    /// <para>Default: null (standard hardware-aware selection).</para>
    /// </summary>
    public string? PreferredAutoModelId { get; set; }

    /// <summary>
    /// Gets or sets llama-server acquisition policy (version pinning, a pre-provisioned binary
    /// path, auto-update behavior) for GGUF model loading.
    /// If null, the process-wide default (<see cref="LlamaServerUpdateService.Instance"/>,
    /// unpinned "latest" resolution) is used. Only applies to GGUF models.
    /// </summary>
    public LlamaServerUpdateOptions? ServerUpdateOptions { get; set; }

    /// <summary>
    /// Gets or sets whether to disable automatic model download.
    /// When true, loading uses only the local cache and throws <see cref="LMSupply.Exceptions.ModelNotFoundException"/>
    /// if a model file is not there, writing nothing to the cache.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>
    /// A copy with every option carried over (<see cref="LlamaOptions"/> and <see cref="ServerUpdateOptions"/>
    /// are shared, not copied). Load paths that need to adjust one option — a factory applying its default
    /// provider — copy first so the caller's instance is left alone and no option is dropped on the way.
    /// </summary>
    public GeneratorOptions Clone() => new()
    {
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        ThreadCount = ThreadCount,
        LogLevel = LogLevel,
        QuantizationHint = QuantizationHint,
        ChatFormat = ChatFormat,
        Verbose = Verbose,
        MaxContextLength = MaxContextLength,
        MaxConcurrentRequests = MaxConcurrentRequests,
        LlamaOptions = LlamaOptions,
        PreferredAutoModelId = PreferredAutoModelId,
        ServerUpdateOptions = ServerUpdateOptions,
        DisableAutoDownload = DisableAutoDownload
    };
}
