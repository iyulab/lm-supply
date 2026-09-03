using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Llama.Server;

namespace LMSupply.Generator;

/// <summary>
/// Fluent builder for creating text generator instances.
/// </summary>
public sealed class TextGeneratorBuilder
{
    private string? _modelPath;
    private string? _modelId;
    private GeneratorOptions _modelOptions = new();
    private Models.GenerationOptions? _defaultGenerationOptions;
    private GeneratorPoolOptions? _poolOptions;
    private MemoryAwareOptions? _memoryOptions;

    /// <summary>
    /// Creates a new TextGeneratorBuilder.
    /// </summary>
    public static TextGeneratorBuilder Create() => new();

    /// <summary>
    /// Sets the local model path.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model directory.</param>
    public TextGeneratorBuilder WithModelPath(string modelPath)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        _modelId = null;
        return this;
    }

    /// <summary>
    /// Sets the HuggingFace model ID to download.
    /// </summary>
    /// <param name="modelId">HuggingFace model identifier (e.g., "microsoft/Phi-3.5-mini-instruct-onnx").</param>
    public TextGeneratorBuilder WithHuggingFaceModel(string modelId)
    {
        _modelId = modelId ?? throw new ArgumentNullException(nameof(modelId));
        _modelPath = null;
        return this;
    }

    /// <summary>
    /// Uses the platform-aware default — hardware-aware auto selection.
    /// Resolves to Gemma 4 GGUF on NVIDIA/CPU/macOS/Linux and Phi-4 Mini ONNX on Windows DirectML + non-NVIDIA.
    /// </summary>
    /// <remarks>
    /// At <see cref="BuildAsync"/> time the builder delegates to
    /// <see cref="LocalGenerator.LoadAsync(string, GeneratorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>
    /// for hardware-aware GGUF/ONNX dispatch, rather than going through the ONNX-only factory.
    /// </remarks>
    public TextGeneratorBuilder WithDefaultModel()
    {
        _modelId = WellKnownModels.Generator.Default;
        _modelPath = null;
        return this;
    }

    /// <summary>
    /// Uses a well-known model by preset.
    /// </summary>
    /// <param name="preset">Model preset from WellKnownModels.</param>
    public TextGeneratorBuilder WithModel(GeneratorModelPreset preset)
    {
        _modelId = preset switch
        {
            GeneratorModelPreset.Default => WellKnownModels.Generator.Default,
            GeneratorModelPreset.Fast => WellKnownModels.Generator.Fast,
            GeneratorModelPreset.Quality => WellKnownModels.Generator.Quality,
            GeneratorModelPreset.Small => WellKnownModels.Generator.Small,
            _ => throw new ArgumentOutOfRangeException(nameof(preset))
        };
        _modelPath = null;
        return this;
    }

    /// <summary>
    /// Sets the execution provider.
    /// </summary>
    /// <param name="provider">Execution provider (Auto, Cpu, Cuda, DirectML, CoreML).</param>
    public TextGeneratorBuilder WithProvider(ExecutionProvider provider)
    {
        _modelOptions.Provider = provider;
        return this;
    }

    /// <summary>
    /// Sets the cache directory for downloaded models.
    /// </summary>
    /// <param name="cacheDirectory">Directory path for model cache.</param>
    public TextGeneratorBuilder WithCacheDirectory(string cacheDirectory)
    {
        _modelOptions.CacheDirectory = cacheDirectory;
        return this;
    }

    /// <summary>
    /// Sets the maximum context length.
    /// </summary>
    /// <param name="maxContextLength">Maximum context length in tokens.</param>
    public TextGeneratorBuilder WithMaxContextLength(int maxContextLength)
    {
        _modelOptions.MaxContextLength = maxContextLength;
        return this;
    }

    /// <summary>
    /// Sets the maximum concurrent requests.
    /// </summary>
    /// <param name="maxConcurrentRequests">Maximum concurrent generation requests.</param>
    public TextGeneratorBuilder WithConcurrency(int maxConcurrentRequests)
    {
        _modelOptions.MaxConcurrentRequests = maxConcurrentRequests;
        return this;
    }

    /// <summary>
    /// Sets fine-grained llama-server tuning options (GPU offload, batch size, RoPE/YaRN
    /// scaling, KV cache quantization, speculative decoding, LoRA, raw CLI passthrough, etc.).
    /// Only used by the GGUF/llama-server path; ignored for ONNX models.
    /// </summary>
    /// <param name="llamaOptions">Advanced llama-server options.</param>
    public TextGeneratorBuilder WithLlamaOptions(LlamaOptions llamaOptions)
    {
        _modelOptions.LlamaOptions = llamaOptions ?? throw new ArgumentNullException(nameof(llamaOptions));
        return this;
    }

    /// <summary>
    /// Sets llama-server acquisition policy (version pinning, a pre-provisioned binary path,
    /// auto-update behavior) for GGUF model loading. Only used by the GGUF/llama-server path;
    /// ignored for ONNX models.
    /// </summary>
    /// <param name="serverUpdateOptions">llama-server acquisition options.</param>
    public TextGeneratorBuilder WithServerUpdateOptions(LlamaServerUpdateOptions serverUpdateOptions)
    {
        _modelOptions.ServerUpdateOptions = serverUpdateOptions ?? throw new ArgumentNullException(nameof(serverUpdateOptions));
        return this;
    }

    /// <summary>
    /// Enables verbose logging.
    /// </summary>
    public TextGeneratorBuilder WithVerboseLogging()
    {
        _modelOptions.Verbose = true;
        return this;
    }

    /// <summary>
    /// Sets the chat format explicitly.
    /// </summary>
    /// <param name="chatFormat">Chat format identifier (phi3, llama3, chatml, gemma).</param>
    public TextGeneratorBuilder WithChatFormat(string chatFormat)
    {
        _modelOptions.ChatFormat = chatFormat;
        return this;
    }

    /// <summary>
    /// Sets default generation options.
    /// </summary>
    /// <param name="options">Default generation options.</param>
    public TextGeneratorBuilder WithDefaultOptions(Models.GenerationOptions options)
    {
        _defaultGenerationOptions = options;
        return this;
    }

    /// <summary>
    /// Configures generation for creative outputs.
    /// </summary>
    public TextGeneratorBuilder ForCreativeGeneration()
    {
        _defaultGenerationOptions = Models.GenerationOptions.Creative;
        return this;
    }

    /// <summary>
    /// Configures generation for precise/deterministic outputs.
    /// </summary>
    public TextGeneratorBuilder ForPreciseGeneration()
    {
        _defaultGenerationOptions = Models.GenerationOptions.Precise;
        return this;
    }

    /// <summary>
    /// Enables model pooling for multi-model scenarios.
    /// </summary>
    /// <param name="options">Optional pool configuration.</param>
    public TextGeneratorBuilder WithPooling(GeneratorPoolOptions? options = null)
    {
        _poolOptions = options ?? new GeneratorPoolOptions();
        return this;
    }

    /// <summary>
    /// Enables memory-aware generation with the specified limit in gigabytes.
    /// </summary>
    /// <param name="limitGB">Maximum memory usage in gigabytes.</param>
    public TextGeneratorBuilder WithMemoryLimit(double limitGB)
    {
        _memoryOptions = MemoryAwareOptions.WithLimitGB(limitGB);
        return this;
    }

    /// <summary>
    /// Enables memory-aware generation with custom options.
    /// </summary>
    /// <param name="options">Memory management options.</param>
    public TextGeneratorBuilder WithMemoryManagement(MemoryAwareOptions options)
    {
        _memoryOptions = options ?? throw new ArgumentNullException(nameof(options));
        return this;
    }

    /// <summary>
    /// Builds the text generator asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The configured text generator.</returns>
    public async Task<IGeneratorModel> BuildAsync(CancellationToken cancellationToken = default)
    {
        // Validate that a model is configured before downloading runtime
        if (string.IsNullOrEmpty(_modelPath) && string.IsNullOrEmpty(_modelId))
        {
            throw new InvalidOperationException(
                "Model path or ID is required. Use WithModelPath(), WithHuggingFaceModel(), or WithDefaultModel().");
        }

        // Aliases that need hardware-aware / GGUF routing must go through
        // LocalGenerator.LoadAsync rather than the ONNX factory. The ONNX factory
        // would treat the literal alias (e.g. "default") as a HuggingFace repo id
        // and 401 against huggingface.co/default, while LocalGenerator dispatches
        // to GGUF aliases or the auto-selection path. See ISSUE-LMSupply.Generator-
        // 20260429-000657 for the diagnosis.
        if (!string.IsNullOrEmpty(_modelId) && NeedsLocalGeneratorRouting(_modelId))
        {
            var localOptions = BuildLocalGeneratorOptions();
            IGeneratorModel localGenerator = await LocalGenerator.LoadAsync(
                _modelId, localOptions, progress: null, cancellationToken).ConfigureAwait(false);

            // Layer memory management on top in the same way as the ONNX path.
            if (_memoryOptions != null)
            {
                localGenerator = new MemoryAwareGenerator(localGenerator, _memoryOptions);
            }

            return localGenerator;
        }

        // Ensure GenAI runtime binaries are available before loading the model
        // This downloads onnxruntime and onnxruntime-genai native binaries on first use
        await OnnxGeneratorBackendRegistry.Require().EnsureRuntimeAsync(
            _modelOptions.Provider,
            progress: null,
            cancellationToken);

        var modelPath = await ResolveModelPathAsync(cancellationToken);
        var modelId = _modelId ?? Path.GetFileName(modelPath);
        var chatFormatter = ResolveChatFormatter(modelId);

        IGeneratorModel generator = OnnxGeneratorBackendRegistry.Require().CreateModel(modelId, modelPath, chatFormatter, _modelOptions);

        // Wrap with memory management if configured
        if (_memoryOptions != null)
        {
            generator = new MemoryAwareGenerator(generator, _memoryOptions);
        }

        return generator;
    }

    /// <summary>
    /// Returns <c>true</c> when <paramref name="modelId"/> must be resolved through
    /// <see cref="LocalGenerator.LoadAsync(string, GeneratorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>
    /// rather than the ONNX factory. Covers the platform-aware default ("default"/"auto")
    /// and any GGUF registry alias (prefixed with "gguf:").
    /// </summary>
    private static bool NeedsLocalGeneratorRouting(string modelId) =>
        modelId.Equals("default", StringComparison.OrdinalIgnoreCase) ||
        modelId.Equals("auto", StringComparison.OrdinalIgnoreCase) ||
        modelId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Builds a <see cref="GeneratorOptions"/> for <see cref="LocalGenerator.LoadAsync"/>
    /// from the builder's accumulated options. Mirrors the fields the ONNX path consumes
    /// so the two paths behave consistently.
    /// </summary>
    private GeneratorOptions BuildLocalGeneratorOptions() => new()
    {
        CacheDirectory = _modelOptions.CacheDirectory,
        Provider = _modelOptions.Provider,
        ChatFormat = _modelOptions.ChatFormat,
        Verbose = _modelOptions.Verbose,
        MaxContextLength = _modelOptions.MaxContextLength,
        MaxConcurrentRequests = _modelOptions.MaxConcurrentRequests,
        LlamaOptions = _modelOptions.LlamaOptions,
        QuantizationHint = _modelOptions.QuantizationHint,
        ServerUpdateOptions = _modelOptions.ServerUpdateOptions,
        PreferredAutoModelId = _modelOptions.PreferredAutoModelId,
    };

    /// <summary>
    /// Builds a generator pool for managing multiple models.
    /// </summary>
    /// <param name="factory">Model factory for creating generator instances.</param>
    /// <returns>The configured generator pool.</returns>
    public GeneratorPool BuildPool(IGeneratorModelFactory factory)
    {
        return new GeneratorPool(factory, _poolOptions ?? new GeneratorPoolOptions());
    }

    private async Task<string> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        if (!string.IsNullOrEmpty(_modelPath))
        {
            return _modelPath;
        }

        if (!string.IsNullOrEmpty(_modelId))
        {
            // Use factory to resolve/download model
            var cacheDir = _modelOptions.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var factory = OnnxGeneratorBackendRegistry.Require().CreateFactory(cacheDir, _modelOptions.Provider);

            // Download if not available
            if (!factory.IsModelAvailable(_modelId))
            {
                await factory.DownloadModelAsync(_modelId, cancellationToken: cancellationToken);
            }

            // Return cache path after download
            return factory.GetModelCachePath(_modelId);
        }

        return string.Empty;
    }

    private IChatFormatter ResolveChatFormatter(string modelId)
    {
        if (!string.IsNullOrEmpty(_modelOptions.ChatFormat))
        {
            return ChatFormatterFactory.Create(_modelOptions.ChatFormat);
        }

        // Auto-detect from model ID
        return ChatFormatterFactory.Create(modelId);
    }
}

/// <summary>
/// Preset model configurations.
/// </summary>
public enum GeneratorModelPreset
{
    /// <summary>Balanced model - Phi-3.5 Mini (MIT license).</summary>
    Default,

    /// <summary>Fast/small model - Llama 3.2 1B.</summary>
    Fast,

    /// <summary>Quality model - Phi-4.</summary>
    Quality,

    /// <summary>Smallest model - Llama 3.2 1B.</summary>
    Small
}
