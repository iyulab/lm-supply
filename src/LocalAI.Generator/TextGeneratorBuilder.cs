using LocalAI.Generator.Abstractions;
using LocalAI.Generator.ChatFormatters;

namespace LocalAI.Generator;

/// <summary>
/// Fluent builder for creating text generator instances.
/// </summary>
public sealed class TextGeneratorBuilder
{
    private string? _modelPath;
    private string? _modelId;
    private GeneratorModelOptions _modelOptions = new();
    private Models.GeneratorOptions? _defaultGeneratorOptions;
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
    /// Uses the default model (Phi-3.5-mini).
    /// </summary>
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
    /// <param name="options">Default generator options.</param>
    public TextGeneratorBuilder WithDefaultOptions(Models.GeneratorOptions options)
    {
        _defaultGeneratorOptions = options;
        return this;
    }

    /// <summary>
    /// Configures generation for creative outputs.
    /// </summary>
    public TextGeneratorBuilder ForCreativeGeneration()
    {
        _defaultGeneratorOptions = Models.GeneratorOptions.Creative;
        return this;
    }

    /// <summary>
    /// Configures generation for precise/deterministic outputs.
    /// </summary>
    public TextGeneratorBuilder ForPreciseGeneration()
    {
        _defaultGeneratorOptions = Models.GeneratorOptions.Precise;
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
        var modelPath = await ResolveModelPathAsync(cancellationToken);

        if (string.IsNullOrEmpty(modelPath))
        {
            throw new InvalidOperationException(
                "Model path or ID is required. Use WithModelPath(), WithHuggingFaceModel(), or WithDefaultModel().");
        }

        var modelId = _modelId ?? Path.GetFileName(modelPath);
        var chatFormatter = ResolveChatFormatter(modelId);

        IGeneratorModel generator = new Internal.OnnxGeneratorModel(modelId, modelPath, chatFormatter, _modelOptions);

        // Wrap with memory management if configured
        if (_memoryOptions != null)
        {
            generator = new MemoryAwareGenerator(generator, _memoryOptions);
        }

        return generator;
    }

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
            var cacheDir = _modelOptions.CacheDirectory ?? GetDefaultCacheDirectory();
            using var factory = new OnnxGeneratorModelFactory(cacheDir, _modelOptions.Provider);

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

    private static string GetDefaultCacheDirectory()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub");
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
