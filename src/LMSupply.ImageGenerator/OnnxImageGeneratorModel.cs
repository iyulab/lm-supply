using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.Core;
using LMSupply.ImageGenerator.Models;
using LMSupply.ImageGenerator.Pipeline;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.ImageGenerator;

/// <summary>
/// ONNX-based image generator model implementation using LCM pipeline.
/// </summary>
internal sealed class OnnxImageGeneratorModel : IImageGeneratorModel
{
    private readonly LcmPipeline _pipeline;
    private readonly ModelDefinition _modelDefinition;
    private readonly ImageGeneratorOptions _options;
    private readonly string _modelPath;
    private bool _disposed;
    private bool _warmedUp;

    public string ModelId => _modelDefinition.RepoId;

    public long? EstimatedMemoryBytes => Directory.Exists(_modelPath)
        ? Directory.GetFiles(_modelPath, "*.onnx", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length) * 2
        : null;

    private OnnxImageGeneratorModel(
        LcmPipeline pipeline,
        ModelDefinition modelDefinition,
        ImageGeneratorOptions options,
        string modelPath)
    {
        _pipeline = pipeline;
        _modelDefinition = modelDefinition;
        _options = options;
        _modelPath = modelPath;
    }

    /// <summary>
    /// Loads an image generator model.
    /// </summary>
    public static async Task<OnnxImageGeneratorModel> LoadAsync(
        ModelDefinition modelDefinition,
        string modelPath,
        ImageGeneratorOptions options,
        CancellationToken cancellationToken = default)
    {
        // Provider selection (Auto chain, DirectML settings, runtime provisioning) is the shared
        // OnnxSessionFactory's job; this callback only carries the caller's log level and threads.
        var pipeline = await LcmPipeline.LoadAsync(
            modelPath,
            options.Provider,
            options.DeviceId,
            sessionOptions =>
            {
                sessionOptions.LogSeverityLevel = (OrtLoggingLevel)(int)options.LogLevel;
                if (options.ThreadCount.HasValue)
                {
                    sessionOptions.IntraOpNumThreads = options.ThreadCount.Value;
                }
            },
            cancellationToken);

        return new OnnxImageGeneratorModel(pipeline, modelDefinition, options, modelPath);
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        if (_warmedUp) return;

        // Run a small test generation to warm up the model
        var warmupOptions = new GenerationOptions
        {
            Width = 64,
            Height = 64,
            Steps = 1,
            GuidanceScale = 1.0f,
            Seed = 0
        };

        try
        {
            _ = await _pipeline.GenerateAsync("warmup", warmupOptions, cancellationToken);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[OnnxImageGeneratorModel] Warmup failed: {ex.Message}");
        }

        _warmedUp = true;
    }

    public ImageGeneratorModelInfo? GetModelInfo()
    {
        return new ImageGeneratorModelInfo
        {
            ModelId = _modelDefinition.RepoId,
            AliasName = _modelDefinition.FriendlyName ?? _modelDefinition.RepoId,
            ModelName = _modelDefinition.FriendlyName,
            Description = $"LCM image generator ({_modelDefinition.RecommendedSteps} steps)",
            Architecture = "LCM",
            Provider = _options.Provider,
            IsFp16 = _options.UseFp16,
            DefaultWidth = 512,
            DefaultHeight = 512,
            RecommendedSteps = _modelDefinition.RecommendedSteps,
            RecommendedGuidanceScale = _modelDefinition.RecommendedGuidanceScale,
            ModelPath = _modelPath
        };
    }

    public async Task<GeneratedImage> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

        options ??= new GenerationOptions();
        options.Validate();

        // Apply model-specific defaults
        var effectiveOptions = ApplyModelDefaults(options);

        return await _pipeline.GenerateAsync(prompt, effectiveOptions, cancellationToken);
    }

    public async Task<GeneratedImage[]> GenerateBatchAsync(
        string prompt,
        int count,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);

        options ??= new GenerationOptions();
        options.Validate();

        var results = new GeneratedImage[count];
        var baseSeed = options.Seed ?? Random.Shared.Next();

        for (int i = 0; i < count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var iterOptions = options.WithSeed(baseSeed + i);
            var effectiveOptions = ApplyModelDefaults(iterOptions);
            results[i] = await _pipeline.GenerateAsync(prompt, effectiveOptions, cancellationToken);
        }

        return results;
    }

    public async IAsyncEnumerable<GenerationStep> GenerateStreamingAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

        options ??= new GenerationOptions();
        options.Validate();

        var effectiveOptions = ApplyModelDefaults(options);

        await foreach (var step in _pipeline.GenerateStreamingAsync(prompt, effectiveOptions, cancellationToken))
        {
            yield return step;
        }
    }

    private GenerationOptions ApplyModelDefaults(GenerationOptions options)
    {
        // Use model's recommended values if not explicitly set
        return new GenerationOptions
        {
            NegativePrompt = options.NegativePrompt,
            Width = options.Width,
            Height = options.Height,
            Steps = options.Steps > 0 ? options.Steps : _modelDefinition.RecommendedSteps,
            GuidanceScale = options.GuidanceScale > 0 ? options.GuidanceScale : _modelDefinition.RecommendedGuidanceScale,
            Seed = options.Seed,
            GeneratePreviews = options.GeneratePreviews
        };
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        await _pipeline.DisposeAsync();
    }
}
