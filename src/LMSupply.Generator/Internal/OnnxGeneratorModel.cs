using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntimeGenAI;
using OnnxGenerator = Microsoft.ML.OnnxRuntimeGenAI.Generator;

namespace LMSupply.Generator.Internal;

/// <summary>
/// ONNX Runtime GenAI implementation of the text generator.
/// </summary>
internal sealed class OnnxGeneratorModel : IGeneratorModel
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorOptions _options;
    private readonly string _modelPath;
    private readonly ExecutionProvider _resolvedProvider;
    private readonly SemaphoreSlim _concurrencyLimiter;
    private bool _disposed;

    public OnnxGeneratorModel(
        string modelId,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorOptions options)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _chatFormatter = chatFormatter;
        _options = options;

        // Resolve Auto to actual provider using hardware detection
        _resolvedProvider = HardwareDetector.ResolveProvider(options.Provider);

        // Load model and tokenizer
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);

        // Initialize concurrency limiter
        _concurrencyLimiter = new SemaphoreSlim(
            Math.Max(1, options.MaxConcurrentRequests),
            Math.Max(1, options.MaxConcurrentRequests));

        // Detect max context length from model config, or use provided/default value
        MaxContextLength = options.MaxContextLength
            ?? GenAiConfigReader.ReadMaxContextLength(modelPath);
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int MaxContextLength { get; }

    /// <inheritdoc />
    public IChatFormatter ChatFormatter => _chatFormatter;

    /// <inheritdoc />
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        // Acquire concurrency slot
        await _concurrencyLimiter.WaitAsync(cancellationToken);
        try
        {
            var sequences = _tokenizer.Encode(prompt);
            using var generatorParams = CreateGeneratorParams(options);

            using var tokenizerStream = _tokenizer.CreateStream();
            using var generator = new OnnxGenerator(_model, generatorParams);
            generator.AppendTokenSequences(sequences);

            var generatedTokenCount = 0;
            var maxNewTokens = options.MaxNewTokens ?? int.MaxValue;

            // Accumulate generated text for multi-token stop sequence detection
            var accumulatedText = new StringBuilder();

            while (!generator.IsDone())
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Check MaxNewTokens limit
                if (generatedTokenCount >= maxNewTokens)
                {
                    yield break;
                }

                generator.GenerateNextToken();
                generatedTokenCount++;

                var outputTokens = generator.GetSequence(0);
                var newToken = outputTokens[^1];
                var decoded = tokenizerStream.Decode(newToken);

                // Append to accumulated text for stop sequence detection
                accumulatedText.Append(decoded);

                // Check stop sequences using accumulated text (handles multi-token sequences)
                if (ShouldStop(accumulatedText.ToString(), options.StopSequences))
                {
                    yield break;
                }

                yield return decoded;

                // Yield to allow other tasks
                await Task.Yield();
            }
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

    /// <inheritdoc />
    public IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.FormatPrompt(messages);

        // Merge stop sequences from formatter
        options ??= GenerationOptions.Default;
        var mergedOptions = MergeStopSequences(options);

        return GenerateAsync(prompt, mergedOptions, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<string> GenerateCompleteAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateAsync(prompt, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public async Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateChatAsync(messages, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Perform a minimal generation to warm up the model
        // Note: MaxTokens must be > 1 for DirectML compatibility (shape mismatch bug with MaxTokens=1)
        return GenerateCompleteAsync(
            "Hi",
            new GenerationOptions { MaxTokens = 5 },
            cancellationToken);
    }

    /// <inheritdoc />
    public GeneratorModelInfo GetModelInfo() => new(
        ModelId,
        _modelPath,
        MaxContextLength,
        _chatFormatter.FormatName,
        _resolvedProvider.ToString());

    private GeneratorParams CreateGeneratorParams(GenerationOptions options)
    {
        var generatorParams = new GeneratorParams(_model);

        generatorParams.SetSearchOption("max_length", options.MaxTokens);
        generatorParams.SetSearchOption("temperature", options.Temperature);
        generatorParams.SetSearchOption("top_p", options.TopP);
        generatorParams.SetSearchOption("top_k", options.TopK);
        generatorParams.SetSearchOption("repetition_penalty", options.RepetitionPenalty);
        generatorParams.SetSearchOption("do_sample", options.DoSample);

        // Beam search configuration
        if (options.NumBeams > 1)
        {
            generatorParams.SetSearchOption("num_beams", options.NumBeams);
            // Beam search requires separate past/present buffers
            generatorParams.SetSearchOption("past_present_share_buffer", false);
        }
        else
        {
            generatorParams.SetSearchOption("past_present_share_buffer", options.PastPresentShareBuffer);
        }

        return generatorParams;
    }

    private GenerationOptions MergeStopSequences(GenerationOptions options)
    {
        var merged = new List<string>();
        
        // 1. Stop sequences from chat formatter (model-specific tokens like <|end|>)
        merged.AddRange(_chatFormatter.GetStopSequences());
        
        // 2. Stop sequences from config file (if defined in genai_config.json)
        var configStops = GenAiConfigReader.ReadStopSequences(_modelPath);
        foreach (var stop in configStops)
        {
            if (!merged.Contains(stop, StringComparer.Ordinal))
            {
                merged.Add(stop);
            }
        }
        
        // 3. User-provided stop sequences
        var userStops = options.StopSequences ?? [];
        foreach (var stop in userStops)
        {
            if (!merged.Contains(stop, StringComparer.Ordinal))
            {
                merged.Add(stop);
            }
        }

        return new GenerationOptions
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK,
            RepetitionPenalty = options.RepetitionPenalty,
            StopSequences = merged,
            IncludePromptInOutput = options.IncludePromptInOutput,
            DoSample = options.DoSample,
            NumBeams = options.NumBeams,
            PastPresentShareBuffer = options.PastPresentShareBuffer,
            MaxNewTokens = options.MaxNewTokens
        };
    }

    private static bool ShouldStop(string accumulatedText, IReadOnlyList<string>? stopSequences)
    {
        if (stopSequences == null || stopSequences.Count == 0)
        {
            return false;
        }

        // Check if any stop sequence appears at the end of accumulated text
        // This handles multi-token stop sequences like "<|end|>" properly
        foreach (var stop in stopSequences)
        {
            if (accumulatedText.EndsWith(stop, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return ValueTask.CompletedTask;
        }

        _disposed = true;
        _concurrencyLimiter.Dispose();
        _tokenizer.Dispose();
        _model.Dispose();

        return ValueTask.CompletedTask;
    }
}
