using LocalAI.Generator.Abstractions;
using LocalAI.Generator.Models;
using Microsoft.ML.OnnxRuntimeGenAI;
using OnnxGenerator = Microsoft.ML.OnnxRuntimeGenAI.Generator;

namespace LocalAI.Generator.Internal;

/// <summary>
/// ONNX Runtime GenAI implementation of the text generator.
/// </summary>
internal sealed class OnnxGeneratorModel : IGeneratorModel
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorModelOptions _options;
    private readonly string _modelPath;
    private bool _disposed;

    public OnnxGeneratorModel(
        string modelId,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorModelOptions options)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _chatFormatter = chatFormatter;
        _options = options;

        // Load model and tokenizer
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);

        // TODO: Detect from model config
        MaxContextLength = options.MaxContextLength ?? 4096;
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
        GeneratorOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GeneratorOptions.Default;

        var sequences = _tokenizer.Encode(prompt);
        using var generatorParams = CreateGeneratorParams(options);

        using var tokenizerStream = _tokenizer.CreateStream();
        using var generator = new OnnxGenerator(_model, generatorParams);
        generator.AppendTokenSequences(sequences);

        while (!generator.IsDone())
        {
            cancellationToken.ThrowIfCancellationRequested();

            generator.GenerateNextToken();

            var outputTokens = generator.GetSequence(0);
            var newToken = outputTokens[^1];
            var decoded = tokenizerStream.Decode(newToken);

            // Check stop sequences
            if (ShouldStop(decoded, options.StopSequences))
            {
                yield break;
            }

            yield return decoded;

            // Yield to allow other tasks
            await Task.Yield();
        }
    }

    /// <inheritdoc />
    public IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.FormatPrompt(messages);

        // Merge stop sequences from formatter
        options ??= GeneratorOptions.Default;
        var mergedOptions = MergeStopSequences(options);

        return GenerateAsync(prompt, mergedOptions, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<string> GenerateCompleteAsync(
        string prompt,
        GeneratorOptions? options = null,
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
        GeneratorOptions? options = null,
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
        return GenerateCompleteAsync(
            "Hello",
            new GeneratorOptions { MaxTokens = 1 },
            cancellationToken);
    }

    /// <inheritdoc />
    public GeneratorModelInfo GetModelInfo() => new(
        ModelId,
        _modelPath,
        MaxContextLength,
        _chatFormatter.FormatName,
        "CPU"); // TODO: Detect actual provider

    private GeneratorParams CreateGeneratorParams(GeneratorOptions options)
    {
        var generatorParams = new GeneratorParams(_model);

        generatorParams.SetSearchOption("max_length", options.MaxTokens);
        generatorParams.SetSearchOption("temperature", options.Temperature);
        generatorParams.SetSearchOption("top_p", options.TopP);
        generatorParams.SetSearchOption("top_k", options.TopK);
        generatorParams.SetSearchOption("repetition_penalty", options.RepetitionPenalty);

        return generatorParams;
    }

    private GeneratorOptions MergeStopSequences(GeneratorOptions options)
    {
        var formatterStops = _chatFormatter.GetStopSequences();
        var userStops = options.StopSequences ?? [];

        var merged = new List<string>(formatterStops);
        merged.AddRange(userStops);

        return new GeneratorOptions
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK,
            RepetitionPenalty = options.RepetitionPenalty,
            StopSequences = merged,
            IncludePromptInOutput = options.IncludePromptInOutput
        };
    }

    private static bool ShouldStop(string token, IReadOnlyList<string>? stopSequences)
    {
        if (stopSequences == null || stopSequences.Count == 0)
        {
            return false;
        }

        foreach (var stop in stopSequences)
        {
            if (token.Contains(stop, StringComparison.Ordinal))
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
        _tokenizer.Dispose();
        _model.Dispose();

        return ValueTask.CompletedTask;
    }
}
