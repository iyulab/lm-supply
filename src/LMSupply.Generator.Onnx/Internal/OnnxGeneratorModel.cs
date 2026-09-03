using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using LMSupply.Exceptions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntimeGenAI;
using OnnxGenerator = Microsoft.ML.OnnxRuntimeGenAI.Generator;

namespace LMSupply.Generator.Internal;

/// <summary>
/// ONNX Runtime GenAI implementation of the text generator.
/// </summary>
internal sealed class OnnxGeneratorModel : IGeneratorModel, IDiagnosticsSink
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorOptions _options;
    private readonly string _modelPath;
    private readonly string? _configBasePath;
    private readonly ExecutionProvider _resolvedProvider;
    private readonly SemaphoreSlim _concurrencyLimiter;
    private SelectionDiagnostics? _diagnostics;
    private bool _disposed;

    public void SetDiagnostics(SelectionDiagnostics diagnostics) => _diagnostics = diagnostics;

    public OnnxGeneratorModel(
        string modelId,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        string? configBasePath = null)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _configBasePath = configBasePath;
        _chatFormatter = chatFormatter;
        _options = options;

        // Resolve Auto to actual provider using hardware detection
        _resolvedProvider = HardwareDetector.ResolveProvider(options.Provider);

        // Load model and tokenizer with proper cleanup on failure
        // This prevents resource leaks when Tokenizer creation fails after Model is created
        Model? model = null;
        Tokenizer? tokenizer = null;
        try
        {
            model = new Model(modelPath);
            tokenizer = new Tokenizer(model);

            _model = model;
            _tokenizer = tokenizer;
        }
        catch
        {
            tokenizer?.Dispose();
            model?.Dispose();
            throw;
        }

        // Initialize concurrency limiter
        _concurrencyLimiter = new SemaphoreSlim(
            Math.Max(1, options.MaxConcurrentRequests),
            Math.Max(1, options.MaxConcurrentRequests));

        // Detect max context length from model config, capped by hardware capability
        if (options.MaxContextLength is not null)
        {
            MaxContextLength = options.MaxContextLength.Value;
        }
        else
        {
            var configContextLength = GenAiConfigReader.ReadMaxContextLength(modelPath, configBasePath);
            var hwMaxContext = HardwareDetector.GetRecommendation().MaxContextLength;
            if (configContextLength > hwMaxContext)
            {
                Trace.TraceInformation(
                    $"[OnnxGeneratorModel] Context length capped: {configContextLength} -> {hwMaxContext} (hardware limit)");
                configContextLength = hwMaxContext;
            }
            MaxContextLength = configContextLength;
        }
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int MaxContextLength { get; }

    /// <inheritdoc />
    public IChatFormatter ChatFormatter => _chatFormatter;

    /// <inheritdoc />
    public bool IsGpuActive => _resolvedProvider is not ExecutionProvider.Cpu;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _resolvedProvider switch
    {
        ExecutionProvider.Cuda => new[] { "CUDAExecutionProvider", "CPUExecutionProvider" },
        ExecutionProvider.DirectML => new[] { "DmlExecutionProvider", "CPUExecutionProvider" },
        ExecutionProvider.CoreML => new[] { "CoreMLExecutionProvider", "CPUExecutionProvider" },
        _ => new[] { "CPUExecutionProvider" }
    };

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => Directory.Exists(_modelPath)
        ? Directory.GetFiles(_modelPath, "*.onnx").Sum(f => new FileInfo(f).Length) * 2
        : null;

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
            // NOTE: OGA has known memory leak issues, particularly with DirectML backend.
            // Even with proper disposal, some internal resources may not be fully released.
            // See: https://github.com/microsoft/onnxruntime-genai/issues/590
            // See: https://github.com/microsoft/onnxruntime-genai/issues/1677
            using var sequences = _tokenizer.Encode(prompt);

            // Use MaxContextLength as the upper bound for max_length
            // This allows the model to use its full context capacity
            var effectiveMaxLength = MaxContextLength;
            using var generatorParams = CreateGeneratorParams(options, effectiveMaxLength);

            using var tokenizerStream = _tokenizer.CreateStream();
            using var generator = new OnnxGenerator(_model, generatorParams);

            try
            {
                generator.AppendTokenSequences(sequences);
            }
            catch (Microsoft.ML.OnnxRuntimeGenAI.OnnxRuntimeGenAIException ex)
                when (ex.Message.Contains("exceeds max length"))
            {
                // Parse token count from error message: "input_ids size (2066) + current sequence length (0) exceeds max length (2048)"
                int? tokenCount = null;
                var match = System.Text.RegularExpressions.Regex.Match(ex.Message, @"input_ids size \((\d+)\)");
                if (match.Success && int.TryParse(match.Groups[1].Value, out var parsed))
                    tokenCount = parsed;

                throw new ContextLengthExceededException(tokenCount, MaxContextLength, ex);
            }

            var generatedTokenCount = 0;
            // Honor the MaxTokens safety-net contract on the ONNX path too: previously this fell back
            // to int.MaxValue when MaxNewTokens was unset, ignoring MaxTokens and letting low-end models
            // run on to the full context window. ResolveMaxOutputTokens mirrors the llama-server path.
            var maxNewTokens = options.ResolveMaxOutputTokens();

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
    public async Task<ChatCompletionResult> GenerateChatWithToolsAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        // Build messages with tool definitions injected into the system prompt
        var messageList = messages.ToList();
        if (options.Tools is { Count: > 0 })
        {
            messageList = InjectToolDefinitions(messageList, options.Tools);
        }

        // Generate the full text response
        var responseText = await GenerateChatCompleteAsync(messageList, options, cancellationToken);

        // Try to parse tool calls from the response
        var toolCalls = ToolCallTextParser.TryParse(responseText);

        if (toolCalls != null)
        {
            return new ChatCompletionResult
            {
                ToolCalls = toolCalls,
                FinishReason = "tool_calls"
            };
        }

        return new ChatCompletionResult
        {
            Content = responseText,
            FinishReason = "stop"
        };
    }

    /// <summary>
    /// Injects tool definitions into the system message so the ONNX model
    /// knows which tools are available and the expected response format.
    /// </summary>
    private static List<ChatMessage> InjectToolDefinitions(
        List<ChatMessage> messages,
        IReadOnlyList<ChatToolDefinition> tools)
    {
        var toolDescriptions = string.Join("\n", tools.Select(t =>
        {
            var parts = new List<string> { $"{{\"name\": \"{t.Name}\"" };
            if (t.Description != null)
                parts.Add($"\"description\": \"{t.Description}\"");
            if (t.Parameters.HasValue)
                parts.Add($"\"parameters\": {t.Parameters.Value}");
            return string.Join(", ", parts) + "}";
        }));

        var toolSystemPrompt =
            "You have access to the following tools. When you need to call a tool, respond ONLY with a valid JSON object in this exact format:\n" +
            "{\"tool_calls\": [{\"id\": \"call_<unique_id>\", \"type\": \"function\", \"function\": {\"name\": \"<tool_name>\", \"arguments\": \"<json_arguments>\"}}]}\n" +
            "Only call a tool when the user's request requires it. If no tool call is needed, respond normally.\n\n" +
            "Available tools:\n" +
            toolDescriptions;

        var result = new List<ChatMessage>(messages.Count + 1);

        // Find existing system message and prepend tool instructions
        var hasSystem = false;
        foreach (var msg in messages)
        {
            if (msg.Role == ChatRole.System && !hasSystem)
            {
                hasSystem = true;
                result.Add(ChatMessage.System(toolSystemPrompt + "\n\n" + msg.Content));
            }
            else
            {
                result.Add(msg);
            }
        }

        // If no system message existed, add one
        if (!hasSystem)
        {
            result.Insert(0, ChatMessage.System(toolSystemPrompt));
        }

        return result;
    }

    /// <inheritdoc />
    /// <remarks>
    /// ONNX models emit tool calls as a single JSON object in the text stream rather
    /// than via native streaming markers. To make tool calls reach upstream consumers
    /// while still supporting streamed text for plain conversational responses, this
    /// method:
    /// <list type="number">
    /// <item>Injects tool definitions into the system prompt (matching the non-streaming
    /// <see cref="GenerateChatWithToolsAsync"/> path) so the model is aware of available tools.</item>
    /// <item>Sniffs the first non-whitespace character of the model output. If it is
    /// <c>{</c>, the entire response is buffered for tool-call parsing at end of stream;
    /// otherwise tokens are forwarded immediately as streaming text.</item>
    /// <item>On stream completion, attempts to parse buffered text via
    /// <see cref="ToolCallTextParser"/>. Successful parses are emitted as
    /// <see cref="ChatToolCallDelta"/> with <c>finish_reason = "tool_calls"</c>; failed
    /// parses fall back to flushing the buffered text verbatim.</item>
    /// </list>
    /// Without this, ONNX-backed models (e.g. Phi-4-mini-instruct) silently bypass
    /// tool calling under the streaming code path that filer-host's OpenAI-compatible
    /// endpoint uses, leaving search_knowledge / RAG tools effectively unreachable.
    /// </remarks>
    public async IAsyncEnumerable<ChatStreamChunk> GenerateChatStreamAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= GenerationOptions.Default;

        var messageList = messages.ToList();
        var hasTools = options.Tools is { Count: > 0 };
        if (hasTools)
        {
            messageList = InjectToolDefinitions(messageList, options.Tools!);
        }

        var buffer = new StringBuilder();
        var bufferingForToolCall = hasTools;
        var firstNonWhitespaceDecided = false;

        await foreach (var token in GenerateChatAsync(messageList, options, cancellationToken))
        {
            if (bufferingForToolCall)
            {
                buffer.Append(token);

                if (!firstNonWhitespaceDecided)
                {
                    var probe = buffer.ToString().AsSpan().TrimStart();
                    if (probe.Length > 0)
                    {
                        firstNonWhitespaceDecided = true;
                        if (probe[0] != '{')
                        {
                            // Plain text response — flush the buffer and continue streaming.
                            bufferingForToolCall = false;
                            yield return new ChatStreamChunk { Text = buffer.ToString() };
                            buffer.Clear();
                        }
                    }
                }
            }
            else
            {
                yield return new ChatStreamChunk { Text = token };
            }
        }

        if (bufferingForToolCall && buffer.Length > 0)
        {
            var fullText = buffer.ToString();
            var toolCalls = ToolCallTextParser.TryParse(fullText);
            if (toolCalls is { Count: > 0 })
            {
                // Validate against advertised tool names — Phi-4-mini and similar
                // small ONNX models occasionally hallucinate tool names that do
                // not exist in the request's tools array (e.g. invented
                // <c>search_memoized_documents</c> when the legitimate tool is
                // <c>search_knowledge</c>). Forwarding a hallucinated name to
                // the upstream FunctionInvokingChatClient triggers an infinite
                // retry loop because the dispatcher cannot resolve the name and
                // re-prompts the model. Drop unknown names before they leave
                // the generator boundary so the caller surfaces the assistant
                // text instead. Sprint-RR1 RR-N evidence (2026-05-02).
                var validNames = options.Tools is { Count: > 0 } toolDefs
                    ? new HashSet<string>(toolDefs.Select(t => t.Name), StringComparer.Ordinal)
                    : null;

                var deltas = new List<ChatToolCallDelta>(toolCalls.Count);
                for (var i = 0; i < toolCalls.Count; i++)
                {
                    var call = toolCalls[i];
                    if (validNames is not null && !string.IsNullOrEmpty(call.FunctionName)
                        && !validNames.Contains(call.FunctionName))
                    {
                        continue;
                    }
                    deltas.Add(new ChatToolCallDelta
                    {
                        Index = deltas.Count,
                        Id = call.Id,
                        Name = call.FunctionName,
                        Arguments = call.Arguments
                    });
                }

                if (deltas.Count > 0)
                {
                    yield return new ChatStreamChunk
                    {
                        ToolCalls = deltas,
                        FinishReason = "tool_calls"
                    };
                    yield break;
                }
            }

            // Either nothing parsed, or every parsed call referenced an unknown
            // tool — surface the raw text so the user still sees what the model
            // produced (the assistant message often contains useful prose mixed
            // with the malformed tool envelope).
            yield return new ChatStreamChunk { Text = fullText };
        }

        yield return new ChatStreamChunk { FinishReason = "stop" };
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
        _resolvedProvider.ToString())
    {
        Diagnostics = _diagnostics
    };

    /// <inheritdoc />
    public Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(text))
            return Task.FromResult(0);

        using var sequences = _tokenizer.Encode(text);
        var tokenCount = sequences[0].Length;
        return Task.FromResult(tokenCount);
    }

    /// <inheritdoc />
    public Task<int> CountTokensAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.FormatPrompt(messages);
        return CountTokensAsync(prompt, cancellationToken);
    }

    private GeneratorParams CreateGeneratorParams(GenerationOptions options, int effectiveMaxLength)
    {
        var generatorParams = new GeneratorParams(_model);

        // max_length includes both input and output tokens
        generatorParams.SetSearchOption("max_length", effectiveMaxLength);
        generatorParams.SetSearchOption("temperature", options.Temperature);
        generatorParams.SetSearchOption("top_p", options.TopP);
        generatorParams.SetSearchOption("top_k", options.TopK);
        generatorParams.SetSearchOption("repetition_penalty", options.RepetitionPenalty);
        // no_repeat_ngram_size is an ONNX Runtime GenAI search option (not supported by llama-server).
        // Applied only when set so default behavior is unchanged. Hard-blocks any n-gram of this size
        // from recurring — a standard verbatim-loop defense.
        if (options.NoRepeatNgramSize is { } noRepeatNgram && noRepeatNgram > 0)
            generatorParams.SetSearchOption("no_repeat_ngram_size", noRepeatNgram);
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
        var configStops = GenAiConfigReader.ReadStopSequences(_modelPath, _configBasePath);
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

    /// <summary>
    /// Disposes the model and releases all resources.
    /// </summary>
    /// <remarks>
    /// NOTE: Due to known OGA memory leak issues (particularly with DirectML), you may see
    /// "OGA Error: N instances of struct X were leaked" warnings even after proper disposal.
    /// This is an upstream issue being tracked at:
    /// - https://github.com/microsoft/onnxruntime-genai/issues/590
    /// - https://github.com/microsoft/onnxruntime-genai/issues/1677
    /// These warnings do not affect functionality.
    /// </remarks>
    public ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return ValueTask.CompletedTask;
        }

        _disposed = true;

        // Dispose in reverse order of creation to avoid orphaned references
        _concurrencyLimiter.Dispose();
        _tokenizer.Dispose();
        _model.Dispose();

        return ValueTask.CompletedTask;
    }
}
