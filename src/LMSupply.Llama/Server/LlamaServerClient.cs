using System.Diagnostics;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Exceptions;

namespace LMSupply.Llama.Server;

/// <summary>
/// OpenAI-compatible HTTP client for llama-server.
/// </summary>
public sealed partial class LlamaServerClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly bool _ownsHttpClient;
    private readonly int _maxContextLength;
    private readonly string? _apiKey;
    private readonly TimeSpan? _requestTimeout;
    private Func<ServerState>? _serverState;

    /// <summary>The wire options every request and response uses (snake_case, nulls omitted, source-generated metadata).</summary>
    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        TypeInfoResolver = LlamaJsonContext.Default
    };

    /// <summary>
    /// Creates a new client for the specified server URL.
    /// </summary>
    /// <param name="baseUrl">llama-server base URL.</param>
    /// <param name="httpClient">Optional shared HttpClient. If null, a new one is created and owned.
    /// <paramref name="requestTimeout"/> only applies to the client created in that case — a caller
    /// supplying their own <paramref name="httpClient"/> owns its timeout too.</param>
    /// <param name="maxContextLength">Model context window size — carried in ContextLengthExceededException on overflow.</param>
    /// <param name="requestTimeout">Limit on each request this client sends when it owns its
    /// HttpClient. Defaults to 5 minutes, not .NET's 100-second default — local/CPU-bound inference,
    /// especially a multi-step tool-calling loop where each round's prompt grows with prior tool
    /// results, routinely exceeds 100 seconds per completion on hardware without a GPU.
    /// <see cref="Timeout.InfiniteTimeSpan"/> means no limit. It covers the same span
    /// <see cref="HttpClient.Timeout"/> would — up to the response headers for streamed calls, the
    /// whole body otherwise — and expiry throws the same shape (<see cref="TaskCanceledException"/>
    /// with an inner <see cref="TimeoutException"/>). Ignored when <paramref name="httpClient"/> is
    /// supplied.</param>
    /// <param name="apiKey">Key the server was started with (<see cref="LlamaServerProcess.ApiKey"/>); sent as
    /// <c>Authorization: Bearer</c> on every request. <see langword="null"/> sends none.</param>
    public LlamaServerClient(
        string baseUrl,
        HttpClient? httpClient = null,
        int maxContextLength = 0,
        TimeSpan? requestTimeout = null,
        string? apiKey = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _maxContextLength = maxContextLength;
        _apiKey = string.IsNullOrEmpty(apiKey) ? null : apiKey;

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            // The limit is applied per request (SendWithTimeoutAsync), not on the HttpClient, so that
            // views over this client (WithRequestTimeout) can carry their own limit on the same
            // connection pool.
            _httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
            _ownsHttpClient = true;
            _requestTimeout = ValidateRequestTimeout(requestTimeout ?? LlamaServerConfig.DefaultRequestTimeout);
        }
    }

    private LlamaServerClient(LlamaServerClient owner, TimeSpan requestTimeout)
    {
        _baseUrl = owner._baseUrl;
        _maxContextLength = owner._maxContextLength;
        _apiKey = owner._apiKey;
        _httpClient = owner._httpClient;
        _ownsHttpClient = false;
        _requestTimeout = ValidateRequestTimeout(requestTimeout);
        _serverState = owner._serverState;
    }

    /// <summary>
    /// A client for the same server that shares this client's connections but limits each request
    /// to <paramref name="requestTimeout"/>. Disposing the view does not dispose the connections.
    /// </summary>
    /// <remarks>
    /// A pooled server is shared by every caller that loads the same model; each caller's lease
    /// gets its own view, so one caller's limit never becomes another's.
    /// </remarks>
    internal LlamaServerClient WithRequestTimeout(TimeSpan requestTimeout) => new(this, requestTimeout);

    /// <summary>The per-request limit this client applies; <see langword="null"/> when the caller supplied the HttpClient (its own timeout governs).</summary>
    internal TimeSpan? RequestTimeout => _requestTimeout;

    private static TimeSpan ValidateRequestTimeout(TimeSpan value) =>
        value == Timeout.InfiniteTimeSpan || value > TimeSpan.Zero
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value,
                "Request timeout must be positive or Timeout.InfiniteTimeSpan.");

    private Task<HttpResponseMessage> PostWithTimeoutAsync(string url, HttpContent content, CancellationToken cancellationToken) =>
        SendWithTimeoutAsync(
            new HttpRequestMessage(HttpMethod.Post, url) { Content = content },
            HttpCompletionOption.ResponseContentRead,
            cancellationToken);

    /// <summary>
    /// Sends with this client's request limit, over exactly the span <see cref="HttpClient.Timeout"/>
    /// covers, and reports expiry the way HttpClient does.
    /// </summary>
    private async Task<HttpResponseMessage> SendWithTimeoutAsync(
        HttpRequestMessage request,
        HttpCompletionOption completionOption,
        CancellationToken cancellationToken)
    {
        // Per request, not DefaultRequestHeaders: a caller-supplied HttpClient is not ours to change.
        if (_apiKey is not null)
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        if (_requestTimeout is not { } limit || limit == Timeout.InfiniteTimeSpan)
            return await _httpClient.SendAsync(request, completionOption, cancellationToken);

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(limit);
        try
        {
            return await _httpClient.SendAsync(request, completionOption, cts.Token);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested && cts.IsCancellationRequested)
        {
            var message = $"The request to llama-server was canceled because the request timeout of {limit.TotalSeconds:0.###} seconds elapsed.";
            throw new TaskCanceledException(message, new TimeoutException(message, ex));
        }
    }

    /// <summary>
    /// Checks if the server is healthy.
    /// </summary>
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await SendWithTimeoutAsync(
                new HttpRequestMessage(HttpMethod.Get, $"{_baseUrl}/health"),
                HttpCompletionOption.ResponseContentRead,
                cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[LlamaServerClient] Health check failed: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Ties this client to the server process it talks to, so a request that fails because that process exited reports
    /// <see cref="InferenceBackendExitedException"/> (exit code and the server's last output) instead of the bare
    /// transport failure. Views made afterwards share it.
    /// </summary>
    internal void AttachServer(LlamaServerProcess server) =>
        AttachServerState(() => new ServerState(server.IsRunning, server.ExitCode, server.RecentLog));

    /// <summary>The seam behind <see cref="AttachServer"/>: reads the server's state when a request fails.</summary>
    internal void AttachServerState(Func<ServerState> state) => _serverState = state;

    /// <summary>What a failed request needs to know about its server.</summary>
    internal readonly record struct ServerState(bool IsRunning, int? ExitCode, string RecentLog);

    /// <summary>
    /// The exception to throw for <paramref name="failure"/>: <see cref="InferenceBackendExitedException"/> when the
    /// attached server has exited, otherwise <see langword="null"/> (rethrow as is). The process may still be dying when
    /// the connection is refused or cut, so its exit gets a moment to register.
    /// </summary>
    private async Task<Exception?> TranslateFailureAsync(Exception failure)
    {
        if (_serverState is not { } read || failure is not (HttpRequestException or IOException))
            return null;

        var state = read();
        for (var i = 0; i < 20 && state.IsRunning; i++)
        {
            await Task.Delay(100).ConfigureAwait(false);
            state = read();
        }

        return state.IsRunning ? null : new InferenceBackendExitedException(state.ExitCode, state.RecentLog, failure);
    }

    private async Task<T> GuardAsync<T>(Func<Task<T>> call)
    {
        try
        {
            return await call().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            if (await TranslateFailureAsync(ex).ConfigureAwait(false) is { } translated)
                throw translated;
            throw;
        }
    }

    private async IAsyncEnumerable<T> GuardStream<T>(
        IAsyncEnumerable<T> source,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var enumerator = source.GetAsyncEnumerator(cancellationToken);
        try
        {
            while (true)
            {
                T item;
                try
                {
                    if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
                        yield break;
                    item = enumerator.Current;
                }
                catch (Exception ex)
                {
                    if (await TranslateFailureAsync(ex).ConfigureAwait(false) is { } translated)
                        throw translated;
                    throw;
                }

                yield return item;
            }
        }
        finally
        {
            await enumerator.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <inheritdoc cref="GenerateChatCoreAsync"/>
    public IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardStream(GenerateChatCoreAsync(messages, options, cancellationToken), cancellationToken);

    /// <inheritdoc cref="GenerateChatStreamCoreAsync"/>
    public IAsyncEnumerable<ChatStreamData> GenerateChatStreamAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardStream(GenerateChatStreamCoreAsync(messages, options, cancellationToken), cancellationToken);

    /// <inheritdoc cref="GenerateChatWithToolsCoreAsync"/>
    public Task<ChatCompletionFullResponse> GenerateChatWithToolsAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardAsync(() => GenerateChatWithToolsCoreAsync(messages, options, cancellationToken));

    /// <inheritdoc cref="GenerateStreamCoreAsync"/>
    public IAsyncEnumerable<CompletionStreamData> GenerateStreamAsync(
        string prompt,
        CompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardStream(GenerateStreamCoreAsync(prompt, options, cancellationToken), cancellationToken);

    /// <inheritdoc cref="CountTokensCoreAsync"/>
    public Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default) =>
        GuardAsync(() => CountTokensCoreAsync(text, cancellationToken));

    /// <inheritdoc cref="TokenizeCoreAsync"/>
    public Task<IReadOnlyList<int>> TokenizeAsync(string text, bool addSpecial = false, CancellationToken cancellationToken = default) =>
        GuardAsync(() => TokenizeCoreAsync(text, addSpecial, cancellationToken));

    /// <inheritdoc cref="CountChatPromptTokensCoreAsync"/>
    public Task<int?> CountChatPromptTokensAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardAsync(() => CountChatPromptTokensCoreAsync(messages, options, cancellationToken));

    /// <inheritdoc cref="GenerateEmbeddingsBatchCoreAsync"/>
    public Task<float[][]> GenerateEmbeddingsBatchAsync(IReadOnlyList<string> inputs, CancellationToken cancellationToken = default) =>
        GuardAsync(() => GenerateEmbeddingsBatchCoreAsync(inputs, cancellationToken));

    /// <inheritdoc cref="RerankCoreAsync"/>
    public Task<IReadOnlyList<RerankResult>> RerankAsync(
        string query,
        IReadOnlyList<string> documents,
        int topN = 10,
        CancellationToken cancellationToken = default) =>
        GuardAsync(() => RerankCoreAsync(query, documents, topN, cancellationToken));

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }
}
