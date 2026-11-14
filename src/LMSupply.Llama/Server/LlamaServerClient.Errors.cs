using System.Text.Json;
using System.Text.RegularExpressions;
using LMSupply.Exceptions;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerClient
{
    /// <summary>
    /// Ensures the HTTP response is successful, converting context overflow errors
    /// to <see cref="ContextLengthExceededException"/>.
    /// </summary>
    private static async Task EnsureSuccessOrThrowContextExceptionAsync(
        HttpResponseMessage response,
        int maxContextLength,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
            return;

        // Read error body for context overflow detection
        var errorBody = await response.Content.ReadAsStringAsync(cancellationToken);

        ThrowBackendFailure(response.StatusCode, errorBody, maxContextLength);
    }

    /// <summary>
    /// The exception for a failure the server reported: <see cref="ContextLengthExceededException"/> for a context or
    /// batch overflow, else <see cref="InferenceBackendException"/> carrying the status and the server's own body.
    /// Never returns. One classification for a refused request and for a failure that arrives inside a stream.
    /// </summary>
    private static void ThrowBackendFailure(System.Net.HttpStatusCode statusCode, string errorBody, int maxContextLength)
    {
        if (statusCode == System.Net.HttpStatusCode.BadRequest &&
            IsContextOverflowError(errorBody))
        {
            var (promptTokens, serverContext) = ReadOverflowCounts(errorBody);
            throw new ContextLengthExceededException(promptTokens, serverContext ?? maxContextLength);
        }

        // An embedding or rerank input longer than the physical batch is refused with a 500. The embedders size that
        // batch to the model's sequence length, so it is the input limit the caller has to stay under.
        if (ReadPhysicalBatchOverflow(errorBody) is { } overflow)
            throw new ContextLengthExceededException(overflow.InputTokens, overflow.BatchSize ?? maxContextLength);

        // Not a recognized context-overflow shape — surface the body we already read instead
        // of discarding it and falling through to EnsureSuccessStatusCode()'s generic message.
        throw new InferenceBackendException(statusCode, errorBody);
    }

    /// <summary>
    /// Throws when a streamed event carries an <c>error</c> object. llama-server answers a failure that happens after the
    /// response headers went out (<c>Compute error.</c> when the machine runs short of memory) as an ordinary
    /// <c>data:</c> event <c>{"error":{"code":500,"message":...,"type":...}}</c>, not as an HTTP status; a reader that
    /// looks only for <c>choices</c> ends the stream as if the model had said nothing. The exception is the one the same
    /// failure gets without streaming: the <c>code</c> as the status (500 when absent or not a status) and the event's JSON
    /// as the body.
    /// </summary>
    private static void ThrowIfStreamedError(System.Text.Json.Nodes.JsonNode? error, string data, int maxContextLength)
    {
        if (error is null)
            return;

        var status = error is System.Text.Json.Nodes.JsonObject fields
            && fields["code"] is System.Text.Json.Nodes.JsonValue code
            && code.TryGetValue<int>(out var number)
            && number is >= 400 and <= 599
                ? (System.Net.HttpStatusCode)number
                : System.Net.HttpStatusCode.InternalServerError;

        ThrowBackendFailure(status, data, maxContextLength);
    }

    /// <summary>
    /// The prompt's token count and the server's context from an overflow refusal, when the server states them —
    /// llama-server's <c>exceed_context_size_error</c> carries <c>n_prompt_tokens</c> and <c>n_ctx</c> in its
    /// <c>error</c> object. Either is null when absent or the body is not JSON.
    /// </summary>
    internal static (int? PromptTokens, int? Context) ReadOverflowCounts(string errorBody)
    {
        try
        {
            using var doc = JsonDocument.Parse(errorBody);
            var root = doc.RootElement;
            var error = root.ValueKind == JsonValueKind.Object && root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.Object
                ? e
                : root;
            return (PositiveInt(error, "n_prompt_tokens"), PositiveInt(error, "n_ctx"));
        }
        catch (JsonException)
        {
            return (null, null);
        }

        static int? PositiveInt(JsonElement element, string name) =>
            element.ValueKind == JsonValueKind.Object
            && element.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.Number
            && value.TryGetInt32(out var number)
            && number > 0
                ? number
                : null;
    }

    /// <summary>
    /// The input's token count and the physical batch size from llama-server's refusal of an input that does not fit one
    /// batch — «input (N tokens) is too large to process. increase the physical batch size (current batch size: M)» —
    /// which a non-causal model (an embedder, a reranker) must process whole. Null when the body is not that refusal.
    /// </summary>
    internal static (int InputTokens, int? BatchSize)? ReadPhysicalBatchOverflow(string errorBody)
    {
        var match = PhysicalBatchOverflow().Match(errorBody);
        if (!match.Success || !int.TryParse(match.Groups["tokens"].ValueSpan, out var tokens))
            return null;

        int? batch = int.TryParse(match.Groups["batch"].ValueSpan, out var size) && size > 0 ? size : null;
        return (tokens, batch);
    }

    [GeneratedRegex(@"input \((?<tokens>\d+) tokens\) is too large to process(?:.*?current batch size: (?<batch>\d+))?", RegexOptions.IgnoreCase | RegexOptions.Singleline)]
    private static partial Regex PhysicalBatchOverflow();

    private static bool IsContextOverflowError(string errorBody)
    {
        // llama-server returns errors like:
        // "the prompt is too long" / "input exceeds context" / "context length exceeded"
        return errorBody.Contains("too long", StringComparison.OrdinalIgnoreCase) ||
               errorBody.Contains("context", StringComparison.OrdinalIgnoreCase) &&
               (errorBody.Contains("exceed", StringComparison.OrdinalIgnoreCase) ||
                errorBody.Contains("overflow", StringComparison.OrdinalIgnoreCase) ||
                errorBody.Contains("too large", StringComparison.OrdinalIgnoreCase));
    }
}
