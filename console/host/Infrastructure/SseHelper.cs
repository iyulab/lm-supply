using System.Text.Json;
using LMSupply.Console.Host.Models.OpenAI;

namespace LMSupply.Console.Host.Infrastructure;

/// <summary>
/// Server-Sent Events helper for OpenAI-compatible streaming
/// </summary>
public static class SseHelper
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Stream chat completions in OpenAI format
    /// </summary>
    public static async Task StreamChatCompletionAsync(
        HttpContext context,
        string model,
        IAsyncEnumerable<string> tokens,
        CancellationToken cancellationToken = default)
    {
        await SetSseHeadersAsync(context);

        var id = $"chatcmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        // First chunk with role
        var firstChunk = new ChatCompletionChunk
        {
            Id = id,
            Model = model,
            Created = created,
            Choices =
            [
                new ChatCompletionChunkChoice
                {
                    Index = 0,
                    Delta = new ChatCompletionDelta { Role = "assistant" }
                }
            ]
        };
        await WriteDataAsync(context, firstChunk, cancellationToken);

        // Content chunks
        await foreach (var token in tokens.WithCancellation(cancellationToken))
        {
            var chunk = new ChatCompletionChunk
            {
                Id = id,
                Model = model,
                Created = created,
                Choices =
                [
                    new ChatCompletionChunkChoice
                    {
                        Index = 0,
                        Delta = new ChatCompletionDelta { Content = token }
                    }
                ]
            };
            await WriteDataAsync(context, chunk, cancellationToken);
        }

        // Final chunk with finish_reason
        var finalChunk = new ChatCompletionChunk
        {
            Id = id,
            Model = model,
            Created = created,
            Choices =
            [
                new ChatCompletionChunkChoice
                {
                    Index = 0,
                    Delta = new ChatCompletionDelta(),
                    FinishReason = "stop"
                }
            ]
        };
        await WriteDataAsync(context, finalChunk, cancellationToken);

        await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        await context.Response.Body.FlushAsync(cancellationToken);
    }

    /// <summary>
    /// Stream any object as SSE
    /// </summary>
    public static async Task StreamAsync<T>(
        HttpContext context,
        IAsyncEnumerable<T> source,
        CancellationToken cancellationToken = default)
    {
        await SetSseHeadersAsync(context);

        await foreach (var item in source.WithCancellation(cancellationToken))
        {
            await WriteDataAsync(context, item, cancellationToken);
        }

        await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        await context.Response.Body.FlushAsync(cancellationToken);
    }

    private static async Task SetSseHeadersAsync(HttpContext context)
    {
        var origin = context.Request.Headers.Origin.ToString();
        if (!string.IsNullOrEmpty(origin))
        {
            context.Response.Headers.AccessControlAllowOrigin = origin;
            context.Response.Headers.AccessControlAllowCredentials = "true";
        }

        context.Response.Headers.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers.Connection = "keep-alive";

        // Start the response to prevent CORS middleware from trying to add headers later
        await context.Response.StartAsync();
    }

    private static async Task WriteDataAsync<T>(HttpContext context, T data, CancellationToken ct)
    {
        var json = JsonSerializer.Serialize(data, JsonOptions);
        await context.Response.WriteAsync($"data: {json}\n\n", ct);
        await context.Response.Body.FlushAsync(ct);
    }
}
